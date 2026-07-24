"""Write a "_processed" ROS2 bag: every original message plus the estimated
GT trajectory, so downstream consumers can work from one self-contained bag
instead of the original bag plus a separate TUM file.

Every original message is copied verbatim (raw serialized bytes, no
decode/re-encode) to avoid any risk of altering source data. Two new
topics are added:

- A `geometry_msgs/msg/PoseStamped` per registered frame, at that frame's
  *original* capture timestamp (looked up from the extraction-time
  `rgb.csv`, since trajectory entries only carry `frame_id`, not the real
  ROS timestamp) -- mirrors how odometry/tf are normally published, for
  per-frame lookup/sync with the rest of the bag.
- A single `nav_msgs/msg/Path` summary message (all poses in one message),
  stamped at the last pose's time -- for direct visualization in
  rviz2/foxglove without needing per-message playback.
"""

import csv
from pathlib import Path
from typing import Any

from rosbags.rosbag2 import Reader as ReaderV2, Writer as WriterV2, StoragePlugin
from rosbags.interfaces import MessageDefinitionFormat
from rosbags.typesys import get_typestore, get_types_from_idl, get_types_from_msg
from rosbags.typesys.stores import Stores

from colmap_rgbd_gt.utils.transforms import rotation_matrix_to_quaternion
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def _load_frame_timestamps(rgb_csv_path: Path) -> dict[int, int]:
    """frame_id (parsed from filename) -> original capture timestamp_ns."""
    mapping: dict[int, int] = {}
    with open(rgb_csv_path) as f:
        for row in csv.DictReader(f):
            frame_id = int(Path(row["filename"]).stem)
            mapping[frame_id] = int(row["timestamp_ns"])
    return mapping


def write_processed_bag(
    original_bag_path: Path,
    trajectory: list[dict[str, Any]],
    rgb_csv_path: Path,
    output_bag_path: Path,
    pose_topic: str = "/gt/pose",
    path_topic: str = "/gt/path",
    frame_id: str = "map",
    storage_plugin: StoragePlugin = StoragePlugin.MCAP,
) -> int:
    """Write `output_bag_path` = every message in `original_bag_path` plus
    the GT trajectory on `pose_topic`/`path_topic`.

    Args:
        original_bag_path: source ROS2 bag (mcap or db3).
        trajectory: c2w trajectory entries (as from
            `colmap.pose_extract.extract_trajectory`/`scale_trajectory`),
            each with `frame_id`, `R` (3x3), `t` (3,).
        rgb_csv_path: workspace's `timestamps/rgb.csv`, used to map each
            entry's `frame_id` back to its real capture timestamp.
        output_bag_path: destination bag path. Caller decides where this
            lives (e.g. alongside the source bag, or in the workspace) --
            this function only writes to the given path.
        storage_plugin: output bag format; MCAP by default (matches the
            most common source format this pipeline reads).

    Returns:
        Number of original messages copied (for logging/verification).
    """
    original_bag_path = Path(original_bag_path)
    output_bag_path = Path(output_bag_path)
    output_bag_path.parent.mkdir(parents=True, exist_ok=True)

    frame_timestamps = _load_frame_timestamps(rgb_csv_path)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    PoseStamped = typestore.types["geometry_msgs/msg/PoseStamped"]
    NavPath = typestore.types["nav_msgs/msg/Path"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    Pose = typestore.types["geometry_msgs/msg/Pose"]
    Point = typestore.types["geometry_msgs/msg/Point"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]

    def _ros_time(ts_ns: int):
        return Time(sec=ts_ns // 1_000_000_000, nanosec=ts_ns % 1_000_000_000)

    def _pose_stamped(ts_ns: int, t, q_xyzw):
        return PoseStamped(
            header=Header(stamp=_ros_time(ts_ns), frame_id=frame_id),
            pose=Pose(
                position=Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                orientation=Quaternion(
                    x=float(q_xyzw[0]), y=float(q_xyzw[1]),
                    z=float(q_xyzw[2]), w=float(q_xyzw[3]),
                ),
            ),
        )

    poses: list[tuple[int, Any, Any]] = []
    missing_timestamps = 0
    for entry in sorted(trajectory, key=lambda e: e["frame_id"]):
        ts_ns = frame_timestamps.get(entry["frame_id"])
        if ts_ns is None:
            missing_timestamps += 1
            continue
        q_xyzw = rotation_matrix_to_quaternion(entry["R"])
        poses.append((ts_ns, entry["t"], q_xyzw))

    if missing_timestamps:
        logger.warning(
            f"{missing_timestamps} trajectory frame(s) had no matching "
            f"entry in {rgb_csv_path}; skipped"
        )
    if not poses:
        logger.warning("No trajectory poses with matching timestamps -- GT topics will be empty")

    with ReaderV2(original_bag_path) as reader, \
         WriterV2(output_bag_path, version=9, storage_plugin=storage_plugin) as writer:

        conn_map = {}
        for conn in reader.connections:
            # Pass the ORIGINAL connection's own msgdef/digest through
            # rather than asking `typestore` to look up conn.msgtype --
            # bags commonly carry custom message types (e.g. a robot
            # vendor's own msgs package) that a generic typestore has
            # never heard of and can't generate a definition for, which
            # would otherwise raise TypesysError and abort the whole copy.
            if conn.digest:
                conn_map[conn.id] = writer.add_connection(
                    conn.topic, conn.msgtype,
                    msgdef=conn.msgdef.data, rihs01=conn.digest,
                )
            else:
                # Some source bags (older recordings, or ones written
                # without type_description_hash support) carry a msgdef
                # but no rihs01 digest. add_connection() requires a
                # truthy rihs01 in that case, so register the type's own
                # msgdef into our typestore and let it compute the hash,
                # instead of relying on the (possibly absent) generic
                # typestore lookup for conn.msgtype.
                if conn.msgtype not in typestore.types:
                    if conn.msgdef.format == MessageDefinitionFormat.IDL:
                        types = get_types_from_idl(conn.msgdef.data)
                    else:
                        types = get_types_from_msg(conn.msgdef.data, conn.msgtype)
                    typestore.register(types)
                conn_map[conn.id] = writer.add_connection(
                    conn.topic, conn.msgtype, typestore=typestore,
                )

        pose_conn = writer.add_connection(pose_topic, PoseStamped.__msgtype__, typestore=typestore)
        path_conn = writer.add_connection(path_topic, NavPath.__msgtype__, typestore=typestore)

        n_copied = 0
        for conn, timestamp, rawdata in reader.messages():
            writer.write(conn_map[conn.id], timestamp, rawdata)
            n_copied += 1

        for ts_ns, t, q_xyzw in poses:
            msg = _pose_stamped(ts_ns, t, q_xyzw)
            writer.write(pose_conn, ts_ns, typestore.serialize_cdr(msg, PoseStamped.__msgtype__))

        if poses:
            path_msg = NavPath(
                header=Header(stamp=_ros_time(poses[-1][0]), frame_id=frame_id),
                poses=[_pose_stamped(ts_ns, t, q_xyzw) for ts_ns, t, q_xyzw in poses],
            )
            writer.write(
                path_conn, poses[-1][0],
                typestore.serialize_cdr(path_msg, NavPath.__msgtype__),
            )

    logger.info(
        f"Wrote processed bag to {output_bag_path}: {n_copied} original messages copied, "
        f"{len(poses)} GT poses added on {pose_topic}, path summary on {path_topic}"
    )
    return n_copied
