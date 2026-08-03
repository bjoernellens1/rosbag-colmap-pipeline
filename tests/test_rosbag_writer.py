"""Tests for write_processed_bag (export/rosbag_writer.py)."""

import numpy as np
import pytest

rosbags = pytest.importorskip("rosbags")

from rosbags.rosbag2 import Reader as ReaderV2, Writer as WriterV2, StoragePlugin
from rosbags.typesys import get_typestore
from rosbags.typesys.stores import Stores

from colmap_rgbd_gt.export.rosbag_writer import write_processed_bag


def _write_synthetic_source_bag(path, n_messages=5):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    String = ts.types["std_msgs/msg/String"]
    with WriterV2(path, version=9, storage_plugin=StoragePlugin.MCAP) as w:
        conn = w.add_connection("/chat", String.__msgtype__, typestore=ts)
        for i in range(n_messages):
            msg = String(data=f"hello{i}")
            w.write(conn, i * 1000, ts.serialize_cdr(msg, String.__msgtype__))


def _write_rgb_csv(path, n_frames):
    with open(path, "w") as f:
        f.write("timestamp_ns,filename\n")
        for i in range(n_frames):
            f.write(f"{i * 1000},{i:06d}.png\n")


def test_write_processed_bag_copies_originals_and_adds_gt_topics(tmp_path):
    src_bag = tmp_path / "src.mcap"
    _write_synthetic_source_bag(src_bag, n_messages=5)

    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, n_frames=5)

    trajectory = [
        {"frame_id": i, "R": np.eye(3), "t": np.array([float(i), 0.0, 0.0])}
        for i in range(5)
    ]

    dst_bag = tmp_path / "src_processed.mcap"
    n_copied = write_processed_bag(src_bag, trajectory, rgb_csv, dst_bag)

    assert n_copied == 5
    assert dst_bag.exists()

    with ReaderV2(dst_bag) as r:
        topics = set(r.topics.keys())
        assert "/chat" in topics
        assert "/gt/colmap_pose" in topics
        assert "/gt/path" in topics

        chat_count = 0
        pose_count = 0
        path_count = 0
        for conn, ts_ns, rawdata in r.messages():
            if conn.topic == "/chat":
                chat_count += 1
            elif conn.topic == "/gt/colmap_pose":
                pose_count += 1
            elif conn.topic == "/gt/path":
                path_count += 1

        assert chat_count == 5
        assert pose_count == 5
        assert path_count == 1


def test_write_processed_bag_skips_frames_without_timestamp(tmp_path):
    src_bag = tmp_path / "src.mcap"
    _write_synthetic_source_bag(src_bag, n_messages=2)

    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, n_frames=2)  # only frame_id 0, 1 have timestamps

    trajectory = [
        {"frame_id": i, "R": np.eye(3), "t": np.array([float(i), 0.0, 0.0])}
        for i in range(5)  # 5 trajectory entries, only 2 have timestamps
    ]

    dst_bag = tmp_path / "dst.mcap"
    write_processed_bag(src_bag, trajectory, rgb_csv, dst_bag)

    with ReaderV2(dst_bag) as r:
        pose_count = sum(1 for conn, _, _ in r.messages() if conn.topic == "/gt/colmap_pose")
        assert pose_count == 2


def test_write_processed_bag_preserves_qos_profiles(tmp_path):
    """The original bag's per-topic QoS (reliability/durability/history)
    must survive the copy, not silently fall back to the Writer default --
    see the FIXED 2026-08-03 note in rosbag_writer.py."""
    from rosbags.interfaces import Qos, QosDurability, QosHistory, QosLiveliness, QosReliability, QosTime

    src_bag = tmp_path / "src.mcap"
    ts = get_typestore(Stores.ROS2_HUMBLE)
    String = ts.types["std_msgs/msg/String"]
    custom_qos = Qos(
        history=QosHistory.KEEP_LAST,
        depth=1,
        reliability=QosReliability.BEST_EFFORT,
        durability=QosDurability.VOLATILE,
        deadline=QosTime(sec=0, nsec=0),
        lifespan=QosTime(sec=0, nsec=0),
        liveliness=QosLiveliness.AUTOMATIC,
        liveliness_lease_duration=QosTime(sec=0, nsec=0),
        avoid_ros_namespace_conventions=False,
    )
    with WriterV2(src_bag, version=9, storage_plugin=StoragePlugin.MCAP) as w:
        conn = w.add_connection(
            "/chat", String.__msgtype__, typestore=ts, offered_qos_profiles=[custom_qos],
        )
        w.write(conn, 0, ts.serialize_cdr(String(data="hi"), String.__msgtype__))

    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, n_frames=0)
    dst_bag = tmp_path / "dst.mcap"
    write_processed_bag(src_bag, [], rgb_csv, dst_bag)

    with ReaderV2(dst_bag) as r:
        chat_conn = next(c for c in r.connections if c.topic == "/chat")
        profiles = chat_conn.ext.offered_qos_profiles
        assert len(profiles) == 1
        assert profiles[0].reliability == QosReliability.BEST_EFFORT
        assert profiles[0].durability == QosDurability.VOLATILE
        assert profiles[0].depth == 1


def test_pose_convention_matches_splatograph_consumer(tmp_path):
    """Round-trip a known, non-trivial c2w pose through write_processed_bag
    and decode it EXACTLY the way splatograph's streaming_frames.py decodes
    /camera_pose (position -> translation, quaternion -> rotation matrix,
    no inversion) -- must reproduce the original c2w, not its inverse."""
    src_bag = tmp_path / "src.mcap"
    _write_synthetic_source_bag(src_bag, n_messages=1)
    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, n_frames=1)

    # A rotation that is NOT its own inverse/transpose-equal-to-self in any
    # trivial way, plus a translation clearly off-origin, so a c2w/w2c
    # mixup would be caught (wrong position AND wrong rotation).
    angle = np.pi / 3  # 60 degrees about Z
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    t = np.array([1.5, -2.5, 0.75])
    trajectory = [{"frame_id": 0, "R": R, "t": t}]

    dst_bag = tmp_path / "dst.mcap"
    write_processed_bag(src_bag, trajectory, rgb_csv, dst_bag)

    with ReaderV2(dst_bag) as r:
        pose_msg = None
        for conn, ts_ns, rawdata in r.messages():
            if conn.topic == "/gt/colmap_pose":
                ts = get_typestore(Stores.ROS2_HUMBLE)
                pose_msg = ts.deserialize_cdr(rawdata, conn.msgtype)
        assert pose_msg is not None

        # Exactly splatograph's streaming_frames.py decode (position ->
        # translation, quaternion -> R, direct assembly into c2w, NO
        # inversion) -- see arguments/__init__.py:122-123's /camera_pose
        # default and streaming_frames.py's pose_source="camera_pose" path.
        p = pose_msg.pose.position
        o = pose_msg.pose.orientation
        qx, qy, qz, qw = o.x, o.y, o.z, o.w
        R_decoded = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        c2w_decoded = np.eye(4)
        c2w_decoded[:3, :3] = R_decoded
        c2w_decoded[:3, 3] = [p.x, p.y, p.z]

        np.testing.assert_allclose(c2w_decoded[:3, :3], R, atol=1e-5)
        np.testing.assert_allclose(c2w_decoded[:3, 3], t, atol=1e-5)
        # Sanity: confirm this ISN'T just accidentally passing because R
        # happens to be close to its own inverse -- assert the decoded
        # translation would NOT match if a w2c convention had been used
        # instead (t_w2c = -R^T @ t_c2w for this R/t is a different vector).
        t_w2c = -R.T @ t
        assert not np.allclose(c2w_decoded[:3, 3], t_w2c, atol=1e-3)
