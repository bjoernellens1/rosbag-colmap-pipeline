"""Bag file reading using rosbags library."""

from pathlib import Path
from typing import Any, Iterator
import numpy as np
from rosbags.rosbag1 import Reader as ReaderV1
from rosbags.rosbag2 import Reader as ReaderV2
from rosbags.typesys.stores.ros1_noetic import (
    sensor_msgs__msg__Image as ImageV1,
    sensor_msgs__msg__CompressedImage as CompressedImageV1,
    sensor_msgs__msg__CameraInfo as CameraInfoV1,
    sensor_msgs__msg__PointCloud2 as PointCloud2V1,
)
from rosbags.typesys.stores.ros2_humble import (
    sensor_msgs__msg__Image as ImageV2,
    sensor_msgs__msg__CompressedImage as CompressedImageV2,
    sensor_msgs__msg__CameraInfo as CameraInfoV2,
    sensor_msgs__msg__PointCloud2 as PointCloud2V2,
)

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


class BagReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._reader = None
        self._is_ros2 = None
        self._storage_id = None
        self._open()

    def _open(self):
        suffix = self.path.suffix.lower()
        if suffix == ".bag":
            self._reader = ReaderV1(self.path)
            self._is_ros2 = False
            self._storage_id = "ros1"
        elif suffix in (".db3", ".mcap"):
            self._reader = ReaderV2(self.path)
            self._is_ros2 = True
            self._storage_id = "sqlite3" if suffix == ".db3" else "mcap"
        else:
            raise ValueError(f"Unknown bag format: {suffix}")
        self._reader.open()

    @property
    def storage_id(self) -> str:
        return self._storage_id

    @property
    def is_ros2(self) -> bool:
        return self._is_ros2

    def close(self):
        if self._reader:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_topics(self) -> list[str]:
        return [conn.topic for conn in self._reader.connections]

    def get_connections(self) -> list:
        return list(self._reader.connections)

    def get_message_count(self, topic: str) -> int:
        for conn in self._reader.connections:
            if conn.topic == topic:
                return conn.msgcount
        return 0

    def get_time_range(self) -> tuple[int, int]:
        start = self._reader.start_time
        end = self._reader.end_time
        return start, end

    def get_messages(self, topic: str) -> Iterator[tuple[int, Any]]:
        from rosbags.typesys import get_typestore
        from rosbags.typesys.stores import Stores

        typestore_name = Stores.ROS2_HUMBLE if self._is_ros2 else Stores.ROS1_NOETIC
        typestore = get_typestore(typestore_name)

        connections = [c for c in self._reader.connections if c.topic == topic]
        warned_trailing_bytes = False
        for conn, timestamp, rawdata in self._reader.messages(connections=connections):
            if self._is_ros2:
                msg = typestore.deserialize_cdr(ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
            else:
                msg, trimmed = _deserialize_ros1_lenient(typestore, rawdata, conn.msgtype)
                if trimmed and not warned_trailing_bytes:
                    logger.warning(
                        f"Topic {topic!r}: message payload has {trimmed} trailing "
                        f"byte(s) beyond what {conn.msgtype!r}'s schema declares "
                        "(deserialized by trimming them; not re-warning per message). "
                        "Seen on bags recorded directly by librealsense's native "
                        "bag-writer (e.g. realsense-viewer's 'record' feature), which "
                        "appends a few unschematized bytes per sensor_msgs/Image "
                        "message -- not a standard rclpy/roscpp `rosbag record` "
                        "capture, so `rosbags`' strict "
                        "`assert pos == len(rawdata)` check in deserialize_ros1() "
                        "fails on the untrimmed payload even though the bag's own "
                        "embedded message definition (conn.msgdef) matches the "
                        "standard schema byte-for-byte -- this is a wire-level "
                        "writer quirk, not a schema mismatch, so re-registering "
                        "types from conn.msgdef would not have fixed it."
                    )
                    warned_trailing_bytes = True
            yield timestamp, msg


def ros1_to_cdr(data: bytes, msgtype: str) -> bytes:
    return data


# Max trailing bytes we'll silently trim before giving up and re-raising the
# real error. Root-caused 2026-08-02 on a RealSense-Viewer-native-recorded
# .bag (topics like /device_0/sensor_1/Color_0/image/data): every
# sensor_msgs/Image message carried exactly 4 unschematized trailing bytes
# (verified across both the Color and Depth topics, 3 messages each --
# rawdata was consistently header+data_field length + 4). `rosbags`'
# deserialize_ros1() hard-asserts `pos == len(rawdata)` after consuming the
# schema's own fields, so any writer that appends even a few bytes the
# schema doesn't declare fails outright rather than skipping them. 8 is a
# small safety margin above the one concretely observed case (4), not itself
# empirically verified beyond 4 -- if a future bag needs more, raise this or
# make it configurable rather than assuming this covers every writer quirk.
_MAX_TRAILING_TRIM = 8


def _deserialize_ros1_lenient(typestore, rawdata: bytes, msgtype: str):
    """Deserialize a ROS1 message, tolerating a small number of trailing
    bytes the message's own schema doesn't account for.

    `rosbags.typesys.store.Typestore.deserialize_ros1()` internally asserts
    the schema consumes the ENTIRE raw payload (`assert pos ==
    len(rawdata)`) -- appropriate for a compliant `rosbag record` capture,
    but too strict for bags written by tools with their own non-standard
    serializer (see `_MAX_TRAILING_TRIM`'s docstring for the concrete case
    this was built for). Retries with progressively fewer trailing bytes
    rather than assuming a fixed offset, since the exact slack is a property
    of the writer, not something to hardcode as a universal constant.

    Returns ``(message, trimmed_byte_count)``.
    """
    try:
        return typestore.deserialize_ros1(rawdata, msgtype), 0
    except AssertionError:
        pass
    for trim in range(1, _MAX_TRAILING_TRIM + 1):
        try:
            return typestore.deserialize_ros1(rawdata[:-trim], msgtype), trim
        except AssertionError:
            continue
    # No trim in range fixed it -- re-raise the original, untrimmed failure
    # so the caller sees the real error instead of a confusing "trim=8 also
    # failed" message.
    typestore.deserialize_ros1(rawdata, msgtype)
    raise AssertionError("unreachable: deserialize_ros1 should have raised above")
