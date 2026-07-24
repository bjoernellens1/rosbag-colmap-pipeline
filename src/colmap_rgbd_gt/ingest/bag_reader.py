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
        for conn, timestamp, rawdata in self._reader.messages(connections=connections):
            if self._is_ros2:
                msg = typestore.deserialize_cdr(ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
            else:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            yield timestamp, msg


def ros1_to_cdr(data: bytes, msgtype: str) -> bytes:
    return data
