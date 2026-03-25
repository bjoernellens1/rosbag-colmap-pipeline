"""Topic discovery from rosbag files."""

from dataclasses import dataclass
from typing import Any
from pathlib import Path

from colmap_rgbd_gt.ingest.bag_reader import BagReader
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TopicInfo:
    name: str
    msgtype: str
    message_count: int


IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/Image",
}

COMPRESSED_IMAGE_TYPES = {
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/CompressedImage",
}

CAMERA_INFO_TYPES = {
    "sensor_msgs/msg/CameraInfo",
    "sensor_msgs/CameraInfo",
}

POINTCLOUD_TYPES = {
    "sensor_msgs/msg/PointCloud2",
    "sensor_msgs/PointCloud2",
}

DEPTH_ENCODINGS = {"16UC1", "32FC1", "mono16"}


def discover_rgb_topics(reader: BagReader) -> list[TopicInfo]:
    rgb_topics = []
    compressed_topics = []

    for conn in reader.get_connections():
        msgtype = conn.msgtype
        if msgtype in IMAGE_TYPES:
            rgb_topics.append(TopicInfo(
                name=conn.topic,
                msgtype=msgtype,
                message_count=conn.msgcount,
            ))
        elif msgtype in COMPRESSED_IMAGE_TYPES:
            compressed_topics.append(TopicInfo(
                name=conn.topic,
                msgtype=msgtype,
                message_count=conn.msgcount,
            ))

    return rgb_topics + compressed_topics


def discover_depth_topics(reader: BagReader) -> list[TopicInfo]:
    depth_topics = []

    for conn in reader.get_connections():
        msgtype = conn.msgtype
        if msgtype in IMAGE_TYPES:
            topic_lower = conn.topic.lower()
            if "depth" in topic_lower:
                depth_topics.append(TopicInfo(
                    name=conn.topic,
                    msgtype=msgtype,
                    message_count=conn.msgcount,
                ))

    return depth_topics


def discover_camera_info_topics(reader: BagReader) -> list[TopicInfo]:
    info_topics = []

    for conn in reader.get_connections():
        if conn.msgtype in CAMERA_INFO_TYPES:
            info_topics.append(TopicInfo(
                name=conn.topic,
                msgtype=conn.msgtype,
                message_count=conn.msgcount,
            ))

    return info_topics


def discover_pointcloud_topics(reader: BagReader) -> list[TopicInfo]:
    pc_topics = []

    for conn in reader.get_connections():
        if conn.msgtype in POINTCLOUD_TYPES:
            pc_topics.append(TopicInfo(
                name=conn.topic,
                msgtype=conn.msgtype,
                message_count=conn.msgcount,
            ))

    return pc_topics


def select_best_topics(reader: BagReader, config: dict[str, Any]) -> dict[str, str]:
    topics_config = config.get("topics", {})
    result = {}

    if "rgb" in topics_config:
        result["rgb"] = topics_config["rgb"]
    else:
        rgb_topics = discover_rgb_topics(reader)
        for t in rgb_topics:
            if t.msgtype in IMAGE_TYPES:
                result["rgb"] = t.name
                break
        if "rgb" not in result and rgb_topics:
            result["rgb"] = rgb_topics[0].name

    if "depth" in topics_config:
        result["depth"] = topics_config["depth"]
    else:
        depth_topics = discover_depth_topics(reader)
        if depth_topics:
            result["depth"] = depth_topics[0].name
        else:
            pc_topics = discover_pointcloud_topics(reader)
            if pc_topics:
                result["pointcloud"] = pc_topics[0].name

    if "camera_info" in topics_config:
        result["camera_info"] = topics_config["camera_info"]
    else:
        info_topics = discover_camera_info_topics(reader)
        if info_topics:
            rgb_ns = ""
            if "rgb" in result:
                parts = result["rgb"].split("/")
                if len(parts) > 1:
                    rgb_ns = "/".join(parts[:-1])

            for t in info_topics:
                if rgb_ns and t.name.startswith(rgb_ns):
                    result["camera_info"] = t.name
                    break

            if "camera_info" not in result and info_topics:
                result["camera_info"] = info_topics[0].name

    return result
