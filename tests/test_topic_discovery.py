"""Test topic discovery."""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_topic_info():
    from colmap_rgbd_gt.ingest.topic_discovery import TopicInfo
    info = TopicInfo(name="/camera/rgb", msgtype="sensor_msgs/Image", message_count=100)
    assert info.name == "/camera/rgb"
    assert info.message_count == 100


def test_image_types():
    from colmap_rgbd_gt.ingest.topic_discovery import IMAGE_TYPES
    assert "sensor_msgs/msg/Image" in IMAGE_TYPES
    assert "sensor_msgs/Image" in IMAGE_TYPES


def test_compressed_image_types():
    from colmap_rgbd_gt.ingest.topic_discovery import COMPRESSED_IMAGE_TYPES
    assert "sensor_msgs/msg/CompressedImage" in COMPRESSED_IMAGE_TYPES


class _FakeConnection:
    def __init__(self, topic, msgtype, msgcount=10):
        self.topic = topic
        self.msgtype = msgtype
        self.msgcount = msgcount


class _FakeReader:
    def __init__(self, connections):
        self._connections = connections

    def get_connections(self):
        return self._connections


def _rgbd_bag_connections():
    # Mirrors a real-world layout: compressed color + compressed depth,
    # depth's connection listed before color's -- regression coverage for
    # the "depth topic misclassified as rgb" and "wrong camera_info picked"
    # bugs found running this pipeline on real bags.
    return [
        _FakeConnection("/camera/depth/image_raw/compressed", "sensor_msgs/msg/CompressedImage"),
        _FakeConnection("/camera/depth/camera_info", "sensor_msgs/msg/CameraInfo"),
        _FakeConnection("/camera/color/image_raw/compressed", "sensor_msgs/msg/CompressedImage"),
        _FakeConnection("/camera/color/camera_info", "sensor_msgs/msg/CameraInfo"),
    ]


def test_discover_rgb_topics_excludes_depth_named_topics():
    from colmap_rgbd_gt.ingest.topic_discovery import discover_rgb_topics
    reader = _FakeReader(_rgbd_bag_connections())

    rgb_topics = discover_rgb_topics(reader)

    names = [t.name for t in rgb_topics]
    assert "/camera/color/image_raw/compressed" in names
    assert "/camera/depth/image_raw/compressed" not in names


def test_discover_depth_topics_includes_compressed():
    from colmap_rgbd_gt.ingest.topic_discovery import discover_depth_topics
    reader = _FakeReader(_rgbd_bag_connections())

    depth_topics = discover_depth_topics(reader)

    assert [t.name for t in depth_topics] == ["/camera/depth/image_raw/compressed"]


def test_select_best_topics_picks_matching_camera_info():
    from colmap_rgbd_gt.ingest.topic_discovery import select_best_topics
    reader = _FakeReader(_rgbd_bag_connections())

    topics = select_best_topics(reader, {})

    assert topics["rgb"] == "/camera/color/image_raw/compressed"
    assert topics["depth"] == "/camera/depth/image_raw/compressed"
    assert topics["camera_info"] == "/camera/color/camera_info"


def test_select_best_topics_null_config_values_fall_back_to_discovery():
    # configs/*.yaml ship an explicit `topics: {rgb: null, depth: null,
    # camera_info: null}` placeholder section. A config key being PRESENT
    # (with value None) must not be treated as an explicit override -- that
    # would short-circuit auto-discovery for every shipped config.
    from colmap_rgbd_gt.ingest.topic_discovery import select_best_topics
    reader = _FakeReader(_rgbd_bag_connections())

    config = {"topics": {"rgb": None, "rgb_compressed": None, "depth": None,
                          "pointcloud": None, "camera_info": None}}
    topics = select_best_topics(reader, config)

    assert topics["rgb"] == "/camera/color/image_raw/compressed"
    assert topics["depth"] == "/camera/depth/image_raw/compressed"
    assert topics["camera_info"] == "/camera/color/camera_info"


def _orbbec_femto_bag_connections():
    # Uncompressed raw-image driver (Orbbec Femto Mega): color, depth, AND
    # an IR stream that is also plain sensor_msgs/Image -- indistinguishable
    # from the real color topic by message type alone.
    return [
        _FakeConnection("/camera/ir/image_raw", "sensor_msgs/msg/Image"),
        _FakeConnection("/camera/ir/camera_info", "sensor_msgs/msg/CameraInfo"),
        _FakeConnection("/camera/color/image_raw", "sensor_msgs/msg/Image"),
        _FakeConnection("/camera/color/camera_info", "sensor_msgs/msg/CameraInfo"),
        _FakeConnection("/camera/depth/image_raw", "sensor_msgs/msg/Image"),
        _FakeConnection("/camera/depth/camera_info", "sensor_msgs/msg/CameraInfo"),
        _FakeConnection("/camera/depth/points", "sensor_msgs/msg/PointCloud2"),
    ]


def test_discover_rgb_topics_excludes_ir_stream():
    from colmap_rgbd_gt.ingest.topic_discovery import discover_rgb_topics
    reader = _FakeReader(_orbbec_femto_bag_connections())

    rgb_topics = discover_rgb_topics(reader)

    names = [t.name for t in rgb_topics]
    assert "/camera/color/image_raw" in names
    assert "/camera/ir/image_raw" not in names


def test_select_best_topics_picks_color_not_ir():
    from colmap_rgbd_gt.ingest.topic_discovery import select_best_topics
    reader = _FakeReader(_orbbec_femto_bag_connections())

    topics = select_best_topics(reader, {})

    assert topics["rgb"] == "/camera/color/image_raw"
    assert topics["depth"] == "/camera/depth/image_raw"
    assert topics["camera_info"] == "/camera/color/camera_info"


def test_select_best_topics_explicit_override_still_honored():
    from colmap_rgbd_gt.ingest.topic_discovery import select_best_topics
    reader = _FakeReader(_rgbd_bag_connections())

    config = {"topics": {"rgb": "/custom/rgb"}}
    topics = select_best_topics(reader, config)

    assert topics["rgb"] == "/custom/rgb"
