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
