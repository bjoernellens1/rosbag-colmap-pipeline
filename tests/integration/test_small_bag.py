"""Integration test placeholder for ROS1 bags."""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.skip(reason="Requires sample bag file")
def test_full_pipeline_ros1_bag():
    from colmap_rgbd_gt.pipelines.extract_only import extract_pipeline

    bag_path = Path("data/samples/test_ros1.bag")
    workspace = Path("data/workspaces/test_ros1")

    if not bag_path.exists():
        pytest.skip(f"Sample bag not found: {bag_path}")

    config = {
        "topics": {
            "rgb": "/camera/rgb/image_raw",
            "depth": "/camera/depth/image_raw",
            "camera_info": "/camera/rgb/camera_info",
        }
    }

    success = extract_pipeline(bag_path, workspace, config)
    assert success


@pytest.mark.skip(reason="Requires sample bag file")
def test_full_pipeline_ros2_bag():
    from colmap_rgbd_gt.pipelines.extract_only import extract_pipeline

    bag_path = Path("data/samples/test_ros2.db3")
    workspace = Path("data/workspaces/test_ros2")

    if not bag_path.exists():
        pytest.skip(f"Sample bag not found: {bag_path}")

    config = {
        "topics": {
            "rgb": "/camera/color/image_raw",
            "depth": "/camera/depth/image_raw",
            "camera_info": "/camera/color/camera_info",
        }
    }

    success = extract_pipeline(bag_path, workspace, config)
    assert success
