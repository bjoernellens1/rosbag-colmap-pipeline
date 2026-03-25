"""
COLMAP RGBD GT - Pipeline for converting rosbag recordings to metric pseudo-GT trajectories.

This package provides a Docker-based pipeline that:
1. Extracts RGB, depth, and camera calibration data from rosbag files
2. Runs COLMAP on RGB images to estimate camera poses
3. Estimates metric scale from depth data
4. Exports trajectories in TUM and CSV formats

Supports both ROS1 (.bag) and ROS2 (.db3) bag formats.
"""

__version__ = "0.1.0"
__author__ = "COLMAP RGBD GT Contributors"

from colmap_rgbd_gt.logging import get_logger, setup_logging

__all__ = [
    "__version__",
    "get_logger",
    "setup_logging",
]
