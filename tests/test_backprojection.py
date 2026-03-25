"""Test backprojection utilities."""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_backproject_depth_image():
    from colmap_rgbd_gt.scaling.backproject import backproject_depth_image
    from colmap_rgbd_gt.utils.camera import CameraIntrinsics

    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    intrinsics = CameraIntrinsics(
        fx=500.0, fy=500.0,
        cx=1.0, cy=1.0,
        width=2, height=2
    )

    points = backproject_depth_image(depth, intrinsics)

    assert points.shape[1] == 3
    assert points.shape[0] == 4
    assert np.all(points[:, 2] > 0)


def test_transform_to_world():
    from colmap_rgbd_gt.scaling.backproject import transform_to_world
    import numpy as np

    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    R = np.eye(3)
    t = np.array([10, 0, 0], dtype=np.float64)

    transformed = transform_to_world(points, (R, t))

    expected = points + t
    np.testing.assert_array_almost_equal(transformed, expected)


def test_filter_invalid_depth():
    from colmap_rgbd_gt.scaling.backproject import filter_invalid_depth

    points = np.array([
        [0.1, 0, 0],
        [1.0, 0, 0],
        [5.0, 0, 0],
        [10.0, 0, 0],
    ], dtype=np.float64)

    filtered = filter_invalid_depth(points, min_m=0.5, max_m=8.0)

    assert filtered.shape[0] == 2
