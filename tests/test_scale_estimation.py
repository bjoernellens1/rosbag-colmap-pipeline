"""Test scale estimation methods."""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_estimate_scale_median():
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_median

    scale_true = 2.5
    depth_points = np.random.rand(100, 3)
    colmap_points = depth_points * scale_true

    estimate = estimate_scale_median(depth_points, colmap_points)

    assert abs(estimate.scale - scale_true) < 0.1
    assert estimate.method == "median"
    assert estimate.num_samples == 100


def test_estimate_scale_umeyama():
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_umeyama

    scale_true = 3.0
    depth_points = np.random.rand(100, 3) * 5

    R_true = np.array([
        [0.9, -0.1, 0.0],
        [0.1, 0.9, -0.1],
        [0.0, 0.1, 0.9]
    ])
    t_true = np.array([1.0, 2.0, 3.0])

    colmap_points = scale_true * (R_true @ depth_points.T).T + t_true

    estimate = estimate_scale_umeyama(depth_points, colmap_points)

    assert abs(estimate.scale - scale_true) < 0.2
    assert estimate.method == "umeyama"


def test_estimate_scale_ransac():
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_ransac

    scale_true = 2.0
    depth_points = np.random.rand(100, 3) * 3
    colmap_points = depth_points * scale_true

    estimate = estimate_scale_ransac(depth_points, colmap_points)

    assert abs(estimate.scale - scale_true) < 0.3
    assert estimate.method == "ransac"


def test_scale_estimate_confidence():
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_median

    depth_points = np.random.rand(10, 3)
    colmap_points = depth_points * 2.0

    estimate = estimate_scale_median(depth_points, colmap_points)

    assert estimate.num_samples == 10
