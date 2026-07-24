"""Test scale estimation methods.

Convention: `scale` is the multiplier applied to COLMAP's arbitrary-unit
quantities to recover metric units, i.e. depth (metric) ~= scale * colmap.
This matches how `scale_trajectory` applies the estimated scale directly to
COLMAP-frame translations to produce a metric trajectory.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_estimate_scale_median():
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_median

    scale_true = 2.5
    colmap_points = np.random.rand(100, 3)
    depth_points = colmap_points * scale_true

    estimate = estimate_scale_median(depth_points, colmap_points)

    assert abs(estimate.scale - scale_true) < 0.1
    assert estimate.method == "median"
    assert estimate.num_samples == 100


def test_estimate_scale_umeyama():
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_umeyama

    from scipy.spatial.transform import Rotation

    scale_true = 3.0
    colmap_points = np.random.rand(100, 3) * 5

    # A proper (orthogonal, det=1) rotation matrix -- the hand-crafted
    # matrix previously used here was not orthogonal (det != 1), which
    # biases the scale Umeyama recovers regardless of scale-direction
    # convention.
    R_true = Rotation.from_euler("xyz", [5, -8, 3], degrees=True).as_matrix()
    t_true = np.array([1.0, 2.0, 3.0])

    depth_points = scale_true * (R_true @ colmap_points.T).T + t_true

    estimate = estimate_scale_umeyama(depth_points, colmap_points)

    assert abs(estimate.scale - scale_true) < 0.2
    assert estimate.method == "umeyama"


def test_estimate_scale_ransac():
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_ransac

    scale_true = 2.0
    colmap_points = np.random.rand(100, 3) * 3
    depth_points = colmap_points * scale_true

    estimate = estimate_scale_ransac(depth_points, colmap_points)

    assert abs(estimate.scale - scale_true) < 0.3
    assert estimate.method == "ransac"


def test_scale_estimate_confidence():
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_median

    colmap_points = np.random.rand(10, 3)
    depth_points = colmap_points * 2.0

    estimate = estimate_scale_median(depth_points, colmap_points)

    assert estimate.num_samples == 10
