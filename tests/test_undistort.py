"""Tests for point-level lens undistortion (rectify/undistort.py)."""

import numpy as np
import cv2

from colmap_rgbd_gt.rectify.undistort import undistort_points, get_pinhole_k
from colmap_rgbd_gt.utils.camera import CameraIntrinsics

FX = FY = 500.0
CX, CY = 320.0, 240.0


def test_undistort_points_noop_when_no_distortion():
    intrinsics = CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480)
    uv = np.array([[100.0, 150.0], [300.0, 200.0]])

    result = undistort_points(uv, intrinsics)

    np.testing.assert_array_equal(result, uv)


def test_undistort_points_noop_when_zero_distortion():
    intrinsics = CameraIntrinsics(
        fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480,
        distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    uv = np.array([[100.0, 150.0]])

    result = undistort_points(uv, intrinsics)

    np.testing.assert_array_equal(result, uv)


def test_undistort_points_roundtrip_with_known_distortion():
    k1, k2, p1, p2 = -0.2, 0.05, 0.001, -0.001
    intrinsics = CameraIntrinsics(
        fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480,
        distortion_coeffs=[k1, k2, p1, p2, 0.0],
    )
    K = intrinsics.K
    D = np.array([k1, k2, p1, p2, 0.0])

    # Start from known undistorted points, distort them via cv2.projectPoints
    # (points in front of the camera, expressed as normalized 3D rays),
    # then confirm undistort_points recovers the original undistorted uv.
    undistorted_uv_true = np.array(
        [[350.0, 260.0], [280.0, 200.0], [400.0, 300.0]]
    )
    x_n = (undistorted_uv_true[:, 0] - CX) / FX
    y_n = (undistorted_uv_true[:, 1] - CY) / FY
    points_3d = np.stack([x_n, y_n, np.ones(len(x_n))], axis=-1)

    distorted_uv, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        K,
        D,
    )
    distorted_uv = distorted_uv.reshape(-1, 2)

    recovered = undistort_points(distorted_uv, intrinsics)

    np.testing.assert_allclose(recovered, undistorted_uv_true, atol=0.5)


def test_undistort_points_empty_input():
    intrinsics = CameraIntrinsics(
        fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480,
        distortion_coeffs=[-0.1, 0.0, 0.0, 0.0, 0.0],
    )
    result = undistort_points(np.zeros((0, 2)), intrinsics)
    assert result.shape == (0, 2)


def test_get_pinhole_k_matches_intrinsics_K():
    intrinsics = CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480)
    np.testing.assert_array_equal(get_pinhole_k(intrinsics), intrinsics.K)
