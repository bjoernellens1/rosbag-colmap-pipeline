"""Tests for point-level lens undistortion (rectify/undistort.py)."""

from pathlib import Path

import numpy as np
import cv2

from colmap_rgbd_gt.dataset.schema import Workspace
from colmap_rgbd_gt.rectify.undistort import (
    undistort_points,
    get_pinhole_k,
    undistort_image,
    rectify_workspace_images,
)
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


def _checkerboard(width: int = 640, height: int = 480) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[::20, :] = 255
    img[:, ::20] = 255
    return img


def test_undistort_image_noop_when_no_distortion():
    intrinsics = CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480)
    image = _checkerboard()

    result = undistort_image(image, intrinsics)

    np.testing.assert_array_equal(result, image)


def test_undistort_image_preserves_shape_with_distortion():
    intrinsics = CameraIntrinsics(
        fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480,
        distortion_coeffs=[-0.2, 0.05, 0.001, -0.001, 0.0],
    )
    image = _checkerboard()

    result = undistort_image(image, intrinsics)

    assert result.shape == image.shape
    # A genuinely distorted image should differ from the input somewhere.
    assert not np.array_equal(result, image)


def test_undistort_image_corrects_known_barrel_distortion():
    # Synthesize a distorted checkerboard by distorting a known-undistorted
    # one via cv2.projectPoints-equivalent remap, then confirm
    # undistort_image recovers something close to the original near the
    # center (edges can clip/stretch, so only check a central patch).
    k1 = -0.3
    intrinsics = CameraIntrinsics(
        fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480,
        distortion_coeffs=[k1, 0.0, 0.0, 0.0, 0.0],
    )
    undistorted = _checkerboard()
    K = intrinsics.K
    D = np.array([k1, 0.0, 0.0, 0.0, 0.0])
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (undistorted.shape[1], undistorted.shape[0]), cv2.CV_32FC1
    )
    # initUndistortRectifyMap's maps go undistorted->distorted lookup
    # direction; apply the inverse distortion to build a synthetic
    # "captured distorted" frame from the known-undistorted one.
    distorted = cv2.remap(
        undistorted, map1, map2, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    recovered = undistort_image(distorted, intrinsics)

    center = np.s_[180:300, 240:400]
    assert np.mean(np.abs(recovered[center].astype(float) - undistorted[center].astype(float))) < 30


def test_rectify_workspace_images_writes_same_filenames(tmp_path: Path):
    ws = Workspace(tmp_path / "scene").create()
    image = _checkerboard()
    for name in ("000000.png", "000001.png"):
        cv2.imwrite(str(ws.layout.rgb / name), image)

    intrinsics = CameraIntrinsics(
        fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480,
        distortion_coeffs=[-0.2, 0.05, 0.0, 0.0, 0.0],
    )

    out_dir = rectify_workspace_images(ws, intrinsics)

    assert out_dir == ws.layout.rgb_rectified
    for name in ("000000.png", "000001.png"):
        rectified_path = ws.layout.rgb_rectified / name
        assert rectified_path.exists()
        rectified = cv2.imread(str(rectified_path))
        assert rectified.shape == image.shape
