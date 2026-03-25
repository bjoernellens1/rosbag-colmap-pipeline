"""Depth backprojection utilities."""

import numpy as np
from colmap_rgbd_gt.utils.camera import CameraIntrinsics
from colmap_rgbd_gt.utils.transforms import Transform


def backproject_depth_image(depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    uv = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float64)
    depth_flat = depth.ravel().astype(np.float64)

    valid = depth_flat > 0
    uv_valid = uv[valid]
    depth_valid = depth_flat[valid]

    z = depth_valid
    x = (uv_valid[:, 0] - intrinsics.cx) * z / intrinsics.fx
    y = (uv_valid[:, 1] - intrinsics.cy) * z / intrinsics.fy

    return np.stack([x, y, z], axis=-1)


def backproject_points(
    depth: np.ndarray,
    uv: np.ndarray,
    intrinsics: CameraIntrinsics
) -> np.ndarray:
    d = np.array([depth[int(v), int(u)] for u, v in uv])
    valid = d > 0

    uv_valid = uv[valid]
    d_valid = d[valid]

    x = (uv_valid[:, 0] - intrinsics.cx) * d_valid / intrinsics.fx
    y = (uv_valid[:, 1] - intrinsics.cy) * d_valid / intrinsics.fy
    z = d_valid

    return np.stack([x, y, z], axis=-1)


def transform_to_world(points_cam: np.ndarray, pose_c2w: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    R, t = pose_c2w
    return (R @ points_cam.T).T + t


def filter_invalid_depth(
    points: np.ndarray,
    min_m: float = 0.2,
    max_m: float = 8.0
) -> np.ndarray:
    depths = np.linalg.norm(points, axis=1)
    valid = (depths >= min_m) & (depths <= max_m)
    return points[valid]


def sample_depth_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    stride: int = 8,
    min_depth: float = 0.2,
    max_depth: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape

    v, u = np.mgrid[0:height:stride, 0:width:stride]
    uv = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float64)

    depth_samples = np.array([depth[int(v), int(u)] for u, v in uv])
    valid = (depth_samples > min_depth) & (depth_samples < max_depth)

    uv_valid = uv[valid]
    depth_valid = depth_samples[valid]

    if len(uv_valid) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    z = depth_valid
    x = (uv_valid[:, 0] - intrinsics.cx) * z / intrinsics.fx
    y = (uv_valid[:, 1] - intrinsics.cy) * z / intrinsics.fy

    return uv_valid, np.stack([x, y, z], axis=-1)
