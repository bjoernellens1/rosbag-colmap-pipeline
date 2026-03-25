"""3D correspondence finding for scale estimation."""

from typing import Any
import numpy as np

from colmap_rgbd_gt.colmap.reconstruction import load_sparse_model, get_image_id_by_name
from colmap_rgbd_gt.utils.camera import CameraIntrinsics
from colmap_rgbd_gt.utils.transforms import quaternion_to_rotation_matrix, Transform
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def find_colmap_points_in_image(
    model: dict[str, Any],
    image_name: str
) -> np.ndarray:
    image_id = get_image_id_by_name(model, image_name)
    if image_id is None:
        return np.array([]).reshape(0, 3)

    points = []
    for point_data in model["points3d"].values():
        if image_id in point_data.get("image_ids", []):
            points.append(point_data["xyz"])

    return np.array(points, dtype=np.float64) if points else np.array([]).reshape(0, 3)


def project_colmap_points_to_image(
    points_3d: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray],
    intrinsics: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    qvec, tvec = pose
    R = quaternion_to_rotation_matrix(qvec)

    points_cam = (R @ points_3d.T).T + tvec

    valid = points_cam[:, 2] > 0
    points_valid = points_cam[valid]

    if len(points_valid) == 0:
        return np.array([]).reshape(0, 2), np.array([])

    u = points_valid[:, 0] * intrinsics.fx / points_valid[:, 2] + intrinsics.cx
    v = points_valid[:, 1] * intrinsics.fy / points_valid[:, 2] + intrinsics.cy
    uv = np.stack([u, v], axis=-1)
    depths = points_valid[:, 2]

    valid_idx = np.where(valid)[0]
    return uv, depths, valid_idx


def find_valid_correspondences(
    depth: np.ndarray,
    colmap_points_3d: np.ndarray,
    uv_projected: np.ndarray,
    depths_projected: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_tolerance: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    if len(uv_projected) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    height, width = depth.shape

    u_int = np.round(uv_projected[:, 0]).astype(np.int32)
    v_int = np.round(uv_projected[:, 1]).astype(np.int32)

    valid_pixel = (
        (u_int >= 0) & (u_int < width) &
        (v_int >= 0) & (v_int < height)
    )

    depth_meas = np.zeros(len(uv_projected))
    for i in range(len(uv_projected)):
        if valid_pixel[i]:
            depth_meas[i] = depth[v_int[i], u_int[i]]

    valid_depth = depth_meas > 0

    depth_diff = np.abs(depth_meas - depths_projected)
    valid_assoc = depth_diff < depth_tolerance * depths_projected

    valid = valid_pixel & valid_depth & valid_assoc

    depth_points = []
    colmap_points = []

    for i in range(len(uv_projected)):
        if valid[i]:
            u, v = uv_projected[i]
            d = depth_meas[i]
            x = (u - intrinsics.cx) * d / intrinsics.fx
            y = (v - intrinsics.cy) * d / intrinsics.fy
            z = d
            depth_points.append([x, y, z])
            colmap_points.append(colmap_points_3d[i])

    return np.array(depth_points), np.array(colmap_points)
