"""Point cloud to depth proxy conversion."""

import numpy as np
from typing import Any
import cv2

from colmap_rgbd_gt.utils.camera import CameraIntrinsics


def decode_pointcloud2(msg: Any) -> np.ndarray:
    points = []
    point_step = msg.point_step
    data = msg.data

    x_offset = y_offset = z_offset = None
    for field in msg.fields:
        if field.name == "x":
            x_offset = field.offset
        elif field.name == "y":
            y_offset = field.offset
        elif field.name == "z":
            z_offset = field.offset

    if x_offset is None or y_offset is None or z_offset is None:
        raise ValueError("Point cloud missing x, y, or z field")

    for i in range(0, len(data), point_step):
        x = np.frombuffer(data[i + x_offset:i + x_offset + 4], dtype=np.float32)[0]
        y = np.frombuffer(data[i + y_offset:i + y_offset + 4], dtype=np.float32)[0]
        z = np.frombuffer(data[i + z_offset:i + z_offset + 4], dtype=np.float32)[0]
        points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def project_points_to_camera(
    points: np.ndarray,
    intrinsics: CameraIntrinsics,
    extrinsics: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if extrinsics is not None:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points = (R @ points.T).T + t

    valid = points[:, 2] > 0
    points_valid = points[valid]

    u = points_valid[:, 0] * intrinsics.fx / points_valid[:, 2] + intrinsics.cx
    v = points_valid[:, 1] * intrinsics.fy / points_valid[:, 2] + intrinsics.cy
    depth = points_valid[:, 2]

    uv = np.stack([u, v], axis=-1)
    return uv, depth


def rasterize_depth(
    uv: np.ndarray,
    depth: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    depth_image = np.zeros((height, width), dtype=np.float32)

    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)

    valid = (
        (u_int >= 0) & (u_int < width) &
        (v_int >= 0) & (v_int < height)
    )

    u_valid = u_int[valid]
    v_valid = v_int[valid]
    d_valid = depth[valid]

    for i in range(len(u_valid)):
        u, v, d = u_valid[i], v_valid[i], d_valid[i]
        if depth_image[v, u] == 0 or d < depth_image[v, u]:
            depth_image[v, u] = d

    return depth_image


def pointcloud_to_depth_proxy(
    msg: Any,
    intrinsics: CameraIntrinsics,
    extrinsics: np.ndarray | None = None
) -> np.ndarray:
    points = decode_pointcloud2(msg)
    uv, depth = project_points_to_camera(points, intrinsics, extrinsics)
    return rasterize_depth(uv, depth, intrinsics.height, intrinsics.width)
