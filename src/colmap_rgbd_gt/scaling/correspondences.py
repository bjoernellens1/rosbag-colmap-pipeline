"""3D correspondence finding for scale estimation and depth-aware bundle adjustment."""

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

from colmap_rgbd_gt.colmap.reconstruction import load_sparse_model, get_image_id_by_name
from colmap_rgbd_gt.utils.camera import CameraIntrinsics
from colmap_rgbd_gt.utils.transforms import quaternion_to_rotation_matrix, Transform
from colmap_rgbd_gt.rectify.undistort import undistort_points
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def _sample_depth_at_pixels(
    depth: np.ndarray,
    uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-pixel depth lookup for a batch of pixel coordinates.

    Returns:
        depth_meas: (N,) sampled depth values (0.0 where out of bounds).
        valid_pixel: (N,) bool mask, True where (u, v) was in-bounds.
    """
    height, width = depth.shape

    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)

    valid_pixel = (
        (u_int >= 0) & (u_int < width) &
        (v_int >= 0) & (v_int < height)
    )

    depth_meas = np.zeros(len(uv))
    for i in range(len(uv)):
        if valid_pixel[i]:
            depth_meas[i] = depth[v_int[i], u_int[i]]

    return depth_meas, valid_pixel


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-frame 3D points into an image.

    Args:
        points_3d: (N, 3) points in COLMAP world frame.
        pose: (qvec, tvec) world-to-camera pose (COLMAP convention).
        intrinsics: pinhole camera intrinsics.

    Returns:
        uv: (M, 2) projected pixel coordinates for points with positive depth.
        depths: (M,) camera-frame z (depth) for those points.
        valid_idx: (M,) indices into the original `points_3d` array that
            `uv`/`depths` correspond to (points_3d[valid_idx] realigns the
            input array to match).
    """
    qvec, tvec = pose
    R = quaternion_to_rotation_matrix(qvec)

    points_cam = (R @ points_3d.T).T + tvec

    valid = points_cam[:, 2] > 0
    points_valid = points_cam[valid]

    if len(points_valid) == 0:
        return np.array([]).reshape(0, 2), np.array([]), np.array([], dtype=np.int64)

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
    """Match projected COLMAP points against measured depth pixels.

    `colmap_points_3d` must already be filtered/aligned to `uv_projected`/
    `depths_projected` — i.e. the caller passes `points_3d[valid_idx]` where
    `valid_idx` came from `project_colmap_points_to_image`. This function
    indexes `colmap_points_3d` positionally against `uv_projected`.

    Returns:
        depth_points: (K, 3) matched points backprojected from measured
            depth, in CAMERA frame.
        colmap_points: (K, 3) the corresponding COLMAP points, in WORLD
            frame (whatever frame `colmap_points_3d` was expressed in).
        The two arrays are 1:1 index-aligned true correspondences.
    """
    if len(uv_projected) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    depth_meas, valid_pixel = _sample_depth_at_pixels(depth, uv_projected)

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


def build_pose_and_point_arrays(
    model: dict[str, Any],
    image_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Build the pose/point arrays kornia_rs.k3d.bundle_adjust expects.

    `rotations`/`translations` are pulled directly from COLMAP's images.txt
    qvec/tvec, which are already world-to-camera -- the same convention
    kornia_rs expects. No c2w conversion (e.g. `colmap_pose_to_c2w`) is
    applied here; doing so would be wrong for this API.

    Returns:
        rotations: (P, 3, 3) float64, world->camera, one per image_names entry.
        translations: (P, 3) float64, world->camera.
        points: (N, 3) float64, world frame, one per COLMAP point3d.
        point_id_to_idx: maps COLMAP point3d_id -> dense index into `points`.
    """
    rotations = []
    translations = []

    for name in image_names:
        image_id = get_image_id_by_name(model, name)
        if image_id is None:
            raise ValueError(f"Image not found in sparse model: {name}")
        img = model["images"][image_id]
        qvec = np.array(img["qvec"], dtype=np.float64)
        tvec = np.array(img["tvec"], dtype=np.float64)
        rotations.append(quaternion_to_rotation_matrix(qvec))
        translations.append(tvec)

    rotations_arr = np.array(rotations, dtype=np.float64).reshape(-1, 3, 3)
    translations_arr = np.array(translations, dtype=np.float64).reshape(-1, 3)

    point_ids_sorted = sorted(model["points3d"].keys())
    points_arr = np.array(
        [model["points3d"][pid]["xyz"] for pid in point_ids_sorted],
        dtype=np.float64,
    ).reshape(-1, 3)
    point_id_to_idx = {pid: i for i, pid in enumerate(point_ids_sorted)}

    return rotations_arr, translations_arr, points_arr, point_id_to_idx


@dataclass
class BAObservations:
    observations: np.ndarray    # (M, 4) float64 [pose_idx, point_idx, u, v], undistorted
    obs_depths: np.ndarray      # (M,) float32, <= 0 means "no depth residual"
    obs_sigmas: np.ndarray      # (M,) float32
    n_depth_observations: int


def build_ba_observations(
    model: dict[str, Any],
    image_names: list[str],
    points: np.ndarray,
    point_id_to_idx: dict[int, int],
    depth_loader: Callable[[str], np.ndarray | None],
    intrinsics: CameraIntrinsics,
    depth_tolerance: float = 0.1,
    obs_sigma_base: float = 0.01,
    obs_sigma_quadratic: float = 0.0,
) -> BAObservations:
    """Build the full kornia_rs `observations`/`obs_depths`/`obs_sigmas` arrays.

    Unlike `find_valid_correspondences` (which returns only the subset of
    correspondences that pass a depth-tolerance gate, correct for scale
    estimation), this returns EVERY valid 2D-3D reprojection observation
    from the COLMAP model's POINTS2D tracks. The depth-tolerance gate is
    applied per-row to decide whether to attach a depth residual
    (`obs_depth > 0`) or leave the row reprojection-only (`obs_depth <= 0`,
    kornia_rs's convention for "no depth residual on this observation").

    Ordering rule: the DISTORTED (u, v) stored by COLMAP is used to sample
    the depth image (which remains in raw/distorted pixel space); only the
    coordinate placed into the returned `observations` array is undistorted
    (see `rectify.undistort.undistort_points`).

    Args:
        model: COLMAP sparse model (from `load_sparse_model`), with
            `read_images_text`-parsed `xys`/`point3d_ids` per image.
        image_names: ordered list; index is the pose_idx used in `observations`.
        points: (N, 3) world-frame points, as returned by
            `build_pose_and_point_arrays` -- used to compute each
            observation's expected (COLMAP-frame) depth for the tolerance gate.
        point_id_to_idx: COLMAP point3d_id -> dense index into `points`.
        depth_loader: image_name -> depth image in meters, or None if missing.
        intrinsics: camera intrinsics (K + distortion) for undistortion.
        depth_tolerance: relative tolerance for the depth-vs-reprojection gate.
        obs_sigma_base: constant term (meters) of the depth measurement sigma.
        obs_sigma_quadratic: quadratic term (meters per meter^2) of sigma,
            i.e. sigma = obs_sigma_base + obs_sigma_quadratic * depth**2.
    """
    obs_rows: list[list[float]] = []
    obs_depths: list[float] = []
    obs_sigmas: list[float] = []

    for pose_idx, image_name in enumerate(image_names):
        image_id = get_image_id_by_name(model, image_name)
        if image_id is None:
            continue
        img = model["images"][image_id]
        xys = img.get("xys", np.zeros((0, 2)))
        point3d_ids = img.get("point3d_ids", np.zeros((0,), dtype=np.int64))
        if len(xys) == 0:
            continue

        valid_pt_mask = np.array([pid in point_id_to_idx for pid in point3d_ids])
        if not np.any(valid_pt_mask):
            continue

        xys_valid = xys[valid_pt_mask]
        point_idxs = np.array([point_id_to_idx[pid] for pid in point3d_ids[valid_pt_mask]])

        qvec = np.array(img["qvec"], dtype=np.float64)
        tvec = np.array(img["tvec"], dtype=np.float64)
        R = quaternion_to_rotation_matrix(qvec)

        points_3d = points[point_idxs]
        colmap_depth = (R @ points_3d.T).T[:, 2] + tvec[2]

        depth_img = depth_loader(image_name)
        if depth_img is not None:
            depth_meas, valid_pixel = _sample_depth_at_pixels(depth_img, xys_valid)
        else:
            depth_meas = np.zeros(len(xys_valid))
            valid_pixel = np.zeros(len(xys_valid), dtype=bool)

        valid_depth = depth_meas > 0
        depth_diff = np.abs(depth_meas - colmap_depth)
        valid_assoc = (colmap_depth > 0) & (depth_diff < depth_tolerance * np.abs(colmap_depth))
        has_depth = valid_pixel & valid_depth & valid_assoc

        undistorted_uv = undistort_points(xys_valid, intrinsics)

        for i in range(len(xys_valid)):
            obs_rows.append([float(pose_idx), float(point_idxs[i]), float(undistorted_uv[i, 0]), float(undistorted_uv[i, 1])])
            if has_depth[i]:
                d = float(depth_meas[i])
                obs_depths.append(d)
                obs_sigmas.append(obs_sigma_base + obs_sigma_quadratic * d * d)
            else:
                obs_depths.append(-1.0)
                obs_sigmas.append(0.0)

    observations_arr = (
        np.array(obs_rows, dtype=np.float64) if obs_rows else np.zeros((0, 4), dtype=np.float64)
    )
    obs_depths_arr = np.array(obs_depths, dtype=np.float32)
    obs_sigmas_arr = np.array(obs_sigmas, dtype=np.float32)
    n_depth_observations = int(np.sum(obs_depths_arr > 0))

    return BAObservations(
        observations=observations_arr,
        obs_depths=obs_depths_arr,
        obs_sigmas=obs_sigmas_arr,
        n_depth_observations=n_depth_observations,
    )
