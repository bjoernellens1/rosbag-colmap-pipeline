"""Scale estimation methods."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScaleEstimate:
    scale: float
    confidence: float
    method: str
    num_samples: int
    inlier_ratio: float
    per_frame_scales: list[float] | None = None


def estimate_scale_median(
    depth_points: np.ndarray,
    colmap_points: np.ndarray
) -> ScaleEstimate:
    depth_norms = np.linalg.norm(depth_points, axis=1)
    colmap_norms = np.linalg.norm(colmap_points, axis=1)

    valid = (depth_norms > 0) & (colmap_norms > 0)
    if np.sum(valid) < 3:
        return ScaleEstimate(
            scale=1.0,
            confidence=0.0,
            method="median",
            num_samples=len(depth_points),
            inlier_ratio=0.0,
        )

    ratios = colmap_norms[valid] / depth_norms[valid]
    scale = float(np.median(ratios))

    mad = np.median(np.abs(ratios - scale))
    inlier_threshold = 2.5 * mad
    inliers = np.abs(ratios - scale) < inlier_threshold
    inlier_ratio = float(np.mean(inliers))

    confidence = min(1.0, inlier_ratio * np.sqrt(np.sum(valid) / 100))

    return ScaleEstimate(
        scale=scale,
        confidence=confidence,
        method="median",
        num_samples=int(np.sum(valid)),
        inlier_ratio=inlier_ratio,
        per_frame_scales=ratios.tolist(),
    )


def estimate_scale_umeyama(
    depth_points: np.ndarray,
    colmap_points: np.ndarray
) -> ScaleEstimate:
    if len(depth_points) < 3 or len(colmap_points) < 3:
        return ScaleEstimate(
            scale=1.0,
            confidence=0.0,
            method="umeyama",
            num_samples=0,
            inlier_ratio=0.0,
        )

    depth_mean = np.mean(depth_points, axis=0)
    colmap_mean = np.mean(colmap_points, axis=0)

    depth_centered = depth_points - depth_mean
    colmap_centered = colmap_points - colmap_mean

    depth_var = np.mean(np.sum(depth_centered ** 2, axis=1))

    cov = colmap_centered.T @ depth_centered / len(depth_points)

    U, S, Vt = np.linalg.svd(cov)

    d = np.sign(np.linalg.det(U @ Vt))
    S_prime = np.diag([1, 1, d])

    R = U @ S_prime @ Vt
    scale = np.trace(np.diag(S) @ S_prime) / depth_var if depth_var > 0 else 1.0

    t = colmap_mean - scale * R @ depth_mean

    transformed = scale * (R @ depth_points.T).T + t
    residuals = np.linalg.norm(colmap_points - transformed, axis=1)

    inlier_threshold = np.median(residuals) * 2
    inliers = residuals < inlier_threshold
    inlier_ratio = float(np.mean(inliers))

    confidence = min(1.0, inlier_ratio * np.sqrt(len(depth_points) / 100))

    return ScaleEstimate(
        scale=float(scale),
        confidence=confidence,
        method="umeyama",
        num_samples=len(depth_points),
        inlier_ratio=inlier_ratio,
    )


def estimate_scale_ransac(
    depth_points: np.ndarray,
    colmap_points: np.ndarray,
    iterations: int = 1000,
    inlier_threshold: float = 0.05
) -> ScaleEstimate:
    if len(depth_points) < 3:
        return ScaleEstimate(
            scale=1.0,
            confidence=0.0,
            method="ransac",
            num_samples=0,
            inlier_ratio=0.0,
        )

    depth_norms = np.linalg.norm(depth_points, axis=1)
    colmap_norms = np.linalg.norm(colmap_points, axis=1)

    valid = (depth_norms > 0) & (colmap_norms > 0)
    depth_norms = depth_norms[valid]
    colmap_norms = colmap_norms[valid]
    valid_indices = np.where(valid)[0]

    best_scale = 1.0
    best_inliers = 0
    best_inlier_mask = np.zeros(len(valid), dtype=bool)

    np.random.seed(42)

    for _ in range(iterations):
        idx = np.random.choice(len(valid_indices), min(3, len(valid_indices)), replace=False)
        sample_scales = colmap_norms[idx] / depth_norms[idx]
        candidate_scale = np.median(sample_scales)

        if candidate_scale <= 0:
            continue

        ratios = colmap_norms / depth_norms
        residuals = np.abs(ratios - candidate_scale) / candidate_scale
        inlier_mask = residuals < inlier_threshold
        num_inliers = np.sum(inlier_mask)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_scale = candidate_scale
            best_inlier_mask = inlier_mask

    inlier_ratio = best_inliers / len(valid) if len(valid) > 0 else 0.0
    confidence = min(1.0, inlier_ratio * np.sqrt(len(valid) / 100))

    return ScaleEstimate(
        scale=float(best_scale),
        confidence=confidence,
        method="ransac",
        num_samples=int(best_inliers),
        inlier_ratio=inlier_ratio,
    )


def estimate_global_scale(
    workspace: Path,
    config: dict[str, Any]
) -> ScaleEstimate:
    from colmap_rgbd_gt.dataset.schema import Workspace
    from colmap_rgbd_gt.dataset.manifest import Manifest
    from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory
    from colmap_rgbd_gt.colmap.reconstruction import load_sparse_model
    from colmap_rgbd_gt.scaling.correspondences import find_colmap_points_in_image
    from colmap_rgbd_gt.scaling.backproject import backproject_depth_image, transform_to_world
    from colmap_rgbd_gt.utils.camera import CameraIntrinsics
    from colmap_rgbd_gt.utils.transforms import colmap_pose_to_c2w

    ws = Workspace(workspace)
    manifest = Manifest.load(ws.layout.manifest)

    intrinsics_data = manifest.camera_info
    if not intrinsics_data:
        raise ValueError("No camera info in manifest")

    intrinsics = CameraIntrinsics(
        fx=intrinsics_data["K"][0],
        fy=intrinsics_data["K"][4],
        cx=intrinsics_data["K"][2],
        cy=intrinsics_data["K"][5],
        width=intrinsics_data["width"],
        height=intrinsics_data["height"],
    )

    sparse_dir = ws.layout.colmap / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = ws.layout.colmap / "sparse"

    model = load_sparse_model(sparse_dir)
    trajectory = extract_trajectory(workspace)

    method = config.get("method", "umeyama")
    max_frames = config.get("max_frames_for_scale", 100)
    min_points = config.get("min_points_per_frame", 200)
    sample_stride = config.get("sample_stride", 8)

    all_depth_points = []
    all_colmap_points = []
    per_frame_scales = []

    for i, entry in enumerate(trajectory[:max_frames]):
        frame_id = entry["frame_id"]
        image_name = f"{frame_id:06d}.png"

        colmap_pts = find_colmap_points_in_image(model, image_name)
        if len(colmap_pts) < min_points:
            continue

        depth_path = ws.get_depth_path(frame_id)
        if not depth_path.exists():
            continue

        import cv2
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float64) / 1000.0

        R_c2w, t_c2w = entry["R"], entry["t"]

        depth_pts = backproject_depth_image(depth, intrinsics)
        depth_pts_world = transform_to_world(depth_pts, (R_c2w, t_c2w))

        min_depth = config.get("min_depth_m", 0.2)
        max_depth = config.get("max_depth_m", 8.0)

        depth_norms = np.linalg.norm(depth_pts, axis=1)
        valid = (depth_norms >= min_depth) & (depth_norms <= max_depth)

        if np.sum(valid) < min_points:
            continue

        all_depth_points.extend(depth_pts_world[valid].tolist())
        all_colmap_points.extend(colmap_pts.tolist())

    if len(all_depth_points) < 100:
        logger.warning(f"Insufficient correspondences: {len(all_depth_points)}")
        return ScaleEstimate(
            scale=1.0,
            confidence=0.0,
            method=method,
            num_samples=len(all_depth_points),
            inlier_ratio=0.0,
        )

    depth_points = np.array(all_depth_points)
    colmap_points = np.array(all_colmap_points)

    if method == "median":
        return estimate_scale_median(depth_points, colmap_points)
    elif method == "ransac":
        return estimate_scale_ransac(depth_points, colmap_points)
    else:
        return estimate_scale_umeyama(depth_points, colmap_points)
