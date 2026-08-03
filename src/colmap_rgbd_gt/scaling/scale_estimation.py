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
    """Estimate scale s such that metric (depth) ~= s * colmap.

    `scale` is the multiplier applied to COLMAP's arbitrary-unit
    quantities to recover metric units (i.e. `depth_norm / colmap_norm`),
    consistent with how callers (e.g. `scale_trajectory`) apply it directly
    to COLMAP-frame translations.
    """
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

    ratios = depth_norms[valid] / colmap_norms[valid]
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
    """Estimate the similarity transform mapping COLMAP points to metric
    depth points: depth ~= scale * R @ colmap + t.

    `scale` is therefore the multiplier applied to COLMAP-frame quantities
    to recover metric units, consistent with `estimate_scale_median`/
    `estimate_scale_ransac` and with how callers apply the result directly
    to COLMAP-frame translations.
    """
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

    colmap_var = np.mean(np.sum(colmap_centered ** 2, axis=1))

    cov = depth_centered.T @ colmap_centered / len(depth_points)

    U, S, Vt = np.linalg.svd(cov)

    d = np.sign(np.linalg.det(U @ Vt))
    S_prime = np.diag([1, 1, d])

    R = U @ S_prime @ Vt
    scale = np.trace(np.diag(S) @ S_prime) / colmap_var if colmap_var > 0 else 1.0

    t = depth_mean - scale * R @ colmap_mean

    transformed = scale * (R @ colmap_points.T).T + t
    residuals = np.linalg.norm(depth_points - transformed, axis=1)

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
    inlier_threshold: float = 0.05,
    seed: int = 42,
) -> ScaleEstimate:
    """Estimate scale s such that metric (depth) ~= s * colmap, matching the
    convention used by `estimate_scale_median`/`estimate_scale_umeyama`."""
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

    rng = np.random.default_rng(seed)

    for _ in range(iterations):
        idx = rng.choice(len(valid_indices), min(3, len(valid_indices)), replace=False)
        sample_scales = depth_norms[idx] / colmap_norms[idx]
        candidate_scale = np.median(sample_scales)

        if candidate_scale <= 0:
            continue

        ratios = depth_norms / colmap_norms
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
    from colmap_rgbd_gt.colmap.reconstruction import load_sparse_model, get_image_id_by_name
    from colmap_rgbd_gt.scaling.correspondences import (
        find_colmap_points_in_image,
        project_colmap_points_to_image,
        find_valid_correspondences,
    )
    from colmap_rgbd_gt.scaling.backproject import transform_to_world
    from colmap_rgbd_gt.utils.camera import CameraIntrinsics

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
        distortion_model=intrinsics_data.get("distortion_model", "plumb_bob"),
        distortion_coeffs=intrinsics_data.get("D", []),
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

    # First pass: gather (depth image, projected uv/depth) per frame without
    # any depth-tolerance filtering. `find_valid_correspondences` compares
    # the *measured* depth (metric) against the *projected* COLMAP depth
    # (arbitrary COLMAP units) directly, which implicitly assumes those two
    # are already close in scale -- true for floor2/floor3 (scale ~1.1-1.2)
    # but not in general (e.g. this trolley bag's COLMAP scale is ~0.5x
    # metric), where every correspondence would be rejected before scale
    # is even estimated. So compute a coarse global scale first from loose
    # (pixel-valid, positive-depth) ratios, and use it to bring the
    # projected depths into roughly metric units before the real
    # depth-tolerance-gated matching below.
    frame_cache = []
    coarse_ratios = []
    for entry in trajectory[:max_frames]:
        frame_id = entry["frame_id"]
        image_name = f"{frame_id:06d}.png"

        colmap_pts_all = find_colmap_points_in_image(model, image_name)
        if len(colmap_pts_all) < min_points:
            continue

        if sample_stride > 1:
            colmap_pts_all = colmap_pts_all[::sample_stride]

        depth_path = ws.get_depth_path(frame_id)
        if not depth_path.exists():
            continue

        import cv2
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float64) / 1000.0

        # `project_colmap_points_to_image` expects a world-to-camera (w2c)
        # pose as (qvec, tvec) and applies R @ X + t. Pull this directly
        # from the COLMAP image entry rather than the trajectory's c2w
        # (R, t), to avoid any ambiguity about pose convention.
        image_id = get_image_id_by_name(model, image_name)
        if image_id is None:
            continue
        img_entry = model["images"][image_id]
        qvec_w2c = np.array(img_entry["qvec"], dtype=np.float64)
        tvec_w2c = np.array(img_entry["tvec"], dtype=np.float64)

        uv, depths_proj, valid_idx = project_colmap_points_to_image(
            colmap_pts_all, (qvec_w2c, tvec_w2c), intrinsics
        )
        if len(valid_idx) == 0:
            continue

        h, w = depth.shape
        u, v = uv[:, 0], uv[:, 1]
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        frame_coarse_scale = None
        if np.any(in_bounds):
            ui = u[in_bounds].astype(int)
            vi = v[in_bounds].astype(int)
            d_meas = depth[vi, ui]
            d_proj = depths_proj[in_bounds]
            nonzero = d_meas > 0
            if np.any(nonzero):
                frame_ratios = (d_meas[nonzero] / d_proj[nonzero]).tolist()
                coarse_ratios.extend(frame_ratios)
                # Per-frame local ratio, used below to gate this frame's own
                # tolerance-check rather than a single pooled/global ratio --
                # see note above the loop this feeds into for why.
                frame_coarse_scale = float(np.median(frame_ratios))

        frame_cache.append((colmap_pts_all[valid_idx], uv, depths_proj, depth, entry, frame_coarse_scale))

    coarse_scale = float(np.median(coarse_ratios)) if coarse_ratios else 1.0
    logger.info(f"Coarse pre-scale estimate for correspondence matching: {coarse_scale:.4f}")

    # NOTE (2026-08-03): gating every frame's tolerance check against one
    # single pooled `coarse_scale` assumes the whole trajectory shares one
    # COLMAP-to-metric ratio. That assumption broke on trolley_femto's
    # global_mapper reconstruction, which has an internal ~4x scale jump
    # between two weakly-connected trajectory segments (frames ~0-281 ratio
    # ~0.83, frames ~357+ ratio ~3.23) -- a real reconstruction-quality
    # defect (likely global positioning drift across a weak-overlap
    # boundary), not a scale-estimation bug per se. Gating each frame
    # against its OWN locally-estimated ratio (falling back to the pooled
    # value only when a frame has no usable in-bounds ratio of its own)
    # restores correspondences within each locally-consistent segment
    # instead of silently zeroing everything against a global value that
    # fits neither segment. The final scale/confidence computed below is
    # still a single global number pooled across all matched points, so a
    # low inlier_ratio/confidence on a reconstruction like this is now a
    # correct signal of real internal scale drift -- investigate the
    # reconstruction (e.g. via `colmap model_analyzer` / per-segment
    # scale checks) rather than trusting the single scale blindly.
    for colmap_pts_valid, uv, depths_proj, depth, entry, frame_coarse_scale in frame_cache:
        local_scale = frame_coarse_scale if frame_coarse_scale is not None else coarse_scale
        depth_pts_cam, colmap_pts_matched = find_valid_correspondences(
            depth,
            colmap_pts_valid,
            uv,
            depths_proj * local_scale,
            intrinsics,
            depth_tolerance=config.get("depth_tolerance", 0.1),
        )
        if len(depth_pts_cam) == 0:
            continue

        min_depth = config.get("min_depth_m", 0.2)
        max_depth = config.get("max_depth_m", 8.0)

        depth_norms = np.linalg.norm(depth_pts_cam, axis=1)
        valid = (depth_norms >= min_depth) & (depth_norms <= max_depth)

        if np.sum(valid) < 3:
            continue

        R_c2w, t_c2w = entry["R"], entry["t"]
        depth_pts_world = transform_to_world(depth_pts_cam[valid], (R_c2w, t_c2w))

        all_depth_points.extend(depth_pts_world.tolist())
        all_colmap_points.extend(colmap_pts_matched[valid].tolist())

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
