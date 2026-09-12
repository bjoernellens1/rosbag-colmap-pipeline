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
    # Genuine per-frame ScaleEstimate objects (one per trajectory frame that
    # yielded usable correspondences), populated by `estimate_global_scale`.
    # This is what should be handed to `diagnostics.compute_diagnostics` --
    # NOT `[scale_estimate]` (a length-1 list wrapping the pooled/global
    # estimate), which used to make `num_valid_frames` report 1 regardless
    # of how many frames actually contributed. See `estimate_global_scale`'s
    # docstring for how these are computed/aggregated.
    frame_estimates: list["ScaleEstimate"] | None = None


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


@dataclass
class SimilarityEstimate:
    """Full Sim(3): metric_point ~= scale * R @ colmap_point + t. Used by
    colmap.scale_regime_correction to actually rewrite a reconstruction's
    points/poses (scalar-only `ScaleEstimate` isn't enough there -- see
    that module's docstring for why: two weakly-connected regions of the
    same reconstruction can end up related by a genuine rigid+scale
    transform, not just a scale factor, once retriangulation has settled
    into two different local optima)."""
    scale: float
    R: np.ndarray
    t: np.ndarray
    confidence: float
    num_samples: int
    inlier_ratio: float


def estimate_similarity_umeyama(
    depth_points: np.ndarray,
    colmap_points: np.ndarray,
) -> SimilarityEstimate:
    """Umeyama similarity fit: depth ~= scale * R @ colmap + t. Returns the
    FULL transform (R, scale, t), not just the scalar scale -- see
    `estimate_scale_umeyama` (a thin wrapper around this, kept for
    backward compatibility with existing scalar-scale callers)."""
    if len(depth_points) < 3 or len(colmap_points) < 3:
        return SimilarityEstimate(
            scale=1.0, R=np.eye(3), t=np.zeros(3),
            confidence=0.0, num_samples=0, inlier_ratio=0.0,
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

    return SimilarityEstimate(
        scale=float(scale), R=R, t=t,
        confidence=confidence, num_samples=len(depth_points), inlier_ratio=inlier_ratio,
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
    to COLMAP-frame translations. (Scalar-only wrapper around
    `estimate_similarity_umeyama` -- use that directly if R/t are needed.)
    """
    sim = estimate_similarity_umeyama(depth_points, colmap_points)
    return ScaleEstimate(
        scale=sim.scale,
        confidence=sim.confidence,
        method="umeyama",
        num_samples=sim.num_samples,
        inlier_ratio=sim.inlier_ratio,
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
    from colmap_rgbd_gt.utils.camera import CameraIntrinsics
    from colmap_rgbd_gt.utils.transforms import quaternion_to_rotation_matrix
    from colmap_rgbd_gt.scaling.robust_statistics import remove_outliers_mad
    from colmap_rgbd_gt.ingest.camera_info import resolve_camera_info

    ws = Workspace(workspace)
    manifest = Manifest.load(ws.layout.manifest)

    # FIXED 2026-08-03: this used to trust manifest.camera_info verbatim,
    # with no validation -- broke completely (0 correspondences) on
    # hallway's bag, whose camera_info topic published an unpopulated
    # identity K (fx=fy=1). Route through the same validate-else-fallback
    # resolution feature_extractor already uses (see colmap_only.py) so a
    # garbage/implausible bag camera_info doesn't silently poison every
    # reprojection computation here too.
    intrinsics_data, intrinsics_source = resolve_camera_info(
        manifest.camera_info, config.get("camera_fallback_profile")
    )
    if not intrinsics_data:
        raise ValueError("No valid camera info in manifest and no fallback profile configured")
    logger.info(f"Scale estimation camera intrinsics source: {intrinsics_source}")

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
    robust_statistic = config.get("robust_statistic", "median")

    all_depth_points = []
    all_colmap_points = []

    # FIXED 2026-09-12 (Bug 2, part 1): this used to be `trajectory[:max_frames]`
    # -- only the FIRST `max_frames` frames of the whole sequence. On a long
    # trajectory with known local scale drift (see the scale-regime-jump note
    # below), that samples one small, non-representative slice of the
    # sequence rather than genuinely covering it. Take an evenly-spaced
    # sample across the WHOLE trajectory instead.
    if max_frames and max_frames > 0 and len(trajectory) > max_frames:
        stride_frames = max(1, len(trajectory) // max_frames)
        frames_to_use = trajectory[::stride_frames][:max_frames]
    else:
        frames_to_use = trajectory

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
    for entry in frames_to_use:
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

        frame_cache.append((
            colmap_pts_all[valid_idx], uv, depths_proj, depth, entry,
            frame_coarse_scale, qvec_w2c, tvec_w2c,
        ))

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
    # FIXED 2026-09-12 (Bug 1): the old code transformed `depth_pts_cam`
    # (metric, CAMERA frame) into WORLD frame via `t_c2w` -- the camera
    # center in COLMAP's arbitrary, unscaled units -- producing
    # `scale_true * X_colmap + (1 - scale_true) * C_colmap_unscaled`, a
    # geometry-dependent blend of the true scale and 1.0 that made the
    # recovered scale depend on how the camera-center/point covariance
    # happens to look in a given reconstruction (see module-level bug
    # writeup / task description for the full derivation). The fix: do the
    # comparison in the CAMERA frame instead, where no such mixing exists.
    # `colmap_pts_matched` is in COLMAP WORLD frame (arbitrary units);
    # transform it into the SAME camera frame `depth_pts_cam` is already in
    # via the frame's own w2c pose. Both point sets then differ by a pure
    # scalar scale from the camera origin (same ray direction, same
    # correspondence) -- no unit mixing, no world-frame contamination.
    frame_estimates: list[ScaleEstimate] = []
    for colmap_pts_valid, uv, depths_proj, depth, entry, frame_coarse_scale, qvec_w2c, tvec_w2c in frame_cache:
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

        R_w2c = quaternion_to_rotation_matrix(qvec_w2c)
        colmap_pts_cam = (R_w2c @ colmap_pts_matched[valid].T).T + tvec_w2c
        depth_pts_cam_valid = depth_pts_cam[valid]

        all_depth_points.extend(depth_pts_cam_valid.tolist())
        all_colmap_points.extend(colmap_pts_cam.tolist())

        # Bug 2: genuine PER-FRAME scale estimate (not just pooled totals),
        # for real per-frame diagnostics/robustness -- see
        # `scaling.diagnostics.compute_diagnostics`, which expects a list of
        # objects exposing `.scale`/`.num_samples`/`.confidence` and used to
        # be handed a length-1 list wrapping the single pooled estimate
        # (`num_valid_frames` was therefore always 1, a reporting artifact
        # rather than a real "only 1 frame used"). Reuse the same simple
        # ratio-based estimator used for the pooled "median" method so the
        # per-frame numbers are directly comparable to it.
        frame_estimates.append(estimate_scale_median(depth_pts_cam_valid, colmap_pts_cam))

    if len(all_depth_points) < 100:
        logger.warning(f"Insufficient correspondences: {len(all_depth_points)}")
        return ScaleEstimate(
            scale=1.0,
            confidence=0.0,
            method=method,
            num_samples=len(all_depth_points),
            inlier_ratio=0.0,
            frame_estimates=frame_estimates or None,
        )

    depth_points = np.array(all_depth_points)
    colmap_points = np.array(all_colmap_points)

    if method == "median":
        pooled_estimate = estimate_scale_median(depth_points, colmap_points)
    elif method == "ransac":
        pooled_estimate = estimate_scale_ransac(depth_points, colmap_points)
    else:
        pooled_estimate = estimate_scale_umeyama(depth_points, colmap_points)

    # Bug 2 (robust aggregation layer): the pooled fit above is the
    # "headline" estimate as configured via `scaling.method`. Independently,
    # aggregate the genuine PER-FRAME scales with median + MAD outlier
    # rejection -- this is orthogonal to `method` (umeyama/median/ransac
    # pick how the pooled correspondences are fit; this is a cross-check /
    # outlier-frame detector over frames, always using the same robust
    # recipe regardless of `method`). `scaling.robust_statistic` selects the
    # final aggregator: "median" (default) takes the plain median of the
    # (MAD-outlier-filtered) per-frame scales; anything else falls back to
    # the pooled estimate's own scale (MAD rejection is an outlier gate, not
    # an aggregator, so it stays on either way as a sanity signal).
    contributing_frame_scales = np.array(
        [e.scale for e in frame_estimates if e.num_samples > 0]
    )
    final_scale = pooled_estimate.scale
    if len(contributing_frame_scales) >= 3:
        filtered_scales = remove_outliers_mad(contributing_frame_scales)
        if len(filtered_scales) == 0:
            filtered_scales = contributing_frame_scales
        robust_scale = float(np.median(filtered_scales))

        if robust_statistic == "median":
            final_scale = robust_scale

        # Sanity cross-check: if the pooled fit and the per-frame robust
        # median disagree substantially, the pooled fit is likely being
        # skewed by a subset of frames (e.g. a reconstruction-quality scale
        # jump, see the NOTE above `frame_cache`'s consumer loop) -- prefer
        # the more robust per-frame aggregate and log loudly rather than
        # silently trusting the pooled number.
        if pooled_estimate.scale > 0 and abs(robust_scale - pooled_estimate.scale) / pooled_estimate.scale > 0.2:
            logger.warning(
                f"Pooled scale estimate ({pooled_estimate.scale:.4f}, method={method}) disagrees "
                f"with the per-frame robust median ({robust_scale:.4f}) by more than 20% -- "
                "using the per-frame robust median as the final scale. Investigate "
                "per-frame scale diagnostics before trusting this scene's trajectory."
            )
            final_scale = robust_scale

    # `num_samples` stays the total pooled correspondence count (already a
    # meaningful total: the sum of every contributing frame's matched
    # correspondences), consistent with the pre-fix behavior of the
    # umeyama/median/ransac estimators.
    return ScaleEstimate(
        scale=float(final_scale),
        confidence=pooled_estimate.confidence,
        method=pooled_estimate.method,
        num_samples=pooled_estimate.num_samples,
        inlier_ratio=pooled_estimate.inlier_ratio,
        per_frame_scales=contributing_frame_scales.tolist(),
        frame_estimates=frame_estimates or None,
    )
