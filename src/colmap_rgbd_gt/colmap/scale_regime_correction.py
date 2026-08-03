"""Detect and correct point-level internal scale discontinuities in a
COLMAP reconstruction ("scale regimes") -- a DIFFERENT defect class from
pose_outliers.py's disconnected-SEGMENT (position-jump) detection.

Root-caused 2026-08-03 on floor2: after the pose-outlier fix produced one
smooth, continuous camera trajectory, a user directly inspecting
colmap/sparse/0 in COLMAP's own GUI still saw two visually distinct,
parallel/offset "sheets" of 3D points -- the SAME physical corridor
surface reconstructed twice at two different implicit scales. Measured:
frames 304-597 have a COLMAP-to-metric depth ratio ~0.61, frames 616+
have ratio ~3.0 -- a clean ~5x discontinuity, confirmed to spatially
separate the point cloud into exactly the two described sheets (colored
each point by its majority-observing-frame regime and rendered them:
they occupy the same physical region but are offset/distorted relative
to each other).

Root cause is a genuine COLMAP algorithmic limitation, not a bug in this
pipeline: `colmap bundle_adjuster` run with 300 extra iterations directly
on this model did not move the ratio AT ALL (0.6178 -> 0.6178, 3.0082 ->
3.0082) and reported NO_CONVERGENCE with repeated "Matrix not positive
definite" Ceres solver failures -- a genuinely stable, degenerate local
minimum from a weak link in the match graph at the segment boundary, not
an under-iterated one. Retriangulation/BA-level tuning cannot fix this.

The fix implemented here instead uses each weakly-connected region's OWN
real depth sensor data as an INDEPENDENT metric anchor: rather than
trying to solve for the (poorly-constrained, that's exactly why it
drifted) relative transform between the two regions from their shared
COLMAP structure, each region is independently registered into the same
real-world metric frame via Umeyama similarity fit against its own RGBD
depth correspondences (the same machinery scale_estimation.py already
uses for the whole-trajectory case, scoped per-region here). Since both
regions observe the same real corridor, anchoring each one to real depth
independently necessarily brings them into a consistent common frame --
this is a stronger constraint than trying to stitch the COLMAP-internal
structure back together, and doesn't depend on the (already-shown-weak)
match-graph connectivity between them at all.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from colmap_rgbd_gt.colmap.reconstruction import get_image_id_by_name
from colmap_rgbd_gt.colmap.pose_extract import colmap_pose_to_c2w
from colmap_rgbd_gt.scaling.correspondences import find_colmap_points_in_image, project_colmap_points_to_image, find_valid_correspondences
from colmap_rgbd_gt.scaling.backproject import transform_to_world
from colmap_rgbd_gt.scaling.scale_estimation import estimate_similarity_umeyama, SimilarityEstimate
from colmap_rgbd_gt.utils.transforms import rotation_matrix_to_quaternion
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

DEFAULT_JUMP_RATIO_THRESHOLD = 1.5
DEFAULT_MIN_SEGMENT_FRAMES = 10
DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_MIN_POINTS_PER_FRAME = 50


@dataclass
class ScaleRegimeSegment:
    frame_ids: list[int]
    median_ratio: float


@dataclass
class ScaleRegimeCorrectionResult:
    action_taken: bool = False
    reason: str = ""
    n_segments: int = 1
    segments: list[dict[str, Any]] = field(default_factory=list)


def compute_per_frame_scale_ratios(
    model: dict[str, Any],
    trajectory: list[dict[str, Any]],
    ws,
    intrinsics,
    min_points: int = DEFAULT_MIN_POINTS_PER_FRAME,
) -> dict[int, float]:
    """Per-frame median(measured_depth / projected_colmap_depth), i.e. the
    same coarse-ratio computation scale_estimation.py already does, but
    exposed per-frame rather than pooled -- this IS the signal a point-
    level scale-regime discontinuity shows up in."""
    ratios: dict[int, float] = {}
    for entry in trajectory:
        frame_id = entry["frame_id"]
        image_name = f"{frame_id:06d}.png"

        colmap_pts_all = find_colmap_points_in_image(model, image_name)
        if len(colmap_pts_all) < min_points:
            continue

        depth_path = ws.get_depth_path(frame_id)
        if not depth_path.exists():
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float64) / 1000.0

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
        if not np.any(in_bounds):
            continue
        ui = u[in_bounds].astype(int)
        vi = v[in_bounds].astype(int)
        d_meas = depth[vi, ui]
        d_proj = depths_proj[in_bounds]
        nonzero = d_meas > 0
        if np.any(nonzero):
            ratios[frame_id] = float(np.median((d_meas[nonzero] / d_proj[nonzero])))

    return ratios


def detect_scale_regime_segments(
    per_frame_ratios: dict[int, float],
    sorted_frame_ids: list[int],
    jump_ratio_threshold: float = DEFAULT_JUMP_RATIO_THRESHOLD,
    min_segment_frames: int = DEFAULT_MIN_SEGMENT_FRAMES,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> list[ScaleRegimeSegment]:
    """Partition `sorted_frame_ids` into contiguous scale-regime segments
    using a rolling-median-smoothed ratio signal (robust to single-frame
    noise) and detecting sustained multiplicative jumps between
    consecutive smoothed windows. Frames with no computed ratio (e.g. no
    depth-visible points) are carried in whichever segment their
    neighbors belong to (they don't break or define a boundary on their
    own -- no evidence either way)."""
    ratio_frame_ids = [fid for fid in sorted_frame_ids if fid in per_frame_ratios]
    if len(ratio_frame_ids) < 2 * smoothing_window:
        return [ScaleRegimeSegment(frame_ids=list(sorted_frame_ids), median_ratio=(
            float(np.median(list(per_frame_ratios.values()))) if per_frame_ratios else 1.0
        ))]

    ratio_values = np.array([per_frame_ratios[fid] for fid in ratio_frame_ids])
    n = len(ratio_values)
    smoothed = np.array([
        np.median(ratio_values[max(0, i - smoothing_window):min(n, i + smoothing_window + 1)])
        for i in range(n)
    ])

    boundaries = []  # indices into ratio_frame_ids where a NEW segment starts
    last_boundary = 0
    for i in range(smoothing_window, n - smoothing_window):
        before = smoothed[max(0, i - smoothing_window):i]
        after = smoothed[i:min(n, i + smoothing_window)]
        if len(before) == 0 or len(after) == 0:
            continue
        med_before, med_after = np.median(before), np.median(after)
        if med_before <= 0:
            continue
        ratio_of_ratios = med_after / med_before
        is_jump = ratio_of_ratios > jump_ratio_threshold or ratio_of_ratios < 1 / jump_ratio_threshold
        if is_jump and (i - last_boundary) >= min_segment_frames:
            boundaries.append(i)
            last_boundary = i

    if not boundaries:
        return [ScaleRegimeSegment(frame_ids=list(sorted_frame_ids), median_ratio=float(np.median(ratio_values)))]

    # Map ratio-frame-id boundary indices back to a partition of the FULL
    # sorted_frame_ids list (including no-ratio frames, assigned to
    # whichever side of the nearest boundary they fall on by frame_id).
    boundary_frame_ids = [ratio_frame_ids[i] for i in boundaries]
    segments: list[ScaleRegimeSegment] = []
    seg_start = 0
    all_boundaries_idx = [0] + [
        next(j for j, fid in enumerate(sorted_frame_ids) if fid >= bfid) for bfid in boundary_frame_ids
    ] + [len(sorted_frame_ids)]
    for k in range(len(all_boundaries_idx) - 1):
        lo, hi = all_boundaries_idx[k], all_boundaries_idx[k + 1]
        if hi <= lo:
            continue
        seg_fids = sorted_frame_ids[lo:hi]
        seg_ratios = [per_frame_ratios[f] for f in seg_fids if f in per_frame_ratios]
        med = float(np.median(seg_ratios)) if seg_ratios else 1.0
        segments.append(ScaleRegimeSegment(frame_ids=seg_fids, median_ratio=med))

    return segments


def estimate_segment_similarity(
    model: dict[str, Any],
    trajectory_by_frame_id: dict[int, dict[str, Any]],
    ws,
    intrinsics,
    frame_ids: list[int],
    coarse_ratio: float = 1.0,
    depth_tolerance: float = 0.15,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> SimilarityEstimate:
    """Independently register one segment's COLMAP points into real
    metric world coordinates via Umeyama similarity fit against its own
    depth correspondences -- see module docstring for why this (not
    trying to solve the weak relative transform between segments) is the
    right approach.

    FIXED 2026-08-03: `depths_proj` (raw COLMAP-frame depth, arbitrary
    units) must be pre-scaled by the segment's own coarse ratio before
    `find_valid_correspondences`'s tolerance gate -- comparing it directly
    against real depth_tolerance-gated `d_meas` (real meters) at a tight
    tolerance otherwise only accepts the coincidental handful of points
    where raw-COLMAP-units-to-meters happens to already be close to 1:1,
    a biased, non-representative sample that produces a near-identity fit
    regardless of the segment's true scale. Same pattern already fixed in
    scaling/scale_estimation.py's correspondence gathering (`depths_proj *
    local_scale`) -- caught here empirically: without this, both floor2
    segments fit to scale~1.0 (near-identity) instead of the true ~0.6/~3.0
    ratio, and the "corrected" point cloud was visually unchanged.
    """
    all_depth_points, all_colmap_points = [], []
    for frame_id in frame_ids:
        entry = trajectory_by_frame_id.get(frame_id)
        if entry is None:
            continue
        image_name = f"{frame_id:06d}.png"
        colmap_pts_all = find_colmap_points_in_image(model, image_name)
        if len(colmap_pts_all) < DEFAULT_MIN_POINTS_PER_FRAME:
            continue
        depth_path = ws.get_depth_path(frame_id)
        if not depth_path.exists():
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float64) / 1000.0

        image_id = get_image_id_by_name(model, image_name)
        if image_id is None:
            continue
        img_entry = model["images"][image_id]
        qvec_w2c = np.array(img_entry["qvec"], dtype=np.float64)
        tvec_w2c = np.array(img_entry["tvec"], dtype=np.float64)

        uv, depths_proj, valid_idx = project_colmap_points_to_image(colmap_pts_all, (qvec_w2c, tvec_w2c), intrinsics)
        if len(valid_idx) == 0:
            continue
        colmap_pts_valid = colmap_pts_all[valid_idx]

        depth_pts_cam, colmap_pts_matched = find_valid_correspondences(
            depth, colmap_pts_valid, uv, depths_proj * coarse_ratio, intrinsics, depth_tolerance=depth_tolerance,
        )
        if len(depth_pts_cam) == 0:
            continue

        depth_norms = np.linalg.norm(depth_pts_cam, axis=1)
        valid = (depth_norms >= min_depth_m) & (depth_norms <= max_depth_m)
        if np.sum(valid) < 3:
            continue

        R_c2w, t_c2w = entry["R"], entry["t"]
        depth_pts_world = transform_to_world(depth_pts_cam[valid], (R_c2w, t_c2w))
        all_depth_points.extend(depth_pts_world.tolist())
        all_colmap_points.extend(colmap_pts_matched[valid].tolist())

    if len(all_depth_points) < 20:
        logger.warning(
            f"scale_regime_correction: only {len(all_depth_points)} correspondences for "
            f"segment ({frame_ids[0]}-{frame_ids[-1]}, {len(frame_ids)} frames) -- too few for a "
            "reliable similarity fit"
        )
        return SimilarityEstimate(scale=1.0, R=np.eye(3), t=np.zeros(3), confidence=0.0, num_samples=len(all_depth_points), inlier_ratio=0.0)

    return estimate_similarity_umeyama(np.array(all_depth_points), np.array(all_colmap_points))


def _apply_sim3_to_pose(sim: SimilarityEstimate, R_c2w: np.ndarray, t_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return sim.R @ R_c2w, sim.scale * (sim.R @ t_c2w) + sim.t


def _apply_sim3_to_point(sim: SimilarityEstimate, xyz: np.ndarray) -> np.ndarray:
    return sim.scale * (sim.R @ xyz) + sim.t


def _apply_chain_to_pose(chain: list[SimilarityEstimate], R_c2w: np.ndarray, t_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for sim in chain:
        R_c2w, t_c2w = _apply_sim3_to_pose(sim, R_c2w, t_c2w)
    return R_c2w, t_c2w


def _apply_chain_to_point(chain: list[SimilarityEstimate], xyz: np.ndarray) -> np.ndarray:
    for sim in chain:
        xyz = _apply_sim3_to_point(sim, xyz)
    return xyz


def _compute_chained_transforms(
    segments: list[ScaleRegimeSegment],
    segment_transforms: list[SimilarityEstimate],
    trajectory_by_frame_id: dict[int, dict[str, Any]],
) -> list[list[SimilarityEstimate]]:
    """Each segment's OWN independently-fit Sim3 correctly fixes its
    internal scale, but two independently-fit transforms have no reason
    to agree at the shared boundary -- applying them as-is reintroduces a
    NEW pose discontinuity where none existed before (found empirically:
    the pose-outlier filter had already confirmed this exact boundary was
    continuous in the original, uncorrected trajectory). Fix: chain each
    non-reference segment's own Sim3 with an additional RIGID (rotation +
    translation, no extra scale -- the segment's OWN fit already set its
    scale correctly) correction, solved so the boundary pose continues
    from the previous (already-corrected) segment's boundary pose using
    the ORIGINAL small, already-confirmed-real relative motion between
    those two temporally-adjacent frames (not forced to exactly coincide
    for no physical reason -- the real camera really did move a small,
    known amount between them).
    """
    chains: list[list[SimilarityEstimate]] = [[segment_transforms[0]]]
    for i in range(1, len(segments)):
        prev_seg, prev_chain = segments[i - 1], chains[i - 1]
        cur_seg, cur_sim = segments[i], segment_transforms[i]

        prev_entry = trajectory_by_frame_id[prev_seg.frame_ids[-1]]
        cur_entry = trajectory_by_frame_id[cur_seg.frame_ids[0]]
        R_prev_orig, t_prev_orig = prev_entry["R"], prev_entry["t"]
        R_cur_orig, t_cur_orig = cur_entry["R"], cur_entry["t"]

        R_prev_corr, t_prev_corr = _apply_chain_to_pose(prev_chain, R_prev_orig, t_prev_orig)
        R_cur_sim, t_cur_sim = _apply_sim3_to_pose(cur_sim, R_cur_orig, t_cur_orig)

        # Original small relative motion prev_boundary -> cur_boundary (a
        # real, already-confirmed-continuous camera motion -- preserve it,
        # not collapse it to zero).
        R_delta = R_prev_orig.T @ R_cur_orig
        t_delta = R_prev_orig.T @ (t_cur_orig - t_prev_orig)
        R_cur_target = R_prev_corr @ R_delta
        t_cur_target = R_prev_corr @ t_delta + t_prev_corr

        R_extra = R_cur_target @ R_cur_sim.T
        t_extra = t_cur_target - R_extra @ t_cur_sim
        extra = SimilarityEstimate(scale=1.0, R=R_extra, t=t_extra, confidence=1.0, num_samples=0, inlier_ratio=1.0)

        chains.append([cur_sim, extra])
    return chains


def apply_segment_similarities(
    model: dict[str, Any],
    segments: list[ScaleRegimeSegment],
    segment_transforms: list[SimilarityEstimate],
    trajectory_by_frame_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Rewrite every image's pose and every point3d's xyz according to
    which scale-regime segment it belongs to, using each segment's
    boundary-continuity-CHAINED transform (see `_compute_chained_
    transforms`), not its raw independent fit. A point observed by images
    from multiple segments (a boundary/shared point) is assigned by
    majority vote among its observing images' segments (ties broken
    toward the lower segment index, deterministic)."""
    chains = _compute_chained_transforms(segments, segment_transforms, trajectory_by_frame_id)

    frame_id_to_segment_idx: dict[int, int] = {}
    for seg_idx, seg in enumerate(segments):
        for fid in seg.frame_ids:
            frame_id_to_segment_idx[fid] = seg_idx

    new_model = {
        "cameras": dict(model["cameras"]),
        "images": {k: dict(v) for k, v in model["images"].items()},
        "points3d": {k: dict(v) for k, v in model["points3d"].items()},
    }

    image_id_to_segment_idx: dict[int, int] = {}
    for image_id, img in new_model["images"].items():
        frame_id = int(img["name"].split(".")[0])
        seg_idx = frame_id_to_segment_idx.get(frame_id)
        if seg_idx is None:
            continue
        image_id_to_segment_idx[image_id] = seg_idx

        chain = chains[seg_idx]
        qvec_w2c = np.array(img["qvec"], dtype=np.float64)
        tvec_w2c = np.array(img["tvec"], dtype=np.float64)
        R_c2w, t_c2w = colmap_pose_to_c2w(qvec_w2c, tvec_w2c)

        R_c2w_new, t_c2w_new = _apply_chain_to_pose(chain, R_c2w, t_c2w)

        R_w2c_new = R_c2w_new.T
        t_w2c_new = -R_w2c_new @ t_c2w_new
        q_xyzw = rotation_matrix_to_quaternion(R_w2c_new)
        img["qvec"] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        img["tvec"] = t_w2c_new.tolist()

    for point in new_model["points3d"].values():
        observer_segments = [
            image_id_to_segment_idx[iid] for iid in point.get("image_ids", []) if iid in image_id_to_segment_idx
        ]
        if not observer_segments:
            continue
        counts = np.bincount(observer_segments)
        seg_idx = int(np.argmax(counts))
        chain = chains[seg_idx]
        xyz = np.array(point["xyz"], dtype=np.float64)
        point["xyz"] = _apply_chain_to_point(chain, xyz).tolist()

    return new_model


def correct_scale_regimes(
    workspace,
    config: dict[str, Any] | None = None,
) -> ScaleRegimeCorrectionResult:
    """Orchestrates the full detect -> estimate -> apply -> write-back flow
    for one workspace's colmap/sparse/0 model. Conservative like
    pose_outliers.py: only rewrites the model when >1 scale-regime segment
    is actually detected; a single-segment (no discontinuity) result is a
    safe no-op. Unlike the pose-outlier filter, there's no "ambiguous,
    needs manual review" case here by construction -- every detected
    segment gets its OWN independent metric anchor via real depth, so
    there's no minority-segment-to-drop decision to get wrong; segments
    that turn out to have too few correspondences for a reliable fit are
    logged clearly and left un-corrected (their original pose/points kept)
    rather than guessed at.
    """
    from pathlib import Path
    import shutil
    import tempfile

    from colmap_rgbd_gt.dataset.schema import Workspace
    from colmap_rgbd_gt.dataset.manifest import Manifest
    from colmap_rgbd_gt.colmap.reconstruction import load_sparse_model, ensure_text_model
    from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory
    from colmap_rgbd_gt.colmap.colmap_io import write_cameras_text, write_images_text, write_points3d_text
    from colmap_rgbd_gt.utils.camera import CameraIntrinsics
    from colmap_rgbd_gt.ingest.camera_info import resolve_camera_info

    config = config or {}
    workspace = Path(workspace)
    ws = Workspace(workspace)

    sparse_dir = ws.layout.colmap / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = ws.layout.colmap / "sparse"
    if not sparse_dir.exists():
        return ScaleRegimeCorrectionResult(action_taken=False, reason="no sparse model found")

    manifest = Manifest.load(ws.layout.manifest)
    intrinsics_data, source = resolve_camera_info(manifest.camera_info, config.get("camera_fallback_profile"))
    if not intrinsics_data:
        return ScaleRegimeCorrectionResult(action_taken=False, reason="no valid camera intrinsics available")
    intrinsics = CameraIntrinsics(
        fx=intrinsics_data["K"][0], fy=intrinsics_data["K"][4],
        cx=intrinsics_data["K"][2], cy=intrinsics_data["K"][5],
        width=intrinsics_data["width"], height=intrinsics_data["height"],
        distortion_model=intrinsics_data.get("distortion_model", "plumb_bob"),
        distortion_coeffs=intrinsics_data.get("D", []),
    )

    model = load_sparse_model(sparse_dir)
    trajectory = sorted(extract_trajectory(workspace), key=lambda e: e["frame_id"])
    if len(trajectory) < 2:
        return ScaleRegimeCorrectionResult(action_taken=False, reason="fewer than 2 poses")
    sorted_frame_ids = [e["frame_id"] for e in trajectory]
    trajectory_by_frame_id = {e["frame_id"]: e for e in trajectory}

    logger.info("scale_regime_correction: computing per-frame COLMAP-to-metric depth ratios...")
    per_frame_ratios = compute_per_frame_scale_ratios(model, trajectory, ws, intrinsics)
    segments = detect_scale_regime_segments(
        per_frame_ratios, sorted_frame_ids,
        jump_ratio_threshold=config.get("jump_ratio_threshold", DEFAULT_JUMP_RATIO_THRESHOLD),
        min_segment_frames=config.get("min_segment_frames", DEFAULT_MIN_SEGMENT_FRAMES),
    )

    segments_info = [
        {"n_frames": len(s.frame_ids), "frame_id_range": [s.frame_ids[0], s.frame_ids[-1]], "median_ratio": s.median_ratio}
        for s in segments
    ]

    if len(segments) <= 1:
        return ScaleRegimeCorrectionResult(
            action_taken=False, reason="no scale-regime discontinuity found; one consistent segment",
            n_segments=1, segments=segments_info,
        )

    logger.warning(
        f"scale_regime_correction: found {len(segments)} internally-inconsistent scale regimes "
        f"in the reconstruction -- {segments_info}. Independently anchoring each to real depth."
    )

    transforms = []
    for seg in segments:
        sim = estimate_segment_similarity(
            model, trajectory_by_frame_id, ws, intrinsics, seg.frame_ids, coarse_ratio=seg.median_ratio,
        )
        logger.info(
            f"scale_regime_correction: segment ({seg.frame_ids[0]}-{seg.frame_ids[-1]}, "
            f"{len(seg.frame_ids)} frames, raw ratio {seg.median_ratio:.3f}) -> similarity fit "
            f"scale={sim.scale:.4f} confidence={sim.confidence:.2%} n_samples={sim.num_samples}"
        )
        transforms.append(sim)

    low_confidence = [i for i, t in enumerate(transforms) if t.confidence < 0.3]
    if low_confidence:
        logger.warning(
            f"scale_regime_correction: segment(s) {low_confidence} have low-confidence similarity "
            "fits (too few real-depth correspondences for a reliable anchor) -- applying anyway since "
            "leaving them at their original (known-inconsistent) scale is strictly worse, but flagging "
            "for awareness."
        )

    new_model = apply_segment_similarities(model, segments, transforms, trajectory_by_frame_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        write_cameras_text(tmpdir / "cameras.txt", new_model["cameras"])
        write_images_text(tmpdir / "images.txt", new_model["images"])
        write_points3d_text(tmpdir / "points3D.txt", new_model["points3d"])

        import subprocess
        import shutil as _shutil
        colmap_exe = _shutil.which(config.get("colmap_path", "colmap"))
        if colmap_exe:
            result = subprocess.run(
                [colmap_exe, "model_converter", "--input_path", str(tmpdir),
                 "--output_path", str(tmpdir), "--output_type", "BIN"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.error(f"scale_regime_correction: model_converter to BIN failed: {result.stderr.strip()}")
                return ScaleRegimeCorrectionResult(
                    action_taken=False, reason="failed to write corrected binary model",
                    n_segments=len(segments), segments=segments_info,
                )
        else:
            logger.warning("scale_regime_correction: colmap not found, writing TEXT-only corrected model")

        for pattern in ("*.bin", "*.txt"):
            for f in sparse_dir.glob(pattern):
                f.unlink()
        for f in tmpdir.iterdir():
            if f.name in ("cameras.txt", "images.txt", "points3D.txt", "cameras.bin", "images.bin", "points3D.bin"):
                shutil.copy2(f, sparse_dir / f.name)

    ensure_text_model(sparse_dir, colmap_path=config.get("colmap_path", "colmap"))

    logger.info(
        f"scale_regime_correction: colmap/sparse model at {sparse_dir} corrected -- "
        f"{len(segments)} scale regimes independently anchored to real depth"
    )
    return ScaleRegimeCorrectionResult(
        action_taken=True,
        reason=f"corrected {len(segments)} scale-regime segments",
        n_segments=len(segments),
        segments=segments_info,
    )
