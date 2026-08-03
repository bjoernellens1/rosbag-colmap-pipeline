"""Scale estimation pipeline."""

from pathlib import Path
from typing import Any

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.scaling.scale_estimation import estimate_global_scale
from colmap_rgbd_gt.scaling.diagnostics import compute_diagnostics, plot_scale_histogram, export_scale_report
from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory, scale_trajectory
from colmap_rgbd_gt.export.tum import export_trajectory_tum
from colmap_rgbd_gt.export.evo import plot_trajectory
from colmap_rgbd_gt.export.report import generate_report, save_report
from colmap_rgbd_gt.export.scene_metadata import compute_scene_metadata, save_scene_metadata
from colmap_rgbd_gt.export.rosbag_writer import _load_frame_timestamps
from colmap_rgbd_gt.colmap.pose_outliers import filter_disconnected_trajectory_segments
from colmap_rgbd_gt.dataset.schema import Workspace

logger = get_logger(__name__)


def scale_pipeline(workspace: Path, config: dict[str, Any]) -> bool:
    workspace = Path(workspace)

    ws = Workspace(workspace)
    if not ws.validate():
        logger.error(f"Invalid workspace: {workspace}")
        return False

    scaling_config = config.get("scaling", {})

    logger.info("Estimating metric scale from depth...")
    scale_estimate = estimate_global_scale(workspace, scaling_config)

    logger.info(
        f"Estimated scale: {scale_estimate.scale:.6f} "
        f"(confidence: {scale_estimate.confidence:.2%}, method: {scale_estimate.method})"
    )

    diagnostics = compute_diagnostics([scale_estimate])

    try:
        plot_scale_histogram(
            diagnostics,
            ws.layout.outputs / "scale_histogram.png"
        )
    except Exception as e:
        logger.warning(f"Could not plot histogram: {e}")

    export_scale_report(diagnostics, scale_estimate, ws.layout.outputs / "scale_report.json")

    logger.info("Scaling trajectory to metric units...")
    trajectory = extract_trajectory(workspace)

    if not trajectory:
        logger.error("No trajectory found")
        return False

    metric_trajectory = scale_trajectory(trajectory, scale_estimate.scale)

    # FIXED 2026-08-03: global_mapper's global positioning solve can place a
    # small, weakly-connected sub-scene at a wrong absolute position even
    # though it's internally self-consistent -- root-caused on a real scene
    # (floor2): two frames 0.43s apart in real capture time, showing the
    # IDENTICAL physical location, ended up ~7.4m apart (implied ~17 m/s,
    # physically impossible). scene_metadata.json's leading-frames mechanism
    # already flags this pattern; act on it here by dropping a minority
    # segment that's disconnected from a much larger majority segment by an
    # implausible-speed jump, so the exported GT trajectory (and everything
    # downstream: TUM export, scene_metadata, depth-ba) doesn't carry a
    # known-bad chunk of poses. Conservative: only auto-drops when the
    # majority segment dominates (see pose_outliers.py); an ambiguous split
    # is left untouched and logged loudly for manual review instead.
    pose_filter_result = None
    try:
        frame_id_to_ts_ns = _load_frame_timestamps(ws.layout.timestamps / "rgb.csv")
        pose_filter_result = filter_disconnected_trajectory_segments(
            metric_trajectory, frame_id_to_ts_ns,
        )
        if pose_filter_result.action_taken:
            logger.warning(
                f"Dropped {len(pose_filter_result.dropped_frame_ids)} disconnected-segment "
                f"frame(s) from the exported trajectory: {pose_filter_result.dropped_frame_ids}"
            )
            metric_trajectory = pose_filter_result.filtered_trajectory

            # FIXED 2026-08-03: dropping frames from the EXPORTED trajectory
            # only (trajectory_metric_tum.txt / scene_metadata.json /
            # depth-ba's inputs) left colmap/sparse/0 itself untouched --
            # anyone opening the raw sparse model directly in COLMAP's own
            # GUI (a normal, expected sanity-check workflow) still saw the
            # original, uncleaned reconstruction with the mis-positioned
            # minority segment intact. Remove the same frames from the
            # actual sparse model on disk too, via COLMAP's own
            # image_deleter (see reconstruction.remove_images_from_sparse_
            # model's docstring for why this, not a hand-rolled binary
            # edit), so the raw model matches every exported artifact.
            sparse_dir = ws.layout.colmap / "sparse" / "0"
            if not sparse_dir.exists():
                sparse_dir = ws.layout.colmap / "sparse"
            image_names_to_delete = [f"{fid:06d}.png" for fid in pose_filter_result.dropped_frame_ids]
            colmap_path = config.get("colmap", {}).get("colmap_path", "colmap")
            from colmap_rgbd_gt.colmap.reconstruction import remove_images_from_sparse_model
            if remove_images_from_sparse_model(sparse_dir, image_names_to_delete, colmap_path=colmap_path):
                logger.info(f"colmap/sparse model at {sparse_dir} updated to match the filtered trajectory")
            else:
                logger.error(
                    f"Failed to remove dropped frames from the sparse model at {sparse_dir} -- "
                    "exported trajectory artifacts ARE clean, but the raw COLMAP model still "
                    "contains the disconnected segment. Investigate before trusting a direct "
                    "inspection of colmap/sparse/0 for this scene."
                )
    except Exception as e:
        logger.warning(f"Could not run disconnected-segment pose filter: {e}")

    # FIXED 2026-08-03: even after the pose-outlier filter above produces
    # one smooth, continuous camera trajectory, colmap/sparse/0's 3D
    # POINTS can still carry a DIFFERENT defect: two weakly-connected
    # regions of the reconstruction settling into two different implicit
    # scales during retriangulation, even though their camera poses stay
    # positionally continuous. Root-caused on a real scene (floor2): a
    # user directly inspecting the raw sparse model saw two visually
    # distinct, offset "sheets"/a floating slab of points -- the same
    # physical surface reconstructed twice at inconsistent scale. Verified
    # `colmap bundle_adjuster` with 300 extra iterations does NOT fix this
    # (NO_CONVERGENCE, ratio unchanged) -- it's a genuinely stable,
    # degenerate local minimum from a weak match-graph link, not an
    # under-iterated one. See colmap/scale_regime_correction.py's module
    # docstring for the full root-cause writeup and the independent-
    # per-region-depth-anchoring fix.
    regime_result = None
    try:
        from colmap_rgbd_gt.colmap.scale_regime_correction import correct_scale_regimes
        regime_result = correct_scale_regimes(workspace, {
            "camera_fallback_profile": scaling_config.get("camera_fallback_profile"),
            "colmap_path": config.get("colmap", {}).get("colmap_path", "colmap"),
        })
        if regime_result.action_taken:
            logger.warning(
                f"Corrected {regime_result.n_segments} internally-inconsistent scale regime(s) "
                f"in colmap/sparse model: {regime_result.segments}"
            )
        try:
            import json
            with open(ws.layout.outputs / "scale_regime_correction.json", "w") as f:
                json.dump({
                    "action_taken": regime_result.action_taken,
                    "reason": regime_result.reason,
                    "n_segments": regime_result.n_segments,
                    "segments": regime_result.segments,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save scale_regime_correction.json: {e}")

        if regime_result.action_taken:
            # The in-memory metric_trajectory was extracted+scaled BEFORE
            # this correction touched colmap/sparse/0's poses -- it's now
            # stale relative to the corrected model. Re-extract so the
            # exported TUM/scene_metadata/depth-ba artifacts stay
            # consistent with what's actually in the sparse model, not a
            # pre-correction snapshot.
            #
            # IMPORTANT: do NOT re-apply scale_trajectory(..., scale_estimate
            # .scale) here -- each corrected segment's Sim3 was fit directly
            # against real depth (estimate_similarity_umeyama(depth_points_
            # METRIC, colmap_points_RAW)), so colmap/sparse/0's poses are
            # ALREADY in real metric units after correction. Applying the
            # earlier (pre-correction, single-global-scale) scale_estimate
            # on top would double-scale them.
            refreshed = extract_trajectory(workspace)
            if refreshed:
                metric_trajectory = refreshed
                logger.info("Re-extracted (already-metric) trajectory from colmap/sparse/0 after scale-regime correction")
    except Exception as e:
        logger.warning(f"Could not run scale-regime correction: {e}")

    # QC-only (2026-08-03, DETECTION not correction -- see module docstring
    # in duplicate_surface_detection.py): investigating floor2's reported
    # "floating slab" after the scale-regime fix above, that specific
    # cluster turned out to be the REAL corridor floor (confirmed by
    # color, sensor-depth backprojection, and image-space reprojection),
    # not a duplicate -- a naive "planar cluster offset in Y" heuristic is
    # NOT a valid detector, it fires on every healthy floor. This runs the
    # correctly-gated version (requires a genuine vertical gap AND shared
    # image-space reprojection footprint between two clusters, not just an
    # offset) purely for visibility -- it does not modify colmap/sparse/0.
    try:
        sparse_dir_qc = ws.layout.colmap / "sparse" / "0"
        if not sparse_dir_qc.exists():
            sparse_dir_qc = ws.layout.colmap / "sparse"
        from colmap_rgbd_gt.colmap.duplicate_surface_detection import run_duplicate_surface_qc
        # This module's thresholds are calibrated in real meters (see its
        # docstring). colmap/sparse/0 is only already-metric when the
        # scale-regime correction above actually rewrote it in place;
        # otherwise it is still in COLMAP's raw/unscaled unit space and
        # must be rescaled by the scene's own estimated scale factor first
        # -- found 2026-08-03 on table1, where unscaled candidates read as
        # a suspicious ~0.7 apart but were really ~9cm apart once scaled.
        dup_points_scale = 1.0 if (regime_result is not None and regime_result.action_taken) else scale_estimate.scale
        dup_result = run_duplicate_surface_qc(sparse_dir_qc, points_scale=dup_points_scale)
        if dup_result.detected:
            logger.warning(
                f"duplicate_surface_detection: {len(dup_result.candidates)} candidate duplicate "
                f"planar-surface pair(s) found in {sparse_dir_qc} -- inspect before trusting this "
                "scene's point cloud; see duplicate_surface_qc.json"
            )
        try:
            import json
            with open(ws.layout.outputs / "duplicate_surface_qc.json", "w") as f:
                json.dump({
                    "detected": dup_result.detected,
                    "n_clusters_found": dup_result.n_clusters_found,
                    "reason": dup_result.reason,
                    "candidates": dup_result.candidates,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save duplicate_surface_qc.json: {e}")
    except Exception as e:
        logger.warning(f"Could not run duplicate-surface QC: {e}")

    metric_tum_path = ws.layout.outputs / "trajectory_metric_tum.txt"
    export_trajectory_tum(metric_trajectory, metric_tum_path)

    try:
        plot_trajectory(
            metric_tum_path,
            ws.layout.outputs / "trajectory_plot.png",
            title="Metric trajectory (top-down XY)",
        )
    except Exception as e:
        logger.warning(f"Could not plot trajectory: {e}")

    report = None
    try:
        report = generate_report(workspace, scale_estimate, diagnostics)
        save_report(report, ws.layout.outputs / "export_report.json")
    except Exception as e:
        logger.warning(f"Could not generate report: {e}")

    if pose_filter_result is not None:
        try:
            import json
            with open(ws.layout.outputs / "pose_outlier_filter.json", "w") as f:
                json.dump({
                    "action_taken": pose_filter_result.action_taken,
                    "reason": pose_filter_result.reason,
                    "dropped_frame_ids": pose_filter_result.dropped_frame_ids,
                    "segments": pose_filter_result.segments,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save pose_outlier_filter.json: {e}")

    try:
        scene_metadata = compute_scene_metadata(
            metric_trajectory,
            ws.layout.timestamps / "rgb.csv",
            frame_count=report.frame_count if report else None,
            # len(metric_trajectory), not report.registered_images: the
            # latter is COLMAP's raw registration count, computed BEFORE
            # the disconnected-segment filter above -- registered_count
            # here must reflect what's actually in the exported trajectory.
            registered_count=len(metric_trajectory),
        )
        save_scene_metadata(scene_metadata, ws.layout.outputs / "scene_metadata.json")
    except Exception as e:
        logger.warning(f"Could not compute scene metadata: {e}")

    logger.info(f"Scale pipeline complete: scale={scale_estimate.scale:.6f}")
    return True
