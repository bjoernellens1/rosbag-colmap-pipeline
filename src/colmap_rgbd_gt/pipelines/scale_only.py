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
    except Exception as e:
        logger.warning(f"Could not run disconnected-segment pose filter: {e}")

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
