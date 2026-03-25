"""Scale estimation pipeline."""

from pathlib import Path
from typing import Any

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.scaling.scale_estimation import estimate_global_scale
from colmap_rgbd_gt.scaling.diagnostics import compute_diagnostics, plot_scale_histogram, export_scale_report
from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory, scale_trajectory
from colmap_rgbd_gt.export.tum import export_trajectory_tum
from colmap_rgbd_gt.export.report import generate_report, save_report
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

    export_trajectory_tum(metric_trajectory, ws.layout.outputs / "trajectory_metric_tum.txt")

    try:
        report = generate_report(workspace, scale_estimate, diagnostics)
        save_report(report, ws.layout.outputs / "export_report.json")
    except Exception as e:
        logger.warning(f"Could not generate report: {e}")

    logger.info(f"Scale pipeline complete: scale={scale_estimate.scale:.6f}")
    return True
