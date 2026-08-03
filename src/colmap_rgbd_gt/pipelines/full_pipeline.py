"""Full pipeline combining extraction, COLMAP, and scale estimation."""

from pathlib import Path
from typing import Any
from datetime import datetime

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.pipelines.extract_only import extract_pipeline
from colmap_rgbd_gt.pipelines.colmap_only import colmap_pipeline
from colmap_rgbd_gt.pipelines.scale_only import scale_pipeline

logger = get_logger(__name__)


def full_pipeline(
    bag_path: Path,
    workspace: Path | None,
    config: dict[str, Any]
) -> bool:
    bag_path = Path(bag_path)

    if not bag_path.exists():
        logger.error(f"Bag file not found: {bag_path}")
        return False

    if workspace is None:
        workspace = bag_path.with_suffix("")

    workspace = Path(workspace)

    start_time = datetime.now()
    logger.info(f"Starting full pipeline at {start_time.isoformat()}")

    logger.info("=" * 60)
    logger.info("Stage 1/3: Extraction")
    logger.info("=" * 60)
    if not extract_pipeline(bag_path, workspace, config):
        logger.error("Extraction stage failed")
        return False

    logger.info("=" * 60)
    logger.info("Stage 2/3: COLMAP Reconstruction")
    logger.info("=" * 60)
    if not colmap_pipeline(workspace, config):
        logger.error("COLMAP stage failed")
        return False

    depth_ba_enabled = config.get("depth_ba", {}).get("enabled", False)
    total_stages = 4 if depth_ba_enabled else 3

    logger.info("=" * 60)
    logger.info(f"Stage 3/{total_stages}: Scale Estimation")
    logger.info("=" * 60)
    if not scale_pipeline(workspace, config):
        logger.error("Scale estimation stage failed")
        return False

    if depth_ba_enabled:
        logger.info("=" * 60)
        logger.info(f"Stage 4/{total_stages}: Depth-Aware Bundle Adjustment")
        logger.info("=" * 60)
        from colmap_rgbd_gt.pipelines.depth_ba_pipeline import depth_ba_pipeline
        # FIXED 2026-08-03: kornia_rs's solver can raise (not just return
        # False) on a genuinely ill-conditioned problem -- e.g. a
        # "Reduced camera Cholesky failed (likely rank-deficient)"
        # ValueError, observed on a floor2 rerun whose scale estimation
        # had degenerated to confidence=0.0 (a separate root cause, fixed
        # upstream in configs/navigation.yaml's keyframe_selection
        # wiring). Before this fix, an uncaught exception here crashed
        # the ENTIRE `full` pipeline with exit 1, discarding the already-
        # successful scale-only stage's output (export_report.json/
        # scale_report.json/trajectory_metric_tum.txt) even though the
        # comment below already documents the intent of falling back to
        # it gracefully -- that fallback only ever triggered on a `False`
        # return, never on an exception. depth-ba is explicitly a
        # best-effort refinement (see optimization/depth_ba.py's own
        # "young solver, no dedicated upstream tests" docstring note), so
        # any failure mode here must degrade to the scale-only trajectory,
        # not take down a pipeline run that had otherwise fully succeeded.
        try:
            depth_ba_ok = depth_ba_pipeline(workspace, config)
        except Exception as e:
            logger.warning(f"Depth BA stage raised {type(e).__name__}: {e}")
            depth_ba_ok = False
        if not depth_ba_ok:
            logger.warning(
                "Depth BA stage failed; keeping scale-only trajectory as the "
                "pipeline's final output"
            )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Total duration: {duration:.1f} seconds")
    logger.info(f"Output directory: {workspace}")

    from colmap_rgbd_gt.dataset.schema import Workspace
    ws = Workspace(workspace)
    output_file = ws.layout.outputs / "trajectory_metric_tum.txt"
    logger.info(f"Metric trajectory: {output_file}")

    return True
