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

    logger.info("=" * 60)
    logger.info("Stage 3/3: Scale Estimation")
    logger.info("=" * 60)
    if not scale_pipeline(workspace, config):
        logger.error("Scale estimation stage failed")
        return False

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
