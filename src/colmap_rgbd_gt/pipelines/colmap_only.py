"""COLMAP reconstruction pipeline."""

from pathlib import Path
from typing import Any

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.colmap.runner import COLMAPRunner
from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory
from colmap_rgbd_gt.export.tum import export_trajectory_tum
from colmap_rgbd_gt.dataset.schema import Workspace

logger = get_logger(__name__)


def colmap_pipeline(workspace: Path, config: dict[str, Any]) -> bool:
    workspace = Path(workspace)

    ws = Workspace(workspace)
    if not ws.validate():
        logger.error(f"Invalid workspace: {workspace}")
        return False

    colmap_config = config.get("colmap", {})

    runner = COLMAPRunner(workspace, colmap_config.get("colmap_path", "colmap"))

    logger.info("Running COLMAP reconstruction...")
    success = runner.run_full_pipeline(colmap_config)

    if not success:
        logger.error("COLMAP reconstruction failed")
        return False

    logger.info("Extracting poses from COLMAP output...")
    trajectory = extract_trajectory(workspace)

    if not trajectory:
        logger.error("No poses extracted from COLMAP")
        return False

    unscaled_path = ws.layout.outputs / "trajectory_colmap_unscaled.txt"
    export_trajectory_tum(trajectory, unscaled_path, scale=1.0)

    logger.info(f"COLMAP pipeline complete: {len(trajectory)} poses")
    return True
