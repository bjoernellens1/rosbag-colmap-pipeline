"""Write a '_processed' bag (original messages + GT trajectory topics)."""

import json
from pathlib import Path
from typing import Any

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.dataset.schema import Workspace
from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory, scale_trajectory
from colmap_rgbd_gt.export.rosbag_writer import write_processed_bag

logger = get_logger(__name__)


def export_bag_pipeline(
    bag_path: Path,
    workspace: Path,
    config: dict[str, Any],
) -> bool:
    bag_path = Path(bag_path)
    workspace = Path(workspace)

    if not bag_path.exists():
        logger.error(f"Bag file not found: {bag_path}")
        return False

    ws = Workspace(workspace)
    if not ws.validate():
        logger.error(f"Invalid workspace: {workspace}")
        return False

    scale_report_path = ws.layout.outputs / "scale_report.json"
    if not scale_report_path.exists():
        logger.error(
            f"No scale report found at {scale_report_path}; run 'scale-depth' first"
        )
        return False

    with open(scale_report_path) as f:
        scale_report = json.load(f)
    scale = scale_report["scale_estimate"]["scale"]

    trajectory = extract_trajectory(workspace)
    if not trajectory:
        logger.error("No trajectory found; run 'run-colmap' first")
        return False

    metric_trajectory = scale_trajectory(trajectory, scale)

    export_config = config.get("export_bag", {})
    output_path = export_config.get("output_path")
    if output_path:
        output_path = Path(output_path)
    else:
        # ROS2 bags are directories (the storage plugin writes its own
        # "<dir_name>.mcap" file inside), so the default output path must
        # NOT carry the original bag's file suffix -- doing so produced a
        # directory literally named "*_processed.mcap" containing a
        # "*_processed.mcap.mcap" file.
        output_path = bag_path.parent / f"{bag_path.stem}_processed"

    n_copied = write_processed_bag(
        bag_path,
        metric_trajectory,
        ws.layout.timestamps / "rgb.csv",
        output_path,
        pose_topic=export_config.get("pose_topic", "/gt/colmap_pose"),
        path_topic=export_config.get("path_topic", "/gt/path"),
        frame_id=export_config.get("frame_id", "map"),
    )

    logger.info(f"Processed bag written to {output_path} ({n_copied} original messages)")
    return True
