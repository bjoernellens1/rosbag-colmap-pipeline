"""JSON report generation."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from datetime import datetime

from colmap_rgbd_gt.utils.io import save_json
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExportReport:
    bag_path: str
    frame_count: int
    registered_images: int
    estimated_scale: float
    scale_confidence: float
    scale_method: str
    colmap_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    processing_time_s: float = 0.0
    timestamp: str = ""


def generate_report(
    workspace: Path,
    scale_estimate: Any,
    diagnostics: Any
) -> ExportReport:
    from colmap_rgbd_gt.dataset.schema import Workspace
    from colmap_rgbd_gt.dataset.manifest import Manifest

    ws = Workspace(workspace)
    manifest = Manifest.load(ws.layout.manifest)

    colmap_stats = {
        "sparse_dir_exists": (ws.layout.colmap / "sparse").exists(),
        "num_cameras": 0,
        "num_images": 0,
        "num_points": 0,
    }

    try:
        from colmap_rgbd_gt.colmap.reconstruction import load_sparse_model
        sparse_dir = ws.layout.colmap / "sparse" / "0"
        if not sparse_dir.exists():
            sparse_dir = ws.layout.colmap / "sparse"

        if sparse_dir.exists():
            model = load_sparse_model(sparse_dir)
            colmap_stats["num_cameras"] = len(model.get("cameras", {}))
            colmap_stats["num_images"] = len(model.get("images", {}))
            colmap_stats["num_points"] = len(model.get("points3d", {}))
    except Exception as e:
        logger.warning(f"Could not load COLMAP stats: {e}")

    return ExportReport(
        bag_path=manifest.bag_path,
        frame_count=manifest.frame_count,
        registered_images=colmap_stats.get("num_images", 0),
        estimated_scale=scale_estimate.scale,
        scale_confidence=scale_estimate.confidence,
        scale_method=scale_estimate.method,
        colmap_stats=colmap_stats,
        errors=[],
        warnings=[],
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


def save_report(report: ExportReport, path: Path) -> None:
    data = {
        "bag_path": report.bag_path,
        "frame_count": report.frame_count,
        "registered_images": report.registered_images,
        "estimated_scale": report.estimated_scale,
        "scale_confidence": report.scale_confidence,
        "scale_method": report.scale_method,
        "colmap_stats": report.colmap_stats,
        "errors": report.errors,
        "warnings": report.warnings,
        "processing_time_s": report.processing_time_s,
        "timestamp": report.timestamp,
    }

    save_json(path, data)
    logger.info(f"Report saved to {path}")
