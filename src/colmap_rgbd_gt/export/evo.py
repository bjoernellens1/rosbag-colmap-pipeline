"""evo tool compatibility utilities."""

from pathlib import Path
from typing import Any
import numpy as np

from colmap_rgbd_gt.export.tum import export_tum
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def export_for_evo(
    poses: list[dict[str, Any]],
    path: Path,
    format_type: str = "tum"
) -> None:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".tum", "") or format_type == "tum":
        tum_path = path.with_suffix(".tum") if ext == "" else path
        export_tum(poses, tum_path)
        logger.info(f"Exported evo-compatible TUM file to {tum_path}")

    elif ext == ".kitti" or format_type == "kitti":
        export_kitti(poses, path)
        logger.info(f"Exported evo-compatible KITTI file to {path}")

    else:
        export_tum(poses, path)


def export_kitti(
    poses: list[dict[str, Any]],
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for pose in poses:
            R = pose["R"]
            t = pose["t"]

            mat = np.eye(4)
            mat[:3, :3] = R
            mat[:3, 3] = t

            flat = mat[:3, :].flatten()
            f.write(" ".join(f"{v:.6f}" for v in flat) + "\n")

    logger.info(f"Exported {len(poses)} poses in KITTI format to {path}")
