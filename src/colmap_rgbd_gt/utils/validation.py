"""Input validation utilities."""

from pathlib import Path
from typing import Any


def validate_bag_path(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    suffix = path.suffix.lower()
    return suffix in (".bag", ".db3", ".mcap")


def validate_workspace(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    if not path.is_dir():
        return False
    required = ["manifest.json", "rgb", "depth"]
    return all((path / r).exists() for r in required)


def validate_config(config: dict[str, Any]) -> list[str]:
    errors = []

    if "bag" in config:
        bag_config = config["bag"]
        if "path" not in bag_config:
            errors.append("Missing 'bag.path' in config")

    if "topics" in config:
        topics = config["topics"]
        if "rgb" not in topics and "rgb_compressed" not in topics:
            errors.append("No RGB topic specified in config")
        if "depth" not in topics and "pointcloud" not in topics:
            errors.append("No depth or pointcloud topic specified in config")

    if "sync" in config:
        sync = config["sync"]
        if "max_rgb_depth_dt_sec" in sync:
            if sync["max_rgb_depth_dt_sec"] <= 0:
                errors.append("sync.max_rgb_depth_dt_sec must be positive")
        if "max_rgb_info_dt_sec" in sync:
            if sync["max_rgb_info_dt_sec"] <= 0:
                errors.append("sync.max_rgb_info_dt_sec must be positive")

    return errors


def validate_colmap_output(workspace: Path) -> list[str]:
    errors = []
    colmap_dir = workspace / "colmap"

    if not colmap_dir.exists():
        errors.append(f"COLMAP directory not found: {colmap_dir}")
        return errors

    sparse_dir = colmap_dir / "sparse"
    if not sparse_dir.exists():
        errors.append(f"Sparse reconstruction not found: {sparse_dir}")
        return errors

    required_files = ["cameras.txt", "images.txt", "points3D.txt"]
    for f in required_files:
        if not (sparse_dir / f).exists():
            errors.append(f"Missing COLMAP output file: {f}")

    return errors
