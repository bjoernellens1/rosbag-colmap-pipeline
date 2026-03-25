"""COLMAP reconstruction utilities."""

from pathlib import Path
from typing import Any
import numpy as np

from colmap_rgbd_gt.colmap.colmap_io import read_cameras_text, read_images_text, read_points3d_text
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def load_sparse_model(path: Path) -> dict[str, Any]:
    path = Path(path)

    cameras = read_cameras_text(path / "cameras.txt")
    images = read_images_text(path / "images.txt")
    points3d = read_points3d_text(path / "points3D.txt")

    return {
        "cameras": cameras,
        "images": images,
        "points3d": points3d,
    }


def get_image_names(model: dict[str, Any]) -> list[str]:
    return [img["name"] for img in model["images"].values()]


def get_camera(model: dict[str, Any], camera_id: int) -> dict[str, Any]:
    return model["cameras"].get(camera_id, {})


def get_image_pose(model: dict[str, Any], image_name: str) -> tuple[np.ndarray, np.ndarray]:
    for img in model["images"].values():
        if img["name"] == image_name:
            qvec = np.array(img["qvec"], dtype=np.float64)
            tvec = np.array(img["tvec"], dtype=np.float64)
            from colmap_rgbd_gt.utils.transforms import quaternion_to_rotation_matrix
            R = quaternion_to_rotation_matrix(qvec)
            return R, tvec

    raise ValueError(f"Image not found: {image_name}")


def get_points_observed_in_image(model: dict[str, Any], image_id: int) -> np.ndarray:
    points = []

    for point_id, point in model["points3d"].items():
        if image_id in point.get("image_ids", []):
            points.append(point["xyz"])

    return np.array(points, dtype=np.float64) if points else np.array([]).reshape(0, 3)


def get_image_id_by_name(model: dict[str, Any], name: str) -> int | None:
    for img_id, img in model["images"].items():
        if img["name"] == name:
            return img_id
    return None
