"""Pose extraction from COLMAP reconstruction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from colmap_rgbd_gt.colmap.reconstruction import load_sparse_model, get_image_id_by_name
from colmap_rgbd_gt.utils.transforms import (
    quaternion_to_rotation_matrix,
    get_camera_center,
)
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


@dataclass
class COLMAPPose:
    image_name: str
    image_id: int
    quaternion: np.ndarray
    translation: np.ndarray
    camera_id: int

    @property
    def qvec(self) -> np.ndarray:
        return self.quaternion

    @property
    def tvec(self) -> np.ndarray:
        return self.translation


def extract_poses(model_path: Path) -> list[COLMAPPose]:
    model = load_sparse_model(model_path)

    poses = []
    for img_id, img in model["images"].items():
        pose = COLMAPPose(
            image_name=img["name"],
            image_id=img_id,
            quaternion=np.array(img["qvec"], dtype=np.float64),
            translation=np.array(img["tvec"], dtype=np.float64),
            camera_id=img["camera_id"],
        )
        poses.append(pose)

    return poses


def colmap_pose_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = quaternion_to_rotation_matrix(qvec)
    R_c2w = R.T
    t_c2w = -R.T @ tvec
    return R_c2w, t_c2w


def extract_trajectory(workspace: Path) -> list[dict[str, Any]]:
    from colmap_rgbd_gt.dataset.schema import Workspace

    ws = Workspace(workspace)
    sparse_dir = ws.layout.colmap / "sparse" / "0"

    if not sparse_dir.exists():
        sparse_dir = ws.layout.colmap / "sparse"
        if not sparse_dir.exists():
            logger.error(f"No sparse reconstruction found in {ws.layout.colmap}")
            return []

    poses = extract_poses(sparse_dir)

    trajectory = []
    for pose in sorted(poses, key=lambda p: p.image_name):
        R_c2w, t_c2w = colmap_pose_to_c2w(pose.qvec, pose.tvec)

        frame_id = int(pose.image_name.split(".")[0])

        trajectory.append({
            "frame_id": frame_id,
            "image_name": pose.image_name,
            "R": R_c2w,
            "t": t_c2w,
            "qvec_w2c": pose.qvec,
            "tvec_w2c": pose.tvec,
        })

    return trajectory


def scale_trajectory(trajectory: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    scaled = []
    for entry in trajectory:
        scaled_entry = entry.copy()
        scaled_entry["t"] = entry["t"] * scale
        scaled.append(scaled_entry)
    return scaled
