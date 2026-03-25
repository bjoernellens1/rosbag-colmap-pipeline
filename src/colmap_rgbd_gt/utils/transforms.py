"""SE3 transformation utilities."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Transform:
    """SE3 transformation with rotation and translation."""

    rotation: np.ndarray  # SO3 rotation matrix (3x3)
    translation: np.ndarray  # Translation vector (3,)

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> "Transform":
        rotation = mat[:3, :3]
        translation = mat[:3, 3]
        return cls(rotation=rotation, translation=translation)

    @classmethod
    def from_quaternion_translation(
        cls, q: np.ndarray, t: np.ndarray
    ) -> "Transform":
        rotation = quaternion_to_rotation_matrix(q)
        return cls(rotation=rotation, translation=t)

    def to_matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    def inverse(self) -> "Transform":
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return Transform(rotation=inv_rotation, translation=inv_translation)

    def apply(self, points: np.ndarray) -> np.ndarray:
        if points.ndim == 1:
            return self.rotation @ points + self.translation
        return (self.rotation @ points.T).T + self.translation

    def __matmul__(self, other: "Transform") -> "Transform":
        rotation = self.rotation @ other.rotation
        translation = self.rotation @ other.translation + self.translation
        return Transform(rotation=rotation, translation=translation)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    if q.shape[0] == 4:
        x, y, z, w = q
    else:
        w, x, y, z = q[0], q[1], q[2], q[3]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)
    return R


def normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def colmap_pose_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = quaternion_to_rotation_matrix(qvec)
    R_cam = R.T
    t_cam = -R.T @ tvec
    return R_cam, t_cam


def get_camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = quaternion_to_rotation_matrix(qvec)
    return -R.T @ tvec
