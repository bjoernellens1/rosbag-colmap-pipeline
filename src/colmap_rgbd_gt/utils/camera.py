"""Camera intrinsics handling."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str = "plumb_bob"
    distortion_coeffs: list[float] = field(default_factory=list)

    @classmethod
    def from_camera_info(cls, msg: Any) -> "CameraIntrinsics":
        K = msg.k
        D = list(msg.d) if hasattr(msg, "d") else []
        return cls(
            fx=float(K[0]),
            fy=float(K[4]),
            cx=float(K[2]),
            cy=float(K[5]),
            width=msg.width,
            height=msg.height,
            distortion_model=getattr(msg, "distortion_model", "plumb_bob"),
            distortion_coeffs=[float(d) for d in D],
        )

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def to_colmap_str(self) -> str:
        params = [self.fx, self.fy, self.cx, self.cy]
        params.extend(self.distortion_coeffs[:5] if self.distortion_coeffs else [0, 0, 0, 0, 0])
        return " ".join(map(str, params))

    def get_colmap_model(self) -> str:
        if not self.distortion_coeffs or all(d == 0 for d in self.distortion_coeffs):
            return "PINHOLE"
        if self.distortion_model in ("plumb_bob", "rational_polynomial"):
            return "OPENCV"
        return "SIMPLE_RADIAL"

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        z = depth
        x = (uv[:, 0] - self.cx) * z / self.fx
        y = (uv[:, 1] - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        u = points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx
        v = points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy
        return np.stack([u, v], axis=-1)


def backproject_depth(depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    uv = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float64)
    depth_flat = depth.ravel().astype(np.float64)

    valid = depth_flat > 0
    uv_valid = uv[valid]
    depth_valid = depth_flat[valid]

    points = intrinsics.backproject(uv_valid, depth_valid)
    return points


def create_pixel_grid(height: int, width: int) -> np.ndarray:
    v, u = np.mgrid[0:height, 0:width]
    return np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float64)
