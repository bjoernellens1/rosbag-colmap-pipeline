"""Camera info extraction from rosbag messages."""

from typing import Any
import numpy as np


def extract_camera_info(msg: Any) -> dict:
    K = list(msg.k) if hasattr(msg, "k") else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    D = list(msg.d) if hasattr(msg, "d") else []
    R = list(msg.r) if hasattr(msg, "r") else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    P = list(msg.p) if hasattr(msg, "p") else [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

    return {
        "width": msg.width,
        "height": msg.height,
        "K": K,
        "D": D,
        "R": R,
        "P": P,
        "distortion_model": getattr(msg, "distortion_model", "plumb_bob"),
        "binning_x": getattr(msg, "binning_x", 1),
        "binning_y": getattr(msg, "binning_y", 1),
    }


def validate_camera_info(info: dict) -> list[str]:
    errors = []

    if info["width"] <= 0:
        errors.append(f"Invalid width: {info['width']}")
    if info["height"] <= 0:
        errors.append(f"Invalid height: {info['height']}")

    K = info["K"]
    if len(K) != 9:
        errors.append(f"K matrix must have 9 elements, got {len(K)}")
    else:
        if K[0] <= 0 or K[4] <= 0:
            errors.append(f"Invalid focal length in K: fx={K[0]}, fy={K[4]}")
        if K[1] != 0 or K[3] != 0:
            errors.append("K matrix should have zero skew")

    return errors


def get_intrinsics_matrix(info: dict) -> np.ndarray:
    K = info["K"]
    return np.array(K).reshape(3, 3)


def get_distortion_coeffs(info: dict) -> np.ndarray:
    D = info["D"]
    return np.array(D) if D else np.zeros(5)
