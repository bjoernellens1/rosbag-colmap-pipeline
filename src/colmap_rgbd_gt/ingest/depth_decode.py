"""Depth image decoding from rosbag messages."""

import numpy as np
from typing import Any


def decode_depth_image(msg: Any, scale: float = 0.001) -> np.ndarray:
    encoding = msg.encoding.lower()
    height = msg.height
    width = msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("16uc1", "mono16", "16sc1"):
        depth = data.view(dtype=np.uint16).reshape((height, width))
        depth_m = depth.astype(np.float64) * scale
    elif encoding in ("32fc1",):
        depth = data.view(dtype=np.float32).reshape((height, width))
        depth_m = depth.astype(np.float64)
    elif encoding in ("typeivar", "16bit",):
        depth = data.view(dtype=np.uint16).reshape((height, width))
        depth_m = depth.astype(np.float64) * scale
    else:
        depth = data.view(dtype=np.uint16).reshape((height, width))
        depth_m = depth.astype(np.float64) * scale

    return depth_m


def apply_depth_scale(depth: np.ndarray, scale: float) -> np.ndarray:
    return depth * scale


def filter_depth_range(
    depth: np.ndarray,
    min_m: float = 0.2,
    max_m: float = 8.0
) -> np.ndarray:
    result = depth.copy()
    result[depth < min_m] = 0
    result[depth > max_m] = 0
    return result


def depth_to_uint16(depth: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    depth_scaled = np.clip(depth * scale, 0, 65535)
    return depth_scaled.astype(np.uint16)


def depth_to_visual(depth: np.ndarray, max_m: float = 10.0) -> np.ndarray:
    valid = depth > 0
    normalized = np.zeros_like(depth)
    normalized[valid] = np.clip(depth[valid] / max_m, 0, 1)
    return (normalized * 255).astype(np.uint8)
