"""Depth image decoding from rosbag messages."""

import numpy as np
from typing import Any

from colmap_rgbd_gt.preprocessing import decompress_image_bytes, parse_compressed_format
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def decode_compressed_depth_image(msg: Any, scale: float = 0.001) -> np.ndarray:
    """Decode a `sensor_msgs/CompressedImage` depth message.

    Unlike RGB compressed transport, ROS depth compression has no single
    standard: some drivers wrap the payload in `image_transport`'s
    compressedDepth 12-byte quantization header, others (as observed here)
    publish a plain PNG of the raw encoding with a `format` string like
    `"16UC1; png compressed "`. This handles the plain-PNG case -- the
    payload is decoded via `preprocessing.decompress_image_bytes`
    (bit-depth-preserving), with the parsed `format` string used only to
    decide whether the decoded array is already metric (32FC1) or needs
    the mm->m `scale` applied (16UC1 and unrecognized formats).
    """
    parsed = parse_compressed_format(msg.format)
    img = decompress_image_bytes(bytes(msg.data))

    if parsed["encoding"] == "32fc1":
        return img.astype(np.float64)

    if parsed["encoding"] not in ("16uc1", "mono16"):
        logger.warning(
            f"Unrecognized compressed depth format '{msg.format}', assuming "
            f"16-bit with scale={scale}"
        )

    return img.astype(np.float64) * scale


def decode_depth_image(msg: Any, scale: float = 0.001) -> np.ndarray:
    if hasattr(msg, "format"):
        return decode_compressed_depth_image(msg, scale=scale)

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
        logger.warning(
            f"Unknown depth encoding '{msg.encoding}', assuming uint16 with scale={scale}"
        )
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
