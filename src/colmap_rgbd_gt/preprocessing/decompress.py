"""Generic byte-level decompression of compressed sensor payloads.

Rosbag drivers publish compressed images (and, in principle, other
compressed sensor modalities) using a variety of container formats and
`format`-string conventions. This module centralizes the "raw bytes ->
decoded array" step so RGB and depth decoders (and future modalities) share
one preprocessing layer instead of each re-implementing codec dispatch.
"""

import numpy as np
import cv2


def decompress_image_bytes(data: bytes, imread_flag: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Decode a compressed image payload (PNG/JPEG/...) into a raw array.

    `cv2.imdecode` auto-detects the container format (PNG, JPEG, ...) from
    the payload's magic bytes, so no explicit codec name is required here.

    Args:
        data: raw compressed bytes (e.g. a CompressedImage message's `data`).
        imread_flag: controls channel/bit-depth handling --
            `cv2.IMREAD_UNCHANGED` preserves the original bit depth/channel
            count (needed for 16-bit depth), `cv2.IMREAD_COLOR` forces a
            3-channel BGR image (RGB payloads).

    Raises:
        ValueError: if the payload could not be decoded (corrupt data or an
            unrecognized container format).
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, imread_flag)
    if img is None:
        raise ValueError("Failed to decompress image payload (unrecognized or corrupt data)")
    return img


def parse_compressed_format(format_str: str) -> dict[str, str]:
    """Parse a ROS `CompressedImage.format` string into its components.

    Handles both `image_transport` conventions seen in practice:
      - RGB: `"rgb8; jpeg compressed bgr8"` -> encoding="rgb8", codec="jpeg"
      - Depth: `"16UC1; png compressed "` -> encoding="16uc1", codec="png"

    Returns:
        dict with 'encoding' (e.g. "16uc1", "rgb8") and 'codec' (e.g.
        "png", "jpeg"), both lowercased. Missing parts are empty strings --
        callers should fall back to sniffing the payload itself when a
        field is empty (e.g. `decompress_image_bytes` already does this
        implicitly via `cv2.imdecode`'s magic-byte detection).
    """
    fmt = (format_str or "").lower()
    parts = [p.strip() for p in fmt.split(";")]

    encoding = parts[0] if parts else ""

    codec = ""
    rest = parts[1] if len(parts) > 1 else fmt
    for candidate in ("png", "jpeg", "jpg"):
        if candidate in rest:
            codec = "jpeg" if candidate == "jpg" else candidate
            break

    return {"encoding": encoding, "codec": codec}
