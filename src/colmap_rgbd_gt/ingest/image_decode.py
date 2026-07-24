"""RGB image decoding from rosbag messages."""

import numpy as np
from typing import Any
import cv2

from colmap_rgbd_gt.preprocessing import decompress_image_bytes


def decode_rgb_image(msg: Any) -> np.ndarray:
    encoding = msg.encoding.lower()

    if encoding in ("rgb8", "rgba", "bgr8", "bgra"):
        height = msg.height
        width = msg.width
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == "rgb8":
            img = data.reshape((height, width, 3))
            img = img[:, :, ::-1].copy()
        elif encoding == "rgba":
            img = data.reshape((height, width, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif encoding == "bgr8":
            img = data.reshape((height, width, 3)).copy()
        elif encoding == "bgra":
            img = data.reshape((height, width, 4))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif encoding == "mono8":
            img = data.reshape((height, width)).copy()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = data.reshape((height, width, 3)).copy()

        return img

    elif encoding in ("16uc1", "mono16"):
        height = msg.height
        width = msg.width
        data = np.frombuffer(msg.data, dtype=np.uint16)
        img = data.reshape((height, width))
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(img_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    else:
        raise ValueError(f"Unsupported RGB encoding: {encoding}")


def decode_compressed_image(msg: Any) -> np.ndarray:
    return decompress_image_bytes(bytes(msg.data), imread_flag=cv2.IMREAD_COLOR)


def decode_image(msg: Any) -> np.ndarray:
    msgtype = type(msg).__name__
    if msgtype == "CompressedImage" or hasattr(msg, "format"):
        return decode_compressed_image(msg)
    return decode_rgb_image(msg)
