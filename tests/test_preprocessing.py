"""Tests for the generic input-preprocessing decompression module."""

import numpy as np
import cv2
import pytest

from colmap_rgbd_gt.preprocessing import decompress_image_bytes, parse_compressed_format


def test_decompress_image_bytes_png_roundtrip_color():
    img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok

    decoded = decompress_image_bytes(buf.tobytes(), imread_flag=cv2.IMREAD_COLOR)

    np.testing.assert_array_equal(decoded, img)


def test_decompress_image_bytes_png_roundtrip_uint16_unchanged():
    depth = np.random.randint(0, 65535, (20, 30), dtype=np.uint16)
    ok, buf = cv2.imencode(".png", depth)
    assert ok

    decoded = decompress_image_bytes(buf.tobytes(), imread_flag=cv2.IMREAD_UNCHANGED)

    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, depth)


def test_decompress_image_bytes_raises_on_garbage():
    with pytest.raises(ValueError, match="Failed to decompress"):
        decompress_image_bytes(b"not an image", imread_flag=cv2.IMREAD_UNCHANGED)


def test_parse_compressed_format_depth():
    parsed = parse_compressed_format("16UC1; png compressed ")
    assert parsed == {"encoding": "16uc1", "codec": "png"}


def test_parse_compressed_format_rgb():
    parsed = parse_compressed_format("rgb8; jpeg compressed bgr8")
    assert parsed == {"encoding": "rgb8", "codec": "jpeg"}


def test_parse_compressed_format_no_semicolon():
    parsed = parse_compressed_format("jpeg")
    assert parsed["encoding"] == "jpeg"
    assert parsed["codec"] == "jpeg"


def test_parse_compressed_format_empty():
    parsed = parse_compressed_format("")
    assert parsed == {"encoding": "", "codec": ""}
