"""Tests for COLMAP text I/O, including POINTS2D parsing (needed for depth-BA)."""

import numpy as np

from colmap_rgbd_gt.colmap.colmap_io import (
    read_images_text,
    read_points3d_text,
    write_images_text,
    write_points3d_text,
)


def test_read_images_text_parses_points2d(tmp_path):
    content = (
        "# comment\n"
        "1 1 0 0 0 0 0 0 1 000001.png\n"
        "100.5 200.5 3 150.0 250.0 -1\n"
        "2 1 0 0 0 1 0 0 1 000002.png\n"
        "\n"
    )
    path = tmp_path / "images.txt"
    path.write_text(content)

    images = read_images_text(path)

    np.testing.assert_array_almost_equal(
        images[1]["xys"], np.array([[100.5, 200.5], [150.0, 250.0]])
    )
    np.testing.assert_array_equal(images[1]["point3d_ids"], np.array([3, -1]))

    assert images[2]["xys"].shape == (0, 2)
    assert images[2]["point3d_ids"].shape == (0,)


def test_write_images_text_then_read_roundtrip(tmp_path):
    images = {
        1: {
            "image_id": 1,
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 0.0],
            "camera_id": 1,
            "name": "000001.png",
        }
    }
    path = tmp_path / "images.txt"
    write_images_text(path, images)

    parsed = read_images_text(path)
    assert parsed[1]["name"] == "000001.png"
    assert parsed[1]["xys"].shape == (0, 2)
    assert parsed[1]["point3d_ids"].shape == (0,)


def test_write_points3d_text_roundtrip(tmp_path):
    points = {
        1: {
            "point_id": 1,
            "xyz": [1.0, 2.0, 3.0],
            "rgb": [10, 20, 30],
            "error": 0.25,
            "image_ids": [1, 2],
            "point2d_idxs": [0, 1],
        },
        2: {
            "point_id": 2,
            "xyz": [4.0, 5.0, 6.0],
            "rgb": [1, 1, 1],
            "error": 0.1,
            "image_ids": [1],
            "point2d_idxs": [3],
        },
    }
    path = tmp_path / "points3D.txt"
    write_points3d_text(path, points)

    parsed = read_points3d_text(path)

    assert parsed[1]["xyz"] == [1.0, 2.0, 3.0]
    assert parsed[1]["rgb"] == [10, 20, 30]
    assert parsed[1]["error"] == 0.25
    assert parsed[1]["image_ids"] == [1, 2]
    assert parsed[1]["point2d_idxs"] == [0, 1]
    assert parsed[2]["xyz"] == [4.0, 5.0, 6.0]
