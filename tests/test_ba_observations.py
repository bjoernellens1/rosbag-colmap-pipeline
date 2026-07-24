"""Tests for build_pose_and_point_arrays / build_ba_observations (depth-BA input building)."""

import numpy as np

from colmap_rgbd_gt.scaling.correspondences import (
    build_pose_and_point_arrays,
    build_ba_observations,
)
from colmap_rgbd_gt.utils.camera import CameraIntrinsics

FX = FY = 500.0
CX, CY = 320.0, 240.0


def _synthetic_model():
    # Two images, identity w2c pose, three 3D points at known metric depth.
    points_world = {
        1: {"point_id": 1, "xyz": [0.0, 0.0, 2.0], "image_ids": [1, 2]},
        2: {"point_id": 2, "xyz": [0.1, 0.1, 2.5], "image_ids": [1]},
        3: {"point_id": 3, "xyz": [-0.1, -0.1, 3.0], "image_ids": [2]},
    }

    def project(xyz):
        x, y, z = xyz
        return [x * FX / z + CX, y * FY / z + CY]

    images = {
        1: {
            "image_id": 1,
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 0.0],
            "camera_id": 1,
            "name": "000001.png",
            "xys": np.array([project(points_world[1]["xyz"]), project(points_world[2]["xyz"])]),
            "point3d_ids": np.array([1, 2]),
        },
        2: {
            "image_id": 2,
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 0.0],
            "camera_id": 1,
            "name": "000002.png",
            "xys": np.array([project(points_world[1]["xyz"]), project(points_world[3]["xyz"])]),
            "point3d_ids": np.array([1, 3]),
        },
    }

    return {"cameras": {}, "images": images, "points3d": points_world}


def test_build_pose_and_point_arrays():
    model = _synthetic_model()
    rotations, translations, points, point_id_to_idx = build_pose_and_point_arrays(
        model, ["000001.png", "000002.png"]
    )

    assert rotations.shape == (2, 3, 3)
    assert translations.shape == (2, 3)
    np.testing.assert_array_almost_equal(rotations[0], np.eye(3))
    assert points.shape == (3, 3)
    assert set(point_id_to_idx.keys()) == {1, 2, 3}


def test_build_ba_observations_depth_gate():
    model = _synthetic_model()
    intrinsics = CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480)
    rotations, translations, points, point_id_to_idx = build_pose_and_point_arrays(
        model, ["000001.png", "000002.png"]
    )

    depth_images = {
        "000001.png": np.zeros((480, 640), dtype=np.float64),
        "000002.png": np.zeros((480, 640), dtype=np.float64),
    }
    # Paint correct depth for point 1 (visible in both), wrong depth for point 2.
    for name, img in model["images"].items():
        for xy, pid in zip(img["xys"], img["point3d_ids"]):
            u, v = int(round(xy[0])), int(round(xy[1]))
            true_depth = model["points3d"][pid]["xyz"][2]
            if pid == 2:
                depth_images[img["name"]][v, u] = true_depth * 5.0  # bad depth -> should be gated out
            else:
                depth_images[img["name"]][v, u] = true_depth

    result = build_ba_observations(
        model,
        ["000001.png", "000002.png"],
        points,
        point_id_to_idx,
        depth_loader=lambda name: depth_images[name],
        intrinsics=intrinsics,
        depth_tolerance=0.1,
    )

    # 4 total observations (2 per image), all present as reprojection rows.
    assert result.observations.shape == (4, 4)
    # point 1 (x2) and point 3 get correct depth painted -> 3 depth observations;
    # point 2 has deliberately wrong depth painted -> gated out by depth_tolerance.
    assert result.n_depth_observations == 3
    assert np.sum(result.obs_depths > 0) == 3


def test_build_ba_observations_no_depth_image():
    model = _synthetic_model()
    intrinsics = CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=640, height=480)
    rotations, translations, points, point_id_to_idx = build_pose_and_point_arrays(
        model, ["000001.png", "000002.png"]
    )

    result = build_ba_observations(
        model,
        ["000001.png", "000002.png"],
        points,
        point_id_to_idx,
        depth_loader=lambda name: None,
        intrinsics=intrinsics,
    )

    assert result.observations.shape == (4, 4)
    assert result.n_depth_observations == 0
    assert np.all(result.obs_depths <= 0)
