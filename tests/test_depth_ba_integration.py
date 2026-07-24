"""End-to-end test: depth-aware BA pulls a mis-scaled trajectory toward metric scale.

Mirrors tests/test_scale_estimation_integration.py's known-scale approach,
but exercises run_depth_bundle_adjustment end-to-end instead of the
post-hoc scale estimator.
"""

import numpy as np
import pytest

pytest.importorskip("kornia_rs")

from colmap_rgbd_gt.optimization.depth_ba import DepthBAConfig, run_depth_bundle_adjustment
from colmap_rgbd_gt.utils.camera import CameraIntrinsics

FX = FY = 500.0
CX, CY = 320.0, 240.0
WIDTH, HEIGHT = 640, 480


def _synthetic_scene(mis_scale: float, n_points: int = 30, n_cameras: int = 3):
    """A small scene of `n_cameras` poses observing `n_points` metric-frame
    points, with the "COLMAP" model deliberately scaled by `mis_scale`
    relative to true metric (mis_scale < 1 means COLMAP is too small)."""
    rng = np.random.default_rng(0)

    points_metric = np.stack(
        [
            rng.uniform(-1.0, 1.0, n_points),
            rng.uniform(-1.0, 1.0, n_points),
            rng.uniform(2.0, 4.0, n_points),
        ],
        axis=1,
    )

    cam_translations_metric = np.stack(
        [
            np.linspace(0.0, 0.3, n_cameras),
            np.zeros(n_cameras),
            np.zeros(n_cameras),
        ],
        axis=1,
    )

    # COLMAP model: same rotations (identity), positions divided by mis_scale
    # in a way consistent with points also being divided (uniform Sim3 scale).
    points_colmap = points_metric / mis_scale
    cam_translations_colmap = cam_translations_metric / mis_scale

    image_names = [f"{i+1:06d}.png" for i in range(n_cameras)]
    images = {}
    points3d = {}

    depth_images = {}

    for i, name in enumerate(image_names):
        t_w2c = -cam_translations_colmap[i]  # world->camera translation (identity R)
        xys = []
        point3d_ids = []
        depth_img = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

        for pid in range(n_points):
            X_cam_metric = points_metric[pid] - cam_translations_metric[i]
            if X_cam_metric[2] <= 0:
                continue
            u = X_cam_metric[0] * FX / X_cam_metric[2] + CX
            v = X_cam_metric[1] * FY / X_cam_metric[2] + CY
            if not (0 <= u < WIDTH and 0 <= v < HEIGHT):
                continue
            xys.append([u, v])
            point3d_ids.append(pid + 1)
            depth_img[int(round(v)), int(round(u))] = X_cam_metric[2]

        images[i + 1] = {
            "image_id": i + 1,
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": t_w2c.tolist(),
            "camera_id": 1,
            "name": name,
            "xys": np.array(xys),
            "point3d_ids": np.array(point3d_ids, dtype=np.int64),
        }
        depth_images[name] = depth_img

    for pid in range(n_points):
        points3d[pid + 1] = {
            "point_id": pid + 1,
            "xyz": points_colmap[pid].tolist(),
            "rgb": [128, 128, 128],
            "error": 0.5,
            "image_ids": [i + 1 for i in range(n_cameras)],
            "point2d_idxs": [0] * n_cameras,
        }

    model = {"cameras": {}, "images": images, "points3d": points3d}
    return model, image_names, depth_images, points_metric


def test_depth_ba_pulls_mis_scaled_trajectory_toward_metric():
    mis_scale = 0.6  # COLMAP positions/points are 1/0.6 too large vs metric
    model, image_names, depth_images, points_metric = _synthetic_scene(mis_scale)

    intrinsics = CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=WIDTH, height=HEIGHT)

    config = DepthBAConfig(depth_tolerance=2.0, max_iterations=30, stage="joint")
    result = run_depth_bundle_adjustment(
        model, image_names,
        depth_loader=lambda name: depth_images[name],
        intrinsics=intrinsics,
        config=config,
    )

    assert result.n_depth_observations > 0

    # Compare optimized point cloud scale against ground-truth metric scale.
    optimized_norm = np.linalg.norm(result.points, axis=1).mean()
    metric_norm = np.linalg.norm(points_metric, axis=1).mean()
    colmap_norm = np.linalg.norm(
        np.array([model["points3d"][pid]["xyz"] for pid in model["points3d"]]), axis=1
    ).mean()

    initial_error = abs(colmap_norm - metric_norm) / metric_norm
    optimized_error = abs(optimized_norm - metric_norm) / metric_norm

    assert optimized_error < initial_error
    assert optimized_error < 0.1
