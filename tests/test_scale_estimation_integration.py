"""Integration test for estimate_global_scale using a synthetic COLMAP model + depth image.

This test exercises the full orchestrator (`estimate_global_scale`), not just the
pure-math estimators in `tests/test_scale_estimation.py`. It builds a synthetic
COLMAP sparse model and a synthetic depth image with a KNOWN ground-truth scale
factor, so the recovered scale can be checked against ground truth. It is designed
to fail against the pre-fix orchestrator, which paired unrelated point sets
(dense-backprojected depth pixels vs. sparse COLMAP points) instead of true
1:1 correspondences.
"""

import json

import cv2
import numpy as np
import pytest

from colmap_rgbd_gt.colmap.colmap_io import write_cameras_text, write_images_text
from colmap_rgbd_gt.scaling.scale_estimation import estimate_global_scale

FX = FY = 500.0
CX, CY = 320.0, 240.0
WIDTH, HEIGHT = 640, 480

# Ground-truth metric scale factor: COLMAP's arbitrary-scale reconstruction is
# this many times SMALLER than metric reality, i.e. metric = colmap * SCALE.
#
# NOTE: find_valid_correspondences() gates matches on
# |measured_depth - projected_depth| < depth_tolerance * projected_depth,
# where projected_depth is in COLMAP's (unscaled) units. Since
# measured_depth is metric, this ratio simplifies to |SCALE - 1|, so
# depth_tolerance must exceed |SCALE - 1| for any correspondence to survive
# the gate. Keep SCALE modest here and depth_tolerance generous accordingly.
SCALE = 1.5


def _write_points3d_text(path, points):
    with open(path, "w") as f:
        f.write("# 3D point list\n")
        for pid, (xyz, image_id) in enumerate(points, start=1):
            x, y, z = xyz
            f.write(f"{pid} {x} {y} {z} 128 128 128 0.5 {image_id} 0\n")


def _build_synthetic_workspace(tmp_path):
    """Build a workspace with one image observing points at a known metric scale.

    Camera is at the world origin looking down +Z (identity w2c pose). A grid of
    3D points sits in front of the camera at metric depths; the COLMAP sparse
    model stores those same points *divided by SCALE* (i.e. in COLMAP's
    arbitrary/unscaled units), while the depth image encodes the true metric
    depth. estimate_global_scale must recover SCALE.
    """
    ws_root = tmp_path / "workspace"
    (ws_root / "depth").mkdir(parents=True)
    (ws_root / "colmap" / "sparse" / "0").mkdir(parents=True)
    (ws_root / "outputs").mkdir(parents=True)

    manifest = {
        "camera_info": {
            "K": [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0],
            "width": WIDTH,
            "height": HEIGHT,
        }
    }
    with open(ws_root / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Grid of metric-frame 3D points in front of the camera (identity pose).
    rng = np.random.default_rng(0)
    # estimate_global_scale requires >= 100 total correspondences before it
    # will report a non-trivial scale estimate.
    n_points = 150
    metric_points = np.stack(
        [
            rng.uniform(-1.0, 1.0, n_points),
            rng.uniform(-1.0, 1.0, n_points),
            rng.uniform(2.0, 4.0, n_points),
        ],
        axis=1,
    )

    image_id = 1
    image_name = "000001.png"
    cameras = {
        1: {
            "camera_id": 1,
            "model": "PINHOLE",
            "width": WIDTH,
            "height": HEIGHT,
            "params": [FX, FY, CX, CY],
        }
    }
    write_cameras_text(ws_root / "colmap" / "sparse" / "0" / "cameras.txt", cameras)

    images = {
        image_id: {
            "image_id": image_id,
            "qvec": [1.0, 0.0, 0.0, 0.0],  # identity w2c rotation
            "tvec": [0.0, 0.0, 0.0],
            "camera_id": 1,
            "name": image_name,
        }
    }
    write_images_text(ws_root / "colmap" / "sparse" / "0" / "images.txt", images)

    # COLMAP world-frame points are the metric points divided by SCALE
    # (COLMAP's arbitrary-scale reconstruction).
    colmap_points = metric_points / SCALE
    _write_points3d_text(
        ws_root / "colmap" / "sparse" / "0" / "points3D.txt",
        [(xyz, image_id) for xyz in colmap_points],
    )

    # Depth image encodes true metric depth (z-component of metric_points),
    # painted at the projected pixel of each point (identity pose => camera
    # frame == world frame here).
    depth_m = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    for x, y, z in metric_points:
        u = int(round(x * FX / z + CX))
        v = int(round(y * FY / z + CY))
        if 0 <= u < WIDTH and 0 <= v < HEIGHT:
            depth_m[v, u] = z

    depth_mm = (depth_m * 1000.0).astype(np.uint16)
    cv2.imwrite(str(ws_root / "depth" / "000001.png"), depth_mm)

    return ws_root


@pytest.mark.parametrize("method", ["umeyama", "median", "ransac"])
def test_estimate_global_scale_recovers_known_scale(tmp_path, method):
    ws_root = _build_synthetic_workspace(tmp_path)

    config = {
        "method": method,
        "min_points_per_frame": 1,
        "max_frames_for_scale": 10,
        "sample_stride": 1,
        "depth_tolerance": 1.0,
        "min_depth_m": 0.1,
        "max_depth_m": 10.0,
    }

    result = estimate_global_scale(ws_root, config)

    assert result.num_samples >= 3, "too few correspondences recovered"
    assert abs(result.scale - SCALE) / SCALE < 0.05, (
        f"recovered scale {result.scale} not within 5% of ground truth {SCALE}"
    )
