"""Regression tests for two 2026-09-12 scale-estimation bugs, root-caused on
a real 2056-frame RGB-D sequence (kitchen1):

Bug 1: `estimate_global_scale` used to transform the metric depth point into
WORLD frame via `t_c2w` (the camera center in COLMAP's arbitrary, unscaled
units) before comparing it against the COLMAP world point. That mixes a
metric quantity (the rotated depth point) with an arbitrary-unit quantity
(`t_c2w`), producing `scale_true * X_colmap + (1 - scale_true) * C_colmap`
-- a geometry-dependent blend of the true scale and 1.0 that varies with how
each frame's camera center sits relative to the point cloud. This bias only
shows up once multiple frames with DIFFERENT camera centers are pooled
together (a single frame at the world origin trivially hides it, since
`t_c2w = 0` contributes nothing) -- see `test_scale_estimation_integration.py`,
whose single-frame-at-origin setup does not exercise this bug at all.

Bug 2: `compute_diagnostics([scale_estimate])` wrapped the single pooled
estimate in a length-1 list, so `num_valid_frames` always reported 1
regardless of how many frames actually contributed; and `estimate_global_scale`
only ever sampled `trajectory[:max_frames]` -- the FIRST N frames -- not a
representative sample of a longer sequence.
"""

import json

import cv2
import numpy as np
import pytest

from colmap_rgbd_gt.colmap.colmap_io import write_cameras_text, write_images_text
from colmap_rgbd_gt.scaling.scale_estimation import estimate_global_scale
from colmap_rgbd_gt.scaling.diagnostics import compute_diagnostics
from colmap_rgbd_gt.utils.transforms import rotation_matrix_to_quaternion

FX = FY = 500.0
CX, CY = 320.0, 240.0
WIDTH, HEIGHT = 640, 480

# Ground-truth metric scale: metric = colmap * TRUE_SCALE.
TRUE_SCALE = 2.0


def _write_points3d_text(path, points):
    with open(path, "w") as f:
        f.write("# 3D point list\n")
        for pid, (xyz, image_id) in enumerate(points, start=1):
            x, y, z = xyz
            f.write(f"{pid} {x} {y} {z} 128 128 128 0.5 {image_id} 0\n")


def _camera_frame_points(rng, n_points):
    """A fixed camera-frame point pattern (what the depth sensor sees),
    identical for every synthetic frame below -- only the camera's placement
    in the COLMAP world differs between frames."""
    return np.stack(
        [
            rng.uniform(-1.0, 1.0, n_points),
            rng.uniform(-1.0, 1.0, n_points),
            rng.uniform(2.0, 4.0, n_points),
        ],
        axis=1,
    )


def _write_depth_image(path, metric_pts_cam):
    """Depth image (metric, mm) for a camera whose frame-local view of the
    scene is exactly `metric_pts_cam` (identity intrinsic projection)."""
    depth_m = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    for x, y, z in metric_pts_cam:
        u = int(round(x * FX / z + CX))
        v = int(round(y * FY / z + CY))
        if 0 <= u < WIDTH and 0 <= v < HEIGHT:
            depth_m[v, u] = z
    depth_mm = (depth_m * 1000.0).astype(np.uint16)
    cv2.imwrite(str(path), depth_mm)


def _build_two_camera_workspace(tmp_path, true_scale, camera_center_offset):
    """Two frames observing the SAME camera-frame point pattern (so the
    depth data / pixel layout is identical for both), but the second
    frame's camera sits at a COLMAP-world position offset by
    `camera_center_offset` (identity rotation for both, translation-only)
    from the first frame's camera (which sits at the COLMAP-world origin).

    This is the minimal adversarial case for Bug 1: pooling frame 1 (whose
    `t_c2w = 0`, hiding the bug) with frame 2 (whose `t_c2w` is large and
    unscaled) exposes the world-frame unit-mixing bias described in this
    module's docstring.
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

    rng = np.random.default_rng(0)
    n_points = 150

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

    images = {}
    all_points3d = []
    identity_R_wxyz = rotation_matrix_to_quaternion(np.eye(3))
    identity_qvec = [identity_R_wxyz[3], identity_R_wxyz[0], identity_R_wxyz[1], identity_R_wxyz[2]]

    camera_centers_w = [np.zeros(3), np.asarray(camera_center_offset, dtype=np.float64)]

    for frame_idx, camera_center_w in enumerate(camera_centers_w, start=1):
        image_id = frame_idx
        image_name = f"{frame_idx:06d}.png"

        metric_pts_cam = _camera_frame_points(rng, n_points)
        colmap_pts_cam = metric_pts_cam / true_scale

        # Identity rotation: R_w2c = I, so t_w2c = -camera_center_w
        # (t_c2w = camera_center_w). colmap world point = colmap_pts_cam +
        # camera_center_w (arbitrary COLMAP units throughout).
        colmap_pts_world = colmap_pts_cam + camera_center_w
        t_w2c = -camera_center_w

        images[image_id] = {
            "image_id": image_id,
            "qvec": identity_qvec,
            "tvec": t_w2c.tolist(),
            "camera_id": 1,
            "name": image_name,
        }

        for xyz in colmap_pts_world:
            all_points3d.append((xyz, image_id))

        _write_depth_image(ws_root / "depth" / image_name, metric_pts_cam)

    write_images_text(ws_root / "colmap" / "sparse" / "0" / "images.txt", images)
    _write_points3d_text(ws_root / "colmap" / "sparse" / "0" / "points3D.txt", all_points3d)

    return ws_root


@pytest.mark.parametrize("method", ["umeyama", "median", "ransac"])
def test_bug1_camera_frame_fix_recovers_true_scale_with_offset_cameras(tmp_path, method):
    """Adversarial case: two frames whose camera centers differ hugely in
    COLMAP's arbitrary units (0 vs. a 50-unit offset). Pre-fix (world-frame
    comparison), pooling these two frames mixes metric and unscaled
    quantities and biases the recovered scale away from TRUE_SCALE -- see
    `test_bug1_world_frame_mixing_is_biased_pre_fix` below, which
    reproduces that exact math directly and shows the bias. Post-fix
    (camera-frame comparison), both frames agree on a pure scalar relation
    and the pipeline recovers TRUE_SCALE accurately regardless of how far
    apart the camera centers are.
    """
    ws_root = _build_two_camera_workspace(
        tmp_path, TRUE_SCALE, camera_center_offset=[50.0, 0.0, 0.0]
    )

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

    assert result.num_samples >= 3
    assert abs(result.scale - TRUE_SCALE) / TRUE_SCALE < 0.05, (
        f"recovered scale {result.scale} not within 5% of ground truth {TRUE_SCALE} "
        f"(method={method}) -- Bug 1 (world-frame unit mixing) appears to have regressed"
    )


def test_bug1_world_frame_mixing_is_biased_pre_fix():
    """Directly reproduces the OLD (buggy) world-frame construction math
    -- `R_c2w @ depth_pts_cam_metric + t_c2w` with `t_c2w` in COLMAP's
    unscaled units -- for the same adversarial two-camera geometry used
    above, and confirms it recovers a scale that is NOT close to
    TRUE_SCALE. This demonstrates the bug exists in the old formula
    independent of the rest of the orchestrator, and that
    `estimate_global_scale`'s camera-frame fix is what actually avoids it
    (see the test above, which passes against the real, fixed code)."""
    from colmap_rgbd_gt.scaling.scale_estimation import estimate_scale_umeyama

    rng = np.random.default_rng(0)
    n_points = 150
    true_scale = TRUE_SCALE
    camera_center_offset = np.array([50.0, 0.0, 0.0])

    all_depth_points_buggy = []
    all_colmap_points_buggy = []

    for camera_center_w in (np.zeros(3), camera_center_offset):
        metric_pts_cam = _camera_frame_points(rng, n_points)
        colmap_pts_cam = metric_pts_cam / true_scale
        colmap_pts_world = colmap_pts_cam + camera_center_w

        # OLD buggy formula: R_c2w == I here, so this is just
        # metric_pts_cam + t_c2w (t_c2w = camera_center_w, UNSCALED).
        depth_pts_world_buggy = metric_pts_cam + camera_center_w

        all_depth_points_buggy.extend(depth_pts_world_buggy.tolist())
        all_colmap_points_buggy.extend(colmap_pts_world.tolist())

    biased = estimate_scale_umeyama(
        np.array(all_depth_points_buggy), np.array(all_colmap_points_buggy)
    )

    relative_error = abs(biased.scale - true_scale) / true_scale
    assert relative_error > 0.1, (
        f"expected the pre-fix world-frame formula to be biased away from "
        f"true_scale={true_scale} by more than 10%, got scale={biased.scale} "
        f"(relative error {relative_error:.3f}) -- adversarial geometry may need adjusting"
    )


def _make_synthetic_frame_estimate(scale, num_samples=50, confidence=0.9):
    from colmap_rgbd_gt.scaling.scale_estimation import ScaleEstimate

    return ScaleEstimate(
        scale=scale,
        confidence=confidence,
        method="median",
        num_samples=num_samples,
        inlier_ratio=1.0,
    )


def test_bug2_compute_diagnostics_reports_real_frame_count():
    """`compute_diagnostics` must reflect the ACTUAL number of contributing
    frames, not the length-1 `[scale_estimate]` reporting artifact this
    used to always produce."""
    frame_estimates = [_make_synthetic_frame_estimate(2.0 + 0.01 * i) for i in range(37)]

    diagnostics = compute_diagnostics(frame_estimates)

    assert diagnostics.num_valid_frames == 37
    assert diagnostics.num_total_frames == 37


def test_bug2_estimate_global_scale_returns_multiple_frame_estimates(tmp_path):
    """`estimate_global_scale`'s return value must expose the genuine
    per-frame estimates it computed (via `.frame_estimates`), not force
    every caller to fall back to wrapping the pooled estimate in a
    length-1 list."""
    ws_root = _build_two_camera_workspace(
        tmp_path, TRUE_SCALE, camera_center_offset=[5.0, 0.0, 0.0]
    )

    config = {
        "method": "umeyama",
        "min_points_per_frame": 1,
        "max_frames_for_scale": 10,
        "sample_stride": 1,
        "depth_tolerance": 1.0,
        "min_depth_m": 0.1,
        "max_depth_m": 10.0,
    }

    result = estimate_global_scale(ws_root, config)

    assert result.frame_estimates is not None
    # Both synthetic frames should have yielded usable correspondences.
    assert len(result.frame_estimates) == 2

    diagnostics = compute_diagnostics(result.frame_estimates)
    assert diagnostics.num_total_frames == 2
    assert diagnostics.num_valid_frames >= 1


def test_bug2_frame_sampling_covers_tail_of_long_trajectory(tmp_path, monkeypatch):
    """A synthetic 1000-"frame" trajectory with `max_frames_for_scale=100`
    must include frames from near the END of the sequence, not only frames
    0-99 (the old `trajectory[:max_frames]` behavior).

    Uses a real (trivial, empty) workspace/COLMAP model so `estimate_global_
    scale`'s own local imports resolve normally; only `extract_trajectory`
    (to supply a long fake trajectory) and `find_colmap_points_in_image`
    (to cheaply record which frame_ids were actually considered, since an
    empty model would otherwise make every frame look identical) are
    monkeypatched, at their SOURCE modules -- `estimate_global_scale` does
    `from ... import ...` locally inside the function body, so patching
    those source-module attributes is what actually takes effect.
    """
    ws_root = tmp_path / "workspace"
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

    write_cameras_text(ws_root / "colmap" / "sparse" / "0" / "cameras.txt", {})
    write_images_text(ws_root / "colmap" / "sparse" / "0" / "images.txt", {})
    _write_points3d_text(ws_root / "colmap" / "sparse" / "0" / "points3D.txt", [])

    fake_trajectory = [
        {"frame_id": i, "image_name": f"{i:06d}.png", "R": np.eye(3), "t": np.zeros(3)}
        for i in range(1000)
    ]

    import colmap_rgbd_gt.colmap.pose_extract as pose_extract_module
    import colmap_rgbd_gt.scaling.correspondences as correspondences_module

    monkeypatch.setattr(pose_extract_module, "extract_trajectory", lambda workspace: fake_trajectory)

    seen_via_colmap_lookup = []

    def _record_and_return_empty(model, image_name):
        seen_via_colmap_lookup.append(image_name)
        return np.zeros((0, 3))

    monkeypatch.setattr(correspondences_module, "find_colmap_points_in_image", _record_and_return_empty)

    config = {
        "max_frames_for_scale": 100,
        "min_points_per_frame": 1,
    }

    estimate_global_scale(ws_root, config)

    assert len(seen_via_colmap_lookup) == 100
    max_frame_seen = max(int(name.split(".")[0]) for name in seen_via_colmap_lookup)
    assert max_frame_seen >= 900, (
        f"expected frame sampling to reach near the end of a 1000-frame trajectory, "
        f"but the highest frame_id considered was {max_frame_seen} -- sampling may "
        "still only be taking the first max_frames_for_scale frames"
    )
