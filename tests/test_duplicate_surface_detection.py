"""Tests for colmap/duplicate_surface_detection.py.

Root-cause investigation (2026-08-03, floor2): a naive "large horizontal
point cluster offset in Y" heuristic fires on any healthy floor -- it is
NOT a valid duplicate detector by itself. The real signature is a PAIR of
Y-separated horizontal-planar clusters whose points ALSO reproject into
the SAME image-space pixel region in shared observing frames (same floor
pixels triangulated to two depths). These tests reproduce a synthetic
version of a genuine duplicate (two Y-offset copies of the same floor,
seen by the same frames at the same pixel coordinates) and confirm the
detector does NOT fire on a single real floor, and does NOT fire on
contiguous Y-slices of one continuous tall surface (the false-positive
mode found empirically on kitchen1 before the vertical-gap requirement
was added).
"""

import numpy as np
import pytest

from colmap_rgbd_gt.colmap.duplicate_surface_detection import (
    detect_duplicate_planar_surfaces,
    find_horizontal_planar_clusters,
)


def _make_floor_points(start_id, y, xs, zs, image_ids, pixel_fn, rng):
    """Build a synthetic horizontal-planar cluster: n = len(xs) points at
    height y (+ small noise), each observed by every image in image_ids
    at a pixel position from pixel_fn(x, z) (+ small noise). Returns
    (points_dict, per_image_xy_p3id_list).
    """
    points = {}
    per_image_obs = {iid: [] for iid in image_ids}
    for i, (x, z) in enumerate(zip(xs, zs)):
        pid = start_id + i
        yy = y + rng.normal(0, 0.005)
        points[pid] = {"xyz": np.array([x, yy, z]), "rgb": [150, 120, 80]}
        for iid in image_ids:
            u, v = pixel_fn(x, z)
            u += rng.normal(0, 1.0)
            v += rng.normal(0, 1.0)
            per_image_obs[iid].append((u, v, pid))
    return points, per_image_obs


def _build_images(per_image_obs_list, width=640, height=480):
    """Merge several clusters' per-image observations into COLMAP-style
    image dicts with xys/point3d_ids/width/height."""
    merged: dict[int, list[tuple[float, float, int]]] = {}
    for per_image_obs in per_image_obs_list:
        for iid, obs in per_image_obs.items():
            merged.setdefault(iid, []).extend(obs)

    images = {}
    for iid, obs in merged.items():
        xys = np.array([[o[0], o[1]] for o in obs], dtype=np.float64)
        p3ids = np.array([o[2] for o in obs], dtype=np.int64)
        images[iid] = {"xys": xys, "point3d_ids": p3ids, "width": width, "height": height}
    return images


def test_single_real_floor_is_not_flagged():
    """A lone, healthy floor plane (no second cluster to duplicate
    against) must never be flagged -- this is the exact floor2 case."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(-3, 3, 800)
    zs = rng.uniform(-3, 3, 800)
    image_ids = list(range(20))

    def pixel_fn(x, z):
        # floor pixels land in the bottom half of the frame
        return 320 + x * 20, 400 + z * 10

    points, per_image = _make_floor_points(0, 1.3, xs, zs, image_ids, pixel_fn, rng)
    images = _build_images([per_image])

    result = detect_duplicate_planar_surfaces(points, images)
    assert result.detected is False
    assert result.candidates == []


def test_genuine_duplicate_floor_is_flagged():
    """Two Y-separated copies of the SAME floor, seen by the SAME frames
    at the SAME (noisy) pixel coordinates -- the actual duplicate-
    triangulation signature -- must be flagged."""
    rng = np.random.default_rng(1)
    xs = rng.uniform(-3, 3, 800)
    zs = rng.uniform(-3, 3, 800)
    image_ids = list(range(20))

    def pixel_fn(x, z):
        return 320 + x * 20, 400 + z * 10

    real_points, real_obs = _make_floor_points(0, 1.3, xs, zs, image_ids, pixel_fn, rng)
    # Ghost: same XZ footprint, same pixel projection, but wrongly
    # triangulated ~0.9m away in height.
    ghost_points, ghost_obs = _make_floor_points(100000, 0.4, xs, zs, image_ids, pixel_fn, rng)

    points = {**real_points, **ghost_points}
    images = _build_images([real_obs, ghost_obs])

    result = detect_duplicate_planar_surfaces(points, images, min_cluster_points=200)
    assert result.detected is True
    assert len(result.candidates) >= 1
    c = result.candidates[0]
    assert c["pixel_overlap_fraction"] >= 0.30
    assert c["height_separation_m"] == pytest.approx(0.9, abs=0.2)


def test_adjacent_touching_bands_are_not_flagged():
    """Two directly-adjacent (near-zero vertical gap) horizontal bands
    with full pixel-footprint overlap must NOT be flagged -- this is
    exactly the false-positive mode found empirically on kitchen1 (a
    single continuous surface sliced into multiple windows by the Y-bin
    scan looks like a "duplicate pair" on IoU/pixel-overlap alone; only a
    genuine vertical GAP between the two clusters distinguishes a real
    duplicate from one surface double-counted)."""
    rng = np.random.default_rng(2)
    xs = rng.uniform(-3, 3, 600)
    zs = rng.uniform(-3, 3, 600)
    image_ids = list(range(20))

    def pixel_fn(x, z):
        return 320 + x * 20, 400 + z * 10

    band_a, obs_a = _make_floor_points(0, 0.500, xs, zs, image_ids, pixel_fn, rng)
    band_b, obs_b = _make_floor_points(100000, 0.545, xs, zs, image_ids, pixel_fn, rng)

    points = {**band_a, **band_b}
    images = _build_images([obs_a, obs_b])

    result = detect_duplicate_planar_surfaces(points, images, min_cluster_points=200)
    assert result.detected is False


def test_find_horizontal_planar_clusters_empty_input():
    assert find_horizontal_planar_clusters({}) == []


def _write_sparse_model_text(sparse_dir, points, images, width=640, height=480):
    """Write a minimal but real COLMAP text-format sparse model to disk
    (cameras.txt/images.txt/points3D.txt) so `run_duplicate_surface_qc`
    can be exercised end-to-end, exactly as it is on a real workspace."""
    from colmap_rgbd_gt.colmap.colmap_io import (
        write_cameras_text, write_images_text, write_points3d_text,
    )

    sparse_dir.mkdir(parents=True, exist_ok=True)
    cameras = {1: {"camera_id": 1, "model": "PINHOLE", "width": width, "height": height,
                   "params": [500.0, 500.0, width / 2, height / 2]}}
    write_cameras_text(sparse_dir / "cameras.txt", cameras)

    images_out = {}
    for iid, img in images.items():
        images_out[iid] = {
            "image_id": iid, "qvec": [1.0, 0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0],
            "camera_id": 1, "name": f"{iid:06d}.png",
            "xys": img["xys"], "point3d_ids": img["point3d_ids"],
        }
    write_images_text(sparse_dir / "images.txt", images_out)

    points_out = {
        pid: {"point_id": pid, "xyz": p["xyz"].tolist(), "rgb": p["rgb"], "error": 1.0,
              "image_ids": [], "point2d_idxs": []}
        for pid, p in points.items()
    }
    write_points3d_text(sparse_dir / "points3D.txt", points_out)


def test_run_duplicate_surface_qc_points_scale_converts_raw_units_to_metric(tmp_path):
    """Root-caused 2026-08-03 on table1: this module's thresholds
    (DEFAULT_MIN_HEIGHT_SEPARATION etc.) are calibrated in real meters,
    but colmap/sparse/0 is only actually in meters when scale-regime
    correction rewrote it in place -- otherwise it is still in COLMAP's
    raw/unscaled unit space, and reading it without applying the scene's
    own scale factor silently misinterprets the unit. Build the SAME
    genuine-duplicate geometry twice, once already scaled to real meters
    (0.9m separation, height_separation_m must read ~0.9) and once left
    in raw COLMAP units at 10x scale (9.0 raw units, i.e. really 0.9m
    once points_scale=0.1 is applied) -- both must agree once the right
    scale is supplied, and the raw-unit reading must NOT be
    misinterpreted as a 9m separation."""
    rng = np.random.default_rng(3)
    xs = rng.uniform(-3, 3, 500)
    zs = rng.uniform(-3, 3, 500)
    image_ids = list(range(20))

    def pixel_fn(x, z):
        return 320 + x * 20, 400 + z * 10

    real_points, real_obs = _make_floor_points(0, 1.3, xs, zs, image_ids, pixel_fn, rng)
    ghost_points, ghost_obs = _make_floor_points(100000, 0.4, xs, zs, image_ids, pixel_fn, rng)
    points = {**real_points, **ghost_points}
    images = _build_images([real_obs, ghost_obs])

    # Same scene, but written to disk as if it came out of COLMAP's raw
    # (unscaled) unit space at 10x the real metric size.
    raw_points = {pid: {"xyz": p["xyz"] * 10.0, "rgb": p["rgb"]} for pid, p in points.items()}

    from colmap_rgbd_gt.colmap.duplicate_surface_detection import run_duplicate_surface_qc

    sparse_dir = tmp_path / "sparse" / "0"
    _write_sparse_model_text(sparse_dir, raw_points, images)

    # Without unscaling: thresholds compared against raw units, so the
    # ~9-raw-unit gap reads (wrongly) as a huge, out-of-range separation
    # and DEFAULT_MAX_HEIGHT_SEPARATION (3.0) rejects the pair entirely.
    result_unscaled = run_duplicate_surface_qc(sparse_dir)
    assert result_unscaled.detected is False

    # With the scene's real scale factor (0.1) applied: reads back as the
    # genuine ~0.9m duplicate and IS flagged.
    result_scaled = run_duplicate_surface_qc(sparse_dir, points_scale=0.1)
    assert result_scaled.detected is True
    assert len(result_scaled.candidates) >= 1
    assert result_scaled.candidates[0]["height_separation_m"] == pytest.approx(0.9, abs=0.2)
