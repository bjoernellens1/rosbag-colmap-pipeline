"""Detects candidate duplicate/ghost planar-surface artifacts in a COLMAP
sparse model: the SAME physical (usually large, texture-repetitive)
surface -- most often a floor -- triangulated twice at two different
heights/depths because of a matching ambiguity (reflective or repeating
texture, e.g. parquet/tile grain).

Investigated 2026-08-03 on floor2 after the scale-regime fix (see
`scale_regime_correction.py`): a naive "large horizontal point cluster
offset in Y from the trajectory / from the rest of the point cloud"
heuristic is NOT a valid detector on its own -- it fires on every healthy
scene with a real floor, because a floor is *exactly* a large, dense,
horizontal, Y-offset-from-camera planar cluster by construction. Applied
to floor2's own reported "floating slab" (Y in [1.17, 1.53), ~6,261
points, ~32% of the cloud): color (warm tan, matches the parquet floor
visible in data/workspaces/floor2/rgb/000400.png), sensor-depth
backprojection of real floor pixels (world Y ~1.0-1.07m, same ballpark as
the cluster, nowhere near the camera-height alternative), and image-space
reprojection (bottom third of frame, where a floor projects) all
independently confirmed this cluster IS the real corridor floor, not a
ghost -- removing it would have deleted a third of the real reconstructed
scene.

The genuine duplicate-surface signature found investigating that same
scene is the OPPOSITE pattern: a much smaller (~3,331 point) secondary
cluster of floor-colored points sitting near camera height (Y近0) that
ALSO reprojects predominantly into the floor region of the image (bottom
third) in the SAME observing frames as the real floor cluster -- i.e. the
same floor pixels are being triangulated to two different depths. THAT
reprojection-footprint overlap between two Y-separated, independently
near-planar-horizontal clusters, observed in shared frames, is the actual
diagnostic this module implements. A lone planar cluster (a real floor
with no matching duplicate) never trips it because there is no second
cluster to overlap with.

This module only DETECTS and reports (loud, greppable, following the
project's `pose_outliers.py`/`scale_regime_correction.py` philosophy) --
it does not remove anything. The floor2 investigation intentionally
stopped short of an automated fix: a color/height heuristic alone is not
specific enough to safely delete points without risking real furniture/
structure genuinely near camera height and floor-colored.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

DEFAULT_Y_BIN_WIDTH = 0.05
DEFAULT_MIN_CLUSTER_POINTS = 300
DEFAULT_PLANE_RESIDUAL_MAX = 0.10
DEFAULT_MIN_VERTICALITY = 0.85  # |normal . up_axis|, up_axis assumed to be Y
DEFAULT_XZ_IOU_MIN = 0.25
DEFAULT_MIN_HEIGHT_SEPARATION = 0.2
DEFAULT_MAX_HEIGHT_SEPARATION = 3.0
DEFAULT_PIXEL_GRID_CELLS = 32  # coarse grid per axis for footprint overlap
DEFAULT_PIXEL_OVERLAP_MIN = 0.30
DEFAULT_MIN_SHARED_FRAMES = 5
# Two clusters directly abutting each other (e.g. adjacent Y-bin windows
# sliced out of the SAME continuous tall surface, like a wall) will
# trivially "overlap" in pixel footprint -- that's not a duplicate, it's
# one structure double-counted by the windowing scan. Require a genuine
# vertical GAP between bands, not just centroid separation, before even
# considering a pair a candidate (found empirically on kitchen1: without
# this, contiguous wall slices produced spurious high-IoU/high-overlap
# "duplicate" pairs).
DEFAULT_MIN_VERTICAL_GAP = 0.12


@dataclass
class PlanarCluster:
    y_lo: float
    y_hi: float
    point_ids: list[int]
    centroid: np.ndarray
    normal: np.ndarray
    residual_std: float
    xz_bbox: tuple[float, float, float, float]  # xmin, xmax, zmin, zmax


@dataclass
class DuplicateSurfaceCandidate:
    cluster_a: dict[str, Any]
    cluster_b: dict[str, Any]
    xz_iou: float
    height_separation_m: float
    n_shared_frames: int
    pixel_overlap_fraction: float


@dataclass
class DuplicateSurfaceDetectionResult:
    detected: bool = False
    n_clusters_found: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


def _fit_plane(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - centroid)
    normal = vt[-1]
    residual = (xyz - centroid) @ normal
    return centroid, normal, float(residual.std())


def find_horizontal_planar_clusters(
    points: dict[int, dict[str, Any]],
    up_axis: int = 1,
    y_bin_width: float = DEFAULT_Y_BIN_WIDTH,
    min_cluster_points: int = DEFAULT_MIN_CLUSTER_POINTS,
    plane_residual_max: float = DEFAULT_PLANE_RESIDUAL_MAX,
    min_verticality: float = DEFAULT_MIN_VERTICALITY,
) -> list[PlanarCluster]:
    """Scan the up-axis (default Y) point-density histogram for dense
    bands, then keep only the ones that are genuinely near-planar and
    near-horizontal (a real floor/ceiling/table surface, or a duplicate
    of one) -- a dense-but-non-planar band (e.g. a doorway full of
    vertical edge points) is not a candidate.
    """
    ids = list(points.keys())
    if not ids:
        return []
    xyz_all = np.array([points[i]["xyz"] for i in ids], dtype=np.float64)
    y_all = xyz_all[:, up_axis]

    lo, hi = y_all.min(), y_all.max()
    if hi <= lo:
        return []
    n_bins = max(1, int(np.ceil((hi - lo) / y_bin_width)))
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_all, edges) - 1, 0, n_bins - 1)

    clusters: list[PlanarCluster] = []
    b = 0
    while b < n_bins:
        # Greedily grow a contiguous run of bins whose combined points
        # form a plausible planar-horizontal cluster once large enough to
        # test -- start from any bin with points and extend while the fit
        # keeps improving/staying valid, capped by a max multi-bin span so
        # we don't just re-discover "everything below the ceiling".
        member_mask = bin_idx == b
        if member_mask.sum() == 0:
            b += 1
            continue
        # Expand the window a few bins at a time and re-test; stop
        # expanding once residual blows up (crossed into a different,
        # non-planar structure) or we've swept a generous vertical span.
        span = 1
        max_span = max(1, int(round(0.5 / y_bin_width)))  # ~0.5m max band
        best = None
        while span <= max_span and b + span <= n_bins:
            window_mask = (bin_idx >= b) & (bin_idx < b + span)
            n_pts = window_mask.sum()
            if n_pts >= min_cluster_points:
                sub_ids = [ids[k] for k in np.where(window_mask)[0]]
                sub_xyz = xyz_all[window_mask]
                centroid, normal, resid = _fit_plane(sub_xyz)
                verticality = abs(normal[up_axis])
                if resid <= plane_residual_max and verticality >= min_verticality:
                    best = (sub_ids, sub_xyz, centroid, normal, resid)
                elif best is not None:
                    break
            span += 1
        if best is not None:
            sub_ids, sub_xyz, centroid, normal, resid = best
            other_axes = [a for a in range(3) if a != up_axis]
            xz = sub_xyz[:, other_axes]
            clusters.append(PlanarCluster(
                y_lo=float(sub_xyz[:, up_axis].min()),
                y_hi=float(sub_xyz[:, up_axis].max()),
                point_ids=sub_ids,
                centroid=centroid,
                normal=normal,
                residual_std=resid,
                xz_bbox=(float(xz[:, 0].min()), float(xz[:, 0].max()), float(xz[:, 1].min()), float(xz[:, 1].max())),
            ))
            b += max(span, 1)
        else:
            b += 1

    return clusters


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ax1, az0, az1 = a
    bx0, bx1, bz0, bz1 = b
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iz0, iz1 = max(az0, bz0), min(az1, bz1)
    if ix1 <= ix0 or iz1 <= iz0:
        return 0.0
    inter = (ix1 - ix0) * (iz1 - iz0)
    area_a = max(ax1 - ax0, 1e-9) * max(az1 - az0, 1e-9)
    area_b = max(bx1 - bx0, 1e-9) * max(bz1 - bz0, 1e-9)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _pixel_footprint_overlap(
    cluster_a: PlanarCluster,
    cluster_b: PlanarCluster,
    images: dict[int, dict[str, Any]],
    grid_cells: int = DEFAULT_PIXEL_GRID_CELLS,
    min_shared_frames: int = DEFAULT_MIN_SHARED_FRAMES,
) -> tuple[int, float]:
    """For frames observing BOTH clusters, bin each cluster's 2D keypoints
    into a coarse per-image grid and measure the fraction of cells
    occupied by cluster A that are ALSO occupied by cluster B -- the
    actual "same pixels, two depths" duplicate-triangulation signature.
    Returns (n_shared_frames, mean_overlap_fraction).
    """
    a_ids_arr = np.array(sorted(cluster_a.point_ids), dtype=np.int64)
    b_ids_arr = np.array(sorted(cluster_b.point_ids), dtype=np.int64)

    frames_a: dict[int, list[int]] = {}  # image_id -> point2d indices belonging to cluster A
    frames_b: dict[int, list[int]] = {}
    for image_id, img in images.items():
        p3ids = img.get("point3d_ids")
        if p3ids is None or len(p3ids) == 0:
            continue
        a_mask = np.isin(p3ids, a_ids_arr, assume_unique=False)
        b_mask = np.isin(p3ids, b_ids_arr, assume_unique=False)
        if a_mask.any():
            frames_a[image_id] = a_mask
        if b_mask.any():
            frames_b[image_id] = b_mask

    shared = set(frames_a) & set(frames_b)
    if len(shared) < min_shared_frames:
        return len(shared), 0.0

    overlaps = []
    for image_id in shared:
        img = images[image_id]
        w = img.get("width")
        h = img.get("height")
        xys = img["xys"]
        if w is None or h is None or xys.shape[0] == 0:
            continue
        a_xy = xys[frames_a[image_id]]
        b_xy = xys[frames_b[image_id]]
        if len(a_xy) == 0 or len(b_xy) == 0:
            continue
        gx_a = np.clip((a_xy[:, 0] / max(w, 1) * grid_cells).astype(int), 0, grid_cells - 1)
        gy_a = np.clip((a_xy[:, 1] / max(h, 1) * grid_cells).astype(int), 0, grid_cells - 1)
        gx_b = np.clip((b_xy[:, 0] / max(w, 1) * grid_cells).astype(int), 0, grid_cells - 1)
        gy_b = np.clip((b_xy[:, 1] / max(h, 1) * grid_cells).astype(int), 0, grid_cells - 1)
        cells_a = set(zip(gx_a.tolist(), gy_a.tolist()))
        cells_b = set(zip(gx_b.tolist(), gy_b.tolist()))
        if not cells_a:
            continue
        overlaps.append(len(cells_a & cells_b) / len(cells_a))

    return len(shared), (float(np.mean(overlaps)) if overlaps else 0.0)


def detect_duplicate_planar_surfaces(
    points: dict[int, dict[str, Any]],
    images: dict[int, dict[str, Any]],
    up_axis: int = 1,
    xz_iou_min: float = DEFAULT_XZ_IOU_MIN,
    min_height_separation: float = DEFAULT_MIN_HEIGHT_SEPARATION,
    max_height_separation: float = DEFAULT_MAX_HEIGHT_SEPARATION,
    pixel_overlap_min: float = DEFAULT_PIXEL_OVERLAP_MIN,
    min_vertical_gap: float = DEFAULT_MIN_VERTICAL_GAP,
    **cluster_kwargs: Any,
) -> DuplicateSurfaceDetectionResult:
    """Top-level entry point. `points`/`images` are COLMAP model dicts as
    returned by `colmap_io.read_points3d_text`/`read_images_text` (each
    image dict additionally needs `width`/`height`, which the raw text
    reader does not populate -- callers must merge these in from the
    corresponding `cameras` entry; see `run_duplicate_surface_qc` below
    for the standard way to do this).
    """
    clusters = find_horizontal_planar_clusters(points, up_axis=up_axis, **cluster_kwargs)
    result = DuplicateSurfaceDetectionResult(n_clusters_found=len(clusters))

    if len(clusters) < 2:
        result.reason = f"only {len(clusters)} horizontal planar cluster(s) found -- no pair to compare"
        return result

    candidates: list[DuplicateSurfaceCandidate] = []
    for ca, cb in combinations(clusters, 2):
        height_sep = abs(ca.centroid[up_axis] - cb.centroid[up_axis])
        if not (min_height_separation <= height_sep <= max_height_separation):
            continue
        gap = max(ca.y_lo, cb.y_lo) - min(ca.y_hi, cb.y_hi)
        if gap < min_vertical_gap:
            continue
        iou = _bbox_iou(ca.xz_bbox, cb.xz_bbox)
        if iou < xz_iou_min:
            continue
        n_shared, overlap_frac = _pixel_footprint_overlap(ca, cb, images)
        if overlap_frac < pixel_overlap_min:
            continue
        candidates.append(DuplicateSurfaceCandidate(
            cluster_a={"y_lo": ca.y_lo, "y_hi": ca.y_hi, "n_points": len(ca.point_ids), "centroid": ca.centroid.tolist()},
            cluster_b={"y_lo": cb.y_lo, "y_hi": cb.y_hi, "n_points": len(cb.point_ids), "centroid": cb.centroid.tolist()},
            xz_iou=iou,
            height_separation_m=height_sep,
            n_shared_frames=n_shared,
            pixel_overlap_fraction=overlap_frac,
        ))

    result.candidates = [
        {
            "cluster_a": c.cluster_a,
            "cluster_b": c.cluster_b,
            "xz_iou": c.xz_iou,
            "height_separation_m": c.height_separation_m,
            "n_shared_frames": c.n_shared_frames,
            "pixel_overlap_fraction": c.pixel_overlap_fraction,
        }
        for c in candidates
    ]
    result.detected = len(candidates) > 0
    if result.detected:
        result.reason = (
            f"{len(candidates)} cluster pair(s) with overlapping XZ footprint AND overlapping "
            "image-space reprojection in shared frames -- same surface likely triangulated twice "
            "at two depths (duplicate/ghost surface, e.g. reflective or repetitive-texture floor)"
        )
        logger.warning(
            f"duplicate_surface_detection: DETECTED {len(candidates)} candidate duplicate planar "
            f"surface pair(s) -- {result.reason}"
        )
    else:
        result.reason = f"{len(clusters)} horizontal planar cluster(s) found, no pair shares a reprojection footprint"
    return result


def run_duplicate_surface_qc(sparse_dir) -> DuplicateSurfaceDetectionResult:
    """Convenience wrapper: reads a COLMAP text-format sparse model
    directly from `sparse_dir` (must contain cameras.txt/images.txt/
    points3D.txt) and runs `detect_duplicate_planar_surfaces` on it.
    """
    from pathlib import Path
    from colmap_rgbd_gt.colmap.colmap_io import read_cameras_text, read_images_text, read_points3d_text

    sparse_dir = Path(sparse_dir)
    cameras = read_cameras_text(sparse_dir / "cameras.txt")
    images = read_images_text(sparse_dir / "images.txt")
    points_raw = read_points3d_text(sparse_dir / "points3D.txt")

    points = {
        pid: {"xyz": np.array(p["xyz"], dtype=np.float64), "rgb": p["rgb"]}
        for pid, p in points_raw.items()
    }
    for img in images.values():
        cam = cameras.get(img["camera_id"])
        if cam is not None:
            img["width"] = cam["width"]
            img["height"] = cam["height"]

    return detect_duplicate_planar_surfaces(points, images)
