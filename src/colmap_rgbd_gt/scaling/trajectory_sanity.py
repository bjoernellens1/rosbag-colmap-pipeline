"""Independent scene-extent/path-length sanity check on an exported metric
trajectory -- a second, cause-agnostic backstop alongside
`colmap.pose_outliers.assess_fragmentation_severity`.

Root-caused on kitchen1 (2026-08-05): a genuinely corrupted trajectory (43
spatially-disconnected segments from a global_mapper weak-link failure, see
pose_outliers.py's assess_fragmentation_severity docstring for the mechanism)
produced a real, checkable symptom independent of segment analysis: 108.98m
of cumulative path length inside a room whose own bounding-box diagonal was
only ~16m (a 6.8x ratio) -- the kind of number a human reviewing the scene
would immediately flag as physically implausible ("that's way too much
walking for a room this size"), which this module checks automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_PATH_TO_EXTENT_RATIO = 15.0


@dataclass
class PlausibilityResult:
    severe: bool
    reason: str
    bbox_diagonal_m: float
    path_length_m: float
    path_to_extent_ratio: float
    max_step_m: float
    max_step_speed_mps: float | None
    n_positions: int


def check_trajectory_plausibility(
    trajectory: list[dict[str, Any]],
    frame_id_to_ts_ns: dict[int, int] | None = None,
    max_path_to_extent_ratio: float = DEFAULT_MAX_PATH_TO_EXTENT_RATIO,
) -> PlausibilityResult:
    """Flag a trajectory whose cumulative path length is implausibly large
    relative to its own spatial extent -- independent of (and a backstop
    for) the segment-jump-based `assess_fragmentation_severity` check, which
    can miss a trajectory that's incoherent without cleanly separating into
    detectable segments.

    `max_path_to_extent_ratio` is deliberately generous (15x) -- a real
    back-and-forth walkthrough of a room can legitimately retrace itself
    several times over without being wrong.
    """
    sorted_traj = sorted(trajectory, key=lambda e: e["frame_id"])
    n = len(sorted_traj)
    if n < 2:
        return PlausibilityResult(
            severe=False, reason="fewer than 2 positions, nothing to check",
            bbox_diagonal_m=0.0, path_length_m=0.0, path_to_extent_ratio=0.0,
            max_step_m=0.0, max_step_speed_mps=None, n_positions=n,
        )

    positions = np.asarray([e["t"] for e in sorted_traj], dtype=np.float64)
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = float(steps.sum())
    max_step_idx = int(np.argmax(steps))
    max_step = float(steps[max_step_idx])

    max_step_speed = None
    if frame_id_to_ts_ns is not None:
        fid_a = sorted_traj[max_step_idx]["frame_id"]
        fid_b = sorted_traj[max_step_idx + 1]["frame_id"]
        ts_a = frame_id_to_ts_ns.get(fid_a)
        ts_b = frame_id_to_ts_ns.get(fid_b)
        if ts_a is not None and ts_b is not None and ts_b > ts_a:
            dt_s = (ts_b - ts_a) / 1e9
            max_step_speed = max_step / dt_s if dt_s > 0 else None

    ratio = path_length / bbox_diagonal if bbox_diagonal > 1e-6 else float("inf")
    severe = ratio > max_path_to_extent_ratio

    reason = (
        f"path_length={path_length:.2f}m bbox_diagonal={bbox_diagonal:.2f}m "
        f"ratio={ratio:.2f}x (threshold {max_path_to_extent_ratio}x), "
        f"max_step={max_step:.2f}m"
        + (f" ({max_step_speed:.2f} m/s implied)" if max_step_speed is not None else "")
    )
    if severe:
        reason = "implausible trajectory: " + reason

    return PlausibilityResult(
        severe=severe, reason=reason,
        bbox_diagonal_m=bbox_diagonal, path_length_m=path_length,
        path_to_extent_ratio=ratio, max_step_m=max_step,
        max_step_speed_mps=max_step_speed, n_positions=n,
    )
