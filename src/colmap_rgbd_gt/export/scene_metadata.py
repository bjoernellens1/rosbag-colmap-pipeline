"""Quick-glance scene QC metadata: trajectory length/speed/rotation,
bounding-box extent/volume, and a coarse revisit/loop-closure signal --
computed from the exported (metric, c2w) GT trajectory, for judging "was
this reconstruction constructed properly" without opening a 3D viewer.

Reuses the same trajectory list-of-dicts format
(`colmap.pose_extract.extract_trajectory`/`scale_trajectory`: each entry
has `frame_id`, `R` (3x3 c2w), `t` (3,)) and the same rgb.csv real-
timestamp lookup pattern already used by `export/rosbag_writer.py`
(imported directly from there rather than re-derived, per that module's
own `_load_frame_timestamps`).
"""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from colmap_rgbd_gt.export.rosbag_writer import _load_frame_timestamps
from colmap_rgbd_gt.utils.transforms import rotation_angle_deg
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

# A pose only counts as a "revisit" of an earlier one if they're this close
# in space (typical indoor loop-closure distance) AND separated by at least
# this much wall-clock time (so simply-adjacent, slow-moving frames along
# the trajectory don't trivially count as "revisiting" each other).
DEFAULT_REVISIT_DISTANCE_M = 0.5
DEFAULT_REVISIT_MIN_TIME_GAP_S = 5.0

# FIXED 2026-08-03 (found investigating table1's real scene_metadata.json):
# a max_mps/max_deg_per_frame spike driven by the trajectory's first couple
# of pose transitions is a qualitatively different (much less concerning)
# signal than a mid-trajectory spike -- COLMAP's earliest registered poses
# haven't accumulated much multi-view constraint yet and are characteristically
# less stable, not evidence of a genuine tracking glitch/teleport. Verified
# on table1: frames 0->1 (9.499 m/s) and 1->2 (9.027 m/s) were the only two
# outliers: everything else in the top-8 speeds was <=1.16 m/s, consistent
# with a normal handheld pace. Without flagging *where* the max occurred, a
# reader has to re-derive this distinction by hand every time. If this shows
# up consistently across scenes it's a general COLMAP-early-registration
# characteristic; check the "is_leading" flag on other scenes' output before
# assuming it's scene-specific.
LEADING_STEP_COUNT = 2


def compute_scene_metadata(
    trajectory: list[dict[str, Any]],
    rgb_csv_path: Path,
    frame_count: int | None = None,
    registered_count: int | None = None,
    revisit_distance_m: float = DEFAULT_REVISIT_DISTANCE_M,
    revisit_min_time_gap_s: float = DEFAULT_REVISIT_MIN_TIME_GAP_S,
) -> dict[str, Any]:
    """Compute QC summary metrics from a (metric, c2w) trajectory.

    Args:
        trajectory: c2w trajectory entries (frame_id/R/t), as returned by
            `colmap.pose_extract.extract_trajectory`/`scale_trajectory`.
        rgb_csv_path: workspace's `timestamps/rgb.csv`, for real capture
            timestamps (frame_id is an index, not a time -- speed/duration
            metrics need the real clock).
        frame_count: total frames in the source (for the registration
            ratio); omitted if unknown.
        registered_count: registered frames (defaults to len(trajectory)
            if not given -- trajectory only contains registered poses).

    Returns:
        A JSON-serializable dict; see the module docstring / README for
        the field list. `pose_count == 0`/`1` degenerate cases return
        zeroed/None metrics rather than raising, so this never blocks the
        export pipeline.
    """
    sorted_traj = sorted(trajectory, key=lambda e: e["frame_id"])
    n_poses = len(sorted_traj)
    registered_count = registered_count if registered_count is not None else n_poses

    result: dict[str, Any] = {
        "n_poses": n_poses,
        "registered_frames": registered_count,
        "total_frames": frame_count,
        "registration_ratio": (
            registered_count / frame_count if frame_count else None
        ),
    }

    if n_poses == 0:
        result.update({
            "trajectory_length_m": 0.0,
            "duration_s": None,
            "speed": None,
            "rotation": None,
            "extent": None,
            "revisit": None,
        })
        return result

    positions = np.array([e["t"] for e in sorted_traj], dtype=np.float64)
    rotations = [np.asarray(e["R"], dtype=np.float64) for e in sorted_traj]

    # --- Trajectory length ---
    if n_poses >= 2:
        deltas = np.diff(positions, axis=0)
        step_lengths = np.linalg.norm(deltas, axis=1)
        trajectory_length_m = float(np.sum(step_lengths))
    else:
        step_lengths = np.array([])
        trajectory_length_m = 0.0
    result["trajectory_length_m"] = trajectory_length_m

    # --- Timestamps / speed ---
    frame_timestamps = _load_frame_timestamps(rgb_csv_path)
    ts_ns = np.array([
        frame_timestamps.get(e["frame_id"], -1) for e in sorted_traj
    ], dtype=np.int64)
    have_ts = np.all(ts_ns >= 0)

    speed_block: dict[str, Any] | None = None
    if have_ts and n_poses >= 2:
        ts_s = ts_ns.astype(np.float64) / 1e9
        duration_s = float(ts_s[-1] - ts_s[0])
        dt = np.diff(ts_s)
        # Guard against duplicate/out-of-order timestamps (dt<=0) rather
        # than dividing by zero/negative -- exclude those steps from the
        # per-step speed stats but keep them in trajectory_length_m (the
        # path length itself doesn't depend on timing).
        valid_dt = dt > 0
        if np.any(valid_dt):
            step_speeds = step_lengths[valid_dt] / dt[valid_dt]
            # Map back from the valid_dt-filtered array to the original
            # step index (0 = the transition from sorted_traj[0] to
            # sorted_traj[1]), so max_mps_step_index means what it says
            # even when some steps were excluded for bad timestamps.
            valid_step_indices = np.where(valid_dt)[0]
            max_i = int(valid_step_indices[np.argmax(step_speeds)])
            speed_block = {
                "avg_mps": trajectory_length_m / duration_s if duration_s > 0 else None,
                "max_mps": float(np.max(step_speeds)),
                "std_mps": float(np.std(step_speeds)),
                # Which step (sorted_traj[max_mps_step_index] ->
                # sorted_traj[max_mps_step_index+1]) produced the max, and
                # whether it falls within the first LEADING_STEP_COUNT
                # transitions -- see the module-level note above on why
                # that distinction matters (early-registration noise vs a
                # genuine mid-trajectory glitch/teleport).
                "max_mps_step_index": max_i,
                "max_mps_frame_ids": [
                    sorted_traj[max_i]["frame_id"], sorted_traj[max_i + 1]["frame_id"],
                ],
                "max_mps_is_leading_frames": max_i < LEADING_STEP_COUNT,
            }
        result["duration_s"] = duration_s
    else:
        result["duration_s"] = None
        if not have_ts:
            logger.warning(
                f"scene_metadata: some frame_ids missing from {rgb_csv_path}; "
                "speed metrics unavailable (trajectory_length_m/rotation/extent unaffected)"
            )
    result["speed"] = speed_block

    # --- Rotation ---
    rotation_block: dict[str, Any] | None = None
    if n_poses >= 2:
        step_angles_deg = np.array([
            rotation_angle_deg(rotations[i], rotations[i + 1])
            for i in range(n_poses - 1)
        ])
        max_i = int(np.argmax(step_angles_deg))
        rotation_block = {
            "total_deg": float(np.sum(step_angles_deg)),
            "avg_deg_per_frame": float(np.mean(step_angles_deg)),
            "max_deg_per_frame": float(np.max(step_angles_deg)),
            # Same leading-frames distinction as speed.max_mps_* above.
            "max_deg_step_index": max_i,
            "max_deg_frame_ids": [
                sorted_traj[max_i]["frame_id"], sorted_traj[max_i + 1]["frame_id"],
            ],
            "max_deg_is_leading_frames": max_i < LEADING_STEP_COUNT,
        }
    result["rotation"] = rotation_block

    # --- Scene extent / bbox volume ---
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extent = maxs - mins
    result["extent"] = {
        "min_m": mins.tolist(),
        "max_m": maxs.tolist(),
        "size_m": extent.tolist(),
        # COARSE proxy only: this is the axis-aligned bounding-box volume
        # of the camera trajectory's positions, NOT a real occupied-space
        # estimate. A long, thin, diagonally-oriented corridor trajectory
        # can have a large AABB volume despite occupying comparatively
        # little actual space (and conversely a trajectory that happens to
        # align with the axes underestimates less) -- treat this as a
        # rough scale indicator, not a precision volume measurement.
        "bbox_volume_m3_COARSE_PROXY": float(np.prod(extent)),
    }

    # --- Revisit / coarse loop-closure signal ---
    revisit_block = None
    if have_ts and n_poses >= 2:
        ts_s = ts_ns.astype(np.float64) / 1e9
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=revisit_distance_m)
        revisited = np.zeros(n_poses, dtype=bool)
        n_qualifying_pairs = 0
        for i, j in pairs:
            if abs(ts_s[i] - ts_s[j]) >= revisit_min_time_gap_s:
                revisited[i] = True
                revisited[j] = True
                n_qualifying_pairs += 1
        revisit_block = {
            "distance_threshold_m": revisit_distance_m,
            "min_time_gap_s": revisit_min_time_gap_s,
            "revisited_pose_fraction": float(np.mean(revisited)),
            "n_revisit_pairs": n_qualifying_pairs,
        }
    result["revisit"] = revisit_block

    return result


def save_scene_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    import json
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Scene metadata saved to {output_path}")
