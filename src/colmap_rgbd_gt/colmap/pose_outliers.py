"""Disconnected-segment detection/rejection for a c2w trajectory.

Root-caused 2026-08-03 on a real scene (floor2): global_mapper's global
positioning solve can place a small, weakly-connected sub-scene (e.g. a
brief kitchenette visit at the very start of a scan) at a wrong absolute
position relative to the rest of the trajectory, even though every pose
within that sub-scene is internally self-consistent -- the images
themselves (297 and 304) showed the IDENTICAL physical location, ~0.43s
apart in real capture time, yet were placed ~7.4m apart (implied ~17 m/s,
physically impossible for handheld capture). scene_metadata.json's
speed/rotation leading-frames mechanism (see export/scene_metadata.py)
already flags exactly this pattern (`max_mps_is_leading_frames=False`,
i.e. a spike well past the trajectory's start) -- this module acts on
that signal instead of just reporting it: it partitions the trajectory at
implausible-speed jumps and drops minority segments that are dominated by
a much larger, internally-coherent majority segment.

This is deliberately conservative: it only auto-drops a segment when the
majority segment is overwhelmingly larger (default 3x), so a genuinely
ambiguous case (e.g. two comparably-sized segments, which could indicate
a real two-room scene connected only by a weak/missing link, not
necessarily one wrong segment) is left alone and flagged loudly for
manual review rather than silently discarding potentially-legitimate
data.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_PLAUSIBLE_SPEED_MPS = 3.0
DEFAULT_MIN_MAJORITY_RATIO = 3.0


@dataclass
class SegmentFilterResult:
    filtered_trajectory: list[dict[str, Any]]
    dropped_frame_ids: list[int] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)  # all segments found, for logging
    action_taken: bool = False
    reason: str = ""


def _segment_by_speed_jumps(
    sorted_trajectory: list[dict[str, Any]],
    frame_id_to_ts_ns: dict[int, int],
    max_plausible_speed_mps: float,
) -> list[list[dict[str, Any]]]:
    """Split a frame_id-sorted trajectory into contiguous segments at any
    step whose implied speed (real capture time, not frame index) exceeds
    `max_plausible_speed_mps`. Steps with a missing timestamp on either
    side are treated as non-jumps (can't compute a speed) -- this filter
    only acts where it has real evidence."""
    if len(sorted_trajectory) < 2:
        return [list(sorted_trajectory)]

    segments: list[list[dict[str, Any]]] = [[sorted_trajectory[0]]]
    for prev, curr in zip(sorted_trajectory[:-1], sorted_trajectory[1:]):
        ts_prev = frame_id_to_ts_ns.get(prev["frame_id"])
        ts_curr = frame_id_to_ts_ns.get(curr["frame_id"])
        is_jump = False
        if ts_prev is not None and ts_curr is not None and ts_curr > ts_prev:
            dt_s = (ts_curr - ts_prev) / 1e9
            dist_m = float(np.linalg.norm(np.asarray(curr["t"]) - np.asarray(prev["t"])))
            speed_mps = dist_m / dt_s if dt_s > 0 else 0.0
            is_jump = speed_mps > max_plausible_speed_mps
        if is_jump:
            segments.append([curr])
        else:
            segments[-1].append(curr)
    return segments


def filter_disconnected_trajectory_segments(
    trajectory: list[dict[str, Any]],
    frame_id_to_ts_ns: dict[int, int],
    max_plausible_speed_mps: float = DEFAULT_MAX_PLAUSIBLE_SPEED_MPS,
    min_majority_ratio: float = DEFAULT_MIN_MAJORITY_RATIO,
) -> SegmentFilterResult:
    """Detect and drop minority trajectory segments that are disconnected
    from a much larger majority segment by an implausible-speed jump.

    Conservative by design: only acts when exactly the majority segment's
    frame count is >= `min_majority_ratio` times the SUM of all other
    segments' frame counts. Otherwise returns the trajectory unchanged
    with `action_taken=False` and a `reason` explaining why (ambiguous
    split, or no jump found at all), so an ambiguous case is surfaced for
    manual review rather than silently resolved either way.
    """
    sorted_traj = sorted(trajectory, key=lambda e: e["frame_id"])
    if not sorted_traj:
        return SegmentFilterResult(filtered_trajectory=[], action_taken=False, reason="empty trajectory")
    segments = _segment_by_speed_jumps(sorted_traj, frame_id_to_ts_ns, max_plausible_speed_mps)

    segments_info = [
        {
            "n_frames": len(seg),
            "frame_id_range": [seg[0]["frame_id"], seg[-1]["frame_id"]],
        }
        for seg in segments
    ]

    if len(segments) == 1:
        return SegmentFilterResult(
            filtered_trajectory=sorted_traj,
            segments=segments_info,
            action_taken=False,
            reason="no implausible-speed jump found; trajectory is one segment",
        )

    sizes = [len(seg) for seg in segments]
    majority_idx = int(np.argmax(sizes))
    majority_size = sizes[majority_idx]
    other_size = sum(sizes) - majority_size

    if other_size == 0 or majority_size < min_majority_ratio * other_size:
        logger.warning(
            f"pose_outliers: found {len(segments)} trajectory segments split by an "
            f"implausible-speed jump (>{max_plausible_speed_mps} m/s), but the largest "
            f"({majority_size} frames) is not dominant enough over the rest ({other_size} "
            f"frames, ratio {majority_size / max(1, other_size):.1f}x < required "
            f"{min_majority_ratio}x) to auto-resolve. Leaving trajectory UNCHANGED -- "
            f"segments: {segments_info}. Needs manual review."
        )
        return SegmentFilterResult(
            filtered_trajectory=sorted_traj,
            segments=segments_info,
            action_taken=False,
            reason=(
                f"{len(segments)} segments found but not dominant enough to auto-resolve "
                f"(largest {majority_size} vs rest {other_size}, need {min_majority_ratio}x)"
            ),
        )

    dropped_frame_ids = [
        e["frame_id"] for i, seg in enumerate(segments) if i != majority_idx for e in seg
    ]
    logger.warning(
        f"pose_outliers: dropping {other_size} frame(s) in {len(segments) - 1} minority "
        f"segment(s) disconnected from the {majority_size}-frame majority segment by an "
        f"implausible-speed jump (>{max_plausible_speed_mps} m/s) -- dropped frame_ids: "
        f"{dropped_frame_ids}. Segments: {segments_info}"
    )
    return SegmentFilterResult(
        filtered_trajectory=list(segments[majority_idx]),
        dropped_frame_ids=dropped_frame_ids,
        segments=segments_info,
        action_taken=True,
        reason=(
            f"dropped {other_size} frame(s) in minority segment(s), kept {majority_size}-frame "
            f"majority (ratio {majority_size / max(1, other_size):.1f}x >= {min_majority_ratio}x)"
        ),
    )
