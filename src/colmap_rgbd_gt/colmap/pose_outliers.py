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

Spatial-adjacency gating (added 2026-08-03, root-caused on hallway and
tableware1's manual reviews -- see outputs/pose_outlier_filter.json for
both scenes): the speed-jump test alone conflates two DIFFERENT physical
situations that both produce a momentary "implied speed above threshold"
reading --
  1. a genuine relocation: the two sides of the jump are also far apart
     in SPACE (floor2's real case: 7.4m apart), i.e. the trajectory
     actually teleports to a different place.
  2. tracking noise / a missed registration / fast-but-real motion: the
     two sides of the jump are spatially close (tableware1's close-range
     orbital scan taking a 10-25cm step slightly too fast for the fixed
     3.0 m/s threshold; hallway's dropped-frame gap and single-frame pose
     jitter, both landing back within the same small room), i.e. nothing
     actually relocated, the trajectory just took an oddly-timed step.
A fixed global speed threshold cannot tell these apart on its own --
scene (2) is a false positive that a purely temporal/speed-based test
will always flag, no matter how the threshold is tuned, because the
"jump" is real in the (distance, time) sense even though it is not a
real relocation in the (majority-scene-extent) sense. This module now
gates the jump test on spatial adjacency: before computing the
majority-ratio auto-resolve, each minority segment produced by the
speed-jump split is checked against the majority segment's actual point
cloud (nearest-point distance, not just its two adjacent boundary
frames, since a folded/looping trajectory can leave the majority
segment's own points on both sides of a minority segment). A minority
segment found within `spatial_adjacency_threshold_m` of the majority
segment is treated as noise-driven over-splitting and MERGED back into
the kept trajectory (not dropped) before the majority-ratio decision is
made at all. Only segments that remain spatially far from the majority
after this pass are ever candidates for dropping -- this is what lets
floor2's genuine 7.4m relocation still get dropped while tableware1's
25cm orbital step and hallway's 12cm/1.29m in-room noise no longer do.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_PLAUSIBLE_SPEED_MPS = 3.0
DEFAULT_MIN_MAJORITY_RATIO = 3.0
# Real-world evidence backing this default (2026-08-03 reviews): genuine
# relocations found so far are meters apart (floor2: ~7.4m). Noise-driven
# over-splits found so far are well under a meter (tableware1: 10-25cm;
# hallway: ~12cm jitter, ~1.29m dropped-frame gap). 2.0m sits between
# those two clusters -- comparable to "a step or two", well below any
# observed genuine relocation, well above any observed noise jump.
DEFAULT_SPATIAL_ADJACENCY_THRESHOLD_M = 2.0


@dataclass
class SegmentFilterResult:
    filtered_trajectory: list[dict[str, Any]]
    dropped_frame_ids: list[int] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)  # all segments found, for logging
    action_taken: bool = False
    reason: str = ""
    spatially_merged_frame_ids: list[int] = field(default_factory=list)


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


def _min_point_distance(segment_a: list[dict[str, Any]], segment_b: list[dict[str, Any]]) -> float:
    """Nearest-point distance between two segments' full point sets (not
    just their two temporally-adjacent boundary frames) -- a folded or
    looping trajectory can leave the majority segment's own points on
    both sides of a minority segment in space, even though only one
    boundary frame pair is temporally adjacent to it."""
    pts_a = np.asarray([e["t"] for e in segment_a], dtype=np.float64)
    pts_b = np.asarray([e["t"] for e in segment_b], dtype=np.float64)
    diffs = pts_a[:, None, :] - pts_b[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return float(dists.min())


def filter_disconnected_trajectory_segments(
    trajectory: list[dict[str, Any]],
    frame_id_to_ts_ns: dict[int, int],
    max_plausible_speed_mps: float = DEFAULT_MAX_PLAUSIBLE_SPEED_MPS,
    min_majority_ratio: float = DEFAULT_MIN_MAJORITY_RATIO,
    spatial_adjacency_threshold_m: float = DEFAULT_SPATIAL_ADJACENCY_THRESHOLD_M,
) -> SegmentFilterResult:
    """Detect and drop minority trajectory segments that are disconnected
    from a much larger majority segment by an implausible-speed jump AND
    spatially separated from it by more than `spatial_adjacency_threshold_m`.

    Two-stage, conservative by design:
    1. Partition the trajectory at implausible-speed jumps (as before).
    2. Spatial-adjacency gate: any minority segment whose nearest point to
       the majority segment is within `spatial_adjacency_threshold_m` is
       treated as a noise-driven over-split (fast-but-real motion, a
       missed-registration gap, single-frame jitter that lands back in
       the same physical place) and MERGED BACK into the kept trajectory,
       not dropped -- it never reaches the majority-ratio test at all.
    3. Only segments that remain spatially far from the majority after
       step 2 are evaluated by the majority-ratio test: dropped if the
       majority (now including anything merged in step 2) dominates by
       >= `min_majority_ratio`, otherwise left untouched and flagged for
       manual review (ambiguous, comparably-sized split -- could be a
       real two-room scene connected only by a weak/missing link).
    """
    sorted_traj = sorted(trajectory, key=lambda e: e["frame_id"])
    if not sorted_traj:
        return SegmentFilterResult(filtered_trajectory=[], action_taken=False, reason="empty trajectory")
    segments = _segment_by_speed_jumps(sorted_traj, frame_id_to_ts_ns, max_plausible_speed_mps)

    if len(segments) == 1:
        segments_info = [{
            "n_frames": len(segments[0]),
            "frame_id_range": [segments[0][0]["frame_id"], segments[0][-1]["frame_id"]],
        }]
        return SegmentFilterResult(
            filtered_trajectory=sorted_traj,
            segments=segments_info,
            action_taken=False,
            reason="no implausible-speed jump found; trajectory is one segment",
        )

    sizes = [len(seg) for seg in segments]
    majority_idx = int(np.argmax(sizes))
    majority_seg = segments[majority_idx]

    # Spatial-adjacency gate: separate minority segments into "spatially
    # embedded in the majority's extent" (noise -> merge back) vs
    # "genuinely spatially separated" (real relocation candidate).
    merged_segment_idxs: list[int] = []
    separated_segment_idxs: list[int] = []
    distances_to_majority: dict[int, float] = {}
    for i, seg in enumerate(segments):
        if i == majority_idx:
            continue
        d = _min_point_distance(seg, majority_seg)
        distances_to_majority[i] = d
        if d <= spatial_adjacency_threshold_m:
            merged_segment_idxs.append(i)
        else:
            separated_segment_idxs.append(i)

    segments_info = [
        {
            "n_frames": len(seg),
            "frame_id_range": [seg[0]["frame_id"], seg[-1]["frame_id"]],
            **({"min_distance_to_majority_m": round(distances_to_majority[i], 3),
                "spatially_adjacent_to_majority": i in merged_segment_idxs}
               if i != majority_idx else {}),
        }
        for i, seg in enumerate(segments)
    ]

    spatially_merged_frame_ids = [
        e["frame_id"] for i in merged_segment_idxs for e in segments[i]
    ]
    if merged_segment_idxs:
        logger.warning(
            f"pose_outliers: {len(merged_segment_idxs)} of {len(segments) - 1} minority "
            f"segment(s) are within {spatial_adjacency_threshold_m}m of the majority "
            f"segment's own points (spatially embedded, not a real relocation) -- merging "
            f"back into the kept trajectory instead of treating as split candidates. "
            f"Merged frame_ids: {spatially_merged_frame_ids}. Distances: "
            f"{ {i: round(distances_to_majority[i], 3) for i in merged_segment_idxs} }"
        )

    # Everything not spatially separated is now part of the effective
    # majority for the ratio test.
    combined_majority = list(majority_seg)
    for i in merged_segment_idxs:
        combined_majority.extend(segments[i])
    combined_majority_size = len(combined_majority)

    if not separated_segment_idxs:
        # No genuinely separated segment remains after spatial merging --
        # this is one continuous scene over-split by speed-jump noise.
        return SegmentFilterResult(
            filtered_trajectory=sorted_traj,
            segments=segments_info,
            action_taken=False,
            reason=(
                f"{len(segments)} segments found by speed-jump test, but all "
                f"{len(merged_segment_idxs)} minority segment(s) are spatially adjacent "
                f"(<= {spatial_adjacency_threshold_m}m) to the majority segment -- treated "
                f"as noise-driven over-split (fast-but-real motion / registration gap / "
                f"jitter), not a real relocation. Trajectory kept intact."
            ),
            spatially_merged_frame_ids=spatially_merged_frame_ids,
        )

    other_size = sum(len(segments[i]) for i in separated_segment_idxs)

    if other_size == 0 or combined_majority_size < min_majority_ratio * other_size:
        logger.warning(
            f"pose_outliers: after spatial-adjacency merging, {len(separated_segment_idxs)} "
            f"genuinely spatially-separated segment(s) remain split from the "
            f"{combined_majority_size}-frame effective majority by an implausible-speed "
            f"jump (>{max_plausible_speed_mps} m/s AND >{spatial_adjacency_threshold_m}m), "
            f"but not dominant enough over them ({other_size} frames, ratio "
            f"{combined_majority_size / max(1, other_size):.1f}x < required "
            f"{min_majority_ratio}x) to auto-resolve. Leaving trajectory UNCHANGED -- "
            f"segments: {segments_info}. Needs manual review."
        )
        return SegmentFilterResult(
            filtered_trajectory=sorted_traj,
            segments=segments_info,
            action_taken=False,
            reason=(
                f"{len(separated_segment_idxs)} spatially-separated segment(s) found (after "
                f"merging {len(merged_segment_idxs)} spatially-adjacent segment(s) back) but "
                f"not dominant enough to auto-resolve (effective majority "
                f"{combined_majority_size} vs rest {other_size}, need {min_majority_ratio}x)"
            ),
            spatially_merged_frame_ids=spatially_merged_frame_ids,
        )

    dropped_frame_ids = [
        e["frame_id"] for i in separated_segment_idxs for e in segments[i]
    ]
    logger.warning(
        f"pose_outliers: dropping {other_size} frame(s) in {len(separated_segment_idxs)} "
        f"spatially-separated minority segment(s) disconnected from the "
        f"{combined_majority_size}-frame effective majority segment by an implausible-speed "
        f"jump (>{max_plausible_speed_mps} m/s AND >{spatial_adjacency_threshold_m}m) -- "
        f"dropped frame_ids: {dropped_frame_ids}. Segments: {segments_info}"
    )
    combined_majority.sort(key=lambda e: e["frame_id"])
    return SegmentFilterResult(
        filtered_trajectory=combined_majority,
        dropped_frame_ids=dropped_frame_ids,
        segments=segments_info,
        action_taken=True,
        reason=(
            f"dropped {other_size} frame(s) in spatially-separated minority segment(s), kept "
            f"{combined_majority_size}-frame effective majority (ratio "
            f"{combined_majority_size / max(1, other_size):.1f}x >= {min_majority_ratio}x); "
            f"also merged {len(merged_segment_idxs)} spatially-adjacent segment(s) "
            f"({len(spatially_merged_frame_ids)} frame(s)) back into the kept trajectory"
        ),
        spatially_merged_frame_ids=spatially_merged_frame_ids,
    )
