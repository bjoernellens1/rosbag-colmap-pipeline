"""Tests for colmap/pose_outliers.py's disconnected-segment filter.

Root-cause case: floor2's real scene had frames 297 and 304 (0.43s apart
in real capture time, ~7.4m apart in exported position -- implied ~17 m/s,
physically impossible) turn out to be a small (11-frame) weakly-connected
segment glued at the wrong position relative to a much larger (256-frame)
internally-coherent majority segment. This filter must reproduce that
exact resolution on a synthetic version of the same pattern, and must NOT
touch a trajectory with no jump, and must NOT auto-resolve an ambiguous
(comparably-sized) split.
"""

import numpy as np
import pytest

from colmap_rgbd_gt.colmap.pose_outliers import filter_disconnected_trajectory_segments


def _entry(frame_id, t):
    return {"frame_id": frame_id, "R": np.eye(3), "t": np.array(t, dtype=np.float64)}


def _ts_map(frame_ids, dt_s=0.1):
    return {fid: int(i * dt_s * 1e9) for i, fid in enumerate(frame_ids)}


def test_no_jump_leaves_trajectory_unchanged():
    frame_ids = list(range(10))
    trajectory = [_entry(i, [i * 0.05, 0.0, 0.0]) for i in frame_ids]  # slow, plausible walk
    ts = _ts_map(frame_ids, dt_s=1.0)

    result = filter_disconnected_trajectory_segments(trajectory, ts)

    assert result.action_taken is False
    assert result.dropped_frame_ids == []
    assert len(result.filtered_trajectory) == 10
    assert len(result.segments) == 1


def test_floor2_like_minority_segment_is_dropped():
    """Reproduces the real pattern: a small early cluster (11 frames, tight
    together), then one implausible jump, then a much larger (30-frame)
    coherent majority segment -- majority:minority ratio ~2.7x... needs to
    clear the 3x default, so size the majority larger (33 frames) for a
    clean >3x ratio, matching floor2's real ~23x ratio in spirit."""
    minority_ids = list(range(11))  # frame_ids 0..10, tight cluster near origin
    minority = [_entry(i, [0.01 * i, 0.0, 0.0]) for i in minority_ids]

    majority_ids = list(range(11, 44))  # 33 frames, far away, smooth walk
    majority = [_entry(i, [10.0 + 0.05 * (i - 11), 0.0, 0.0]) for i in majority_ids]

    trajectory = minority + majority
    ts = _ts_map(minority_ids + majority_ids, dt_s=0.5)  # 0.5s/frame -> plausible speeds within each cluster

    result = filter_disconnected_trajectory_segments(
        trajectory, ts, max_plausible_speed_mps=3.0,
    )

    assert result.action_taken is True
    assert set(result.dropped_frame_ids) == set(minority_ids)
    assert [e["frame_id"] for e in result.filtered_trajectory] == majority_ids
    assert len(result.segments) == 2


def test_ambiguous_comparable_size_split_is_not_auto_resolved():
    """Two comparably-sized segments (not a dominant majority) must be left
    untouched -- could be a real two-room scene with a weak/missing link,
    not necessarily "one segment is wrong". Flagged, not silently resolved."""
    seg_a_ids = list(range(10))
    seg_a = [_entry(i, [0.01 * i, 0.0, 0.0]) for i in seg_a_ids]
    seg_b_ids = list(range(10, 22))
    seg_b = [_entry(i, [10.0 + 0.01 * (i - 10), 0.0, 0.0]) for i in seg_b_ids]

    trajectory = seg_a + seg_b
    ts = _ts_map(seg_a_ids + seg_b_ids, dt_s=0.5)

    result = filter_disconnected_trajectory_segments(
        trajectory, ts, max_plausible_speed_mps=3.0, min_majority_ratio=3.0,
    )

    assert result.action_taken is False
    assert result.dropped_frame_ids == []
    assert len(result.filtered_trajectory) == len(trajectory)
    assert len(result.segments) == 2


def test_missing_timestamps_are_not_treated_as_jumps():
    frame_ids = [0, 1, 2]
    trajectory = [_entry(i, [i * 5.0, 0.0, 0.0]) for i in frame_ids]  # would be a huge jump if timed
    ts = {}  # no timestamps at all

    result = filter_disconnected_trajectory_segments(trajectory, ts)

    assert result.action_taken is False
    assert len(result.segments) == 1


def test_single_pose_trajectory_does_not_raise():
    result = filter_disconnected_trajectory_segments([_entry(0, [0, 0, 0])], {})
    assert result.action_taken is False
    assert len(result.filtered_trajectory) == 1


def test_empty_trajectory_does_not_raise():
    result = filter_disconnected_trajectory_segments([], {})
    assert result.filtered_trajectory == []
    assert result.action_taken is False


# --- Spatial-adjacency gating (added 2026-08-03, hallway/tableware1 follow-up) ---
#
# The speed-jump test alone can't distinguish a genuine relocation (far
# apart in SPACE, not just briefly fast) from noise that happens to cross
# the speed threshold (a real but small step, a dropped-frame gap, single-
# frame jitter) while staying within the same small physical area. These
# tests cover the three real cases this session's manual reviews found:
# (a) floor2-style genuine disconnection (regression -- must still split),
# (b) tableware1-style close-range fast orbital motion (must now merge
#     back, not split), and (c) hallway-style single-frame teleport glitch
#     mixed with in-room noise (glitch must still be identified/dropped,
#     the noise must not be conflated with it).


def test_floor2_like_genuine_disconnection_still_splits_regression():
    """REGRESSION (must not break with spatial-adjacency gating added):
    two clusters genuinely >2m apart in space, no intermediate points --
    "camera picked up and carried to a different room". This must still
    be detected and the minority segment still dropped."""
    minority_ids = list(range(11))
    minority = [_entry(i, [0.01 * i, 0.0, 0.0]) for i in minority_ids]  # tight cluster near origin

    majority_ids = list(range(11, 44))
    majority = [_entry(i, [10.0 + 0.05 * (i - 11), 0.0, 0.0]) for i in majority_ids]  # far away, coherent

    trajectory = minority + majority
    ts = _ts_map(minority_ids + majority_ids, dt_s=0.5)

    result = filter_disconnected_trajectory_segments(trajectory, ts, max_plausible_speed_mps=3.0)

    assert result.action_taken is True
    assert set(result.dropped_frame_ids) == set(minority_ids)
    assert [e["frame_id"] for e in result.filtered_trajectory] == majority_ids
    assert result.spatially_merged_frame_ids == []


def test_tableware1_like_close_range_fast_motion_merges_back_not_split():
    """Reproduces the real tableware1 pattern: a close-range orbital scan
    takes small (10-25cm) steps that momentarily cross the fixed 3.0 m/s
    speed threshold purely because the steps are quick, not because
    anything relocated. The "minority" segment sits well within the
    majority segment's own spatial extent (a 25cm step vs a >3m-wide
    orbit) -- spatial-adjacency gating must merge it back, keeping the
    full trajectory intact instead of dropping 1/3 of the scene."""
    seg1_ids = list(range(0, 30))  # long arc, radius ~1.5m around origin
    seg1 = [_entry(i, [1.5 * np.cos(i * 0.2), 0.0, 1.5 * np.sin(i * 0.2)]) for i in seg1_ids]

    # A short (5-frame) segment reached by one fast-but-small (25cm/69ms
    # ~ 3.6 m/s) step, then continuing the SAME orbital arc (spatially
    # embedded within seg1's own extent, not a relocation).
    last = seg1[-1]["t"]
    step_dir = np.array([1.0, 0.0, 0.3])
    step_dir /= np.linalg.norm(step_dir)
    jump_point = last + step_dir * 0.25  # 25cm jump, above speed threshold given dt below
    seg2_ids = list(range(30, 35))
    seg2 = [_entry(seg2_ids[0], jump_point)]
    for k, fid in enumerate(seg2_ids[1:], start=1):
        seg2.append(_entry(fid, jump_point + np.array([0.02 * k, 0.0, 0.01 * k])))

    seg3_ids = list(range(35, 60))  # arc continues, still within the same orbit
    seg3 = [_entry(i, [1.5 * np.cos(i * 0.2), 0.0, 1.5 * np.sin(i * 0.2)]) for i in seg3_ids]

    trajectory = seg1 + seg2 + seg3
    all_ids = seg1_ids + seg2_ids + seg3_ids
    # 69ms/frame within the fast step, matching tableware1's real timing
    ts = dict(_ts_map(seg1_ids, dt_s=0.5))
    base_t = ts[seg1_ids[-1]]
    ts[seg2_ids[0]] = base_t + int(0.069 * 1e9)
    for k, fid in enumerate(seg2_ids[1:], start=1):
        ts[fid] = ts[seg2_ids[0]] + int(k * 0.5 * 1e9)
    ts.update(_ts_map(seg3_ids, dt_s=0.5))
    for i, fid in enumerate(seg3_ids):
        ts[fid] = ts[seg2_ids[-1]] + int((i + 1) * 0.5 * 1e9)

    result = filter_disconnected_trajectory_segments(trajectory, ts, max_plausible_speed_mps=3.0)

    assert result.action_taken is False
    assert result.dropped_frame_ids == []
    assert len(result.filtered_trajectory) == len(trajectory)
    assert len(result.segments) >= 2  # the speed-jump test still found a split
    assert set(seg2_ids).issubset(set(result.spatially_merged_frame_ids))


def test_hallway_like_single_frame_teleport_glitch_still_identified_and_dropped():
    """Reproduces the real hallway pattern: a large coherent trajectory
    (mostly confined to a small room) with ONE genuine single-frame
    teleport glitch far outside the room (the >>100 m/s frame that
    matched scene_metadata's reported extent.max_m exactly), PLUS a
    separate, spatially-adjacent small segment caused by a dropped-frame
    gap (must be merged, not conflated with the real glitch). The real
    glitch is far (>2m) from the majority's extent and must still be
    identified and dropped; the spatially-adjacent noise segment must NOT
    be dropped alongside it."""
    room_a_ids = list(range(0, 40))
    room_a = [_entry(i, [0.3 * np.cos(i * 0.3), 0.0, 0.3 * np.sin(i * 0.3)]) for i in room_a_ids]

    # Dropped-frame-gap-style small segment: one fast step (~1.29m analog
    # scaled down for a tight test room) that lands back in the same room.
    gap_ids = [40, 41]
    gap = [_entry(40, [0.5, 0.0, 0.5]), _entry(41, [0.55, 0.0, 0.52])]

    room_b_ids = list(range(42, 80))
    room_b = [_entry(i, [0.3 * np.cos(i * 0.3), 0.0, 0.3 * np.sin(i * 0.3)]) for i in room_b_ids]

    # The real glitch: a single frame teleported ~10m away, then back.
    glitch_ids = [80]
    glitch = [_entry(80, [10.0, 5.0, 8.0])]

    room_c_ids = list(range(81, 120))
    room_c = [_entry(i, [0.3 * np.cos(i * 0.3), 0.0, 0.3 * np.sin(i * 0.3)]) for i in room_c_ids]

    trajectory = room_a + gap + room_b + glitch + room_c
    all_ids = room_a_ids + gap_ids + room_b_ids + glitch_ids + room_c_ids
    ts = _ts_map(all_ids, dt_s=0.1)  # tight timing so any real spatial jump reads as fast

    result = filter_disconnected_trajectory_segments(trajectory, ts, max_plausible_speed_mps=3.0)

    assert result.action_taken is True
    assert result.dropped_frame_ids == glitch_ids
    assert set(gap_ids).issubset(set(result.spatially_merged_frame_ids))
    assert 80 not in result.spatially_merged_frame_ids
    kept_ids = {e["frame_id"] for e in result.filtered_trajectory}
    assert kept_ids == set(all_ids) - set(glitch_ids)
