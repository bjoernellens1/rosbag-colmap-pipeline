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
