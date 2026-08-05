"""Tests for scaling/trajectory_sanity.py -- the independent, cause-agnostic
path-length/extent plausibility backstop.

Root-cause case: kitchen1's real corrupted trajectory had 108.98m of
cumulative path length inside a room whose own bounding-box diagonal was
only ~16m (a ~6.8x ratio) -- this module must flag that shape, must NOT
flag a real (possibly-repeated) walkthrough of a small room, and must not
raise on degenerate (<2 position) input.
"""
import numpy as np
import pytest

from colmap_rgbd_gt.scaling.trajectory_sanity import check_trajectory_plausibility


def _entry(frame_id, t):
    return {"frame_id": frame_id, "t": np.array(t, dtype=np.float64)}


def test_clean_single_cluster_walkthrough_not_severe():
    # A smooth 10m walk down a corridor -- path length ~= extent.
    trajectory = [_entry(i, [i * 0.5, 0.0, 0.0]) for i in range(21)]

    result = check_trajectory_plausibility(trajectory)

    assert result.severe is False
    assert result.path_to_extent_ratio == pytest.approx(1.0, abs=0.05)


def test_kitchen1_like_disconnected_clusters_is_severe():
    # Two clusters ~12m apart with no dominant majority (mirrors the real
    # unresolved kitchen1 split) -- interleaved so bbox_diagonal is small
    # relative to how far apart consecutive-in-frame_id steps actually are.
    cluster_a = [_entry(2 * i, [0.05 * i, 0.0, 0.0]) for i in range(20)]
    cluster_b = [_entry(2 * i + 1, [12.0 + 0.05 * i, 0.0, 0.0]) for i in range(20)]
    trajectory = cluster_a + cluster_b

    result = check_trajectory_plausibility(trajectory)

    assert result.severe is True
    assert result.path_to_extent_ratio > 15.0


def test_legitimate_back_and_forth_within_small_room_not_severe():
    # Real repeated coverage: walk back and forth across a 3m room 4 times.
    positions = []
    for lap in range(4):
        forward = np.linspace(0.0, 3.0, 10)
        backward = np.linspace(3.0, 0.0, 10)
        positions.extend(forward if lap % 2 == 0 else backward)
    trajectory = [_entry(i, [p, 0.0, 0.0]) for i, p in enumerate(positions)]

    result = check_trajectory_plausibility(trajectory)

    assert result.severe is False
    assert result.path_to_extent_ratio < 15.0


def test_high_ratio_zigzag_without_clean_segments_is_severe():
    # Deterministic zig-zag: bounces between x=0 and x=1 many times (small
    # 1m extent) but racks up 40m of cumulative path length -- never cleanly
    # separates into two spatially-disconnected segments, so this is purely
    # a path-length/extent case, not something the segment filter would catch.
    positions = []
    for i in range(40):
        positions.append(1.0 if i % 2 else 0.0)
    trajectory = [_entry(i, [p, 0.0, 0.0]) for i, p in enumerate(positions)]

    result = check_trajectory_plausibility(trajectory)

    assert result.severe is True
    assert result.path_to_extent_ratio > 15.0


def test_fewer_than_two_positions_not_severe():
    result = check_trajectory_plausibility([_entry(0, [0, 0, 0])])
    assert result.severe is False
    assert result.n_positions == 1


def test_max_step_speed_computed_when_timestamps_available():
    trajectory = [_entry(0, [0, 0, 0]), _entry(1, [10.0, 0, 0])]
    ts = {0: 0, 1: int(1e9)}  # 1 second apart, 10m -> 10 m/s

    result = check_trajectory_plausibility(trajectory, ts)

    assert result.max_step_speed_mps == pytest.approx(10.0, abs=0.01)
