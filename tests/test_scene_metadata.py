"""Tests for export/scene_metadata.py -- QC metrics computed from a hand-
constructed synthetic trajectory with values verified by hand, not just
eyeballed against a real scene's output."""

import csv
import numpy as np
import pytest

from colmap_rgbd_gt.export.scene_metadata import compute_scene_metadata
from colmap_rgbd_gt.utils.transforms import rotation_angle_deg


def _rot_z(deg):
    a = np.radians(deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a), np.cos(a), 0.0],
        [0.0, 0.0, 1.0],
    ])


def _write_rgb_csv(path, frame_ids_and_ts_s):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "filename"])
        for fid, ts_s in frame_ids_and_ts_s:
            w.writerow([int(ts_s * 1e9), f"{fid:06d}.png"])


def test_straight_line_then_turn(tmp_path):
    """5 poses: 3 straight-line steps (1m each, no rotation), then an
    in-place 90-degree turn, then 1 more straight step -- every metric is
    hand-computable by construction:
      trajectory_length = 1+1+1+1 = 4m
      rotation: 0, 0, 90, 0 degrees per step -> total=90, avg=22.5, max=90
      extent: x in [0,2], y in [0,2], z=0 -> size (2,2,0), bbox_volume=0
      duration = 4s (1s per step) -> avg_speed = 4m/4s = 1 m/s exactly
      every step is 1m in 1s -> max_speed=1, std_speed=0
    """
    R_id = np.eye(3)
    R_turned = _rot_z(90)
    trajectory = [
        {"frame_id": 0, "R": R_id, "t": np.array([0.0, 0.0, 0.0])},
        {"frame_id": 1, "R": R_id, "t": np.array([1.0, 0.0, 0.0])},
        {"frame_id": 2, "R": R_id, "t": np.array([2.0, 0.0, 0.0])},
        {"frame_id": 3, "R": R_turned, "t": np.array([2.0, 1.0, 0.0])},
        {"frame_id": 4, "R": R_turned, "t": np.array([2.0, 2.0, 0.0])},
    ]
    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, [(i, float(i)) for i in range(5)])

    m = compute_scene_metadata(trajectory, rgb_csv, frame_count=5, registered_count=5)

    assert m["n_poses"] == 5
    assert m["registered_frames"] == 5
    assert m["total_frames"] == 5
    assert m["registration_ratio"] == pytest.approx(1.0)

    assert m["trajectory_length_m"] == pytest.approx(4.0)
    assert m["duration_s"] == pytest.approx(4.0)

    assert m["speed"]["avg_mps"] == pytest.approx(1.0)
    assert m["speed"]["max_mps"] == pytest.approx(1.0)
    assert m["speed"]["std_mps"] == pytest.approx(0.0, abs=1e-9)

    assert m["rotation"]["total_deg"] == pytest.approx(90.0)
    assert m["rotation"]["avg_deg_per_frame"] == pytest.approx(90.0 / 4)
    assert m["rotation"]["max_deg_per_frame"] == pytest.approx(90.0)

    assert m["extent"]["min_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert m["extent"]["max_m"] == pytest.approx([2.0, 2.0, 0.0])
    assert m["extent"]["size_m"] == pytest.approx([2.0, 2.0, 0.0])
    assert m["extent"]["bbox_volume_m3_COARSE_PROXY"] == pytest.approx(0.0)

    # No revisit expected: total duration (4s) is below the default
    # min_time_gap_s (5s), so nothing qualifies even though positions are
    # close together.
    assert m["revisit"]["n_revisit_pairs"] == 0
    assert m["revisit"]["revisited_pose_fraction"] == pytest.approx(0.0)


def test_revisit_detection(tmp_path):
    """A trajectory that returns near its starting position after enough
    wall-clock time must be flagged as a revisit; an identical spatial
    return that happens too QUICKLY (below min_time_gap_s) must not."""
    R_id = np.eye(3)
    # Poses 0 and 3 are close in space (0.1m apart) and 10s apart in time
    # -> should count. Poses 1 and 2 are close (0.05m) but only 1s apart
    # -> should NOT count (same-neighborhood adjacent poses, not a real
    # revisit).
    trajectory = [
        {"frame_id": 0, "R": R_id, "t": np.array([0.0, 0.0, 0.0])},
        {"frame_id": 1, "R": R_id, "t": np.array([5.0, 0.0, 0.0])},
        {"frame_id": 2, "R": R_id, "t": np.array([5.05, 0.0, 0.0])},
        {"frame_id": 3, "R": R_id, "t": np.array([0.1, 0.0, 0.0])},
    ]
    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, [(0, 0.0), (1, 1.0), (2, 2.0), (3, 12.0)])

    m = compute_scene_metadata(
        trajectory, rgb_csv,
        revisit_distance_m=0.5, revisit_min_time_gap_s=5.0,
    )
    assert m["revisit"]["n_revisit_pairs"] == 1
    assert m["revisit"]["revisited_pose_fraction"] == pytest.approx(2 / 4)


def test_empty_trajectory_does_not_raise(tmp_path):
    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, [])
    m = compute_scene_metadata([], rgb_csv)
    assert m["n_poses"] == 0
    assert m["trajectory_length_m"] == 0.0
    assert m["speed"] is None
    assert m["rotation"] is None


def test_single_pose_does_not_raise(tmp_path):
    rgb_csv = tmp_path / "rgb.csv"
    _write_rgb_csv(rgb_csv, [(0, 0.0)])
    trajectory = [{"frame_id": 0, "R": np.eye(3), "t": np.array([1.0, 2.0, 3.0])}]
    m = compute_scene_metadata(trajectory, rgb_csv)
    assert m["n_poses"] == 1
    assert m["trajectory_length_m"] == 0.0
    assert m["speed"] is None
    assert m["rotation"] is None
    assert m["extent"]["size_m"] == pytest.approx([0.0, 0.0, 0.0])


def test_rotation_angle_deg_helper_known_values():
    assert rotation_angle_deg(np.eye(3), np.eye(3)) == pytest.approx(0.0)
    assert rotation_angle_deg(np.eye(3), _rot_z(90)) == pytest.approx(90.0)
    assert rotation_angle_deg(np.eye(3), _rot_z(180)) == pytest.approx(180.0)
    # Order shouldn't matter for the magnitude of a pure rotation delta.
    assert rotation_angle_deg(_rot_z(90), np.eye(3)) == pytest.approx(90.0)
