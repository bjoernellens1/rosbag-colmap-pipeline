"""Tests for optimization/depth_ba.py's validation and array-building logic.

Array-building/validation tests run unconditionally (pure Python/numpy);
tests that actually invoke the kornia_rs solver use importorskip.
"""

import numpy as np
import pytest

from colmap_rgbd_gt.optimization.depth_ba import (
    _select_solver,
    _validate_arrays,
)


def test_select_solver_forces_schur_when_depth_present():
    assert _select_solver(has_depth=True, requested_solver=None) == "schur"
    assert _select_solver(has_depth=True, requested_solver="schur") == "schur"


def test_select_solver_rejects_lm_with_depth():
    with pytest.raises(ValueError, match="schur"):
        _select_solver(has_depth=True, requested_solver="lm")


def test_select_solver_defaults_lm_without_depth():
    assert _select_solver(has_depth=False, requested_solver=None) == "lm"


def test_validate_arrays_accepts_consistent_shapes():
    rotations = np.tile(np.eye(3), (2, 1, 1))
    translations = np.zeros((2, 3))
    points = np.zeros((5, 3))
    observations = np.array([[0, 0, 1.0, 1.0], [1, 4, 2.0, 2.0]])
    k = np.eye(3)
    obs_depths = np.array([1.0, 2.0], dtype=np.float32)
    obs_sigmas = np.array([0.01, 0.01], dtype=np.float32)

    _validate_arrays(rotations, translations, points, observations, k, obs_depths, obs_sigmas)


def test_validate_arrays_rejects_bad_shape():
    rotations = np.tile(np.eye(3), (2, 1, 1))
    translations = np.zeros((3, 3))  # wrong: should be (2,3)
    points = np.zeros((5, 3))
    observations = np.zeros((0, 4))
    k = np.eye(3)
    obs_depths = np.zeros((0,), dtype=np.float32)
    obs_sigmas = np.zeros((0,), dtype=np.float32)

    with pytest.raises(ValueError, match="translations"):
        _validate_arrays(rotations, translations, points, observations, k, obs_depths, obs_sigmas)


def test_validate_arrays_rejects_out_of_range_pose_idx():
    rotations = np.tile(np.eye(3), (2, 1, 1))
    translations = np.zeros((2, 3))
    points = np.zeros((5, 3))
    observations = np.array([[5, 0, 1.0, 1.0]])  # pose_idx=5 out of range
    k = np.eye(3)
    obs_depths = np.array([1.0], dtype=np.float32)
    obs_sigmas = np.array([0.01], dtype=np.float32)

    with pytest.raises(ValueError, match="pose_idx"):
        _validate_arrays(rotations, translations, points, observations, k, obs_depths, obs_sigmas)


def test_validate_arrays_rejects_out_of_range_point_idx():
    rotations = np.tile(np.eye(3), (2, 1, 1))
    translations = np.zeros((2, 3))
    points = np.zeros((5, 3))
    observations = np.array([[0, 99, 1.0, 1.0]])  # point_idx=99 out of range
    k = np.eye(3)
    obs_depths = np.array([1.0], dtype=np.float32)
    obs_sigmas = np.array([0.01], dtype=np.float32)

    with pytest.raises(ValueError, match="point_idx"):
        _validate_arrays(rotations, translations, points, observations, k, obs_depths, obs_sigmas)
