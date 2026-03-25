"""Test pose convention handling."""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_quaternion_rotation_roundtrip():
    from colmap_rgbd_gt.utils.transforms import (
        rotation_matrix_to_quaternion,
        quaternion_to_rotation_matrix,
    )

    R_original = np.array([
        [0.9999, 0.01, 0.0],
        [-0.01, 0.9999, 0.01],
        [0.0, -0.01, 0.9999]
    ])

    q = rotation_matrix_to_quaternion(R_original)
    R_recovered = quaternion_to_rotation_matrix(q)

    np.testing.assert_array_almost_equal(R_original, R_recovered, decimal=4)


def test_colmap_pose_to_c2w():
    from colmap_rgbd_gt.utils.transforms import colmap_pose_to_c2w, get_camera_center

    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    tvec = np.array([1.0, 0.0, 0.0])

    R_c2w, t_c2w = colmap_pose_to_c2w(qvec, tvec)

    center = get_camera_center(qvec, tvec)
    np.testing.assert_array_almost_equal(center, t_c2w)


def test_colmap_pose_to_c2w_rotation():
    from colmap_rgbd_gt.utils.transforms import colmap_pose_to_c2w, quaternion_to_rotation_matrix
    from scipy.spatial.transform import Rotation

    R_w2c = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    qvec = np.roll(Rotation.from_matrix(R_w2c).as_quat(), 1)

    t_w2c = np.array([1.0, 2.0, 3.0])

    R_c2w, t_c2w = colmap_pose_to_c2w(qvec, t_w2c)

    expected_R_c2w = R_w2c.T
    expected_t_c2w = -R_w2c.T @ t_w2c

    np.testing.assert_array_almost_equal(R_c2w, expected_R_c2w, decimal=4)
    np.testing.assert_array_almost_equal(t_c2w, expected_t_c2w, decimal=4)


def test_transform_composition():
    from colmap_rgbd_gt.utils.transforms import Transform

    t1 = Transform(
        rotation=np.eye(3),
        translation=np.array([1.0, 0.0, 0.0])
    )

    t2 = Transform(
        rotation=np.eye(3),
        translation=np.array([0.0, 1.0, 0.0])
    )

    t_combined = t1 @ t2

    np.testing.assert_array_almost_equal(
        t_combined.translation,
        np.array([1.0, 1.0, 0.0])
    )
