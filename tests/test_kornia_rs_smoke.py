"""Smoke test for kornia_rs's Python bundle-adjustment binding directly.

Not testing our wrapper -- this catches upstream API breakage (signature
changes, return-type changes) in isolation, since kornia_rs's ba/ba_schur
modules are young and have no dedicated Python-level upstream tests.
"""

import numpy as np
import pytest

kornia_rs = pytest.importorskip("kornia_rs")
from kornia_rs import k3d  # noqa: E402


def test_bundle_adjust_schur_with_depth_converges():
    rotations = np.tile(np.eye(3), (2, 1, 1))
    translations = np.array([[0.0, 0.0, 0.0], [0.15, 0.02, -0.01]])
    points_true = np.array([[0.0, 0.0, 2.0], [0.1, 0.1, 2.5], [-0.1, 0.05, 3.0]])
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])

    observations = []
    obs_depths = []
    obs_sigmas = []
    for pose_idx in range(2):
        R = rotations[pose_idx]
        t = translations[pose_idx]
        for point_idx, X in enumerate(points_true):
            Xc = R @ X + t
            u = Xc[0] * K[0, 0] / Xc[2] + K[0, 2]
            v = Xc[1] * K[1, 1] / Xc[2] + K[1, 2]
            observations.append([pose_idx, point_idx, u, v])
            obs_depths.append(Xc[2])
            obs_sigmas.append(0.01)

    observations = np.array(observations, dtype=np.float64)
    obs_depths = np.array(obs_depths, dtype=np.float32)
    obs_sigmas = np.array(obs_sigmas, dtype=np.float32)

    # Perturb the initial points away from ground truth.
    points_init = points_true + np.array([0.05, -0.05, 0.1])

    R_opt, t_opt, X_opt, iterations, converged = k3d.bundle_adjust(
        rotations=rotations,
        translations=translations,
        points=points_init,
        observations=observations,
        k=K,
        fixed_pose_indices=[0],
        fix_all_points=False,
        solver="schur",
        obs_depths=obs_depths,
        obs_sigmas=obs_sigmas,
        max_iterations=30,
    )

    np.testing.assert_allclose(X_opt, points_true, atol=1e-2)
