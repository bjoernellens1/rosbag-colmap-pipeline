"""Depth-aware bundle adjustment via kornia_rs.k3d.bundle_adjust.

This is a thin, defensive wrapper. kornia_rs's `ba`/`ba_schur` Rust modules
are young (first added ~2 months before this was written) and have no
dedicated Python-level upstream tests, so this wrapper validates array
shapes/dtypes itself and fails with clear messages rather than trusting the
Rust extension to raise something useful.

Two upstream constraints shape this module's design:

- Depth residuals (`obs_depths`/`obs_sigmas`) and pose-position priors are
  only honored by `solver="schur"` -- the plain `"lm"` path silently
  ignores them. Since this module's entire purpose is depth-aware BA,
  `solver="schur"` is always forced; `_select_solver` exists to fail loudly
  if that invariant is ever violated by a future change.
- The Schur solver does not currently support robust loss kernels (an
  upstream TODO: "Currently supports: identity loss only for schur").
  Because depth requires schur, depth-aware BA cannot combine with
  Huber/Cauchy robustness today -- so this module deliberately does NOT
  expose a `robust` config knob that would silently do nothing. Outlier
  rejection instead happens via the depth-tolerance pre-filtering already
  applied in `build_ba_observations` (rows with inconsistent depth get
  `obs_depth <= 0`, i.e. no depth residual, rather than a down-weighted one).
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from colmap_rgbd_gt.scaling.correspondences import (
    build_pose_and_point_arrays,
    build_ba_observations,
)
from colmap_rgbd_gt.colmap.pose_extract import colmap_pose_to_c2w
from colmap_rgbd_gt.utils.camera import CameraIntrinsics
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)

_MAX_RECOMMENDED_POSES = 300


@dataclass
class DepthBAConfig:
    depth_tolerance: float = 0.1
    obs_sigma_base: float = 0.01
    obs_sigma_quadratic: float = 0.0
    max_iterations: int = 50
    fixed_pose_indices: list[int] | None = None  # default [0] applied at call time
    stage: str = "joint"  # "pose_only" | "joint"


@dataclass
class DepthBAResult:
    rotations: np.ndarray       # (P, 3, 3) float64, world->camera
    translations: np.ndarray    # (P, 3) float64, world->camera
    points: np.ndarray          # (N, 3) float64, world frame
    image_names: list[str]      # pose_idx -> name
    point_ids: list[int]        # point_idx -> original COLMAP point3d_id
    converged: bool
    iterations: int
    n_observations: int
    n_depth_observations: int

    def to_colmap_model(self, original_model: dict[str, Any]) -> dict[str, Any]:
        """Copy of `original_model` with optimized qvec/tvec/xyz written back."""
        from colmap_rgbd_gt.utils.transforms import rotation_matrix_to_quaternion
        from colmap_rgbd_gt.colmap.reconstruction import get_image_id_by_name

        model = {
            "cameras": dict(original_model["cameras"]),
            "images": {k: dict(v) for k, v in original_model["images"].items()},
            "points3d": {k: dict(v) for k, v in original_model["points3d"].items()},
        }

        for pose_idx, name in enumerate(self.image_names):
            image_id = get_image_id_by_name(model, name)
            if image_id is None:
                continue
            R = self.rotations[pose_idx]
            t = self.translations[pose_idx]
            q_xyzw = rotation_matrix_to_quaternion(R)
            qvec_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            model["images"][image_id]["qvec"] = qvec_wxyz
            model["images"][image_id]["tvec"] = t.tolist()

        for point_idx, point_id in enumerate(self.point_ids):
            if point_id in model["points3d"]:
                model["points3d"][point_id]["xyz"] = self.points[point_idx].tolist()

        return model

    def to_tum_trajectory(self) -> list[dict[str, Any]]:
        """Optimized w2c poses converted to the trajectory dict format
        expected by `export.tum.export_trajectory_tum` (c2w, with frame_id
        parsed from the image name, matching `pose_extract.extract_trajectory`)."""
        trajectory = []
        for pose_idx, name in enumerate(self.image_names):
            from colmap_rgbd_gt.utils.transforms import rotation_matrix_to_quaternion

            R = self.rotations[pose_idx]
            t = self.translations[pose_idx]
            q_xyzw = rotation_matrix_to_quaternion(R)
            qvec_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

            R_c2w, t_c2w = colmap_pose_to_c2w(qvec_wxyz, t)
            frame_id = int(name.split(".")[0])
            trajectory.append({
                "frame_id": frame_id,
                "image_name": name,
                "R": R_c2w,
                "t": t_c2w,
                "qvec_w2c": qvec_wxyz,
                "tvec_w2c": t,
            })
        return trajectory


def _select_solver(has_depth: bool, requested_solver: str | None) -> str:
    if has_depth:
        if requested_solver not in (None, "schur"):
            raise ValueError(
                "obs_depths are supplied, but obs_depths/pose position priors "
                "are only honored by solver='schur' (kornia_rs silently "
                f"ignores them under solver={requested_solver!r}). Refusing "
                "to run depth-aware BA with a non-schur solver."
            )
        return "schur"
    return requested_solver or "lm"


def _validate_arrays(
    rotations: np.ndarray,
    translations: np.ndarray,
    points: np.ndarray,
    observations: np.ndarray,
    k: np.ndarray,
    obs_depths: np.ndarray,
    obs_sigmas: np.ndarray,
) -> None:
    p = rotations.shape[0] if rotations.ndim == 3 else None
    n = points.shape[0] if points.ndim == 2 else None
    m = observations.shape[0] if observations.ndim == 2 else None

    checks = [
        ("rotations", rotations, (p, 3, 3)),
        ("translations", translations, (p, 3)),
        ("points", points, (n, 3)),
        ("observations", observations, (m, 4)),
        ("k", k, (3, 3)),
        ("obs_depths", obs_depths, (m,)),
        ("obs_sigmas", obs_sigmas, (m,)),
    ]
    for name, arr, expected_shape in checks:
        if arr.shape != expected_shape:
            raise ValueError(
                f"depth_ba: array '{name}' has shape {arr.shape}, "
                f"expected {expected_shape}"
            )

    if m and m > 0:
        pose_idx = observations[:, 0]
        point_idx = observations[:, 1]
        if pose_idx.min() < 0 or pose_idx.max() >= p:
            raise ValueError(
                f"depth_ba: observations pose_idx out of range [0, {p}): "
                f"[{pose_idx.min()}, {pose_idx.max()}]"
            )
        if point_idx.min() < 0 or point_idx.max() >= n:
            raise ValueError(
                f"depth_ba: observations point_idx out of range [0, {n}): "
                f"[{point_idx.min()}, {point_idx.max()}]"
            )


def _run_bundle_adjust(
    rotations: np.ndarray,
    translations: np.ndarray,
    points: np.ndarray,
    observations: np.ndarray,
    k: np.ndarray,
    obs_depths: np.ndarray,
    obs_sigmas: np.ndarray,
    fixed_pose_indices: list[int],
    fix_all_points: bool,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    try:
        import kornia_rs.k3d as k3d
    except ImportError as e:
        raise RuntimeError(
            "depth-aware bundle adjustment requires the optional 'depth-ba' "
            "extra: pip install colmap-rgbd-gt[depth-ba]"
        ) from e

    rotations = rotations.astype(np.float64, copy=False)
    translations = translations.astype(np.float64, copy=False)
    points = points.astype(np.float64, copy=False)
    observations = observations.astype(np.float64, copy=False)
    k = k.astype(np.float64, copy=False)
    obs_depths = obs_depths.astype(np.float32, copy=False)
    obs_sigmas = obs_sigmas.astype(np.float32, copy=False)

    _validate_arrays(rotations, translations, points, observations, k, obs_depths, obs_sigmas)

    has_depth = bool(np.any(obs_depths > 0))
    solver = _select_solver(has_depth, "schur" if has_depth else "lm")

    R_opt, t_opt, X_opt, iterations, converged = k3d.bundle_adjust(
        rotations=rotations,
        translations=translations,
        points=points,
        observations=observations,
        k=k,
        fixed_pose_indices=fixed_pose_indices,
        fix_all_points=fix_all_points,
        max_iterations=max_iterations,
        solver=solver,
        obs_depths=obs_depths,
        obs_sigmas=obs_sigmas,
    )
    return R_opt, t_opt, X_opt, iterations, converged


def run_depth_bundle_adjustment(
    model: dict[str, Any],
    image_names: list[str],
    depth_loader: Callable[[str], np.ndarray | None],
    intrinsics: CameraIntrinsics,
    config: DepthBAConfig | None = None,
) -> DepthBAResult:
    """Run staged depth-aware bundle adjustment.

    Stage B (pose-only, `fix_all_points=True`) then Stage C (joint,
    `fix_all_points=False`, seeded from Stage B's output) if
    `config.stage == "joint"`; only Stage B if `config.stage == "pose_only"`.

    Scale initialization (the review's "Stage A") is NOT performed here --
    callers are expected to have already run `estimate_global_scale` +
    `scale_trajectory` (Part A) and passed in a `model` whose poses are
    already roughly metric, since kornia_rs's solver is local and a badly
    scaled initialization risks a bad local minimum.
    """
    config = config or DepthBAConfig()

    if len(image_names) > _MAX_RECOMMENDED_POSES:
        logger.warning(
            f"depth-ba: {len(image_names)} poses exceeds the ~{_MAX_RECOMMENDED_POSES} "
            "pose regime kornia_rs's dense Schur solver is validated for; "
            "this stage may become slow. Consider reducing the keyframe count."
        )

    rotations, translations, points, point_id_to_idx = build_pose_and_point_arrays(
        model, image_names
    )
    point_ids = sorted(point_id_to_idx, key=lambda pid: point_id_to_idx[pid])

    obs = build_ba_observations(
        model,
        image_names,
        points,
        point_id_to_idx,
        depth_loader,
        intrinsics,
        depth_tolerance=config.depth_tolerance,
        obs_sigma_base=config.obs_sigma_base,
        obs_sigma_quadratic=config.obs_sigma_quadratic,
    )

    fixed_pose_indices = config.fixed_pose_indices if config.fixed_pose_indices is not None else [0]
    k = intrinsics.K

    logger.info(
        f"depth-ba: {len(image_names)} poses, {len(points)} points, "
        f"{len(obs.observations)} observations ({obs.n_depth_observations} with depth)"
    )

    # Stage B: pose-only.
    R_opt, t_opt, X_opt, iterations, converged = _run_bundle_adjust(
        rotations, translations, points, obs.observations, k,
        obs.obs_depths, obs.obs_sigmas,
        fixed_pose_indices=fixed_pose_indices,
        fix_all_points=True,
        max_iterations=config.max_iterations,
    )

    if config.stage == "joint":
        # Stage C: joint, seeded from Stage B's output.
        R_opt, t_opt, X_opt, iterations, converged = _run_bundle_adjust(
            R_opt, t_opt, X_opt, obs.observations, k,
            obs.obs_depths, obs.obs_sigmas,
            fixed_pose_indices=fixed_pose_indices,
            fix_all_points=False,
            max_iterations=config.max_iterations,
        )

    return DepthBAResult(
        rotations=R_opt,
        translations=t_opt,
        points=X_opt,
        image_names=list(image_names),
        point_ids=point_ids,
        converged=bool(converged),
        iterations=int(iterations),
        n_observations=len(obs.observations),
        n_depth_observations=obs.n_depth_observations,
    )
