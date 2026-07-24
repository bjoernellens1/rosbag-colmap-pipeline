"""Depth-aware bundle adjustment pipeline (optional, requires 'depth-ba' extra)."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.dataset.schema import Workspace
from colmap_rgbd_gt.dataset.manifest import Manifest
from colmap_rgbd_gt.colmap.reconstruction import load_sparse_model
from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory, scale_trajectory
from colmap_rgbd_gt.colmap.colmap_io import write_cameras_text, write_images_text, write_points3d_text
from colmap_rgbd_gt.scaling.scale_estimation import estimate_global_scale
from colmap_rgbd_gt.export.tum import export_trajectory_tum
from colmap_rgbd_gt.export.evo import plot_trajectory
from colmap_rgbd_gt.utils.camera import CameraIntrinsics

logger = get_logger(__name__)


def _apply_scale_to_model(model: dict[str, Any], trajectory_scaled: list[dict[str, Any]]) -> dict[str, Any]:
    """Write scaled tvec back into a copy of the model's images, and scale
    points3d xyz by the same factor (uniform Sim3 scale about COLMAP's
    origin -- consistent with `scale_trajectory`'s translation-only scaling)."""
    model = {
        "cameras": dict(model["cameras"]),
        "images": {k: dict(v) for k, v in model["images"].items()},
        "points3d": {k: dict(v) for k, v in model["points3d"].items()},
    }

    from colmap_rgbd_gt.utils.transforms import rotation_matrix_to_quaternion
    from colmap_rgbd_gt.colmap.reconstruction import get_image_id_by_name

    scale = None
    for entry in trajectory_scaled:
        image_id = get_image_id_by_name(model, entry["image_name"])
        if image_id is None:
            continue
        R_c2w = entry["R"]
        t_c2w = entry["t"]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        q_xyzw = rotation_matrix_to_quaternion(R_w2c)
        model["images"][image_id]["qvec"] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        model["images"][image_id]["tvec"] = t_w2c.tolist()

    return model


def _scale_points3d(model: dict[str, Any], scale: float) -> None:
    for point in model["points3d"].values():
        point["xyz"] = [c * scale for c in point["xyz"]]


def depth_ba_pipeline(workspace: Path, config: dict[str, Any]) -> bool:
    workspace = Path(workspace)
    ws = Workspace(workspace)
    if not ws.validate():
        logger.error(f"Invalid workspace: {workspace}")
        return False

    try:
        from colmap_rgbd_gt.optimization.depth_ba import DepthBAConfig, run_depth_bundle_adjustment
    except ImportError:
        logger.error(
            "depth-ba requires the optional extra: pip install colmap-rgbd-gt[depth-ba]"
        )
        return False

    manifest = Manifest.load(ws.layout.manifest)
    intrinsics_data = manifest.camera_info
    if not intrinsics_data:
        logger.error("No camera info in manifest")
        return False

    intrinsics = CameraIntrinsics(
        fx=intrinsics_data["K"][0],
        fy=intrinsics_data["K"][4],
        cx=intrinsics_data["K"][2],
        cy=intrinsics_data["K"][5],
        width=intrinsics_data["width"],
        height=intrinsics_data["height"],
        distortion_model=intrinsics_data.get("distortion_model", "plumb_bob"),
        distortion_coeffs=intrinsics_data.get("D", []),
    )

    sparse_dir = ws.layout.colmap / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = ws.layout.colmap / "sparse"

    model = load_sparse_model(sparse_dir)

    ba_config_dict = config.get("depth_ba", {})
    scaling_config = config.get("scaling", {})

    # Stage A: initial metric scale (reuse Part A's estimator unchanged),
    # applied to a copy of the model so BA starts from a roughly-correct
    # metric initialization rather than COLMAP's arbitrary scale.
    logger.info("depth-ba Stage A: estimating initial metric scale...")
    scale_estimate = estimate_global_scale(workspace, scaling_config)
    trajectory = extract_trajectory(workspace)
    if not trajectory:
        logger.error("No trajectory found")
        return False
    scaled_trajectory = scale_trajectory(trajectory, scale_estimate.scale)
    model = _apply_scale_to_model(model, scaled_trajectory)
    _scale_points3d(model, scale_estimate.scale)

    max_frames = ba_config_dict.get("max_frames_for_ba", scaling_config.get("max_frames_for_scale", 100))
    image_names = [entry["image_name"] for entry in scaled_trajectory[:max_frames]]

    def depth_loader(image_name: str) -> np.ndarray | None:
        frame_id = int(image_name.split(".")[0])
        depth_path = ws.get_depth_path(frame_id)
        if not depth_path.exists():
            return None
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        return depth.astype(np.float64) / 1000.0

    ba_config = DepthBAConfig(
        depth_tolerance=ba_config_dict.get("depth_tolerance", 0.1),
        obs_sigma_base=ba_config_dict.get("obs_sigma_base", 0.01),
        obs_sigma_quadratic=ba_config_dict.get("obs_sigma_quadratic", 0.0),
        max_iterations=ba_config_dict.get("max_iterations", 50),
        stage="joint",
    )

    logger.info(f"depth-ba Stages B+C: running bundle adjustment on {len(image_names)} poses...")
    result = run_depth_bundle_adjustment(
        model, image_names, depth_loader, intrinsics, ba_config
    )

    refined_model = result.to_colmap_model(model)
    refined_dir = ws.layout.colmap / "sparse" / "0_refined"
    write_cameras_text(refined_dir / "cameras.txt", refined_model["cameras"])
    write_images_text(refined_dir / "images.txt", refined_model["images"])
    write_points3d_text(refined_dir / "points3D.txt", refined_model["points3d"])

    ba_trajectory = result.to_tum_trajectory()
    ba_tum_path = ws.layout.outputs / "trajectory_depth_ba_tum.txt"
    export_trajectory_tum(ba_trajectory, ba_tum_path)

    try:
        metric_tum_path = ws.layout.outputs / "trajectory_metric_tum.txt"
        plot_trajectory(
            ba_tum_path,
            ws.layout.outputs / "trajectory_depth_ba_plot.png",
            title="Depth-BA vs scale-only trajectory (top-down XY)",
            reference_tum_path=metric_tum_path if metric_tum_path.exists() else None,
        )
    except Exception as e:
        logger.warning(f"Could not plot depth-BA trajectory: {e}")

    report = {
        "converged": result.converged,
        "iterations": result.iterations,
        "n_observations": result.n_observations,
        "n_depth_observations": result.n_depth_observations,
        "n_poses": len(result.image_names),
        "n_points": len(result.point_ids),
        "initial_scale_estimate": scale_estimate.scale,
        "initial_scale_confidence": scale_estimate.confidence,
    }
    with open(ws.layout.outputs / "depth_ba_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"depth-ba pipeline complete: converged={result.converged}, "
        f"iterations={result.iterations}, n_observations={result.n_observations}"
    )
    return True
