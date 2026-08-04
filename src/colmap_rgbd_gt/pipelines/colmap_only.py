"""COLMAP reconstruction pipeline."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.colmap.runner import COLMAPRunner
from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory
from colmap_rgbd_gt.export.tum import export_trajectory_tum
from colmap_rgbd_gt.dataset.schema import Workspace
from colmap_rgbd_gt.dataset.manifest import Manifest
from colmap_rgbd_gt.ingest.camera_info import resolve_camera_info
from colmap_rgbd_gt.rectify.undistort import rectify_workspace_images
from colmap_rgbd_gt.utils.camera import CameraIntrinsics

logger = get_logger(__name__)

# Marker file scale_estimation.py checks for to know a workspace's COLMAP
# features live in an undistorted (PINHOLE) frame rather than the raw
# OPENCV-distorted one, so it can zero out distortion consistently -- see
# colmap_pipeline()'s Caspar branch below.
RECTIFIED_MARKER_NAME = "rectified.json"


def _resolve_camera_params(
    workspace: Path, colmap_config: dict[str, Any]
) -> tuple[str | None, str | None, CameraIntrinsics | None]:
    """Real per-recording intrinsics (bag camera_info, else a hardcoded
    Femto Bolt/Mega fallback) -> (--ImageReader.camera_params value,
    camera_model, intrinsics). Returns (None, None, None) if no trustworthy
    source exists, in which case COLMAP falls back to self-calibrating as
    before."""
    manifest_path = Workspace(workspace).layout.manifest
    manifest = Manifest.load(manifest_path) if manifest_path.exists() else None
    bag_camera_info = manifest.camera_info if manifest else None

    fallback_profile = colmap_config.get("camera_fallback_profile")
    info, source = resolve_camera_info(bag_camera_info, fallback_profile)
    if info is None:
        return None, None, None

    intrinsics = CameraIntrinsics(
        fx=info["K"][0], fy=info["K"][4], cx=info["K"][2], cy=info["K"][5],
        width=info["width"], height=info["height"],
        distortion_model=info.get("distortion_model", "plumb_bob"),
        distortion_coeffs=info.get("D", []),
    )
    camera_model = colmap_config.get("camera_model") or intrinsics.get_colmap_model()
    logger.info(f"Camera intrinsics source: {source} (model={camera_model})")
    return intrinsics.to_colmap_str(camera_model), camera_model, intrinsics


def colmap_pipeline(workspace: Path, config: dict[str, Any]) -> bool:
    workspace = Path(workspace)

    ws = Workspace(workspace)
    if not ws.validate():
        logger.error(f"Invalid workspace: {workspace}")
        return False

    colmap_config = dict(config.get("colmap", {}))
    camera_params, resolved_model, intrinsics = _resolve_camera_params(workspace, colmap_config)
    if camera_params:
        colmap_config.setdefault("camera_params", camera_params)
        colmap_config.setdefault("camera_model", resolved_model)

    image_dir_name = "rgb"
    rectified_marker = ws.layout.colmap / RECTIFIED_MARKER_NAME
    # ADDED 2026-08-04: Caspar (COLMAP's native GPU BA solver) only
    # supports PINHOLE/SIMPLE_RADIAL camera models, not this pipeline's
    # OPENCV (real RGBD lens tangential distortion). Rather than
    # approximating the camera model away (tested and rejected -- see
    # docs/cluster-dispatch.md's Caspar section, SIMPLE_RADIAL reproduced
    # a real scale-regime-split regression), undistort every RGB frame to
    # a mathematically exact PINHOLE image first, so no accuracy is lost.
    if colmap_config.get("ba_backend") == "caspar" and colmap_config.get("use_gpu") and intrinsics:
        rectify_workspace_images(ws, intrinsics)
        image_dir_name = "rgb_rectified"
        pinhole_intrinsics = replace(intrinsics, distortion_coeffs=[])
        colmap_config["camera_model"] = "PINHOLE"
        colmap_config["camera_params"] = pinhole_intrinsics.to_colmap_str("PINHOLE")
        rectified_marker.parent.mkdir(parents=True, exist_ok=True)
        rectified_marker.write_text(json.dumps({"rectified": True}))
    elif rectified_marker.exists():
        # Stale marker from a prior Caspar run against this same workspace
        # -- clear it so a later CPU/OPENCV re-run isn't misread as
        # rectified by scale_estimation.py.
        rectified_marker.unlink()

    runner = COLMAPRunner(
        workspace, colmap_config.get("colmap_path", "colmap"), image_dir_name=image_dir_name
    )

    logger.info("Running COLMAP reconstruction...")
    success = runner.run_full_pipeline(colmap_config)

    if not success:
        logger.error("COLMAP reconstruction failed")
        return False

    logger.info("Extracting poses from COLMAP output...")
    trajectory = extract_trajectory(workspace)

    if not trajectory:
        logger.error("No poses extracted from COLMAP")
        return False

    unscaled_path = ws.layout.outputs / "trajectory_colmap_unscaled.txt"
    export_trajectory_tum(trajectory, unscaled_path)

    logger.info(f"COLMAP pipeline complete: {len(trajectory)} poses")
    return True
