"""COLMAP reconstruction pipeline."""

from pathlib import Path
from typing import Any

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.colmap.runner import COLMAPRunner
from colmap_rgbd_gt.colmap.pose_extract import extract_trajectory
from colmap_rgbd_gt.export.tum import export_trajectory_tum
from colmap_rgbd_gt.dataset.schema import Workspace
from colmap_rgbd_gt.dataset.manifest import Manifest
from colmap_rgbd_gt.ingest.camera_info import resolve_camera_info
from colmap_rgbd_gt.utils.camera import CameraIntrinsics

logger = get_logger(__name__)


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
    camera_params, resolved_model, _intrinsics = _resolve_camera_params(workspace, colmap_config)
    if camera_params:
        colmap_config.setdefault("camera_params", camera_params)
        colmap_config.setdefault("camera_model", resolved_model)

    # `ba_backend: caspar` (see colmap/runner.py's bundle_adjuster()) needs no
    # special-casing here: docker/patches/caspar-opencv/ (applied in
    # docker/Dockerfile.cuda's COLMAP build) adds native OPENCV support to
    # Caspar's GPU BA kernels, so it takes this pipeline's normal OPENCV
    # camera model directly -- see that patch's README.md. An earlier
    # approach undistorted RGB to PINHOLE first (rectify/undistort.py's
    # rectify_workspace_images(), still available as a utility) since Caspar
    # upstream only ships PINHOLE/SIMPLE_RADIAL, but that added a real
    # accuracy risk (tested and found a residual scale-regime split on
    # harder scenes) that native OPENCV support avoids entirely.
    runner = COLMAPRunner(workspace, colmap_config.get("colmap_path", "colmap"))

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
