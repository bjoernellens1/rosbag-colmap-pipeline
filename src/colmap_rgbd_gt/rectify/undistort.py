"""Point-level lens undistortion.

`kornia_rs.k3d.bundle_adjust` only implements a pinhole projection model
(fx, fy, cx, cy) and ignores distortion entirely -- pixel observations must
already be undistorted before being passed in. This repo's COLMAP
reconstruction defaults to the distorted `OPENCV` camera model, so
individual (u, v) feature observations need to be undistorted before they
are handed to the depth-aware bundle adjuster.

This module undistorts *points*, not images: `cv2.undistortPoints(uv, K, D,
P=K)` maps a distorted pixel coordinate to where an idealized,
distortion-free pinhole camera *using the same K* would have observed it.
Reusing the same K for both the distorted COLMAP model and kornia_rs's
pinhole model means no image-level rectification or COLMAP re-run is
needed -- only the (u, v) values passed into the depth-BA observation
arrays are affected.
"""

from pathlib import Path

import numpy as np
import cv2

from colmap_rgbd_gt.dataset.schema import Workspace
from colmap_rgbd_gt.utils.camera import CameraIntrinsics
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def get_pinhole_k(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Returns the (3, 3) camera matrix to use as kornia_rs's pinhole `k`.

    This is the SAME K used for the (distorted) COLMAP reconstruction --
    `undistort_points` re-projects distortion-free coordinates back into
    this K's pixel frame, so both the distorted COLMAP model and the
    undistorted kornia_rs observations share one consistent K.
    """
    return intrinsics.K


def undistort_points(uv: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Remove lens distortion from pixel coordinates, keeping them in K's frame.

    Args:
        uv: (N, 2) distorted pixel coordinates.
        intrinsics: camera intrinsics, including K and distortion_coeffs.

    Returns:
        (N, 2) undistorted pixel coordinates, in the same K pixel frame.
        No-op passthrough (returns `uv` unchanged) if the camera has no
        distortion coefficients (already PINHOLE).
    """
    uv = np.asarray(uv, dtype=np.float64)
    if uv.size == 0:
        return uv.reshape(-1, 2)

    coeffs = intrinsics.distortion_coeffs
    if not coeffs or all(d == 0 for d in coeffs):
        return uv

    K = intrinsics.K
    D = np.array(coeffs, dtype=np.float64).reshape(-1, 1)

    pts = uv.reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, K, D, P=K)
    return undistorted.reshape(-1, 2)


def undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Remove lens distortion from an image, keeping the same K and size.

    Same `P=K`/`newCameraMatrix=K` convention as `undistort_points` above,
    so image pixels and any previously-undistorted point observations stay
    in one consistent K frame. No-op passthrough if the camera has no
    distortion coefficients.

    Uses `initUndistortRectifyMap` + `remap` directly (rather than the
    equivalent, simpler `cv2.undistort(image, K, D, newCameraMatrix=K)`)
    so interpolation/border handling can be controlled explicitly:
    `cv2.undistort`'s defaults are bilinear interpolation and
    `BORDER_CONSTANT` (zero-fill) -- the latter paints a hard black edge
    wherever the undistortion map samples outside the original frame,
    which SIFT can pick up as spurious high-contrast features right at
    the image border. `BORDER_REPLICATE` avoids inventing that edge, and
    `INTER_LANCZOS4` reduces resampling blur relative to bilinear, both
    aimed at keeping COLMAP's extracted features closer to what it would
    have found on the original (undistorted-content, still-distorted-
    geometry) image.
    """
    coeffs = intrinsics.distortion_coeffs
    if not coeffs or all(d == 0 for d in coeffs):
        return image

    K = intrinsics.K
    D = np.array(coeffs, dtype=np.float64).reshape(-1, 1)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (image.shape[1], image.shape[0]), cv2.CV_32FC1
    )
    return cv2.remap(
        image, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )


def rectify_workspace_images(workspace: Workspace, intrinsics: CameraIntrinsics) -> Path:
    """Undistort every frame in `rgb/` into `rgb_rectified/`, same filenames.

    Used to feed COLMAP's Caspar GPU BA backend (PINHOLE/SIMPLE_RADIAL only,
    see colmap/runner.py's bundle_adjuster()) a mathematically exact PINHOLE
    input instead of approximating the OPENCV distortion model away.
    """
    layout = workspace.layout
    layout.rgb_rectified.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(layout.rgb.iterdir())
    logger.info(f"Rectifying {len(frame_paths)} frames to {layout.rgb_rectified}...")
    for frame_path in frame_paths:
        if not frame_path.is_file():
            continue
        image = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning(f"rectify_workspace_images: could not read {frame_path}, skipping")
            continue
        rectified = undistort_image(image, intrinsics)
        cv2.imwrite(str(layout.rgb_rectified / frame_path.name), rectified)

    return layout.rgb_rectified
