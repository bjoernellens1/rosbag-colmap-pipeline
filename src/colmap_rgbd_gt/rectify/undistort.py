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

import numpy as np
import cv2

from colmap_rgbd_gt.utils.camera import CameraIntrinsics


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
