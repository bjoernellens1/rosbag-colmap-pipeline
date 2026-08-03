"""Camera info extraction from rosbag messages."""

from typing import Any
import numpy as np


def extract_camera_info(msg: Any) -> dict:
    K = list(msg.k) if hasattr(msg, "k") else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    D = list(msg.d) if hasattr(msg, "d") else []
    R = list(msg.r) if hasattr(msg, "r") else [1, 0, 0, 0, 1, 0, 0, 0, 1]
    P = list(msg.p) if hasattr(msg, "p") else [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

    return {
        "width": msg.width,
        "height": msg.height,
        "K": K,
        "D": D,
        "R": R,
        "P": P,
        "distortion_model": getattr(msg, "distortion_model", "plumb_bob"),
        "binning_x": getattr(msg, "binning_x", 1),
        "binning_y": getattr(msg, "binning_y", 1),
    }


def validate_camera_info(info: dict) -> list[str]:
    errors = []

    if info["width"] <= 0:
        errors.append(f"Invalid width: {info['width']}")
    if info["height"] <= 0:
        errors.append(f"Invalid height: {info['height']}")

    K = info["K"]
    if len(K) != 9:
        errors.append(f"K matrix must have 9 elements, got {len(K)}")
    else:
        if K[0] <= 0 or K[4] <= 0:
            errors.append(f"Invalid focal length in K: fx={K[0]}, fy={K[4]}")
        else:
            # FIXED 2026-08-03: fx=1.0 (or any other tiny positive value)
            # passed the old `<= 0` check but is obviously not a real
            # calibration for any real sensor -- found on hallway's bag:
            # camera_info topic published an identity K ([1,0,0,0,1,0,0,0,1],
            # a driver stub before real calibration loaded), which this
            # check let through as "valid", silently breaking every
            # downstream consumer's reprojection math (scale estimation
            # got exactly 0 correspondences: "Coarse pre-scale estimate...
            # 1.0000" / "Insufficient correspondences: 0" -- the telltale
            # sign of fx=fy=1 poisoning every projected-point computation).
            # A real lens's focal length in pixels is always a sizeable
            # fraction of the image dimension it's paired with (rarely
            # below ~10% for anything from a fisheye to a narrow FoV lens);
            # anything smaller is not a plausible calibration.
            width = info.get("width", 0)
            height = info.get("height", 0)
            min_plausible_fx = 0.1 * width if width > 0 else 1.0
            min_plausible_fy = 0.1 * height if height > 0 else 1.0
            if K[0] < min_plausible_fx or K[4] < min_plausible_fy:
                errors.append(
                    f"Implausible focal length in K: fx={K[0]}, fy={K[4]} "
                    f"for a {width}x{height} image (expected fx >= {min_plausible_fx:.1f}, "
                    f"fy >= {min_plausible_fy:.1f}) -- likely an unpopulated/stub camera_info"
                )
        if K[1] != 0 or K[3] != 0:
            errors.append("K matrix should have zero skew")

    return errors


def get_intrinsics_matrix(info: dict) -> np.ndarray:
    K = info["K"]
    return np.array(K).reshape(3, 3)


def get_distortion_coeffs(info: dict) -> np.ndarray:
    D = info["D"]
    return np.array(D) if D else np.zeros(5)


# Fallback factory-calibration profiles for cameras used across this
# ecosystem's real scenes, for when a bag's own camera_info
# topic is missing or fails `validate_camera_info`. These are NOT invented:
# both come from splatograph's own ORB-SLAM3 frontend configs
# (splatograph-orbslam3-src/config/OrbbecFemto{Bolt,Mega}_RGBD.yaml), which
# already carry real per-unit RGB intrinsics for this project's cameras.
# camera_info from the bag's own topic is always preferred when present and
# valid -- see `resolve_camera_info` -- since it reflects the actual unit
# and settings used for that specific recording; these are a documented
# last-resort fallback only.
KNOWN_CAMERA_PROFILES: dict[str, dict] = {
    "orbbec_femto_bolt": {
        "width": 1280,
        "height": 720,
        "K": [749.820374, 0.0, 640.758850, 0.0, 749.719055, 364.850922, 0.0, 0.0, 1.0],
        "D": [0.07101957, -0.09848151, 0.00043707, 0.00012857, 0.03971413, 0.0, 0.0, 0.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [749.820374, 0.0, 640.758850, 0.0, 0.0, 749.719055, 364.850922, 0.0, 0.0, 0.0, 1.0, 0.0],
        "distortion_model": "plumb_bob",
        "binning_x": 0,
        "binning_y": 0,
    },
    # Same physical RGB sensor/lens as the Bolt in this project's own
    # ORB-SLAM3 configs (OrbbecFemtoMega_RGBD.yaml carries identical
    # values to OrbbecFemtoBolt_RGBD.yaml) -- kept as a separate named
    # profile since Bolt/Mega calibration could diverge for other units.
    "orbbec_femto_mega": {
        "width": 1280,
        "height": 720,
        "K": [749.820374, 0.0, 640.758850, 0.0, 749.719055, 364.850922, 0.0, 0.0, 1.0],
        "D": [0.07101957, -0.09848151, 0.00043707, 0.00012857, 0.03971413, 0.0, 0.0, 0.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [749.820374, 0.0, 640.758850, 0.0, 0.0, 749.719055, 364.850922, 0.0, 0.0, 0.0, 1.0, 0.0],
        "distortion_model": "plumb_bob",
        "binning_x": 0,
        "binning_y": 0,
    },
    # RealSense D435i, 1280x720 RGB -- values from this project's own
    # ros2-jazzy-realsense-fedora repo
    # (config/pyslam_realsense_d435i.yaml), not invented. Added 2026-08-03
    # for hallway's bag, whose own camera_info topic published an
    # unpopulated identity K (see validate_camera_info's implausible-
    # focal-length check).
    "realsense_d435i": {
        "width": 1280,
        "height": 720,
        "K": [907.2616, 0.0, 651.042, 0.0, 906.7314, 370.796, 0.0, 0.0, 1.0],
        "D": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [907.2616, 0.0, 651.042, 0.0, 0.0, 906.7314, 370.796, 0.0, 0.0, 0.0, 1.0, 0.0],
        "distortion_model": "plumb_bob",
        "binning_x": 0,
        "binning_y": 0,
    },
}


def resolve_camera_info(
    camera_info: dict | None,
    fallback_profile: str | None = None,
) -> tuple[dict | None, str]:
    """Pick the camera_info to actually trust: the bag's own camera_info
    topic when present and valid, else a named hardcoded fallback profile.

    Returns (info_or_None, source) where source is one of
    "bag_camera_info", "fallback:<profile>", or "none" (no valid source at
    all -- caller should let COLMAP self-calibrate as before).
    """
    if camera_info:
        errors = validate_camera_info(camera_info)
        if not errors:
            return camera_info, "bag_camera_info"
    if fallback_profile:
        profile = KNOWN_CAMERA_PROFILES.get(fallback_profile)
        if profile is not None:
            return profile, f"fallback:{fallback_profile}"
    return None, "none"
