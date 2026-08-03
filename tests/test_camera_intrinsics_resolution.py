"""Tests for real-intrinsics-as-priors wiring (2026-08-03 fix).

Covers the two pieces that were previously "built but not wired":
- CameraIntrinsics.to_colmap_str() producing a param string COLMAP's
  --ImageReader.camera_params actually accepts for a given camera_model.
- ingest.camera_info.resolve_camera_info() preferring a valid bag
  camera_info over a hardcoded fallback profile, and falling back
  correctly when the bag's camera_info is missing/invalid.
"""

from colmap_rgbd_gt.ingest.camera_info import (
    KNOWN_CAMERA_PROFILES,
    resolve_camera_info,
    validate_camera_info,
)
from colmap_rgbd_gt.utils.camera import CameraIntrinsics


def test_to_colmap_str_opencv_drops_k3_and_orders_params():
    # ROS plumb_bob D = [k1, k2, p1, p2, k3]; COLMAP OPENCV wants
    # fx,fy,cx,cy,k1,k2,p1,p2 (no k3).
    intr = CameraIntrinsics(
        fx=748.6, fy=748.4, cx=636.2, cy=345.3,
        width=1280, height=720,
        distortion_model="plumb_bob",
        distortion_coeffs=[0.076, -0.103, -0.0001, 0.0004, 0.042],
    )
    s = intr.to_colmap_str("OPENCV")
    parts = [float(x) for x in s.split(",")]
    assert len(parts) == 8
    assert parts[:4] == [748.6, 748.4, 636.2, 345.3]
    assert parts[4:] == [0.076, -0.103, -0.0001, 0.0004]  # k3 dropped


def test_to_colmap_str_pinhole_has_no_distortion_params():
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    s = intr.to_colmap_str("PINHOLE")
    parts = s.split(",")
    assert len(parts) == 4


def test_get_colmap_model_picks_opencv_for_plumb_bob_with_distortion():
    intr = CameraIntrinsics(
        fx=500, fy=500, cx=320, cy=240, width=640, height=480,
        distortion_model="plumb_bob", distortion_coeffs=[0.1, -0.05, 0, 0, 0],
    )
    assert intr.get_colmap_model() == "OPENCV"


def test_get_colmap_model_picks_pinhole_when_no_distortion():
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    assert intr.get_colmap_model() == "PINHOLE"


def _valid_info():
    return {
        "width": 1280, "height": 720,
        "K": [748.6, 0, 636.2, 0, 748.4, 345.3, 0, 0, 1],
        "D": [0.076, -0.103, -0.0001, 0.0004, 0.042],
        "distortion_model": "plumb_bob",
    }


def test_validate_camera_info_accepts_real_intrinsics():
    assert validate_camera_info(_valid_info()) == []


def test_validate_camera_info_rejects_zero_focal_length():
    info = _valid_info()
    info["K"] = [0, 0, 636.2, 0, 0, 345.3, 0, 0, 1]
    errors = validate_camera_info(info)
    assert any("focal length" in e for e in errors)


def test_resolve_camera_info_prefers_valid_bag_info_over_fallback():
    info, source = resolve_camera_info(_valid_info(), fallback_profile="orbbec_femto_bolt")
    assert source == "bag_camera_info"
    assert info["K"][0] == 748.6


def test_resolve_camera_info_falls_back_when_bag_info_missing():
    info, source = resolve_camera_info(None, fallback_profile="orbbec_femto_bolt")
    assert source == "fallback:orbbec_femto_bolt"
    assert info == KNOWN_CAMERA_PROFILES["orbbec_femto_bolt"]


def test_resolve_camera_info_falls_back_when_bag_info_invalid():
    bad = _valid_info()
    bad["K"] = [0, 0, 636.2, 0, 0, 345.3, 0, 0, 1]
    info, source = resolve_camera_info(bad, fallback_profile="orbbec_femto_mega")
    assert source == "fallback:orbbec_femto_mega"


def test_resolve_camera_info_returns_none_with_no_source_at_all():
    info, source = resolve_camera_info(None, fallback_profile=None)
    assert info is None
    assert source == "none"
