"""Regression tests for Bug 3 (2026-09-12): depth-BA's optimized trajectory
never became the pipeline's canonical "final" output.

Before this fix, `depth_ba_pipeline` always wrote its result to
`outputs/trajectory_depth_ba_tum.txt` and `colmap/sparse/0_refined`, but
never touched `outputs/trajectory_metric_tum.txt` (the file downstream
tooling and `full_pipeline`'s own "Metric trajectory: ..." log line treat
as canonical) -- even when depth-BA converged successfully. These tests
exercise the real `depth_ba_pipeline()` end to end (with the heavy
kornia_rs solve stubbed out, following the same approach as
`test_full_pipeline_depth_ba_fallback.py`) and check that:

- a converged run promotes the BA trajectory to trajectory_metric_tum.txt
  and preserves the pre-BA (scale-only) trajectory at
  trajectory_scale_only_tum.txt;
- a non-converged run leaves trajectory_metric_tum.txt exactly as
  scale_pipeline produced it (the existing fallback behavior).
"""

from pathlib import Path

import numpy as np
import pytest

from colmap_rgbd_gt.colmap.colmap_io import write_cameras_text, write_images_text, write_points3d_text
from colmap_rgbd_gt.export.tum import export_trajectory_tum
from colmap_rgbd_gt.pipelines import depth_ba_pipeline as depth_ba_pipeline_module
import colmap_rgbd_gt.optimization.depth_ba as depth_ba_module

FX = FY = 500.0
CX, CY = 320.0, 240.0
WIDTH, HEIGHT = 640, 480

# A translation far outside anything the scale-only/synthetic-model math
# below could produce, so it's unambiguous which trajectory ended up in
# trajectory_metric_tum.txt.
BA_MARKER_TRANSLATION = [123.0, 456.0, 789.0]
SCALE_ONLY_MARKER_TRANSLATION = [0.0, 0.0, 0.0]


class _FakeDepthBAResult:
    """Stand-in for `optimization.depth_ba.DepthBAResult` -- exercises
    `depth_ba_pipeline`'s file-writing/promotion logic without running the
    real kornia_rs solve."""

    def __init__(self, converged: bool, image_names: list[str]):
        self.converged = converged
        self.iterations = 5
        self.n_observations = 10
        self.n_depth_observations = 10
        self.image_names = image_names
        self.point_ids = []
        self.rotations = np.tile(np.eye(3), (len(image_names), 1, 1))
        self.translations = np.zeros((len(image_names), 3))
        self.points = np.zeros((0, 3))

    def to_colmap_model(self, original_model):
        return {
            "cameras": dict(original_model["cameras"]),
            "images": {k: dict(v) for k, v in original_model["images"].items()},
            "points3d": {k: dict(v) for k, v in original_model["points3d"].items()},
        }

    def to_tum_trajectory(self):
        return [
            {
                "frame_id": 1,
                "image_name": self.image_names[0] if self.image_names else "000001.png",
                "R": np.eye(3),
                "t": np.array(BA_MARKER_TRANSLATION),
            }
        ]


def _build_workspace(tmp_path):
    ws_root = tmp_path / "workspace"
    for sub in ("rgb", "depth", "outputs"):
        (ws_root / sub).mkdir(parents=True)
    (ws_root / "colmap" / "sparse" / "0").mkdir(parents=True)

    import json
    manifest = {
        "camera_info": {
            "K": [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0],
            "width": WIDTH,
            "height": HEIGHT,
        }
    }
    with open(ws_root / "manifest.json", "w") as f:
        json.dump(manifest, f)

    cameras = {
        1: {"camera_id": 1, "model": "PINHOLE", "width": WIDTH, "height": HEIGHT, "params": [FX, FY, CX, CY]}
    }
    images = {
        1: {
            "image_id": 1,
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 0.0],
            "camera_id": 1,
            "name": "000001.png",
        }
    }
    write_cameras_text(ws_root / "colmap" / "sparse" / "0" / "cameras.txt", cameras)
    write_images_text(ws_root / "colmap" / "sparse" / "0" / "images.txt", images)
    write_points3d_text(ws_root / "colmap" / "sparse" / "0" / "points3D.txt", {})

    # Pre-existing scale-only trajectory, as scale_pipeline() would have
    # already written before depth_ba_pipeline() runs (full_pipeline always
    # runs scale_pipeline first).
    scale_only_trajectory = [
        {"frame_id": 1, "image_name": "000001.png", "R": np.eye(3), "t": np.array(SCALE_ONLY_MARKER_TRANSLATION)}
    ]
    export_trajectory_tum(scale_only_trajectory, ws_root / "outputs" / "trajectory_metric_tum.txt")

    return ws_root


def _read_first_translation(tum_path: Path) -> list[float]:
    line = tum_path.read_text().strip().splitlines()[0]
    parts = line.split()
    # TUM format: timestamp tx ty tz qx qy qz qw
    return [float(parts[1]), float(parts[2]), float(parts[3])]


@pytest.fixture(autouse=True)
def _stub_scale_estimation(monkeypatch):
    """Isolate promotion-logic tests from the real scale estimator --
    depth_ba_pipeline's Stage A just needs *a* ScaleEstimate, not a
    correctly-recovered one."""
    from colmap_rgbd_gt.scaling.scale_estimation import ScaleEstimate

    monkeypatch.setattr(
        depth_ba_pipeline_module, "estimate_global_scale",
        lambda workspace, config: ScaleEstimate(
            scale=1.0, confidence=1.0, method="median", num_samples=10, inlier_ratio=1.0,
        ),
    )
    monkeypatch.setattr(
        depth_ba_pipeline_module, "extract_trajectory",
        lambda workspace: [
            {"frame_id": 1, "image_name": "000001.png", "R": np.eye(3), "t": np.zeros(3)}
        ],
    )
    # Plotting is best-effort (wrapped in try/except in the pipeline) but
    # touches matplotlib/evo machinery we don't need to exercise here.
    monkeypatch.setattr(depth_ba_pipeline_module, "plot_trajectory", lambda *a, **k: None)


def test_converged_depth_ba_promotes_trajectory_and_preserves_scale_only(tmp_path, monkeypatch):
    ws_root = _build_workspace(tmp_path)

    def _fake_run_depth_bundle_adjustment(model, image_names, depth_loader, intrinsics, config):
        return _FakeDepthBAResult(converged=True, image_names=image_names)

    monkeypatch.setattr(depth_ba_module, "run_depth_bundle_adjustment", _fake_run_depth_bundle_adjustment)

    ok = depth_ba_pipeline_module.depth_ba_pipeline(ws_root, {"depth_ba": {}, "scaling": {}})
    assert ok is True

    metric_tum_path = ws_root / "outputs" / "trajectory_metric_tum.txt"
    scale_only_tum_path = ws_root / "outputs" / "trajectory_scale_only_tum.txt"
    ba_tum_path = ws_root / "outputs" / "trajectory_depth_ba_tum.txt"

    assert metric_tum_path.exists()
    assert scale_only_tum_path.exists()
    assert ba_tum_path.exists()

    # trajectory_metric_tum.txt must now hold the BA-refined poses...
    assert _read_first_translation(metric_tum_path) == pytest.approx(BA_MARKER_TRANSLATION)
    # ...and the original scale-only trajectory must be preserved, not lost.
    assert _read_first_translation(scale_only_tum_path) == pytest.approx(SCALE_ONLY_MARKER_TRANSLATION)

    report_path = ws_root / "outputs" / "depth_ba_report.json"
    assert report_path.exists()
    import json
    report = json.loads(report_path.read_text())
    assert report["converged"] is True
    assert report["promoted_to_metric_tum"] is True


def test_non_converged_depth_ba_keeps_scale_only_trajectory(tmp_path, monkeypatch):
    ws_root = _build_workspace(tmp_path)

    def _fake_run_depth_bundle_adjustment(model, image_names, depth_loader, intrinsics, config):
        return _FakeDepthBAResult(converged=False, image_names=image_names)

    monkeypatch.setattr(depth_ba_module, "run_depth_bundle_adjustment", _fake_run_depth_bundle_adjustment)

    ok = depth_ba_pipeline_module.depth_ba_pipeline(ws_root, {"depth_ba": {}, "scaling": {}})
    assert ok is True

    metric_tum_path = ws_root / "outputs" / "trajectory_metric_tum.txt"
    scale_only_tum_path = ws_root / "outputs" / "trajectory_scale_only_tum.txt"

    # Regression-test the existing fallback: trajectory_metric_tum.txt must
    # stay exactly the scale-only trajectory it was before depth_ba_pipeline
    # ran, and no promotion-only artifact should appear.
    assert _read_first_translation(metric_tum_path) == pytest.approx(SCALE_ONLY_MARKER_TRANSLATION)
    assert not scale_only_tum_path.exists()

    report_path = ws_root / "outputs" / "depth_ba_report.json"
    import json
    report = json.loads(report_path.read_text())
    assert report["converged"] is False
    assert report["promoted_to_metric_tum"] is False
