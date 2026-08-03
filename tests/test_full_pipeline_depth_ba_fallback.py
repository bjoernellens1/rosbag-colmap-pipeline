"""Tests for full_pipeline()'s depth-ba fallback behavior.

FIXED 2026-08-03: an exception raised by depth_ba_pipeline (e.g. kornia_rs's
"Reduced camera Cholesky failed (likely rank-deficient)" ValueError, hit on
a real floor2 rerun) used to crash the ENTIRE `full` pipeline with exit 1,
discarding an otherwise-successful scale-only stage's output. depth-ba is
a best-effort refinement and must degrade to the scale-only trajectory on
any failure -- return False, not raise -- exactly like an ordinary
`depth_ba_pipeline() -> False` failure already did before this fix.
"""

import importlib
from pathlib import Path

import pytest

full_pipeline_module = importlib.import_module("colmap_rgbd_gt.pipelines.full_pipeline")
full_pipeline = full_pipeline_module.full_pipeline


@pytest.fixture(autouse=True)
def _stub_pre_depth_ba_stages(monkeypatch, tmp_path):
    """Stub extract/colmap/scale so we can isolate depth-ba's fallback
    behavior without a real bag/COLMAP run."""
    monkeypatch.setattr(
        full_pipeline_module, "extract_pipeline",
        lambda bag_path, workspace, config: True,
    )
    monkeypatch.setattr(
        full_pipeline_module, "colmap_pipeline",
        lambda workspace, config: True,
    )
    monkeypatch.setattr(
        full_pipeline_module, "scale_pipeline",
        lambda workspace, config: True,
    )
    # full_pipeline() requires bag_path.exists()
    bag = tmp_path / "fake.mcap"
    bag.write_bytes(b"")
    return bag


def test_depth_ba_exception_does_not_crash_full_pipeline(monkeypatch, tmp_path, _stub_pre_depth_ba_stages):
    def _raise(*args, **kwargs):
        raise ValueError("Reduced camera Cholesky failed (likely rank-deficient)")

    monkeypatch.setattr(
        "colmap_rgbd_gt.pipelines.depth_ba_pipeline.depth_ba_pipeline", _raise,
    )

    config = {"depth_ba": {"enabled": True}}
    ok = full_pipeline(_stub_pre_depth_ba_stages, tmp_path / "ws", config)

    # The whole pipeline must still report success -- scale-only output
    # from the earlier (stubbed-successful) stage is the valid result.
    assert ok is True


def test_depth_ba_false_return_still_succeeds(monkeypatch, tmp_path, _stub_pre_depth_ba_stages):
    """Unchanged existing behavior: a plain `False` return (no exception)
    from depth_ba_pipeline must also fall back gracefully, not fail the
    whole pipeline."""
    monkeypatch.setattr(
        "colmap_rgbd_gt.pipelines.depth_ba_pipeline.depth_ba_pipeline",
        lambda workspace, config: False,
    )

    config = {"depth_ba": {"enabled": True}}
    ok = full_pipeline(_stub_pre_depth_ba_stages, tmp_path / "ws", config)
    assert ok is True


def test_depth_ba_success_path_unaffected(monkeypatch, tmp_path, _stub_pre_depth_ba_stages):
    monkeypatch.setattr(
        "colmap_rgbd_gt.pipelines.depth_ba_pipeline.depth_ba_pipeline",
        lambda workspace, config: True,
    )

    config = {"depth_ba": {"enabled": True}}
    ok = full_pipeline(_stub_pre_depth_ba_stages, tmp_path / "ws", config)
    assert ok is True
