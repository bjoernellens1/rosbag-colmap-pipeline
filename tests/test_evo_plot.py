"""Tests for the trajectory-plotting contract (export/evo.py::plot_trajectory)."""

import pytest

from colmap_rgbd_gt.export.evo import plot_trajectory

pytest.importorskip("evo")


def _write_tum(path, n=5):
    with open(path, "w") as f:
        for i in range(n):
            f.write(f"{i}.0 {float(i)} 0.0 0.0 0.0 0.0 0.0 1.0\n")


def test_plot_trajectory_creates_png(tmp_path):
    tum_path = tmp_path / "traj.txt"
    _write_tum(tum_path)
    out_path = tmp_path / "plot.png"

    plot_trajectory(tum_path, out_path, title="Test")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_trajectory_with_reference(tmp_path):
    tum_path = tmp_path / "traj.txt"
    ref_path = tmp_path / "ref.txt"
    _write_tum(tum_path)
    _write_tum(ref_path, n=3)
    out_path = tmp_path / "plot.png"

    plot_trajectory(tum_path, out_path, title="Test", reference_tum_path=ref_path)

    assert out_path.exists()


def test_plot_trajectory_missing_file_does_not_raise(tmp_path):
    out_path = tmp_path / "plot.png"

    plot_trajectory(tmp_path / "nonexistent.txt", out_path)

    assert not out_path.exists()
