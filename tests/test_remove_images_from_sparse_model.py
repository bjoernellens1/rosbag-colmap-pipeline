"""Real (not mocked) integration tests for
colmap.reconstruction.remove_images_from_sparse_model -- uses the actual
`colmap` binary (image_deleter/model_converter), matching this project's
"verify against real behavior" standard. Built to close a real gap found
2026-08-03: the disconnected-segment pose filter (pose_outliers.py) only
ever cleaned exported trajectory artifacts, never the actual
colmap/sparse/0 binary model a user might open directly in COLMAP's GUI.
"""

import shutil

import pytest

from colmap_rgbd_gt.colmap.reconstruction import (
    remove_images_from_sparse_model,
    load_sparse_model,
)

pytest.importorskip("shutil")
COLMAP_AVAILABLE = shutil.which("colmap") is not None
pytestmark = pytest.mark.skipif(not COLMAP_AVAILABLE, reason="requires the real colmap binary")


def _write_synthetic_text_model(path, n_images=3):
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "1 PINHOLE 640 480 500 500 320 240\n"
    )
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"]
    for i in range(n_images):
        lines.append(f"{i + 1} 1 0 0 0 {float(i)} 0 0 1 {i:06d}.png\n\n")
    (path / "images.txt").write_text("".join(lines))
    (path / "points3D.txt").write_text("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")


def _to_binary(path):
    import subprocess
    subprocess.run(
        ["colmap", "model_converter", "--input_path", str(path),
         "--output_path", str(path), "--output_type", "BIN"],
        check=True, capture_output=True,
    )


def test_remove_images_from_sparse_model_real_colmap(tmp_path):
    sparse_dir = tmp_path / "sparse" / "0"
    _write_synthetic_text_model(sparse_dir, n_images=3)
    _to_binary(sparse_dir)

    ok = remove_images_from_sparse_model(sparse_dir, ["000001.png"])
    assert ok is True

    model = load_sparse_model(sparse_dir)
    names = sorted(img["name"] for img in model["images"].values())
    assert names == ["000000.png", "000002.png"]
    assert len(model["images"]) == 2


def test_remove_images_from_sparse_model_empty_list_is_noop(tmp_path):
    sparse_dir = tmp_path / "sparse" / "0"
    _write_synthetic_text_model(sparse_dir, n_images=2)
    _to_binary(sparse_dir)

    ok = remove_images_from_sparse_model(sparse_dir, [])
    assert ok is True

    model = load_sparse_model(sparse_dir)
    assert len(model["images"]) == 2


def test_remove_images_from_sparse_model_leaves_original_on_bad_input(tmp_path):
    """A nonexistent image name should not silently corrupt or blank the
    model -- confirm the original 2-image model survives intact either way
    (COLMAP's image_deleter itself decides how to handle an unknown name;
    this test's job is just to confirm no data loss on the real images)."""
    sparse_dir = tmp_path / "sparse" / "0"
    _write_synthetic_text_model(sparse_dir, n_images=2)
    _to_binary(sparse_dir)

    remove_images_from_sparse_model(sparse_dir, ["999999_does_not_exist.png"])

    model = load_sparse_model(sparse_dir)
    names = {img["name"] for img in model["images"].values()}
    assert {"000000.png", "000001.png"}.issubset(names) or len(model["images"]) == 2
