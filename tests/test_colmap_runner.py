"""Tests for COLMAPRunner's disconnected-model consolidation.

Regression coverage: COLMAP's incremental mapper can write multiple
disconnected reconstructions (sparse/0, sparse/1, ...) when the image
sequence doesn't form one fully connected component. Every downstream
consumer hardcodes sparse/0 -- without consolidation, they'd silently read
whichever (possibly tiny) reconstruction happened to be written first.
"""

import struct

from colmap_rgbd_gt.colmap.runner import COLMAPResult, COLMAPRunner, count_registered_images


def _write_images_bin(path, num_images):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", num_images))


def test_sequential_matcher_passes_overlap_and_quadratic(tmp_path, monkeypatch):
    runner = COLMAPRunner(tmp_path / "ws")
    captured_args = {}

    def fake_run_command(args, env=None):
        captured_args["args"] = args
        return COLMAPResult(success=True, return_code=0, stdout="", stderr="")

    monkeypatch.setattr(runner, "_run_command", fake_run_command)

    runner.sequential_matcher({"sequential_overlap": 30, "quadratic_overlap": True})

    args = captured_args["args"]
    assert args[args.index("--SequentialMatching.overlap") + 1] == "30"
    assert args[args.index("--SequentialMatching.quadratic_overlap") + 1] == "1"
    assert "--SequentialMatching.loop_detection" not in args


def test_sequential_matcher_default_overlap(tmp_path, monkeypatch):
    runner = COLMAPRunner(tmp_path / "ws")
    captured_args = {}

    def fake_run_command(args, env=None):
        captured_args["args"] = args
        return COLMAPResult(success=True, return_code=0, stdout="", stderr="")

    monkeypatch.setattr(runner, "_run_command", fake_run_command)

    runner.sequential_matcher({})

    args = captured_args["args"]
    assert args[args.index("--SequentialMatching.overlap") + 1] == "10"
    assert args[args.index("--SequentialMatching.quadratic_overlap") + 1] == "0"
    assert "--SequentialMatching.loop_detection" not in args


def test_sequential_matcher_loop_detection_with_vocab_tree(tmp_path, monkeypatch):
    runner = COLMAPRunner(tmp_path / "ws")
    captured_args = {}

    def fake_run_command(args, env=None):
        captured_args["args"] = args
        return COLMAPResult(success=True, return_code=0, stdout="", stderr="")

    monkeypatch.setattr(runner, "_run_command", fake_run_command)

    runner.sequential_matcher({"loop_detection": True, "vocab_tree_path": "/opt/vocab.bin"})

    args = captured_args["args"]
    assert args[args.index("--SequentialMatching.loop_detection") + 1] == "1"
    assert args[args.index("--SequentialMatching.vocab_tree_path") + 1] == "/opt/vocab.bin"


def test_sequential_matcher_loop_detection_without_vocab_tree_skipped(tmp_path, monkeypatch):
    runner = COLMAPRunner(tmp_path / "ws")
    captured_args = {}

    def fake_run_command(args, env=None):
        captured_args["args"] = args
        return COLMAPResult(success=True, return_code=0, stdout="", stderr="")

    monkeypatch.setattr(runner, "_run_command", fake_run_command)

    runner.sequential_matcher({"loop_detection": True})

    args = captured_args["args"]
    assert "--SequentialMatching.loop_detection" not in args
    assert "--SequentialMatching.vocab_tree_path" not in args


def test_bundle_adjuster_invalidates_stale_text_model(tmp_path, monkeypatch):
    import colmap_rgbd_gt.colmap.runner as runner_module

    workspace = tmp_path / "ws"
    sparse0 = workspace / "colmap" / "sparse" / "0"
    sparse0.mkdir(parents=True)
    # Pre-BA text model, as mapper() would have left it.
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse0 / name).write_text("stale")

    runner = COLMAPRunner(workspace)

    def fake_run_command(args, env=None):
        return COLMAPResult(success=True, return_code=0, stdout="", stderr="")

    ensure_text_model_calls = []
    monkeypatch.setattr(runner, "_run_command", fake_run_command)
    monkeypatch.setattr(
        runner_module, "ensure_text_model",
        lambda path, colmap_path="colmap": ensure_text_model_calls.append(path) or True,
    )

    ok = runner.bundle_adjuster({})

    assert ok is True
    # Stale text files must be gone before re-conversion is triggered.
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        assert not (sparse0 / name).exists()
    assert ensure_text_model_calls == [sparse0]


def test_bundle_adjuster_passes_max_iterations(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    sparse0 = workspace / "colmap" / "sparse" / "0"
    sparse0.mkdir(parents=True)

    runner = COLMAPRunner(workspace)
    captured_args = {}

    def fake_run_command(args, env=None):
        captured_args["args"] = args
        return COLMAPResult(success=True, return_code=0, stdout="", stderr="")

    monkeypatch.setattr(runner, "_run_command", fake_run_command)
    monkeypatch.setattr("colmap_rgbd_gt.colmap.runner.ensure_text_model", lambda *a, **k: True)

    runner.bundle_adjuster({"bundle_adjustment_max_iterations": 250})

    args = captured_args["args"]
    assert args[args.index("--BundleAdjustment.max_num_iterations") + 1] == "250"


def test_count_registered_images_from_binary(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_images_bin(model_dir / "images.bin", 42)

    assert count_registered_images(model_dir) == 42


def test_count_registered_images_from_text_fallback(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "images.txt").write_text(
        "# comment\n"
        "1 1 0 0 0 0 0 0 1 000000.png\n"
        "\n"
        "2 1 0 0 0 0 0 0 1 000001.png\n"
        "\n"
    )

    assert count_registered_images(model_dir) == 2


def test_count_registered_images_missing_returns_zero(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert count_registered_images(model_dir) == 0


def test_consolidate_best_model_picks_largest(tmp_path):
    workspace = tmp_path / "ws"
    sparse_dir = workspace / "colmap" / "sparse"
    _write_images_bin(sparse_dir / "0" / "images.bin", 3)
    _write_images_bin(sparse_dir / "1" / "images.bin", 38)
    _write_images_bin(sparse_dir / "2" / "images.bin", 115)
    (sparse_dir / "2" / "marker.txt").write_text("this is model 2")

    runner = COLMAPRunner(workspace)
    runner._consolidate_best_model()

    assert count_registered_images(sparse_dir / "0") == 115
    assert (sparse_dir / "0" / "marker.txt").exists()
    # The original (worse) model 0 must not be lost, just moved aside.
    alt_dirs = [d for d in sparse_dir.iterdir() if d.name.startswith("0_alt")]
    assert len(alt_dirs) == 1
    assert count_registered_images(alt_dirs[0]) == 3


def test_consolidate_best_model_noop_when_already_best(tmp_path):
    workspace = tmp_path / "ws"
    sparse_dir = workspace / "colmap" / "sparse"
    _write_images_bin(sparse_dir / "0" / "images.bin", 100)
    _write_images_bin(sparse_dir / "1" / "images.bin", 5)

    runner = COLMAPRunner(workspace)
    runner._consolidate_best_model()

    assert count_registered_images(sparse_dir / "0") == 100
    assert not any(d.name.startswith("0_alt") for d in sparse_dir.iterdir())


def test_consolidate_best_model_noop_single_model(tmp_path):
    workspace = tmp_path / "ws"
    sparse_dir = workspace / "colmap" / "sparse"
    _write_images_bin(sparse_dir / "0" / "images.bin", 10)

    runner = COLMAPRunner(workspace)
    runner._consolidate_best_model()

    assert count_registered_images(sparse_dir / "0") == 10
    assert list(sparse_dir.iterdir()) == [sparse_dir / "0"]
