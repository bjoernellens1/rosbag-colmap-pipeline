"""COLMAP CLI runner."""

import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.colmap.reconstruction import ensure_text_model
from colmap_rgbd_gt.utils.io import ensure_dir

logger = get_logger(__name__)


def count_registered_images(model_dir: Path) -> int:
    """Cheaply count registered images in a COLMAP sparse model.

    Reads only the leading uint64 count from `images.bin` (COLMAP's binary
    format starts every such file with the number of entries) rather than
    parsing the full model, so this is safe to call on every candidate
    model without the cost of a full `model_converter` pass. Falls back to
    counting `images.txt` if only a text model is present.
    """
    images_bin = model_dir / "images.bin"
    if images_bin.exists():
        with open(images_bin, "rb") as f:
            data = f.read(8)
        if len(data) == 8:
            return struct.unpack("<Q", data)[0]
        return 0

    images_txt = model_dir / "images.txt"
    if images_txt.exists():
        # Alternating header/POINTS2D line pairs -- POINTS2D may be blank,
        # so only comment lines are filtered, not blank ones.
        with open(images_txt) as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        return len(lines) // 2

    return 0


@dataclass
class COLMAPResult:
    success: bool
    return_code: int
    stdout: str
    stderr: str


class COLMAPRunner:
    def __init__(self, workspace: Path, colmap_path: str = "colmap"):
        self.workspace = Path(workspace)
        self.colmap_path = colmap_path
        self.database = self.workspace / "colmap" / "database.db"
        self.images_dir = self.workspace / "rgb"
        self.sparse_dir = self.workspace / "colmap" / "sparse"
        self.logs_dir = self.workspace / "colmap" / "logs"

    def find_colmap(self) -> str | None:
        return shutil.which(self.colmap_path)

    def _stream_pipe(
        self,
        pipe: TextIO | None,
        *,
        log_path: Path,
        log_method: Any,
    ) -> list[str]:
        captured: list[str] = []
        if pipe is None:
            return captured

        with log_path.open("w", encoding="utf-8") as log_file:
            for raw_line in pipe:
                line = raw_line.rstrip()
                if not line:
                    continue
                captured.append(line)
                print(line, flush=True)
                log_method(line)
                log_file.write(line + "\n")
                log_file.flush()

        return captured

    def _run_command(self, args: list[str], env: dict[str, str] | None = None) -> COLMAPResult:
        import os
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        ensure_dir(self.logs_dir)
        command_name = args[0]
        combined_log = self.logs_dir / f"{command_name}.log"

        try:
            process = subprocess.Popen(
                [self.colmap_path] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.workspace),
                env=full_env,
            )
            combined_lines = self._stream_pipe(
                process.stdout,
                log_path=combined_log,
                log_method=logger.info,
            )
            return_code = process.wait()
            return COLMAPResult(
                success=return_code == 0,
                return_code=return_code,
                stdout="\n".join(combined_lines),
                stderr="",
            )
        except FileNotFoundError:
            logger.error(f"COLMAP not found at {self.colmap_path}")
            return COLMAPResult(success=False, return_code=-1, stdout="", stderr="COLMAP not found")

    def feature_extractor(self, config: dict[str, Any]) -> bool:
        use_gpu = config.get("use_gpu", False)
        camera_model = config.get("camera_model", "OPENCV")
        single_camera = config.get("single_camera", True)

        self.database.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "feature_extractor",
            "--database_path", str(self.database),
            "--image_path", str(self.images_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
        ]

        logger.info("Running COLMAP feature extraction...")
        result = self._run_command(args)

        if result.success:
            logger.info("Feature extraction completed")
        else:
            logger.error(f"Feature extraction failed: {result.stderr}")

        return result.success

    def sequential_matcher(self, config: dict[str, Any]) -> bool:
        use_gpu = config.get("use_gpu", False)
        # COLMAP's default overlap (10) only tries matching each image
        # against its next 10 neighbors in sequence -- a single stretch
        # with degraded overlap (motion blur, low texture) can still break
        # the chain even with motion-adaptive keyframe selection. A wider
        # window gives more chances to bridge such a gap without changing
        # keyframe density.
        overlap = config.get("sequential_overlap", 10)
        quadratic_overlap = config.get("quadratic_overlap", False)
        loop_detection = config.get("loop_detection", False)
        vocab_tree_path = config.get("vocab_tree_path")

        args = [
            "sequential_matcher",
            "--database_path", str(self.database),
            "--SiftMatching.use_gpu", "1" if use_gpu else "0",
            "--SequentialMatching.overlap", str(overlap),
            "--SequentialMatching.quadratic_overlap", "1" if quadratic_overlap else "0",
        ]

        # Loop detection re-matches each frame against visually-similar
        # past frames (not just temporally-adjacent ones), which is
        # COLMAP's own documented mechanism for reconnecting a sequence
        # that's split into disconnected sub-models -- it needs a
        # pretrained vocab tree; silently ignored (with a warning) if none
        # is configured, rather than passing an invalid COLMAP flag combo.
        if loop_detection and vocab_tree_path:
            args.extend([
                "--SequentialMatching.loop_detection", "1",
                "--SequentialMatching.vocab_tree_path", str(vocab_tree_path),
            ])
        elif loop_detection:
            logger.warning(
                "loop_detection requested but no vocab_tree_path configured; "
                "skipping loop detection"
            )

        logger.info(
            f"Running COLMAP sequential matching (overlap={overlap}, "
            f"quadratic_overlap={quadratic_overlap}, loop_detection={loop_detection and bool(vocab_tree_path)})..."
        )
        result = self._run_command(args)

        if result.success:
            logger.info("Sequential matching completed")
        else:
            logger.error(f"Sequential matching failed: {result.stderr}")

        return result.success

    def exhaustive_matcher(self, config: dict[str, Any]) -> bool:
        use_gpu = config.get("use_gpu", False)

        args = [
            "exhaustive_matcher",
            "--database_path", str(self.database),
            "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        ]

        logger.info("Running COLMAP exhaustive matching...")
        result = self._run_command(args)

        if result.success:
            logger.info("Exhaustive matching completed")
        else:
            logger.error(f"Exhaustive matching failed: {result.stderr}")

        return result.success

    def _consolidate_best_model(self) -> None:
        """Ensure the largest reconstruction ends up at `sparse/0`.

        COLMAP's incremental mapper writes a separate numbered model
        (sparse/0, sparse/1, ...) each time the image sequence doesn't form
        one fully connected component -- e.g. insufficient frame-to-frame
        overlap causes registration to stall and restart from a new seed
        pair. Every downstream consumer in this codebase (pose_extract,
        scale_estimation, depth_ba_pipeline) hardcodes `sparse/0` as *the*
        model, so without this, they would silently read whichever
        (possibly tiny, e.g. 3-image) reconstruction happened to be written
        first, ignoring a much larger reconstruction sitting at sparse/1 or
        sparse/2.
        """
        candidates = sorted(
            (d for d in self.sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
        )
        if len(candidates) <= 1:
            return

        counts = {d: count_registered_images(d) for d in candidates}
        best = max(counts, key=counts.get)
        canonical = self.sparse_dir / "0"

        if best == canonical:
            return

        logger.info(
            "COLMAP produced %d disconnected models (%s); using %s "
            "(%d registered images) as the canonical sparse/0 model",
            len(candidates),
            ", ".join(f"{d.name}={n}" for d, n in counts.items()),
            best.name,
            counts[best],
        )

        tmp = self.sparse_dir / "__best_model_tmp"
        best.rename(tmp)

        displaced = self.sparse_dir / f"{canonical.name}_alt"
        i = 0
        while displaced.exists():
            i += 1
            displaced = self.sparse_dir / f"{canonical.name}_alt{i}"
        canonical.rename(displaced)

        tmp.rename(canonical)

    def mapper(self, config: dict[str, Any]) -> bool:
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "mapper",
            "--database_path", str(self.database),
            "--image_path", str(self.images_dir),
            "--output_path", str(self.sparse_dir),
        ]

        if config.get("mapper_preset") == "low":
            args.extend(["--Mapper.ba_global_max_num_iterations", "25"])
        elif config.get("mapper_preset") == "high":
            args.extend(["--Mapper.ba_global_max_num_iterations", "100"])

        logger.info("Running COLMAP mapper...")
        result = self._run_command(args)

        if result.success:
            self._consolidate_best_model()
            sparse_model_dir = self.sparse_dir / "0"
            if sparse_model_dir.exists():
                ensure_text_model(sparse_model_dir, colmap_path=self.colmap_path)
            logger.info("Mapper completed")
        else:
            logger.error(f"Mapper failed: {result.stderr}")

        return result.success

    def bundle_adjuster(self, config: dict[str, Any]) -> bool:
        """Run a final global bundle-adjustment pass on the consolidated model.

        The incremental mapper only runs *periodic* global BA during
        registration; loop-closure constraints discovered late in the
        sequence (via vocab-tree loop detection) may not get fully
        propagated by the time mapping stops. An explicit final pass with
        a generous iteration budget lets any verified loop-closure matches
        actually pull accumulated drift back together, rather than leaving
        a fully-connected-but-still-drifted trajectory (all frames
        registered, but a physically closed loop not visually closing).
        """
        sparse_input = self.sparse_dir / "0"
        if not sparse_input.exists():
            logger.error("No sparse reconstruction found")
            return False

        max_iterations = config.get("bundle_adjustment_max_iterations", 100)

        args = [
            "bundle_adjuster",
            "--input_path", str(sparse_input),
            "--output_path", str(sparse_input),
            "--BundleAdjustment.max_num_iterations", str(max_iterations),
        ]

        logger.info(f"Running COLMAP bundle adjustment (max_iterations={max_iterations})...")
        result = self._run_command(args)

        if result.success:
            # bundle_adjuster overwrites the binary model only; the text
            # model mapper() already generated (via ensure_text_model) is
            # now stale relative to it. Remove it so the next
            # ensure_text_model call regenerates from the fresh binary
            # output instead of silently skipping conversion because
            # cameras.txt/images.txt/points3D.txt already exist.
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                stale = sparse_input / name
                if stale.exists():
                    stale.unlink()
            ensure_text_model(sparse_input, colmap_path=self.colmap_path)
            logger.info("Bundle adjustment completed")
        else:
            logger.error(f"Bundle adjustment failed: {result.stderr}")

        return result.success

    def run_full_pipeline(self, config: dict[str, Any]) -> bool:
        if not self.find_colmap():
            logger.error("COLMAP not found in PATH")
            return False

        matcher = config.get("matcher", "sequential")

        steps = [
            (self.feature_extractor, config),
            (self.sequential_matcher if matcher == "sequential" else self.exhaustive_matcher, config),
            (self.mapper, config),
        ]

        if config.get("run_bundle_adjustment", False):
            steps.append((self.bundle_adjuster, config))

        for step_func, step_config in steps:
            if not step_func(step_config):
                return False

        return True
