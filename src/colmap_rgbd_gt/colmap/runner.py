"""COLMAP CLI runner."""

import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.colmap.loop_closure_filter import (
    DEFAULT_MIN_INLIERS,
    DEFAULT_MIN_INLIER_RATIO,
    filter_loop_closure_matches,
)
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
    def __init__(
        self,
        workspace: Path,
        colmap_path: str = "colmap",
        image_dir_name: str = "rgb",
    ):
        self.workspace = Path(workspace)
        self.colmap_path = colmap_path
        self.database = self.workspace / "colmap" / "database.db"
        # ADDED 2026-08-04: image_dir_name lets a caller point feature
        # extraction at rgb_rectified/ instead of rgb/ -- see
        # rectify/undistort.py's rectify_workspace_images(), used when
        # colmap.ba_backend: caspar needs a true PINHOLE input (Caspar
        # doesn't support OPENCV).
        self.images_dir = self.workspace / image_dir_name
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
        # FIXED 2026-08-03: real per-recording intrinsics (from the bag's
        # own camera_info topic, or a hardcoded Femto Bolt/Mega fallback --
        # see ingest.camera_info.resolve_camera_info) were being extracted
        # into the workspace manifest but never actually handed to COLMAP.
        # Without `--ImageReader.camera_params`, COLMAP treats focal length
        # as unknown-per-camera and self-estimates it during SfM, which is
        # exactly what triggered "Less than 50% of cameras have prior focal
        # lengths" on trolley_femto's global_mapper run -- a real
        # reconstruction-quality hit (weaker reprojection consistency,
        # which loop-closure/view-graph verification depends on), not just
        # a cosmetic warning.
        camera_params = config.get("camera_params")

        self.database.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "feature_extractor",
            "--database_path", str(self.database),
            "--image_path", str(self.images_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            # FIXED 2026-08-02: was --SiftExtraction.use_gpu -- COLMAP 4.x
            # (this pipeline now builds 4.1.1 from source, see
            # docker/Dockerfile) renamed the option group from
            # Sift{Extraction,Matching} to FeatureExtraction/FeatureMatching
            # (COLMAP now supports non-SIFT feature backends). 3.9.1's flag
            # name silently failed as "unrecognised option" once the image
            # switched off apt's 3.9.1 -- caught by an actual `run-colmap`
            # invocation against a real workspace, not assumed.
            "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
        ]
        if camera_params:
            args.extend(["--ImageReader.camera_params", camera_params])
            logger.info(f"Using real camera intrinsics as priors: {camera_params}")
        else:
            logger.warning(
                "No valid camera intrinsics available (bag camera_info missing/invalid "
                "and no fallback profile) -- COLMAP will self-calibrate focal length"
            )

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
            # FeatureMatching, not SiftMatching -- see feature_extractor()'s
            # comment on the same 3.9.1 -> 4.1.1 rename.
            "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
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
            return False

        loop_detection_applied = loop_detection and bool(vocab_tree_path)
        if loop_detection_applied:
            self._filter_loop_closure_matches(config, overlap, quadratic_overlap)

        return True

    def _filter_loop_closure_matches(
        self, config: dict[str, Any], sequential_overlap: int, quadratic_overlap: bool
    ) -> None:
        # Only meaningful when loop_detection actually ran -- sequential-
        # only matching has no loop-closure-retrieved pairs to prune. See
        # colmap/loop_closure_filter.py's module docstring for the full
        # root-cause writeup (kitchen1, 2026-08-05).
        min_inliers = config.get("loop_closure_min_inliers", DEFAULT_MIN_INLIERS)
        min_inlier_ratio = config.get("loop_closure_min_inlier_ratio", DEFAULT_MIN_INLIER_RATIO)

        try:
            result = filter_loop_closure_matches(
                self.database,
                sequential_overlap=sequential_overlap,
                quadratic_overlap=quadratic_overlap,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
            )
            report_path = self.workspace / "outputs" / "loop_closure_filter.json"
            ensure_dir(report_path.parent)
            with open(report_path, "w") as f:
                json.dump(
                    {
                        "action_taken": result.action_taken,
                        "reason": result.reason,
                        "n_pairs_checked": result.n_pairs_checked,
                        "n_loop_pairs": result.n_loop_pairs,
                        "n_sequential_pairs": result.n_sequential_pairs,
                        "n_dropped": result.n_dropped,
                        "dropped_pairs": result.dropped_pairs,
                        "min_inliers": min_inliers,
                        "min_inlier_ratio": min_inlier_ratio,
                    },
                    f,
                    indent=2,
                )
            logger.info(
                f"loop_closure_filter: {result.n_loop_pairs} loop pair(s) checked, "
                f"{result.n_dropped} dropped -- see {report_path}"
            )
        except Exception as e:
            logger.warning(f"loop_closure_filter failed, continuing without it: {e}")

    def exhaustive_matcher(self, config: dict[str, Any]) -> bool:
        use_gpu = config.get("use_gpu", False)

        args = [
            "exhaustive_matcher",
            "--database_path", str(self.database),
            # FeatureMatching, not SiftMatching -- see feature_extractor()'s
            # comment on the same 3.9.1 -> 4.1.1 rename.
            "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
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

    def global_mapper(self, config: dict[str, Any]) -> bool:
        """Global (GLOMAP-derived) reconstruction -- `colmap global_mapper`.

        Added 2026-08-02 after a real production incident: the incremental
        `mapper()` above ran 10+ CPU-hours on a 2056-frame scene without
        finishing (a bundle-adjustment round hit real numerical
        ill-conditioning -- cost frozen across 13+ iterations after a
        "Matrix not positive definite" CHOLMOD warning, not just a slow
        big job), and a SEPARATE scene's incremental run fragmented into 5
        disconnected models with only ~10% of frames in the largest one.
        `global_mapper` on the SAME (already-extracted, already-matched)
        databases reconstructed both scenes with 100% of frames in a single
        connected model, in a small fraction of the wall time (48 minutes
        end-to-end on the 2056-frame scene that stalled incrementally).
        Reads/writes the identical `database.db` and sparse-model format as
        `mapper()` -- confirmed empirically against a real database.db, not
        just from upstream docs -- so this is a drop-in alternative for the
        mapper stage only; feature_extractor/sequential_matcher are
        unchanged either way. Requires COLMAP >=4.0 (this pipeline pins
        4.1.1 in docker/Dockerfile specifically for this).
        """
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "global_mapper",
            "--database_path", str(self.database),
            "--image_path", str(self.images_dir),
            "--output_path", str(self.sparse_dir),
        ]

        # ADDED 2026-08-03: investigating floor3's wall-clock cost found the
        # "Running iterative retriangulation and refinement" phase (AFTER
        # the Ceres BA solve, which itself completed in well under an hour)
        # is the actual dominant cost -- 4+ hours and counting at only
        # ~2-3/32 cores utilized (host otherwise idle), not thread- or
        # contention-limited, just an inherently low-parallelism phase.
        # `--GlobalMapper.skip_retriangulation` is a real, exposed COLMAP
        # flag for it. NOT defaulted on: retriangulation refines the point
        # cloud/tracks after BA, and depth_ba_pipeline's
        # build_ba_observations() gates every depth observation against
        # `colmap_depth` computed FROM that same point cloud -- skipping
        # retriangulation could plausibly degrade depth-ba's usable-
        # observation rate, not just save wall-clock time. Needs an A/B
        # measurement (registration ratio + depth-ba n_depth_observations
        # with vs without) before ever flipping the default; this config
        # knob exists so that test can be run without a source patch.
        if config.get("skip_retriangulation", False):
            args.extend(["--GlobalMapper.skip_retriangulation", "1"])

        # ADDED 2026-08-04/05: global_mapper's own internal 3-iteration BA
        # loop was Ceres-CPU-only in COLMAP 4.1.1 (no backend flag existed
        # at all) -- confirmed the dominant wall-clock cost on large scenes
        # (700-900s on 1600-2700 frame scenes) alongside global positioning
        # and retriangulation. docker/Dockerfile.cuda now builds from
        # colmap/colmap#4484 (merged after 4.1.1), which adds
        # --GlobalMapper.ba_backend, and it IS genuinely fast (11s vs 884s
        # CPU on trolley_femto's internal loop). BUT: deliberately gated on
        # a SEPARATE key from `ba_backend` (the standalone bundle_adjuster
        # Caspar flag, safe and on by default) -- live accuracy sweep found
        # this newer, less-mature upstream feature introduces real
        # scale-regime-split regressions on 2/3 tested scenes (floor2: 0->2
        # segments, trolley_femto: 0->5 segments; only tableware1, the
        # smallest/simplest scene, stayed clean). NOT enabled in
        # configs/navigation.yaml -- opt-in only, until upstream's
        # global-mapper Caspar integration matures. Only takes effect when
        # GPU is actually requested -- a CPU-only binary doesn't have this
        # flag.
        if config.get("global_mapper_ba_backend") == "caspar" and config.get("use_gpu", False):
            args.extend(["--GlobalMapper.ba_backend", "CASPAR"])

        logger.info("Running COLMAP global_mapper (GLOMAP-derived global SfM)...")
        result = self._run_command(args)

        if result.success:
            self._consolidate_best_model()
            sparse_model_dir = self.sparse_dir / "0"
            if sparse_model_dir.exists():
                ensure_text_model(sparse_model_dir, colmap_path=self.colmap_path)
            logger.info("Global mapper completed")
        else:
            logger.error(f"Global mapper failed: {result.stderr}")

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
            # FIXED 2026-08-02: was --BundleAdjustment.max_num_iterations --
            # COLMAP 4.1.1 moved the Ceres-specific solver knobs (including
            # max_num_iterations) under a separate BundleAdjustmentCeres
            # group (COLMAP now supports pluggable BA backends), same
            # 3.9.1->4.1.1 rename class as feature_extractor()'s
            # FeatureExtraction/FeatureMatching fix.
            "--BundleAdjustmentCeres.max_num_iterations", str(max_iterations),
        ]

        # ADDED 2026-08-04: Caspar is COLMAP's own native GPU bundle-
        # adjustment solver (>=4.1.0, docker/Dockerfile.cuda builds with
        # -DCASPAR_ENABLED=ON) -- NOT a Ceres/cuDSS backend, so it avoids
        # the Ceres-master+cuDSS numerical-corruption bug found and
        # documented in docs/cluster-dispatch.md. `--BundleAdjustmentCeres.*`
        # flags above only apply when backend=CERES (the default); Caspar
        # ignores them, so max_iterations has no effect in that path today
        # (only the CLI image's build needs CASPAR_ENABLED -- a CPU-only
        # binary silently rejects this flag, so only send it when GPU is
        # actually requested).
        if config.get("ba_backend") == "caspar" and config.get("use_gpu", False):
            args.extend(["--BundleAdjustment.backend", "CASPAR"])

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
        # "global" (default, colmap global_mapper) or "incremental" (colmap
        # mapper). DEFAULT FLIPPED 2026-08-02 after a direct, controlled
        # comparison on the SAME database.db for two real scenes:
        # incremental `mapper` fragmented both (hallway: 5 disconnected
        # models, largest held ~10% of frames; kitchen1: ran 598.8 CPU-
        # minutes -- ~10 hours -- before giving up with "No good initial
        # image pair found", fragmenting into 3 pieces). `global_mapper` on
        # the IDENTICAL kitchen1 database reconstructed all 2056 images
        # into ONE connected model in 48 minutes. No scene tested this
        # session did better with incremental than global. See
        # global_mapper()'s docstring for the full mechanism writeup.
        # "incremental" is kept available as an explicit opt-in
        # (mapper_type: incremental) in case some future scene's geometry
        # genuinely favors it, but nothing should need to ask for
        # "global" by name anymore -- it's the default.
        mapper_type = config.get("mapper_type", "global")

        steps = [
            (self.feature_extractor, config),
            (self.sequential_matcher if matcher == "sequential" else self.exhaustive_matcher, config),
            (self.global_mapper if mapper_type == "global" else self.mapper, config),
        ]

        if config.get("run_bundle_adjustment", False):
            steps.append((self.bundle_adjuster, config))

        for step_func, step_config in steps:
            if not step_func(step_config):
                return False

        return True
