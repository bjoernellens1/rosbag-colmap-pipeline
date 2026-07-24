"""COLMAP CLI runner."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.colmap.reconstruction import ensure_text_model
from colmap_rgbd_gt.utils.io import ensure_dir

logger = get_logger(__name__)


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

        args = [
            "sequential_matcher",
            "--database_path", str(self.database),
            "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        ]

        logger.info("Running COLMAP sequential matching...")
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
            sparse_model_dir = self.sparse_dir / "0"
            if sparse_model_dir.exists():
                ensure_text_model(sparse_model_dir, colmap_path=self.colmap_path)
            logger.info("Mapper completed")
        else:
            logger.error(f"Mapper failed: {result.stderr}")

        return result.success

    def bundle_adjuster(self, config: dict[str, Any]) -> bool:
        sparse_input = self.sparse_dir / "0"
        if not sparse_input.exists():
            logger.error("No sparse reconstruction found")
            return False

        args = [
            "bundle_adjuster",
            "--input_path", str(sparse_input),
            "--output_path", str(sparse_input),
        ]

        logger.info("Running COLMAP bundle adjustment...")
        result = self._run_command(args)

        if result.success:
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
