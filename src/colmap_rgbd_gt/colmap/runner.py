"""COLMAP CLI runner."""

import subprocess
import shutil
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from colmap_rgbd_gt.logging import get_logger
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

    def _run_command(self, args: list[str], env: dict[str, str] | None = None) -> COLMAPResult:
        import os
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [self.colmap_path] + args,
                capture_output=True,
                text=True,
                cwd=str(self.workspace),
                env=full_env,
            )
            return COLMAPResult(
                success=result.returncode == 0,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
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
