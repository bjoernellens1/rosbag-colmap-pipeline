"""Scale diagnostics utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.utils.io import save_json

logger = get_logger(__name__)


@dataclass
class ScaleDiagnostics:
    per_frame_scales: list[float]
    per_frame_counts: list[int]
    scale_histogram: list[int]
    histogram_bins: list[float]
    confidence_score: float
    outlier_frames: list[int]
    median_scale: float
    std_scale: float
    num_total_frames: int
    num_valid_frames: int


def compute_diagnostics(estimates: list[Any]) -> ScaleDiagnostics:
    valid_scales = []
    valid_counts = []
    outlier_frames = []

    for i, est in enumerate(estimates):
        if est.num_samples > 0 and est.confidence > 0.1:
            valid_scales.append(est.scale)
            valid_counts.append(est.num_samples)
        else:
            outlier_frames.append(i)

    if len(valid_scales) == 0:
        return ScaleDiagnostics(
            per_frame_scales=[],
            per_frame_counts=[],
            scale_histogram=[],
            histogram_bins=[],
            confidence_score=0.0,
            outlier_frames=outlier_frames,
            median_scale=1.0,
            std_scale=0.0,
            num_total_frames=len(estimates),
            num_valid_frames=0,
        )

    scales = np.array(valid_scales)
    median_scale = float(np.median(scales))
    std_scale = float(np.std(scales))

    hist, bins = np.histogram(scales, bins=20)

    inlier_mask = np.abs(scales - median_scale) < 2 * std_scale
    confidence = float(np.mean(inlier_mask))

    return ScaleDiagnostics(
        per_frame_scales=valid_scales,
        per_frame_counts=valid_counts,
        scale_histogram=hist.tolist(),
        histogram_bins=bins[:-1].tolist(),
        confidence_score=confidence,
        outlier_frames=outlier_frames,
        median_scale=median_scale,
        std_scale=std_scale,
        num_total_frames=len(estimates),
        num_valid_frames=len(valid_scales),
    )


def plot_scale_histogram(diagnostics: ScaleDiagnostics, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(
            diagnostics.histogram_bins,
            diagnostics.scale_histogram,
            width=diagnostics.histogram_bins[1] - diagnostics.histogram_bins[0]
                 if len(diagnostics.histogram_bins) > 1 else 0.1,
            alpha=0.7,
            color="steelblue",
        )

        ax.axvline(
            diagnostics.median_scale,
            color="red",
            linestyle="--",
            label=f"Median: {diagnostics.median_scale:.4f}",
        )

        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("Frame Count")
        ax.set_title("Scale Distribution Across Frames")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Scale histogram saved to {path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping histogram plot")


def export_scale_report(
    diagnostics: ScaleDiagnostics,
    scale_estimate: Any,
    path: Path
) -> None:
    report = {
        "scale_estimate": {
            "scale": scale_estimate.scale,
            "confidence": scale_estimate.confidence,
            "method": scale_estimate.method,
            "num_samples": scale_estimate.num_samples,
            "inlier_ratio": scale_estimate.inlier_ratio,
        },
        "diagnostics": {
            "median_scale": diagnostics.median_scale,
            "std_scale": diagnostics.std_scale,
            "confidence_score": diagnostics.confidence_score,
            "num_total_frames": diagnostics.num_total_frames,
            "num_valid_frames": diagnostics.num_valid_frames,
            "num_outlier_frames": len(diagnostics.outlier_frames),
            "outlier_frames": diagnostics.outlier_frames,
            "per_frame_scales": diagnostics.per_frame_scales[:100],
            "per_frame_counts": diagnostics.per_frame_counts[:100],
        },
        "histogram": {
            "counts": diagnostics.scale_histogram,
            "bins": diagnostics.histogram_bins,
        },
    }

    save_json(path, report)
    logger.info(f"Scale report saved to {path}")
