"""Scaling module initialization."""

from colmap_rgbd_gt.scaling.correspondences import (
    find_colmap_points_in_image,
    project_colmap_points_to_image,
    find_valid_correspondences,
)
from colmap_rgbd_gt.scaling.backproject import (
    backproject_depth_image,
    backproject_points,
    transform_to_world,
    filter_invalid_depth,
    sample_depth_points,
)
from colmap_rgbd_gt.scaling.scale_estimation import (
    ScaleEstimate,
    estimate_scale_median,
    estimate_scale_umeyama,
    estimate_scale_ransac,
    estimate_global_scale,
)
from colmap_rgbd_gt.scaling.robust_statistics import (
    median_absolute_deviation,
    huber_weights,
    remove_outliers_iqr,
    robust_mean,
    robust_std,
)
from colmap_rgbd_gt.scaling.diagnostics import (
    ScaleDiagnostics,
    compute_diagnostics,
    plot_scale_histogram,
    export_scale_report,
)

__all__ = [
    "find_colmap_points_in_image",
    "project_colmap_points_to_image",
    "find_valid_correspondences",
    "backproject_depth_image",
    "backproject_points",
    "transform_to_world",
    "filter_invalid_depth",
    "sample_depth_points",
    "ScaleEstimate",
    "estimate_scale_median",
    "estimate_scale_umeyama",
    "estimate_scale_ransac",
    "estimate_global_scale",
    "median_absolute_deviation",
    "huber_weights",
    "remove_outliers_iqr",
    "robust_mean",
    "robust_std",
    "ScaleDiagnostics",
    "compute_diagnostics",
    "plot_scale_histogram",
    "export_scale_report",
]
