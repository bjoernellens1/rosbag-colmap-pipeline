"""evo tool compatibility utilities."""

from pathlib import Path
from typing import Any
import numpy as np

from colmap_rgbd_gt.export.tum import export_tum
from colmap_rgbd_gt.logging import get_logger

logger = get_logger(__name__)


def plot_trajectory(
    tum_path: Path,
    output_path: Path,
    title: str = "Trajectory",
    reference_tum_path: Path | None = None,
) -> None:
    """Render an evo-style trajectory plot (top-down XY view) to a PNG.

    Every pipeline run that produces a TUM trajectory calls this so a
    visual sanity check is always available alongside the raw numbers --
    a scale bug or a broken reconstruction is often obvious at a glance
    (a trajectory folded onto itself, an implausible spatial extent) in a
    way that isn't from `trajectory_metric_tum.txt` alone.

    Args:
        tum_path: primary trajectory to plot (e.g. the metric trajectory).
        output_path: PNG path to write.
        title: plot title.
        reference_tum_path: optional second trajectory (e.g. a ground-truth
            or unscaled trajectory) plotted alongside for comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evo.tools import file_interface, plot as evo_plot

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        traj = file_interface.read_tum_trajectory_file(str(tum_path))
    except Exception as e:
        logger.warning(f"Could not read trajectory for plotting ({tum_path}): {e}")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = evo_plot.prepare_axis(fig, evo_plot.PlotMode.xy)
    evo_plot.traj(ax, evo_plot.PlotMode.xy, traj, style="-", color="tab:blue", label=tum_path.name)

    if reference_tum_path is not None and Path(reference_tum_path).exists():
        try:
            ref_traj = file_interface.read_tum_trajectory_file(str(reference_tum_path))
            evo_plot.traj(
                ax, evo_plot.PlotMode.xy, ref_traj, style="--",
                color="tab:orange", label=Path(reference_tum_path).name,
            )
        except Exception as e:
            logger.warning(f"Could not read reference trajectory for plotting: {e}")

    ax.legend()
    ax.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Trajectory plot saved to {output_path}")


def export_for_evo(
    poses: list[dict[str, Any]],
    path: Path,
    format_type: str = "tum"
) -> None:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".tum", "") or format_type == "tum":
        tum_path = path.with_suffix(".tum") if ext == "" else path
        export_tum(poses, tum_path)
        logger.info(f"Exported evo-compatible TUM file to {tum_path}")

    elif ext == ".kitti" or format_type == "kitti":
        export_kitti(poses, path)
        logger.info(f"Exported evo-compatible KITTI file to {path}")

    else:
        export_tum(poses, path)


def export_kitti(
    poses: list[dict[str, Any]],
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for pose in poses:
            R = pose["R"]
            t = pose["t"]

            mat = np.eye(4)
            mat[:3, :3] = R
            mat[:3, 3] = t

            flat = mat[:3, :].flatten()
            f.write(" ".join(f"{v:.6f}" for v in flat) + "\n")

    logger.info(f"Exported {len(poses)} poses in KITTI format to {path}")
