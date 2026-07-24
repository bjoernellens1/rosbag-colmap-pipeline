"""CLI entry point for gttool."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from colmap_rgbd_gt import __version__
from colmap_rgbd_gt.logging import get_logger, setup_logging

app = typer.Typer(
    name="gttool",
    help="Pipeline for converting rosbag recordings to metric pseudo-GT trajectories.",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"gttool version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """gttool - Pipeline for rosbag to metric pseudo-GT trajectory conversion."""
    pass


@app.command("inspect-bag")
def inspect_bag(
    bag_path: Path = typer.Argument(..., help="Path to rosbag file (.bag or .db3)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
) -> None:
    """
    Inspect a rosbag file and list available topics.

    Shows RGB, depth, camera_info, and pointcloud topics found in the bag.
    """
    from colmap_rgbd_gt.ingest.bag_reader import BagReader

    if not bag_path.exists():
        console.print(f"[red]Error: Bag file not found: {bag_path}[/red]")
        raise typer.Exit(1)

    setup_logging("DEBUG" if verbose else "INFO")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Reading bag metadata...", total=None)
        reader = BagReader(bag_path)

    # Display bag info
    table = Table(title=f"Bag Information: {bag_path.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    start_ns, end_ns = reader.get_time_range()
    duration_s = (end_ns - start_ns) / 1e9

    table.add_row("Storage Type", reader.storage_id)
    table.add_row("Duration", f"{duration_s:.2f} seconds")
    table.add_row("Start Time", f"{start_ns} ns")
    table.add_row("End Time", f"{end_ns} ns")

    console.print(table)

    # Topic discovery
    from colmap_rgbd_gt.ingest.topic_discovery import (
        discover_camera_info_topics,
        discover_depth_topics,
        discover_pointcloud_topics,
        discover_rgb_topics,
    )

    rgb_topics = discover_rgb_topics(reader)
    depth_topics = discover_depth_topics(reader)
    info_topics = discover_camera_info_topics(reader)
    pc_topics = discover_pointcloud_topics(reader)

    topics_table = Table(title="Discovered Topics")
    topics_table.add_column("Type", style="cyan")
    topics_table.add_column("Topic", style="green")
    topics_table.add_column("Messages", style="yellow")

    for t in rgb_topics:
        topics_table.add_row("RGB", t.name, str(t.message_count))
    for t in depth_topics:
        topics_table.add_row("Depth", t.name, str(t.message_count))
    for t in info_topics:
        topics_table.add_row("CameraInfo", t.name, str(t.message_count))
    for t in pc_topics:
        topics_table.add_row("PointCloud2", t.name, str(t.message_count))

    console.print(topics_table)
    reader.close()


@app.command("extract")
def extract(
    bag_path: Path = typer.Argument(..., help="Path to rosbag file"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Output workspace directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration YAML file"),
    rgb_topic: Optional[str] = typer.Option(None, "--rgb", help="RGB topic name"),
    depth_topic: Optional[str] = typer.Option(None, "--depth", help="Depth topic name"),
    camera_info_topic: Optional[str] = typer.Option(None, "--camera-info", help="Camera info topic name"),
) -> None:
    """
    Extract RGB, depth, and camera info from a rosbag.

    Creates a canonical workspace layout with extracted frames.
    """
    setup_logging("INFO")

    if not bag_path.exists():
        console.print(f"[red]Error: Bag file not found: {bag_path}[/red]")
        raise typer.Exit(1)

    if workspace is None:
        workspace = bag_path.with_suffix("")  # Remove extension

    from colmap_rgbd_gt.pipelines.extract_only import extract_pipeline

    # Load config if provided
    config_dict = {}
    if config and config.exists():
        import yaml
        config_dict = yaml.safe_load(config.read_text())

    # Override with CLI options
    if rgb_topic:
        config_dict.setdefault("topics", {})["rgb"] = rgb_topic
    if depth_topic:
        config_dict.setdefault("topics", {})["depth"] = depth_topic
    if camera_info_topic:
        config_dict.setdefault("topics", {})["camera_info"] = camera_info_topic

    console.print(f"[cyan]Extracting from: {bag_path}[/cyan]")
    console.print(f"[cyan]Workspace: {workspace}[/cyan]")

    success = extract_pipeline(bag_path, workspace, config_dict)

    if success:
        console.print("[green]✓ Extraction complete[/green]")
    else:
        console.print("[red]✗ Extraction failed[/red]")
        raise typer.Exit(1)


@app.command("run-colmap")
def run_colmap(
    workspace: Path = typer.Argument(..., help="Path to workspace directory"),
    use_gpu: bool = typer.Option(False, "--gpu", help="Use GPU acceleration (CUDA only)"),
) -> None:
    """
    Run COLMAP reconstruction on extracted RGB frames.

    Uses sequential matching for bag-like sequential data.
    """
    setup_logging("INFO")

    if not workspace.exists():
        console.print(f"[red]Error: Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)

    from colmap_rgbd_gt.pipelines.colmap_only import colmap_pipeline

    config = {"colmap": {"use_gpu": use_gpu}}

    console.print(f"[cyan]Running COLMAP on: {workspace}[/cyan]")

    success = colmap_pipeline(workspace, config)

    if success:
        console.print("[green]✓ COLMAP reconstruction complete[/green]")
    else:
        console.print("[red]✗ COLMAP failed[/red]")
        raise typer.Exit(1)


@app.command("scale-depth")
def scale_depth(
    workspace: Path = typer.Argument(..., help="Path to workspace directory"),
    method: str = typer.Option("umeyama", "--method", "-m", help="Scale estimation method (median, umeyama, ransac)"),
) -> None:
    """
    Estimate metric scale from depth data.

    Aligns COLMAP geometry with depth-based 3D points.
    """
    setup_logging("INFO")

    if not workspace.exists():
        console.print(f"[red]Error: Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)

    from colmap_rgbd_gt.pipelines.scale_only import scale_pipeline

    config = {"scaling": {"method": method}}

    console.print(f"[cyan]Estimating scale for: {workspace}[/cyan]")

    success = scale_pipeline(workspace, config)

    if success:
        console.print("[green]✓ Scale estimation complete[/green]")
    else:
        console.print("[red]✗ Scale estimation failed[/red]")
        raise typer.Exit(1)


@app.command("depth-ba")
def depth_ba(
    workspace: Path = typer.Argument(..., help="Path to workspace directory"),
    depth_tolerance: float = typer.Option(0.1, "--depth-tolerance", help="Relative depth-vs-reprojection tolerance"),
    max_iterations: int = typer.Option(50, "--max-iterations", help="Max LM iterations per BA stage"),
) -> None:
    """
    Run depth-aware bundle adjustment to jointly refine poses and sparse
    structure against metric depth (requires the 'depth-ba' extra:
    pip install colmap-rgbd-gt[depth-ba]).
    """
    setup_logging("INFO")

    if not workspace.exists():
        console.print(f"[red]Error: Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)

    try:
        from colmap_rgbd_gt.pipelines.depth_ba_pipeline import depth_ba_pipeline
    except ImportError:
        console.print(
            "[red]depth-ba requires the optional extra: "
            "pip install colmap-rgbd-gt[depth-ba][/red]"
        )
        raise typer.Exit(1)

    config = {
        "depth_ba": {
            "depth_tolerance": depth_tolerance,
            "max_iterations": max_iterations,
        }
    }

    console.print(f"[cyan]Running depth-aware bundle adjustment for: {workspace}[/cyan]")

    success = depth_ba_pipeline(workspace, config)

    if success:
        console.print("[green]✓ Depth-aware bundle adjustment complete[/green]")
    else:
        console.print("[red]✗ Depth-aware bundle adjustment failed[/red]")
        raise typer.Exit(1)


@app.command("export-bag")
def export_bag(
    bag_path: Path = typer.Argument(..., help="Path to the original rosbag"),
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace directory (must have a completed scale-depth run)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output bag path (default: <bag>_processed alongside the original)"),
    pose_topic: str = typer.Option("/gt/pose", "--pose-topic", help="Topic for per-frame GT PoseStamped messages"),
    path_topic: str = typer.Option("/gt/path", "--path-topic", help="Topic for the summary GT nav_msgs/Path message"),
    frame_id: str = typer.Option("map", "--frame-id", help="Frame ID for the GT pose/path headers"),
) -> None:
    """
    Write a '_processed' copy of the original bag containing every original
    message plus the estimated metric GT trajectory, on a new PoseStamped
    topic (per-frame) and a summary Path topic.
    """
    setup_logging("INFO")

    if not bag_path.exists():
        console.print(f"[red]Error: Bag file not found: {bag_path}[/red]")
        raise typer.Exit(1)
    if not workspace.exists():
        console.print(f"[red]Error: Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)

    from colmap_rgbd_gt.pipelines.export_bag_pipeline import export_bag_pipeline

    config = {
        "export_bag": {
            "output_path": str(output) if output else None,
            "pose_topic": pose_topic,
            "path_topic": path_topic,
            "frame_id": frame_id,
        }
    }

    console.print(f"[cyan]Writing processed bag for: {bag_path}[/cyan]")

    success = export_bag_pipeline(bag_path, workspace, config)

    if success:
        console.print("[green]✓ Processed bag written[/green]")
    else:
        console.print("[red]✗ Failed to write processed bag[/red]")
        raise typer.Exit(1)


@app.command("export-tum")
def export_tum(
    workspace: Path = typer.Argument(..., help="Path to workspace directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output TUM file path"),
) -> None:
    """
    Export trajectory in TUM RGB-D format.

    Uses the metric-scaled trajectory if available.
    """
    setup_logging("INFO")

    if not workspace.exists():
        console.print(f"[red]Error: Workspace not found: {workspace}[/red]")
        raise typer.Exit(1)

    from colmap_rgbd_gt.dataset.schema import Workspace
    from colmap_rgbd_gt.export.tum import export_tum as do_export

    ws = Workspace(workspace)

    # Find trajectory files
    metric_traj = ws.outputs / "trajectory_metric_tum.txt"
    unscaled_traj = ws.outputs / "trajectory_colmap_unscaled.txt"

    if metric_traj.exists():
        console.print(f"[green]Metric trajectory already exists: {metric_traj}[/green]")
        return

    if output is None:
        output = ws.outputs / "trajectory_metric_tum.txt"

    console.print(f"[yellow]No metric trajectory found. Run 'scale-depth' first.[/yellow]")
    console.print(f"[yellow]Use 'run-colmap' to generate unscaled trajectory.[/yellow]")


@app.command("full")
def full_pipeline(
    bag_path: Path = typer.Argument(..., help="Path to rosbag file"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Output workspace directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration YAML file"),
    rgb_topic: Optional[str] = typer.Option(None, "--rgb", help="RGB topic name"),
    depth_topic: Optional[str] = typer.Option(None, "--depth", help="Depth topic name"),
    camera_info_topic: Optional[str] = typer.Option(None, "--camera-info", help="Camera info topic name"),
    depth_ba: bool = typer.Option(False, "--depth-ba/--no-depth-ba", help="Run depth-aware bundle adjustment after scale estimation (requires the 'depth-ba' extra)"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level"),
) -> None:
    """
    Run the complete pipeline: extract, COLMAP, scale, and export.

    This is the primary command for generating metric pseudo-GT trajectories.
    """
    setup_logging(log_level, log_file=None)

    if not bag_path.exists():
        console.print(f"[red]Error: Bag file not found: {bag_path}[/red]")
        raise typer.Exit(1)

    if workspace is None:
        workspace = bag_path.with_suffix("")

    # Load config
    config_dict = {}
    if config and config.exists():
        import yaml
        config_dict = yaml.safe_load(config.read_text())

    # Override with CLI options
    if rgb_topic:
        config_dict.setdefault("topics", {})["rgb"] = rgb_topic
    if depth_topic:
        config_dict.setdefault("topics", {})["depth"] = depth_topic
    if camera_info_topic:
        config_dict.setdefault("topics", {})["camera_info"] = camera_info_topic
    if depth_ba:
        config_dict.setdefault("depth_ba", {})["enabled"] = True

    from colmap_rgbd_gt.pipelines.full_pipeline import full_pipeline as run_full

    console.print(f"[bold cyan]Starting full pipeline for: {bag_path}[/bold cyan]")
    console.print(f"[cyan]Workspace: {workspace}[/cyan]")

    success = run_full(bag_path, workspace, config_dict)

    if success:
        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")
        console.print(f"[green]Output: {workspace / 'outputs' / 'trajectory_metric_tum.txt'}[/green]")
    else:
        console.print("\n[bold red]✗ Pipeline failed[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
