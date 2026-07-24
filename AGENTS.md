# Agent Guidelines for colmap-rgbd-gt

This file provides guidance for agentic coding assistants operating in this repository.

## Repository Overview

**colmap-rgbd-gt** is a Docker-based pipeline for converting rosbag recordings to metric pseudo-ground-truth trajectories. It extracts RGB/depth data from rosbags, runs COLMAP reconstruction, and estimates metric scale to produce TUM-format trajectories.

- **Language**: Python 3.12+
- **Package Manager**: `uv` (preferred) or `pip`
- **Location**: All source code under `src/colmap_rgbd_gt/`
- **CLI Entry Point**: `gttool` (defined in pyproject.toml)

---

## Build, Lint, and Test Commands

### Development Environment Setup

```bash
# Install package with dev dependencies
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Code Quality Tools

```bash
# Run all linters (ruff, black, isort, mypy)
make lint

# Run ruff linter only
uv run ruff check src/ tests/

# Auto-fix ruff issues
uv run ruff check --fix src/ tests/

# Format with black
uv run black src/ tests/

# Check import order with isort
uv run isort --check-only src/ tests/

# Type checking with mypy
uv run mypy src/

# All format + lint (CI equivalent)
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_scale_estimation.py -v

# Run a single test function
uv run pytest tests/test_scale_estimation.py::test_umeyama_3D -v

# Run tests matching a pattern
uv run pytest tests/ -k "scale" -v

# Run with coverage
uv run pytest tests/ --cov=colmap_rgbd_gt --cov-report=term-missing

# Run tests in Docker (if COLMAP dependencies needed)
cd docker && podman compose run --rm dev pytest tests/ -v
```

### CLI Usage

```bash
# Inspect a rosbag
uv run gttool inspect-bag /path/to/bag.bag

# Run full pipeline
uv run gttool full /path/to/bag.bag --config configs/default.yaml --workspace /path/to/workspace

# Run individual pipeline stages
uv run gttool extract /path/to/bag.bag --workspace /path/to/workspace
uv run gttool run-colmap /path/to/workspace
uv run gttool scale-depth /path/to/workspace
```

---

## Code Style Guidelines

### Formatting

- **Line Length**: 100 characters (configured in black, ruff, mypy)
- **Indentation**: 4 spaces (no tabs)
- **String Quotes**: Prefer double quotes for strings; single quotes only for simple cases
- **Line Endings**: LF ( enforced via .gitattributes)

### Import Organization

Imports must be grouped in this order with blank lines between groups:

1. **Standard library** (` pathlib`, `logging`, `json`, etc.)
2. **Third-party packages** (`numpy`, `cv2`, `yaml`, `typer`, `rich`)
3. **Local application imports** (`colmap_rgbd_gt.*`)

```python
# Correct import order
from pathlib import Path
from typing import Any
import numpy as np
import cv2
import yaml
from rich.console import Console
from colmap_rgbd_gt.logging import get_logger
from colmap_rgbd_gt.dataset.schema import Workspace
```

### Type Annotations

- **Required**: All function parameters and return values MUST have type annotations (`disallow_untyped_defs: true` in mypy config)
- **Use `|` for union types**: `str | Path` not `Union[str, Path]` (Python 3.10+ union syntax)
- **Optional types**: `str | None` not `Optional[str]`
- **Use concrete types when possible**: `list[str]` not `List[str]`

```python
# Good
def process_frame(frame_id: int, timestamp_ns: int, config: dict[str, Any]) -> bool:
    ...

# Bad - missing types
def process_frame(frame_id, timestamp_ns, config):
    ...
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `bag_reader.py` |
| Classes | `PascalCase` | `class BagReader` |
| Functions/methods | `snake_case` | `def get_messages()` |
| Variables | `snake_case` | `rgb_topic` |
| Constants | `UPPER_SNAKE` | `MAX_DEPTH_METERS = 10.0` |
| Type aliases | `PascalCase` | `RGBFrame = tuple[int, np.ndarray]` |
| Private attrs/methods | `_leading_underscore` | `self._reader` |

### Docstrings

Use Google-style docstrings with parameter/return descriptions:

```python
def extract_rgb_frames(
    reader: BagReader,
    output_dir: Path,
    topic: str,
    config: dict[str, Any]
) -> list[tuple[int, str]]:
    """
    Extract RGB frames from a rosbag.

    Args:
        reader: Bag file reader instance.
        output_dir: Directory to save extracted frames.
        topic: ROS topic name for RGB images.
        config: Configuration dict with export settings.

    Returns:
        List of (timestamp_ns, frame_path) tuples sorted by timestamp.

    Raises:
        ValueError: If topic is not found in bag.
        IOError: If write fails.
    """
```

### Error Handling

- Use structured logging via `get_logger(__name__)` instead of print statements
- Raise specific exceptions with clear messages
- Let exceptions propagate from internal functions; catch only at boundary layers
- Never swallow exceptions silently

```python
# Good - specific error with context
if not bag_path.exists():
    raise FileNotFoundError(f"Bag file not found: {bag_path}")

# Good - log and return failure indicator
logger.error(f"Extraction failed: {e}")
return False

# Bad - bare except
except:
    pass

# Bad - catching too broad
except Exception:
    logger.error("Something went wrong")
```

### Logging Patterns

```python
# Get logger at module level
logger = get_logger(__name__)

# Use appropriate log levels
logger.debug("Exported RGB frame %d", frame_id)  # Detailed debug info
logger.info("Extraction complete: %d frames", count)  # Normal operations
logger.warning("Depth topic not found, skipping")  # Recoverable issues
logger.error("Failed to open bag: %s", e)  # Errors
```

### File Organization

- One class per file for major classes (exception: tightly coupled small classes)
- Keep modules focused: `bag_reader.py` handles rosbag I/O, `scale_estimation.py` handles scaling
- Private implementation details in dedicated `_impl.py` or as private functions when simple
- Constants grouped at module top after imports

---

## ROS Bags API Notes

The codebase uses `rosbags>=0.11` with these API specifics:

```python
# Correct imports for rosbags 0.11+
from rosbags.typesys.stores import Stores, get_typestore
from rosbags.typesys.stores.ros1_noetic import sensor_msgs__msg__Image as ImageV1

# Getting a typestore
typestore = get_typestore(Stores.ROS1_NOETIC)  # ROS1
typestore = get_typestore(Stores.ROS2_HUMBLE)  # ROS2

# Deserializing messages
msg = typestore.deserialize_ros1(rawdata, msgtype)  # ROS1
msg = typestore.deserialize_cdr(rawdata, msgtype)   # ROS2
```

---

## Testing Guidelines

- Test files in `tests/` with `test_*.py` naming
- Use `pytest` fixtures for shared setup
- Mock external dependencies (rosbag files, COLMAP binary)
- Focus tests on transformation logic, not I/O
- Aim for meaningful assertions over just checking "no exception"

```python
def test_scale_estimation_with_valid_points():
    """Umeyama should return valid scale factor."""
    result = estimate_scale(points_3d, points_colmap)
    assert result.scale > 0
    assert result.scale < 1000  # Reasonable bounds
    assert result.rmse < 0.1  # Good fit
```

---

## Common Patterns

### Path Handling

```python
# Always use Path for file paths, not strings
from pathlib import Path

def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())

# Use / operator for joining paths
output_path = workspace / "outputs" / "trajectory.txt"
```

### Configuration Dicts

Configuration flows through the pipeline as `dict[str, Any]`. Access nested keys safely:

```python
images_config = config.get("images", {})
min_frame_stride = images_config.get("min_frame_stride", 1)
```

### Working with NumPy

```python
import numpy as np

# Type numpy arrays properly in annotations
def process_depth(depth: np.ndarray) -> np.ndarray:
    """Process depth map to meters."""
    depth_meters = depth.astype(np.float32) * 0.001
    return np.clip(depth_meters, 0.2, 8.0)
```
