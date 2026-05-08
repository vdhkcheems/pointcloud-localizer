"""Point cloud loading and saving utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, list[list[float]], tuple[tuple[float, float, float], ...]]


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "open3d is required for file I/O (.ply/.pcd). "
            "Install with: pip install 'pointcloud-localizer[open3d]'"
        ) from exc
    return o3d


def _validate_points(points: np.ndarray, *, name: str = "points") -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}.")
    if points.shape[0] == 0:
        raise ValueError(f"{name} is empty.")
    return points


def load_point_cloud(data: Union[str, Path, ArrayLike]) -> np.ndarray:
    """Load point cloud from .ply/.pcd file path or in-memory array."""
    if isinstance(data, (np.ndarray, list, tuple)):
        return _validate_points(np.asarray(data, dtype=float), name="in-memory cloud")

    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud path not found: {path}")
    if path.suffix.lower() not in {".ply", ".pcd"}:
        raise ValueError("Supported point cloud formats are .ply and .pcd.")

    o3d = _require_open3d()
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=float)
    return _validate_points(points, name=f"cloud at {path}")


def save_point_cloud(points: np.ndarray, path: Union[str, Path]) -> None:
    """Save point cloud to .ply/.pcd using Open3D."""
    path = Path(path)
    if path.suffix.lower() not in {".ply", ".pcd"}:
        raise ValueError("Output format must be .ply or .pcd.")

    points = _validate_points(points, name="points to save")
    o3d = _require_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    ok = o3d.io.write_point_cloud(str(path), cloud)
    if not ok:
        raise RuntimeError(f"Failed to save point cloud to {path}")
