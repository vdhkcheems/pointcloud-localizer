"""Point cloud preprocessing utilities."""

from __future__ import annotations

import numpy as np


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "open3d is required for normal estimation. "
            "Install with: pip install 'pointcloud-localizer[open3d]'"
        ) from exc
    return o3d


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Downsample points by voxel centroid aggregation.

    This implementation is custom (not Open3D voxel_down_sample).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}.")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0.")
    if points.shape[0] == 0:
        return points.copy()

    voxel_idx = np.floor(points / voxel_size).astype(np.int64)

    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for idx, point in zip(voxel_idx, points):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        buckets.setdefault(key, []).append(point)

    downsampled = np.array([np.mean(bucket, axis=0) for bucket in buckets.values()], dtype=float)
    return downsampled


def estimate_normals(points: np.ndarray, radius: float, max_nn: int = 30) -> np.ndarray:
    """Estimate normals with Open3D when needed for point-to-plane variants."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}.")
    o3d = _require_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(cloud.normals, dtype=float)
