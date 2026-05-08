"""Synthetic scene generation for controlled ICP evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation


MISALIGNMENT_LEVELS = {
    "small": (5.0, 0.02),
    "medium": (15.0, 0.10),
    "large": (30.0, 0.25),
}


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "open3d is required for mesh surface sampling. "
            "Install with: pip install 'pointcloud-localizer[open3d]'"
        ) from exc
    return o3d


@dataclass(frozen=True)
class SyntheticPair:
    source: np.ndarray
    target: np.ndarray
    T_gt: np.ndarray


def generate_asymmetric_cloud(num_points: int = 2000, seed: int = 0) -> np.ndarray:
    """
    Generate an asymmetric point cloud to avoid rotational ambiguities.
    """
    rng = np.random.default_rng(seed)
    n1 = num_points // 2
    n2 = num_points // 3
    n3 = num_points - n1 - n2

    cluster1 = np.column_stack(
        [
            rng.normal(0.0, 0.30, size=n1),
            rng.normal(0.6, 0.12, size=n1),
            rng.normal(-0.2, 0.18, size=n1),
        ]
    )
    cluster2 = np.column_stack(
        [
            rng.normal(0.9, 0.08, size=n2),
            rng.normal(-0.4, 0.20, size=n2),
            rng.normal(0.4, 0.10, size=n2),
        ]
    )
    x = rng.uniform(-1.0, 1.0, size=n3)
    cluster3 = np.column_stack([x, 0.25 * x + 0.8, -0.15 * x - 0.5])
    return np.vstack([cluster1, cluster2, cluster3]).astype(float)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform to Nx3 points."""
    points = np.asarray(points, dtype=float)
    T = np.asarray(T, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3).")
    if T.shape != (4, 4):
        raise ValueError("T must be (4, 4).")
    homog = np.hstack([points, np.ones((points.shape[0], 1), dtype=float)])
    transformed = (T @ homog.T).T
    return transformed[:, :3]


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create homogeneous transform from R and t."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def random_transform(
    rng: np.random.Generator,
    max_angle_deg: float,
    max_translation_m: float,
) -> np.ndarray:
    """Sample random source->target transform with bounded rotation and translation."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle_rad = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
    rot = Rotation.from_rotvec(axis * angle_rad).as_matrix()
    trans = rng.uniform(-max_translation_m, max_translation_m, size=3)
    return make_transform(rot, trans)


def add_gaussian_noise(points: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if sigma == 0:
        return points.copy()
    return points + rng.normal(0.0, sigma, size=points.shape)


def sample_mesh_surface(
    mesh_path: str | Path,
    num_points: int,
    noise_sigma: float = 0.0,
    seed: int = 0,
    method: Literal["uniform", "poisson"] = "poisson",
) -> np.ndarray:
    """Sample points from a mesh surface and optionally add Gaussian noise."""
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh path does not exist: {mesh_path}")

    o3d = _require_open3d()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Mesh at {mesh_path} is empty or unreadable.")

    if method == "poisson":
        cloud = mesh.sample_points_poisson_disk(number_of_points=num_points)
    elif method == "uniform":
        cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    else:
        raise ValueError("method must be either 'uniform' or 'poisson'.")

    points = np.asarray(cloud.points, dtype=float)
    rng = np.random.default_rng(seed)
    return add_gaussian_noise(points, noise_sigma, rng)


def generate_synthetic_pair(
    source_points: np.ndarray,
    *,
    noise_sigma: float = 0.0,
    max_angle_deg: float = 15.0,
    max_translation_m: float = 0.10,
    seed: int = 0,
) -> SyntheticPair:
    """
    Generate (source, target, T_gt) with known source->target transform.
    """
    rng = np.random.default_rng(seed)
    source = np.asarray(source_points, dtype=float)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source_points must be (N,3), got {source.shape}")
    if source.shape[0] == 0:
        raise ValueError("source_points is empty.")

    source_noisy = add_gaussian_noise(source, noise_sigma, rng)
    T_gt = random_transform(rng, max_angle_deg=max_angle_deg, max_translation_m=max_translation_m)
    target_clean = transform_points(source_noisy, T_gt)
    target = add_gaussian_noise(target_clean, noise_sigma, rng)
    return SyntheticPair(source=source_noisy, target=target, T_gt=T_gt)


def level_to_misalignment(level: Literal["small", "medium", "large"]) -> tuple[float, float]:
    if level not in MISALIGNMENT_LEVELS:
        raise ValueError(f"Invalid level: {level}. Choose from {tuple(MISALIGNMENT_LEVELS)}")
    return MISALIGNMENT_LEVELS[level]
