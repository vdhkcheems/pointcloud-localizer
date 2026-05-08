"""Custom point-to-point ICP implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform to Nx3 points."""
    points = np.asarray(points, dtype=float)
    T = np.asarray(T, dtype=float)
    homog = np.hstack([points, np.ones((points.shape[0], 1), dtype=float)])
    transformed = (T @ homog.T).T
    return transformed[:, :3]


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate rigid transform T (src->dst) using SVD-based Kabsch alignment."""
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("src and dst must have matching shape (N,3).")
    if src.shape[0] < 3:
        raise ValueError("At least 3 correspondences are required.")

    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)

    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Enforce proper rotation (det(R)=+1).
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = dst_centroid - R @ src_centroid

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass(frozen=True)
class ICPConfig:
    max_iterations: int = 50
    tol_rmse: float = 1e-6
    tol_transform: float = 1e-8
    distance_threshold: Optional[float] = None
    min_correspondences: int = 20
    init_transform: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ICPResult:
    T_est: np.ndarray
    rmse_history: list[float]
    iterations: int
    final_rmse: float
    correspondence_counts: list[int]
    converged: bool
    stop_reason: str


def run_icp(source: np.ndarray, target: np.ndarray, config: Optional[ICPConfig] = None) -> ICPResult:
    """
    Run custom point-to-point ICP where T_est maps source -> target.
    """
    config = config or ICPConfig()
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source must be (N,3), got {source.shape}")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target must be (N,3), got {target.shape}")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("source and target must be non-empty.")
    if config.min_correspondences < 3:
        raise ValueError("min_correspondences must be >= 3.")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be > 0.")

    T_est = np.eye(4, dtype=float) if config.init_transform is None else np.asarray(config.init_transform, dtype=float).copy()
    if T_est.shape != (4, 4):
        raise ValueError("init_transform must be (4,4).")

    tree = cKDTree(target)
    rmse_history: list[float] = []
    corr_counts: list[int] = []
    stop_reason = "max_iterations_reached"
    converged = False

    for _ in range(config.max_iterations):
        src_transformed = transform_points(source, T_est)
        dists, indices = tree.query(src_transformed, k=1)

        valid = np.ones_like(dists, dtype=bool)
        if config.distance_threshold is not None:
            valid &= dists <= config.distance_threshold

        valid_count = int(np.count_nonzero(valid))
        corr_counts.append(valid_count)
        if valid_count < config.min_correspondences:
            stop_reason = "insufficient_correspondences"
            break

        src_corr = src_transformed[valid]
        dst_corr = target[indices[valid]]

        residuals = src_corr - dst_corr
        rmse = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
        rmse_history.append(rmse)

        T_delta = estimate_rigid_transform(src_corr, dst_corr)
        # Source->target convention: compose delta in front.
        T_est = T_delta @ T_est

        delta_norm = float(np.linalg.norm(T_delta - np.eye(4)))
        if delta_norm < config.tol_transform:
            converged = True
            stop_reason = "small_transform_update"
            break

        if len(rmse_history) > 1 and abs(rmse_history[-1] - rmse_history[-2]) < config.tol_rmse:
            converged = True
            stop_reason = "rmse_converged"
            break

    final_rmse = rmse_history[-1] if rmse_history else float("inf")
    return ICPResult(
        T_est=T_est,
        rmse_history=rmse_history,
        iterations=len(rmse_history),
        final_rmse=final_rmse,
        correspondence_counts=corr_counts,
        converged=converged,
        stop_reason=stop_reason,
    )
