"""Evaluation and plotting utilities for point cloud registration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .icp import ICPConfig, run_icp, transform_points
from .synthetic import MISALIGNMENT_LEVELS, generate_synthetic_pair


def compute_transform_errors(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """
    Compare source->target transforms and return:
    - rotation error (degrees)
    - translation error (meters)
    """
    T_est = np.asarray(T_est, dtype=float)
    T_gt = np.asarray(T_gt, dtype=float)
    if T_est.shape != (4, 4) or T_gt.shape != (4, 4):
        raise ValueError("T_est and T_gt must be (4,4).")

    T_err = T_est @ np.linalg.inv(T_gt)
    R_err = T_err[:3, :3]
    t_err = T_err[:3, 3]

    trace_val = float(np.trace(R_err))
    cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos_theta)))
    trans_m = float(np.linalg.norm(t_err))
    return rot_deg, trans_m


def save_rmse_curve(rmse_history: Sequence[float], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(rmse_history) + 1), rmse_history, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("ICP RMSE Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _projection_axes(projection: str) -> tuple[int, int]:
    mapping = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if projection not in mapping:
        raise ValueError("projection must be one of 'xy', 'xz', 'yz'.")
    return mapping[projection]


def save_before_after_projections(
    source: np.ndarray,
    target: np.ndarray,
    source_registered: np.ndarray,
    output_dir: str | Path,
    prefix: str = "registration",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ("X", "Y", "Z")

    for projection in ("xy", "xz", "yz"):
        i, j = _projection_axes(projection)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)

        axes[0].scatter(target[:, i], target[:, j], s=2, alpha=0.65, label="Target")
        axes[0].scatter(source[:, i], source[:, j], s=2, alpha=0.65, label="Source")
        axes[0].set_title(f"Before ({projection.upper()})")
        axes[0].set_xlabel(labels[i])
        axes[0].set_ylabel(labels[j])
        axes[0].legend(markerscale=3)

        axes[1].scatter(target[:, i], target[:, j], s=2, alpha=0.65, label="Target")
        axes[1].scatter(source_registered[:, i], source_registered[:, j], s=2, alpha=0.65, label="Source Registered")
        axes[1].set_title(f"After ({projection.upper()})")
        axes[1].set_xlabel(labels[i])
        axes[1].set_ylabel(labels[j])
        axes[1].legend(markerscale=3)

        plt.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{projection}_before_after.png", dpi=180)
        plt.close(fig)


def run_robustness_sweep(
    base_points: np.ndarray,
    noise_levels: Iterable[float] = (0.0, 0.005, 0.02),
    output_dir: str | Path = "output",
    base_seed: int = 42,
    max_iterations: int = 80,
    tol_rmse: float = 1e-6,
    tol_transform: float = 1e-8,
    distance_threshold: float = 0.08,
    min_correspondences: int = 20,
) -> pd.DataFrame:
    """
    Run 3x3 sweep over noise and misalignment levels.
    Misalignment defaults:
    - small: 5 deg / 0.02 m
    - medium: 15 deg / 0.10 m
    - large: 30 deg / 0.25 m
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str | bool]] = []
    case_id = 0

    for sigma in noise_levels:
        for level_name, (angle_deg, trans_m) in MISALIGNMENT_LEVELS.items():
            pair = generate_synthetic_pair(
                base_points,
                noise_sigma=float(sigma),
                max_angle_deg=float(angle_deg),
                max_translation_m=float(trans_m),
                seed=base_seed + case_id,
            )

            config = ICPConfig(
                max_iterations=max_iterations,
                tol_rmse=tol_rmse,
                tol_transform=tol_transform,
                distance_threshold=distance_threshold,
                min_correspondences=min_correspondences,
                init_transform=np.eye(4),
            )
            result = run_icp(pair.source, pair.target, config=config)
            rot_err_deg, trans_err_m = compute_transform_errors(result.T_est, pair.T_gt)

            rows.append(
                {
                    "noise_sigma_m": float(sigma),
                    "misalignment_level": level_name,
                    "misalignment_angle_deg": float(angle_deg),
                    "misalignment_translation_m": float(trans_m),
                    "iterations": int(result.iterations),
                    "final_rmse": float(result.final_rmse),
                    "rotation_error_deg": float(rot_err_deg),
                    "translation_error_m": float(trans_err_m),
                    "converged": bool(result.converged),
                    "stop_reason": result.stop_reason,
                }
            )
            case_id += 1

    df = pd.DataFrame(rows)
    csv_path = output_dir / "robustness_sweep_summary.csv"
    df.to_csv(csv_path, index=False)

    # Plot: translation error across sweep settings.
    plt.figure(figsize=(8, 4.5))
    for sigma in sorted(df["noise_sigma_m"].unique()):
        sub = df[df["noise_sigma_m"] == sigma]
        order = ["small", "medium", "large"]
        sub = sub.set_index("misalignment_level").loc[order].reset_index()
        plt.plot(sub["misalignment_level"], sub["translation_error_m"], marker="o", label=f"sigma={sigma:.3f}m")
    plt.xlabel("Initial Misalignment")
    plt.ylabel("Translation Error (m)")
    plt.title("Robustness Sweep: Translation Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "robustness_sweep_translation_error.png", dpi=180)
    plt.close()

    return df


def summarize_registration(
    source: np.ndarray,
    target: np.ndarray,
    T_est: np.ndarray,
    rmse_history: Sequence[float],
    output_dir: str | Path,
    prefix: str = "registration",
) -> None:
    """Write core visual artifacts for one registration run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_registered = transform_points(source, T_est)
    save_before_after_projections(source, target, source_registered, output_dir, prefix=prefix)
    save_rmse_curve(rmse_history, output_dir / f"{prefix}_rmse_curve.png")
