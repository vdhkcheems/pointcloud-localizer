"""CLI for pointcloud-localizer workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .evaluate import compute_transform_errors, run_robustness_sweep, summarize_registration
from .icp import ICPConfig, run_icp
from .loader import load_point_cloud
from .preprocess import voxel_downsample
from .synthetic import (
    generate_asymmetric_cloud,
    generate_synthetic_pair,
    sample_mesh_surface,
)


def _load_init_transform(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.load(path)
    if arr.shape != (4, 4):
        raise ValueError("Initial transform file must contain a (4,4) matrix.")
    return arr


def _common_icp_config(args: argparse.Namespace, init_transform: np.ndarray | None = None) -> ICPConfig:
    return ICPConfig(
        max_iterations=args.max_iters,
        tol_rmse=args.tol_rmse,
        tol_transform=args.tol_transform,
        distance_threshold=args.distance_threshold,
        min_correspondences=args.min_correspondences,
        init_transform=init_transform,
    )


def _prepare_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size > 0:
        return voxel_downsample(points, voxel_size=voxel_size)
    return points


def run_register(args: argparse.Namespace) -> None:
    source = load_point_cloud(args.source)
    target = load_point_cloud(args.target)
    source = _prepare_points(source, args.voxel_size)
    target = _prepare_points(target, args.voxel_size)

    config = _common_icp_config(args, init_transform=_load_init_transform(args.init_transform_npy))
    result = run_icp(source, target, config=config)

    output_dir = Path(args.output_dir)
    summarize_registration(source, target, result.T_est, result.rmse_history, output_dir, prefix="register")
    np.save(output_dir / "register_estimated_transform.npy", result.T_est)

    print("ICP implementation: custom from-scratch loop (no external ICP solver).")
    print(f"Iterations: {result.iterations}")
    print(f"Final RMSE: {result.final_rmse:.6f}")
    print(f"Converged: {result.converged} ({result.stop_reason})")
    print("Estimated Transform (source -> target):")
    print(result.T_est)


def run_synthetic_register(args: argparse.Namespace) -> None:
    if args.mesh is not None:
        base = sample_mesh_surface(
            args.mesh,
            num_points=args.num_points,
            noise_sigma=0.0,
            seed=args.seed,
            method=args.mesh_sample_method,
        )
    else:
        base = generate_asymmetric_cloud(num_points=args.num_points, seed=args.seed)

    pair = generate_synthetic_pair(
        base,
        noise_sigma=args.noise_sigma,
        max_angle_deg=args.gt_angle_deg,
        max_translation_m=args.gt_translation_m,
        seed=args.seed,
    )

    source = _prepare_points(pair.source, args.voxel_size)
    target = _prepare_points(pair.target, args.voxel_size)

    config = _common_icp_config(args, init_transform=_load_init_transform(args.init_transform_npy))
    result = run_icp(source, target, config=config)

    rot_err, trans_err = compute_transform_errors(result.T_est, pair.T_gt)
    output_dir = Path(args.output_dir)
    summarize_registration(source, target, result.T_est, result.rmse_history, output_dir, prefix="synthetic")
    np.save(output_dir / "synthetic_estimated_transform.npy", result.T_est)
    np.save(output_dir / "synthetic_ground_truth_transform.npy", pair.T_gt)

    print("ICP implementation: custom from-scratch loop (no external ICP solver).")
    print(f"Iterations: {result.iterations}")
    print(f"Final RMSE: {result.final_rmse:.6f}")
    print(f"Converged: {result.converged} ({result.stop_reason})")
    print(f"Rotation error (deg): {rot_err:.4f}")
    print(f"Translation error (m): {trans_err:.6f}")
    print("Estimated Transform (source -> target):")
    print(result.T_est)
    print("Ground Truth Transform (source -> target):")
    print(pair.T_gt)


def run_sweep(args: argparse.Namespace) -> None:
    if args.mesh is not None:
        base = sample_mesh_surface(
            args.mesh,
            num_points=args.num_points,
            noise_sigma=0.0,
            seed=args.seed,
            method=args.mesh_sample_method,
        )
    else:
        base = generate_asymmetric_cloud(num_points=args.num_points, seed=args.seed)

    base = _prepare_points(base, args.voxel_size)
    df = run_robustness_sweep(
        base_points=base,
        noise_levels=(0.0, 0.005, 0.02),
        output_dir=args.output_dir,
        base_seed=args.seed,
        max_iterations=args.max_iters,
        tol_rmse=args.tol_rmse,
        tol_transform=args.tol_transform,
        distance_threshold=args.distance_threshold if args.distance_threshold is not None else 0.08,
        min_correspondences=args.min_correspondences,
    )

    print("Robustness sweep complete.")
    print("Saved summary CSV and plots to:", args.output_dir)
    print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pointcloud Localizer (custom ICP).")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", default="output", help="Directory for generated outputs.")
        p.add_argument("--voxel-size", type=float, default=0.0, help="Voxel size for downsampling (0 to disable).")
        p.add_argument("--max-iters", type=int, default=80, help="Max ICP iterations.")
        p.add_argument("--tol-rmse", type=float, default=1e-6, help="RMSE convergence tolerance.")
        p.add_argument("--tol-transform", type=float, default=1e-8, help="Transform update convergence tolerance.")
        p.add_argument("--distance-threshold", type=float, default=None, help="Optional max correspondence distance.")
        p.add_argument("--min-correspondences", type=int, default=20, help="Minimum valid correspondences to continue.")
        p.add_argument(
            "--init-transform-npy",
            type=str,
            default=None,
            help="Optional .npy file containing initial 4x4 source->target transform.",
        )

    reg = sub.add_parser("register", help="Register two input clouds (.ply/.pcd).")
    reg.add_argument("--source", required=True, help="Source cloud path (.ply/.pcd).")
    reg.add_argument("--target", required=True, help="Target cloud path (.ply/.pcd).")
    add_common_flags(reg)
    reg.set_defaults(func=run_register)

    syn = sub.add_parser("synthetic-register", help="Generate synthetic pair and run registration.")
    syn.add_argument("--mesh", default=None, help="Optional mesh path (OBJ/PLY/STL).")
    syn.add_argument("--mesh-sample-method", default="poisson", choices=["poisson", "uniform"])
    syn.add_argument("--num-points", type=int, default=3000)
    syn.add_argument("--noise-sigma", type=float, default=0.005, help="Noise sigma in meters.")
    syn.add_argument("--gt-angle-deg", type=float, default=15.0, help="Ground-truth max rotation magnitude.")
    syn.add_argument("--gt-translation-m", type=float, default=0.10, help="Ground-truth max translation magnitude.")
    syn.add_argument("--seed", type=int, default=42)
    add_common_flags(syn)
    syn.set_defaults(func=run_synthetic_register)

    sweep = sub.add_parser("sweep", help="Run robustness sweep over noise and misalignment levels.")
    sweep.add_argument("--mesh", default=None, help="Optional mesh path (OBJ/PLY/STL) for base cloud.")
    sweep.add_argument("--mesh-sample-method", default="poisson", choices=["poisson", "uniform"])
    sweep.add_argument("--num-points", type=int, default=3000)
    sweep.add_argument("--seed", type=int, default=42)
    add_common_flags(sweep)
    sweep.set_defaults(func=run_sweep)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
