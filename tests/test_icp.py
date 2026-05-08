import numpy as np

from pointcloud_localizer.evaluate import compute_transform_errors
from pointcloud_localizer.icp import ICPConfig, run_icp
from pointcloud_localizer.synthetic import generate_asymmetric_cloud, generate_synthetic_pair


def test_icp_recovers_noise_free_transform_within_thresholds() -> None:
    base = generate_asymmetric_cloud(num_points=2500, seed=7)
    pair = generate_synthetic_pair(
        base,
        noise_sigma=0.0,
        max_angle_deg=8.0,
        max_translation_m=0.03,
        seed=11,
    )

    config = ICPConfig(
        max_iterations=120,
        tol_rmse=1e-10,
        tol_transform=1e-12,
        distance_threshold=None,
        min_correspondences=100,
        init_transform=np.eye(4),
    )
    result = run_icp(pair.source, pair.target, config=config)
    rot_err_deg, trans_err_m = compute_transform_errors(result.T_est, pair.T_gt)

    assert result.iterations > 0
    assert result.final_rmse < 1e-3
    assert result.rmse_history[-1] <= result.rmse_history[0]
    assert rot_err_deg < 1.0
    assert trans_err_m < 0.01
