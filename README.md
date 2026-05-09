# pointcloud-localizer

Tooling for registering overlapping point clouds with a custom ICP loop and
evaluating transform accuracy against known ground truth.

## What this project does

- Loads `.ply` and `.pcd` point clouds (or in-memory numpy point arrays).
- Generates synthetic source/target pairs from mesh samples or asymmetric synthetic shapes.
- Applies custom voxel downsampling preprocessing.
- Runs ICP implemented from scratch (no `open3d.pipelines.registration` calls).
- Reports registration quality:
  - final RMSE,
  - rotation error (degrees),
  - translation error (meters).
- Runs a robustness sweep across:
  - noise sigma: `0.0`, `0.005`, `0.02` m,
  - initial misalignment levels:
    - small: `5° / 0.02 m`,
    - medium: `15° / 0.10 m`,
    - large: `30° / 0.25 m`.

## Transform convention

The project uses one transform convention everywhere:

- `T_gt` maps **source -> target**
- `T_est` maps **source -> target**
- evaluation compares `T_est` against `T_gt`

In ICP, incremental updates compose as:

`T_est = T_delta @ T_est`

## Setup

Python `3.10` to `3.12` is supported for full functionality.
Open3D currently does not provide wheels for Python `3.13+`, so use Python `3.12`
if you need `.ply/.pcd` I/O and mesh surface sampling.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,open3d]"
```

## CLI usage

All commands write artifacts to `output/` by default. Artifacts are generated at runtime.

### 1) Register two real clouds

```bash
pointcloud-localizer register \
  --source path/to/source.ply \
  --target path/to/target.pcd \
  --voxel-size 0.01 \
  --max-iters 100 \
  --distance-threshold 0.08
```

### 2) Synthetic pair registration

Using a mesh:

```bash
pointcloud-localizer synthetic-register \
  --mesh path/to/model.stl \
  --num-points 4000 \
  --noise-sigma 0.005 \
  --gt-angle-deg 15 \
  --gt-translation-m 0.10
```

Without a mesh (uses asymmetric synthetic cloud):

```bash
pointcloud-localizer synthetic-register --num-points 3000 --noise-sigma 0.0
```

### 3) Robustness sweep

```bash
pointcloud-localizer sweep --num-points 3000 --distance-threshold 0.08
```

## Outputs

Generated in `output/`:

- XY/XZ/YZ before/after registration projection plots
- RMSE convergence curve
- estimated transform (`.npy`) and synthetic ground truth transform (`.npy`)
- robustness sweep summary:
  - `robustness_sweep_summary.csv`
  - sweep plot (`robustness_sweep_translation_error.png`)

## Design notes

- **ICP method**: point-to-point with SVD-based rigid alignment (Kabsch/Umeyama style).
- **Correspondence search**: `scipy.spatial.cKDTree`.
- **Outlier handling**: optional max correspondence distance threshold.
- **Degenerate iteration stop**: early stop when valid correspondences drop below a minimum count.
- **Normals**: Open3D normal estimation utility is included for future point-to-plane extension.

## Test report (brief)

Testing was run end-to-end with Python 3.12 and Open3D installed.

- **Unit baseline**: `python -m pytest -q` passed (`1 passed`), verifying the custom ICP loop recovers a known noise-free synthetic transform within `1 degree` rotation and `0.01 m` translation.
- **Single-run synthetic check**: `synthetic-register` in noise-free mode produced near-zero RMSE and an estimated transform matching ground truth.
- **Robustness sweep (single seed)**: all 9 required cases were generated (`sigma in {0, 0.005, 0.02}` x `{small, medium, large}`), with expected degradation in harder conditions.
- **Multi-seed robustness (10 seeds)**:
  - small misalignment: `100%` success across all noise levels,
  - medium misalignment: `70%` (`sigma=0`), `90%` (`sigma=0.005`), `100%` (`sigma=0.02`),
  - large misalignment: `40-50%` success depending on noise.
- **Observed behavior**: as expected for point-to-point ICP, large initial misalignment remains the main failure mode due to local minima/initialization sensitivity.