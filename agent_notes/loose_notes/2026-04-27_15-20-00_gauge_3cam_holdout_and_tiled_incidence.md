# Gauge 3-Camera Holdout and Tiled Incidence Probe

Date: 2026-04-27

## What changed

Added a more honest DeepView validation path for the gauge-field harness:

- train on two calibrated DeepView cameras: `camera_0001,camera_0015`
- hold out a third camera: `camera_0040`
- keep physical time separate from view index, so training samples `(view, t)` rather than flattening cameras into time
- save train camera metadata in checkpoints and expose train/heldout camera columns in summaries

Added a first Torch-side tiled candidate path for ray-Gaussian incidence:

- `render.line_candidate_mode = all_pairs | projected_bbox`
- `projected_bbox` culls line-integral candidates per screen tile using a conservative projected covariance radius
- existing all-pairs behavior remains the default

## Files

- `research_experiments/gauge_fields/train.py`
- `research_experiments/gauge_fields/data.py`
- `research_experiments/gauge_fields/summarize_runs.py`
- `research_experiments/gauge_fields/run_deepview_incidence_matrix.py`
- `research_experiments/gauge_fields/run_deepview_3cam_holdout.py`
- `src/train/multicam_val_data.py`
- `src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_smoke_32_2f_64el.jsonc`
- `src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_128_16f_2048el.jsonc`
- `src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_mass_candidate_tiled_128_16f_2048el.jsonc`

## Verification

Compile:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/data.py \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/summarize_runs.py \
  research_experiments/gauge_fields/run_deepview_incidence_matrix.py \
  research_experiments/gauge_fields/run_deepview_3cam_holdout.py \
  src/train/multicam_val_data.py
```

Tests:

```bash
uv run --with pytest python -m pytest tests/test_gauge_incidence.py
```

Result: `4 passed`.

Scoped whitespace check:

```bash
git diff --check -- \
  research_experiments/gauge_fields/data.py \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/summarize_runs.py \
  research_experiments/gauge_fields/run_deepview_incidence_matrix.py \
  research_experiments/gauge_fields/run_deepview_3cam_holdout.py \
  src/train/multicam_val_data.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_smoke_32_2f_64el.jsonc \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_128_16f_2048el.jsonc \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_mass_candidate_tiled_128_16f_2048el.jsonc
```

Result: passed.

## Smoke Results

Tiny tiled ray-mass smoke:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_incidence_smoke_32_2f_64el.jsonc \
  --device mps \
  --steps 1 \
  --incidence-mode ray_gaussian_line_mass \
  --line-candidate-mode projected_bbox \
  --no-wandb \
  --output-dir /tmp/gauge_line_tiled_mass_smoke
```

Result:

- `eval_psnr`: `5.8330`
- `heldout_eval_psnr`: `5.0674`

Tiny 3-camera train-2/test-1 smoke:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_smoke_32_2f_64el.jsonc \
  --device mps \
  --steps 1 \
  --no-wandb \
  --output-dir /tmp/gauge_3cam_smoke
```

Result:

- train cameras: `camera_0001,camera_0015`
- heldout camera: `camera_0040`
- `heldout_eval_psnr`: `5.5725`

Full-size 3-camera runner smoke, one step, screen-disk control:

```bash
uv run python research_experiments/gauge_fields/run_deepview_3cam_holdout.py \
  --steps 1 \
  --device mps \
  --no-wandb \
  --only screen_disk_projected_conic \
  --output-root /tmp/gauge_3cam_runner_smoke
```

Result:

- wall clock: `109.18s`
- train cameras: `camera_0001,camera_0015`
- heldout camera: `camera_0040`
- `eval_psnr`: `20.4352`
- `heldout_eval_psnr`: `7.3421`

Full-size 3-camera tiled ray-mass one-step probe was stopped after roughly seven minutes without completing:

```bash
uv run python research_experiments/gauge_fields/run_deepview_3cam_holdout.py \
  --steps 1 \
  --device mps \
  --no-wandb \
  --only rank_adaptive_metric_ray_gaussian_line_mass_candidate_tiled \
  --output-root /tmp/gauge_3cam_runner_tiled_smoke
```

Interpretation: the Torch-side tiled path is useful as a correctness/candidate-culling prototype, but the Python tile loop is not a real speed solution. The next performance step is fused rasterization or vectorized tile binning, not more full-size runs through this path.

## Next

Use the 3-camera train-2/test-1 runner as the main representation selector. It is stricter than the old train-one/test-one setup because it pressures the model to fit multiple observed camera fibers while holding out a third calibrated camera.

Do not spend the full benchmark budget on `ray_gaussian_line_mass` through the current Python tiled implementation. If ray-integrated incidence remains interesting, move it into a fused tile rasterizer or a vectorized candidate gather first.
