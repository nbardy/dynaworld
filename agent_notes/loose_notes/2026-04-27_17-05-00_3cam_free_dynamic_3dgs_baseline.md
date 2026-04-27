# 3-Camera Free Dynamic 3DGS Baseline

Date: 2026-04-27

## What Changed

Extended `train_splat_baseline.py` to train on the same train-2-cameras/test-1-camera DeepView split as the gauge-field runner.

The baseline now:

- loads `bundle.train_videos`, `bundle.train_K`, and `bundle.train_w2c` when configured
- samples `(view, frame)` during training
- keeps one shared per-frame 3DGS state for all training cameras
- evaluates the first train camera and the held-out camera as before
- records `train_camera_count`, train camera tensors, and heldout camera names in the checkpoint

Added a matching config:

- `src/train_configs/local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_free_dynamic_3dgs_128_16f_2048splats.jsonc`

Added `free_dynamic_3dgs` to:

- `research_experiments/gauge_fields/run_deepview_3cam_holdout.py`

## Verification

Compile:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/train_splat_baseline.py \
  research_experiments/gauge_fields/run_deepview_3cam_holdout.py
```

Smoke:

```bash
uv run python research_experiments/gauge_fields/run_deepview_3cam_holdout.py \
  --steps 1 \
  --device mps \
  --no-wandb \
  --only free_dynamic_3dgs \
  --output-root /tmp/splat_3cam_runner_smoke
```

Result:

- `eval_psnr`: `19.1329`
- `heldout_eval_psnr`: `13.4893`
- wall clock: `81.92s`

Full 80-step baseline:

```bash
uv run python research_experiments/gauge_fields/run_deepview_3cam_holdout.py \
  --steps 80 \
  --device mps \
  --no-wandb \
  --only free_dynamic_3dgs \
  --output-root outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step_fast_modes
```

Result:

- train cameras: `camera_0001,camera_0015`
- heldout camera: `camera_0040`
- `eval_psnr`: `16.4423`
- `heldout_eval_psnr`: `13.2940`
- wall clock: `184.31s`

Updated summary:

```bash
uv run python research_experiments/gauge_fields/summarize_runs.py \
  outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step_fast_modes \
  --sort-by heldout_eval_psnr \
  --out-md outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step_fast_modes/summary.md \
  --out-json outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step_fast_modes/summary.json
```

## Comparison

Same 128px / 16f / 2048 primitives / 80-step budget:

| model | heldout PSNR | wall clock |
| --- | ---: | ---: |
| `free_dynamic_3dgs` | `13.2940` | `184.31s` |
| `rank_adaptive_metric/projected_conic` | `7.7890` | `478.65s` |
| `screen_disk/projected_conic` | `7.4607` | `206.51s` |

## Takeaway

For this calibrated 3-camera DeepView split, ordinary per-frame 3DGS is the stronger baseline by a wide margin. The gauge-field variants remain useful as representation experiments, but they are not beating the splat control on held-out camera generalization here.

This also changes the next research pressure: if gauges are still worth pursuing, they need either a stronger camera/geometry initialization, more expressive optimization, or a more targeted diagnostic claim. They should not be framed as currently beating dynamic splats on this benchmark.
