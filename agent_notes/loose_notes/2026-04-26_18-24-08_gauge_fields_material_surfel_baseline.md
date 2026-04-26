# Gauge Fields Material-Surfel Baseline

## Context

We tested the new gauge-fields / material-surfel direction against the same
small Dynaworld 128px video used by the recent local overfit baselines:

```text
test_data/test_video_small_128_4fps.mp4
```

The goal was deliberately narrow: prove that a persistent material field with a
T-parameterized low-rank motion basis can overfit the clip before adding real
camera machinery, cheat probes, holonomy, or a production tiled rasterizer.

## What Changed

Added a new experimental harness:

```text
research_experiments/gauge_fields/train.py
research_experiments/gauge_fields/smiley_smoke.py
research_experiments/gauge_fields/README.md
```

Added local JSONC configs:

```text
src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc
src/train_configs/local_mac_gauge_fields_material_surfel_128_16f_512el.jsonc
src/train_configs/local_mac_gauge_fields_material_surfel_128_16f_512el_long.jsonc
src/train_configs/local_mac_gauge_fields_material_surfel_static_128_1f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc
```

The harness reuses Dynaworld's existing JSONC config loader, video sequence
loader, and W&B video helpers, but keeps the renderer simple: pure Torch
projected disks with chunked pixel evaluation.

## Hypotheses

1. The first bad W&B image was probably a capacity / coverage failure, not a
   renderer failure.
2. A one-frame dense coverage gate should pass before a 16-frame motion gate is
   trusted.
3. The stable baseline should be the smallest config that visibly overfits the
   128px video without introducing production renderer work.

## Tests

Renderer-only smiley smoke:

```bash
uv run python research_experiments/gauge_fields/smiley_smoke.py \
  --device cpu \
  --size 96 \
  --frames 8 \
  --pixel-chunk 2048 \
  --output-dir /tmp/gauge_fields_smiley_smoke
```

Preflight train smoke:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc \
  --device cpu \
  --no-wandb \
  --output-dir /tmp/gauge_fields_preflight_smoke
```

One-frame dense gate:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_static_128_1f_2048el.jsonc \
  --device mps \
  --steps 25 \
  --no-wandb \
  --output-dir /tmp/gauge_fields_static_gate_25
```

Sixteen-frame motion gate:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc \
  --device mps \
  --steps 25 \
  --no-wandb \
  --output-dir /tmp/gauge_fields_motion_gate_25
```

Reference W&B overfit:

```bash
WANDB_SILENT=true uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc \
  --device mps \
  --steps 100 \
  --output-dir outputs/gauge_fields/material_surfel_motion_128_16f_2048el_100step
```

## Results

The renderer smoke succeeded. The saved strip columns are:

```text
rgb | alpha | depth
```

The bad 32-element/512-element visual was explained by insufficient coverage:

```text
512 elements, 16f, 25 steps:
  eval_l1: 0.3739
  eval_psnr: 7.56
  alpha_mean: 0.174
  alpha_coverage_050: 0.0
```

The dense one-frame gate passed:

```text
2048 elements, static 1f, 25 steps:
  eval_l1: 0.0457
  eval_psnr: 22.93
  alpha_mean: 0.944
  alpha_coverage_050: 1.0
```

The 16-frame low-rank motion gate passed well enough to become the current
baseline candidate:

```text
2048 elements, motion 16f, 25 steps:
  eval_l1: 0.0913
  eval_psnr: 16.68
  alpha_mean: 0.949
  alpha_coverage_050: 1.0
```

The 100-step reference run improved further:

```text
W&B run: https://wandb.ai/nbardy/dynaworld/runs/ajy46erb
local output: outputs/gauge_fields/material_surfel_motion_128_16f_2048el_100step

eval_l1: 0.0759
eval_mse: 0.0157
eval_psnr: 18.03
alpha_mean: 0.940
alpha_coverage_050: 0.999
model_radius_mean: 0.0727
model_motion_smooth: 0.0123
```

## Conclusion

Yes: we now have a stable baseline of T-parameterized material gauges for the
small 128px video overfit task.

The representation is still crude. It smears texture, has dirty background
behavior, and uses a toy fixed image-plane camera. But the important first gate
is crossed: the renderer and low-rank time parameterization can fit real video
frames rather than only a synthetic smiley.

## Next Steps

1. Promote `local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc`
   as the baseline candidate.
2. Run a 250-500 step baseline on the same config.
3. Add SSIM/DSSIM and temporal motion metrics before comparing against
   Dynaworld Gaussian baselines.
4. Add a small capacity / radius sweep:

```text
elements: 1024, 2048, 4096
radius: 0.07, 0.09
```

5. Keep camera mode fixed until the overfit baseline is stable. Then replace
   the hand-rolled camera with Dynaworld camera helpers.
6. Consider a static background layer or foreground-biased sampling if dirty
   background remains the main visual failure.

