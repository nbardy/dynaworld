# STAR UVT RGB-Pyramid Target Bridge Smoke

Date: 2026-05-19 03:44 ICT

## Goal

After the bridge audit proved that the selected fast STAR feature route was
still RGB-target `FeatureToColor` training, add the smallest executable
cached-feature target path and benchmark it.

## Code Change

Updated `src/train/train_star_uvt_feature_overfit.py` with an opt-in
`feature_target` section:

- existing configs keep the old RGB reconstruction behavior
- `feature_target.enabled=true` requires a top-level `features` config
- the trainer loads the video once as `SequenceData`
- `VideoFeatureCache` loads or bakes the configured feature layer
- cached tensors are adapted into `[T,F,H,W]`
- token-shaped cached tensors require explicit
  `feature_target.token_grid_shape=[T,H,W]`
- the training loss can target `render.feature_image` directly
- `rgb_loss_weight=0` disables colorizer gradients intentionally

Added smoke config:

```text
src/train_configs/star_uvt_feature_testvideo_8f_64_rgbpyramid_target_gradcache_reduce_vec4_10step.jsonc
```

## Benchmark

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_8f_64_rgbpyramid_target_gradcache_reduce_vec4_10step.jsonc
```

First run baked:

```text
outputs/feature_cache/star_uvt_feature_targets/rgb_pyramid/f6ab7b0e079ec0072d8ad2fa.pt
```

Second run hit the cache and passed:

```text
pass=true
loss 0.3400550410 -> 0.2480939664
mean step 93.55ms
mean render forward 18.38ms
mean target-loss prep 6.67ms
mean backward 42.99ms
tile overflow 0
max tile 75/128
```

Target adapter:

```text
rgb_x1 source [1,3,8,64,64] -> repeated/truncated target [8,32,64,64]
synthetic token-grid adapter smoke: [1,4,3] + token_grid_shape [1,2,2] -> [1,3,2,2]
```

Gradient flow:

```text
raw_feature, center_uv, center_t, velocity_uv, raw_precision, raw_opacity: present
colorizer: intentionally absent because rgb_loss_weight=0
```

## Output

- `outputs/benchmarks/2026-05-19_star_uvt_feature_rgbpyramid_target_8f_64px_512t_f32_gradcache_reduce_vec4_10step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_rgbpyramid_target_bridge_smoke.md`
- RGB compatibility smoke after the shared `SequenceData` loader change:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_rgb_compat_8f_64_directatomic_2step.json`
  (`pass=true`, loss `0.1860204972 -> 0.1191769093`, colorizer gradients
  present, zero overflow).

## Interpretation

The cached-feature bridge is now real at smoke scale. This does not make the
fast route a V-JEPA run yet. The next gate is a real V-JEPA target config with
an explicit token-grid adapter and a comparison against the existing
Gaussian/token precomputed-feature rows.
