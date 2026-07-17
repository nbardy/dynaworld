# STAR UVT Real V-JEPA Target Bridge Smoke

Date: 2026-05-19 03:57 ICT

## Goal

Move past the `rgb_pyramid` cached-target contract smoke and run the STAR UVT
feature trainer against real cached V-JEPA tokens.

## Setup

The local checkpoint and torchhub repo were already present:

```text
~/.cache/torch/hub/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
~/.cache/torch/hub/facebookresearch_vjepa2_main
```

Shape probe on the tiny test video:

```text
vjepa_tokens: [1, 1024, 768]
cache: outputs/feature_cache/star_uvt_feature_targets/vjepa2_1_vitb_256crop_8f/d36d341998e50c9ae598bce9.pt
```

The token count maps cleanly to `token_grid_shape=[4,16,16]`.

## Config

```text
src/train_configs/star_uvt_feature_testvideo_8f_64_vjepa_target_gradcache_reduce_vec4_10step.jsonc
```

Important target settings:

```text
feature_target.enabled=true
layer=vjepa_tokens
token_grid_shape=[4,16,16]
channel_adapter=truncate_or_pad
temporal_spatial_adapter=trilinear
normalization=channel_standardize
rgb_loss_weight=0
```

## Benchmark

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_8f_64_vjepa_target_gradcache_reduce_vec4_10step.jsonc
```

Result:

```text
pass=true
loss 1.0008157045 -> 0.9004198760
mean step 181.05ms
mean render forward 92.27ms
mean target-loss prep 9.45ms
mean backward 53.84ms
tile overflow 0
max tile 73/128
```

Gradient flow:

```text
raw_feature, center_uv, center_t, velocity_uv, raw_precision, raw_opacity: present
colorizer: intentionally absent because rgb_loss_weight=0
```

## Output

- `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_8f_64px_512t_f32_gradcache_reduce_vec4_10step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_bridge_smoke.md`

## Interpretation

The real V-JEPA target bridge is now executable at smoke scale. This does not
promote STAR feature tubes as a quality path yet. The selected 512px fast route
is still RGB-target `FeatureToColor` training; the next gate is the same
V-JEPA target contract under the selected no-pre-norm
`feature_direct_gradcache_reduce_vec4` 512px renderer, followed by comparison
to the Gaussian/token precomputed V-JEPA rows.
