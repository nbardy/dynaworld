# Best Local Baseline Tweaks

Current as of 2026-04-27.

This is the single operational index for the best tiny local Dynaworld baseline
recipe. Raw chronology belongs in `agent_notes/loose_notes/`; surprising durable
lessons belong in `agent_notes/key_learnings.md`.

## Current Best Run

W&B:

```text
https://wandb.ai/nbardy/dynaworld/runs/mybv736f
```

Run name:

```text
ablate-time-static-dynamic-96-32-crossattn4-precomputed-vjepa2-1-vitb-384-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats-1000step
```

Important correction: the config requested 1000 steps, but the strong visual
result was observed after the step-500 video checkpoint and the run was
interrupted around step 525. Treat it as the best `~500 step` result so far, not
as proof that the full 1000-step schedule has converged.

Latest synced metrics:

```text
Eval/Loss                     0.0547
Eval/L1                       0.0413
Eval/SSIM                     0.7836
Eval/PSNR                     23.69
Eval/TemporalAdjacentL1Ratio  0.8009
Eval/DecodedXYZAdjacentL2     0.1305
Camera/EvalAdjacentRotDeg     0.1827
```

## Rerun Command

Preferred launcher:

```bash
./src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh 1000
```

Direct command:

```bash
PYTHONPATH=src/train uv run python src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc
```

## Checked-In Config

```text
src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc
```

The 250-step sibling is useful for quick comparisons:

```text
src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc
```

## Recipe

Data:

```text
source             test_data/test_video_small_128_4fps.mp4
loaded frames      16
input/render size  128
frame source       explicit_video
```

Feature conditioning:

```text
backend       precomputed
extractor     vjepa_torchhub
model         vjepa2_1_vit_base_384
crop          384
feature dim   768
cache tensor  vjepa_tokens [1, 4608, 768], fp16
```

Cache:

```text
data/feature_cache/ablate_time_static_dynamic_vjepa2_1_vitb_384/b6ba09206f179d4c2cc29d52.pt
```

Model:

```text
variant                  learned_time_orbit_path
tokens                   128
static_tokens            96
dynamic_tokens           32
gaussians_per_token      64
explicit gaussians       8192
model_dim                128
bottleneck_dim           256
num_heads                8
cross_attn_layers        4
dynamic_time_basis_count 8
```

Initialization:

```text
rgb_init                 uniform
query_token_init_std     0.8
head_output_init_std     0.12
scale_init               0.02
scale_init_log_jitter    0.7
opacity_init             0.1
position coverage        0.9
rotation_init            random
```

Camera and renderer:

```text
camera global head        legacy_orbit
lens model                pinhole
base fov                  60 deg
base radius               3.0
renderer                  fast_mac
recon backward strategy   batched
temporal microbatch       4
```

Loss:

```text
l1_weight                 0.8
dssim_weight              0.2
mse_weight                0.2
camera_motion_weight      0.01
camera_temporal_weight    0.02
camera_global_weight      0.005
dynamic rate losses       0.0
```

## What Made It Work

The best current explanation is a three-part interaction:

1. The 96/32 static/dynamic split gives the model a stable scene bank plus a
   small low-rank dynamic bank. This worked better than asking every token to
   be equally time-varying.
2. V-JEPA 2.1 features make query cross-attention useful. Four cross-attention
   layers with only the local encoder froze or underfit; four layers over a
   rich 4608-token V-JEPA memory became productive.
3. Longer optimization mattered. The 250-step V-JEPA split was already the best
   run, but the step-500 checkpoint was the first one that looked visually close
   to a clean fit.

## Evidence Ladder

```text
local static/dynamic:
  Eval/Loss 0.1195, SSIM 0.4287, temporal adjacent ratio 0.3408

V-JEPA static/dynamic, 250 steps:
  Eval/Loss 0.0881, SSIM 0.6109, temporal adjacent ratio 0.6322

V-JEPA static/dynamic, ~525 steps:
  Eval/Loss 0.0547, SSIM 0.7836, temporal adjacent ratio 0.8009
```

This supports the current working belief:

```text
V-JEPA features make the scene readable.
Static/dynamic split gives the decoder the right motion bias.
Longer training sharpens the same-source fit.
```

## Caveats

This is still a same-source/same-view overfit result. It is the right local
baseline to protect, but it does not prove novel-view or held-out-scene
generalization.

Camera motion increased in the best run:

```text
local static/dynamic camera adjacent rotation: 0.0159 deg
250-step V-JEPA split camera adjacent rotation: 0.1309 deg
~525-step V-JEPA split camera adjacent rotation: 0.1827 deg
```

That can be useful, but it also means some visual quality may come from learned
camera path flexibility. Validate against held-out clips and multi-camera
same-time novel-angle ground truth before making general claims.

## Next Tests

1. Finish the full 1000-step run and compare step 250 / 500 / 1000 media.
2. Repeat the same recipe on a scene-distinct train/eval clip from the local
   20/10 dataset.
3. Run the recipe on multi-camera validation once unified camera objects are
   plumbed into the loader.
4. Add a control that keeps the V-JEPA split but clamps/reduces camera motion,
   to separate real dynamic splat motion from camera-path compensation.
5. Add clip-aware feature caching before using this recipe on longer videos;
   current prebake operates on the whole loaded `SequenceData`.
