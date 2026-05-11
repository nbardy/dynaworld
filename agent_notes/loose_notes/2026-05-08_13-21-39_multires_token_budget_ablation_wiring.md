# Multires + Token-Budget Ablation Wiring

## Baseline Forked

Forked from:

`src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc`

This is the current strongest local no-VGGT multicam feature-splatting setup:

- DeepView `03_Dog`
- train cameras `camera_0006`, `camera_0014`
- heldout camera `camera_0005`
- V-JEPA 2.1 ViT-B precomputed features at 256px
- F32 feature splatting through fast-mac `v5_features`
- `alpha_threshold = 1/128`
- random RGB training background / white eval background
- full relative-pose head with nonzero `relpose_output_init_std = 0.12`

## What Changed

Implemented two opt-in ablation surfaces without changing the baseline config behavior.

1. Trainer-level multi-resolution training:
   - `train.multires_render_sizes` samples the render/loss viewport per train step.
   - `train.multires_render_probabilities` optionally gives weighted sampling instead of uniform sampling.
   - The model input/features remain at the baseline config size.
   - Validation media stays at the base `render.render_size` so heldout videos remain comparable.
   - Scalars log `RenderSize`, `Render/BaseSize`, and multires status.

2. Token-layout / dynamic decoded-token budget:
   - `model.token_layout` lets `model.tokens` include non-decoded world/register/detail tokens.
   - Decoded splat tokens are selected from named static/dynamic core and detail spans.
   - `train.multires_token_detail_levels` optionally maps each sampled render size to an active detail level.
   - The all-up config uses 72 decoded tokens at 128px, 104 at 192px, and 128 at 256px, while keeping 4 world tokens, 2 register tokens, and 2 detail delimiter/register tokens in the query bank.

## Configs Added

- `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_1024_1920_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc`
  - pure multires control: `[64, 128, 256, 512, 1024, 1920]`
  - weighted probabilities: `[0.20, 0.40, 0.20, 0.10, 0.05, 0.05]`

- `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_1024_1920_tokenbudget_world4_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc`
  - all-up multires + world/register/detail token layout
  - `model.tokens = 136`
  - max decoded capacity stays `96 static + 32 dynamic`
  - weighted probabilities: `[0.20, 0.40, 0.20, 0.10, 0.05, 0.05]`
  - active decoded token budgets by train render size: `64px -> 72`, `128px -> 72`, `256px -> 104`, `512px -> 128`, `1024px -> 128`, `1920px -> 128`

## Bugs Caught While Wiring

- The base config resolver assumed `static_tokens + dynamic_tokens == model.tokens`. That is correct for legacy layouts, but wrong once `model.tokens` includes non-decoded query tokens. The resolver now allows token-layout configs to keep decoded capacity below total query-token count.
- Renderer mode selection and training startup logs assumed every token becomes splats. They now count active decoded tokens when a token layout is present.
- Browser export sliced static/dynamic tokens contiguously after the camera tokens. That would export world/register tokens as static splat tokens. Export now asks the model for decoded static/dynamic query-token tensors.

## Verification

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_config_factory_helpers.py \
  tests/test_multicam_relative_pose_trainer.py -q
```

Result: `24 passed`.

All-up runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train.py /tmp/dynaworld_multires_tokenbudget_smoke.json
```

Result: passed. This hit cached V-JEPA features, MPS model construction,
multires render schedule `{128: 0, 192: 1, 256: 2}`, validation media
logging, and one train step. Offline W&B run:

`wandb/offline-run-20260508_132357-02jmh5rx`

Pure multires runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train.py /tmp/dynaworld_multires_smoke.json
```

Result: passed. This exercises the legacy static/dynamic layout with
`train.multires_render_sizes` but no token-detail schedule. Offline W&B run:

`wandb/offline-run-20260508_132611-anbj60id`

Compile smoke:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/gs_models/dynamic_video_token_gs_implicit_camera.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train/export_dynaworld_browser_bundle.py \
  src/train/model_factories.py
```

Result: passed.

## Follow-Ups

- If the all-up path trains cleanly, add a `BASELINES.md` row only after a real run with heldout metrics and validation media.
- The current dynamic budget saves splat decode/raster work at lower resolutions but still cross-attends all query tokens. A later speed ablation can add active-query pruning if quality is acceptable.
- `1920px` is likely an expensive curriculum outlier on the current 256px feature/target load path: the loader materializes camera frames at `model.size=256`, so 1920px loss targets are upsampled from 256px unless we add a separate high-res frame target path.
