# Multicam Full Relpose Feature-Splatting Grid

## Context

The no-VGGT full-relative-pose goodset run was still RGB splatting. Its config
did not set `model.feature_dim`, so the model defaulted to `feature_dim=3` and
`src/train/renderers/fast_mac.py` dispatched through the RGB `v5` path. The F32
feature-splatting path is selected by `model.feature_dim != 3`, which routes to
`v5_features`, returns accumulated alpha, and requires the `colorize` section.

## What changed

- Added F32 sibling config:
  `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc`
- Added 256px F32 sibling config:
  `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc`
- Kept the same goodset split and relpose contract:
  - train cameras: `camera_0006`, `camera_0014`
  - heldout camera: `camera_0005`
  - source/anchor: `camera_0006`
  - `relpose_feature_frame_mode: "first_frame"`
- Enabled feature splatting with:
  - `model.feature_dim: 32`
  - `render.fast_mac.feature_variant: "v5_features"`
  - `render.fast_mac.feature_background: 0.0`
  - `colorize`: no hidden layer, sigmoid, LayerNorm pre-norm, Kaiming init,
    gain `4.0`, no view conditioning
  - `logging.feature_pca_log: true`
- Added a new W&B video key:
  `Multicam_Feature_GT_Render_ByCamera_Grid_Video`

The new grid is arranged as media rows by camera columns:

```text
Feature PCA camera_0006 | Feature PCA camera_0014 | Feature PCA camera_0005
GT camera_0006          | GT camera_0014          | GT camera_0005
Render camera_0006      | Render camera_0014      | Render camera_0005
```

The older multicam diagnostic remains intact:
`Multicam_GT_Splat_Alpha_Feature_Grid_Video`, arranged by camera rows with
`GT | Splat/Pred | Alpha | Feature PCA` columns.

## Rasterizer choice

The F32 configs use `v5_features` explicitly. The trainer also supports
`render.fast_mac.feature_variant: "v6_refined_features"`, and the built
extension is present, but prior F32 measurements did not justify promoting it:
the `v6_refined_features` port passed feature/alpha/reference/trainer smokes,
but the measured F32 active path was not a global speed win. The stronger "v6 is
faster" evidence was for the RGB path (`rgb_variant: "v6_refined"`), not for
feature splatting.

Keep `v5_features` as the default feature-splatting rasterizer until a fresh
same-config 256/512px matrix shows `v6_refined_features` winning on this
multicam relpose workload.

## 512px V6 refined feature config update

User asked whether `v6_refined_features` exists and whether configs already use
it. Current state:

- `local_mac_unconditioned_tokens_features_F32_LN_kaiming_g4_v6_refined_features.jsonc`
  is the only checked-in F32 config that already used
  `render.fast_mac.feature_variant: "v6_refined_features"`.
- `local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_v6_refined_8192splats.jsonc`
  and `local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_v6_refined_8192splats.jsonc`
  use the RGB `v6_refined` rasterizer, not the F32 feature rasterizer.
- The built `v6_refined_features` extension is present for Python 3.11:
  `third_party/fast-mac-gsplat/variants/v6_refined_features/torch_gsplat_bridge_v6_refined_features/_C.cpython-311-darwin.so`.

Added a separate 512px experimental config instead of mutating the 256px
baseline:

```text
src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_512_16f_8192splats_v6refined_goodset_train0006_0014_holdout0005.jsonc
```

Key deltas from the 256px config:

- `model.size: 512`
- `render.render_size: 512`
- `render.fast_mac.feature_variant: "v6_refined_features"`
- active-tile V6 knobs are explicit but defaulted conservative:
  `use_active_tiles: false`, `active_policy: "off"`, and
  `stop_count_mode: "adaptive"`; prior F32 measurements showed active mode was
  not a global win, so A/B `active_policy: "auto"` separately before promoting.
- V-JEPA conditioning already had a projector:
  `PrecomputedVideoFeatureAdapter` does `LayerNorm(768)` +
  `Linear(768 -> model_dim)` after token striding and before cross-attention.
  This config keeps `model_dim: 64` and increases
  `video_feature_token_stride` from `9` to `12` to lighten cross-attention.
- Token capacity shifts to more, slimmer tokens while keeping 8192 splats:
  `tokens: 256`, `static_tokens: 192`, `dynamic_tokens: 64`,
  `gaussians_per_token: 32`.
- Depth increases without widening channels:
  `encoder_self_attn_layers: 2`, `bottleneck_self_attn_layers: 4`,
  `cross_attn_layers: 6`.
- 512px memory risk is handled by `temporal_microbatch_size: 2`.

## Verification

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_pipeline_helpers.py \
  tests/test_multicam_relative_pose_trainer.py \
  tests/test_config_factory_helpers.py -q
```

Result: `20 passed`.

Compile gate:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/pipeline/validation_media.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train/train.py
```

Result: passed.

1-step F32 runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  /tmp/dynaworld_full_relpose_features_f32_smoke.json
```

Smoke details:

- offline W&B run:
  `wandb/offline-run-20260506_191708-mjsn3upv`
- used cached V-JEPA features for the three normal clips and three repeated
  frame-0 relpose clips
- hit the F32 feature path with alpha videos and feature PCA videos
- emitted both multicam grid videos:
  - `Multicam_GT_Splat_Alpha_Feature_Grid_Video`
  - `Multicam_Feature_GT_Render_ByCamera_Grid_Video`
- saved temp checkpoint:
  `/tmp/dynaworld_full_relpose_features_f32_smoke_checkpoint.pt`

Smoke final eval after one step:

- `TrainView0`: PSNR `5.2281`, SSIM `0.0594`
- `TrainView1`: PSNR `4.9073`, SSIM `0.0687`
- `Heldout0_camera_0005`: PSNR `5.1835`, SSIM `0.0657`

256px smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  /tmp/dynaworld_full_relpose_features_f32_256_smoke.json
```

Smoke details:

- offline W&B run:
  `wandb/offline-run-20260506_193708-ufmqp2qd`
- completed one 256px F32 step with `v5_features`
- emitted:
  - `Multicam_GT_Splat_Alpha_Feature_Grid_Video`
  - `Multicam_Feature_GT_Render_ByCamera_Grid_Video`
- saved temp checkpoint:
  `/tmp/dynaworld_full_relpose_features_f32_256_smoke_checkpoint.pt`
- after the smoke, the checked-in 256 config cache path/key was corrected to
  use a 256-specific cache dir and `256-16f` sample key.

256px smoke final eval after one step:

- `TrainView0`: PSNR `4.8759`, SSIM `0.0942`
- `TrainView1`: PSNR `4.5864`, SSIM `0.0847`
- `Heldout0_camera_0005`: PSNR `4.8283`, SSIM `0.0881`

512px V6 refined config tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_multicam_relative_pose_trainer.py \
  tests/test_fast_mac_feature_background.py -q
```

Result: `10 passed`.

512px V6 refined smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  /tmp/dynaworld_full_relpose_features_f32_512_v6_smoke.json
```

Smoke details:

- offline W&B run:
  `wandb/offline-run-20260506_195615-lfv2qd1u`
- baked six 512px V-JEPA feature-cache files under
  `data/feature_cache/multicam_deepview_static_dynamic_vjepa2_1_vitb_384_512px/`
  for the three normal clips and three repeated frame-0 relpose clips
- hit the `v6_refined_features` F32 raster path with 512px input/render size
- model summary printed:
  `1 global camera token + 1 path token + 192 static + 64 dynamic 3DGS tokens x 32 gaussians/token = 8192 explicit Gaussians`
- first training step took `138.45s` wall-clock including 512px validation
  media encoding
- saved temp checkpoint:
  `/tmp/dynaworld_full_relpose_features_f32_512_v6_smoke_checkpoint.pt`

512px V6 smoke initial eval:

- `TrainView0`: PSNR `4.7362`, SSIM `0.1584`
- `TrainView1`: PSNR `4.4139`, SSIM `0.1476`
- `Heldout0_camera_0005`: PSNR `4.7131`, SSIM `0.1696`

512px V6 smoke final eval after one step:

- `TrainView0`: PSNR `5.2769`, SSIM `0.1418`
- `TrainView1`: PSNR `4.8494`, SSIM `0.1359`
- `Heldout0_camera_0005`: PSNR `5.1676`, SSIM `0.1522`

## Caveats

- This commit only adds the F32 config and smoke-verifies the path. It does not
  replace the completed RGB full-relpose run `0pdfypqe`.
- The by-camera feature row uses a joint PCA basis across all displayed camera
  feature tensors, so feature colors are comparable within that grid video.
- The relpose-only follow-up from the RGB run was a heldout-negative result; do
  not assume head-only tuning will help the F32 config without a new objective.
