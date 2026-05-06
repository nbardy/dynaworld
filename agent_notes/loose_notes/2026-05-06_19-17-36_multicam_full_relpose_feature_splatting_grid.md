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

## Caveats

- This commit only adds the F32 config and smoke-verifies the path. It does not
  replace the completed RGB full-relpose run `0pdfypqe`.
- The by-camera feature row uses a joint PCA basis across all displayed camera
  feature tensors, so feature colors are comparable within that grid video.
- The relpose-only follow-up from the RGB run was a heldout-negative result; do
  not assume head-only tuning will help the F32 config without a new objective.
