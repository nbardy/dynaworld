# Multicam Full Relpose No-VGGT Setup

## What changed

- Added `src/train/train_multicam_relative_pose_implicit_dynamic.py`, a separate trainer for
  `arch: "multicam_relative_pose_implicit_camera"`.
- Kept the old learned-residual trainer/config intact. The old path is still
  "calibrated source-relative camera plus learned residual"; the new path is
  "predict the full source-relative query camera transform."
- Registered the new arch in `src/train/train.py`.
- Added full-pose helpers in `src/train/relative_pose.py`:
  - `cameras_with_se3_transform(...)`
  - `se3_transform_l2_loss(...)`
- Added `train.relpose_feature_frame_mode`. The checked-in configs use
  `first_frame`, which repeats each camera's global frame 0 before feature
  extraction so the pose head gets initial-frame query evidence rather than
  full-clip query evidence.
- Added focused tests for full-pose helpers, heldout pair semantics, dispatch,
  and config validation.

## Goodset configs

- Joint run:
  `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc`
- Relpose-only follow-up:
  `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_offsetonly_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc`

Both configs use the overlap-goodset split:

- train cameras: `camera_0006`, `camera_0014`
- heldout camera: `camera_0005`
- anchor/condition/source default: `camera_0006`

The joint config saves:

`outputs/multicam_relative_pose/full_relpose_goodset_train0006_0014_holdout0005/checkpoint_final.pt`

The offset-only config loads that checkpoint, freezes the splat/world model,
colorizer, and camera rig, and trains only the relative-pose head. It refuses
to resolve without a checkpoint path or without cross-camera pairs, because
freezing around random splats or self-only pairs is not a meaningful offset fit.

## Training contract

For a train pair `(source, query)`:

```text
W_source = splat decoder(projected features(source))
Delta_source_to_query = relpose(projected features(source), projected features(query))
render(W_source, Delta_source_to_query) -> GT(query)
```

With `relpose_feature_frame_mode: "first_frame"`, the head consumes features
from frame 0 repeated to the configured feature clip length for the source and
query cameras. The head outputs one static source-relative `camera_to_world`
transform for the rendered clip.
Train supervision uses only train cameras, via:

- reconstruction loss through the predicted query camera
- train-pair pose loss against calibrated source-relative train transforms
- identity loss on self pairs
- cycle loss on train cross pairs

Heldout camera `camera_0005` is excluded from training pairs. Validation can
use heldout image features as query evidence for the relpose head, then render
from the predicted heldout camera. That metric is therefore a query-conditioned
heldout-pose evaluation, not a no-query-image camera hallucination metric.

## Verification

Passed:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/relative_pose.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train/train.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_relative_pose.py \
  tests/test_camera_swap_sampling.py \
  tests/test_multicam_relative_pose_trainer.py -q
```

Result: `13 passed`.

Runtime smoke passed with a temporary RGB-pyramid config patched to the goodset
split and `steps=1`:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  /tmp/dynaworld_full_relpose_smoke.jsonc
```

Smoke evidence:

- loaded 2 train sequences and 1 eval sequence
- prebaked train plus heldout feature memories before extractor release
- logged predicted train and heldout validation videos
- logged `BankRate/relpose_full_pose_loss`, identity, and cycle terms
- offline W&B run:
  `wandb/offline-run-20260506_171616-aafo6ut8`

Checkpoint and relpose-only smoke also passed:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  /tmp/dynaworld_full_relpose_smoke_save.jsonc

PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  /tmp/dynaworld_full_relpose_offsetonly_smoke.jsonc
```

The first smoke saved `/tmp/dynaworld_full_relpose_smoke_checkpoint.pt`.
The second loaded that checkpoint, printed `Trainable scope: relpose_only`,
and ran the same predicted heldout validation path. Offline W&B runs:

- `wandb/offline-run-20260506_171826-shx5zqn6`
- `wandb/offline-run-20260506_171850-uyyzu0a8`

After adding `relpose_feature_frame_mode: "first_frame"`, the focused tests
passed again (`14 passed`) and the same three local smokes passed again. The
first-frame smoke showed three extra feature-cache bakes for the repeated
frame-0 train/heldout sequences before extractor release. Offline W&B runs:

- `wandb/offline-run-20260506_174254-gq16otu2`
- `wandb/offline-run-20260506_174351-22uys8vj`
- `wandb/offline-run-20260506_174429-i74kb187`

## Caveats / next checks

- `first_frame` mode still predicts one static source-relative camera transform
  per source/query pair. That is the intended rig-offset contract for this
  static DeepView split, not a moving-camera pose trajectory model.
- The full-pose output is bounded at `45 deg` and `1.0 * rig_radius` in the
  checked-in configs. That is a starting range, not tuned.

## Real V-JEPA runs

Joint full-relpose run:

```bash
PYTHONPATH=src/train uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc
```

- W&B: `0pdfypqe`
  (`https://wandb.ai/nbardy/dynaworld/runs/0pdfypqe`)
- completed: 250 steps
- checkpoint:
  `outputs/multicam_relative_pose/full_relpose_goodset_train0006_0014_holdout0005/checkpoint_final.pt`
- final train view metrics:
  - `camera_0006`: PSNR `18.6174`, SSIM `0.5511`
  - `camera_0014`: PSNR `19.3146`, SSIM `0.5347`
- final heldout `camera_0005`: PSNR `12.6225`, SSIM `0.1117`

The run baked six V-JEPA caches: three normal clips and three repeated frame-0
clips for first-frame relpose evidence.

Relpose-only follow-up:

```bash
PYTHONPATH=src/train uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_offsetonly_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc
```

- W&B: `vrr1a8pg`
  (`https://wandb.ai/nbardy/dynaworld/runs/vrr1a8pg`)
- completed: 100 steps
- loaded:
  `outputs/multicam_relative_pose/full_relpose_goodset_train0006_0014_holdout0005/checkpoint_final.pt`
- checkpoint:
  `outputs/multicam_relative_pose/full_relpose_goodset_train0006_0014_holdout0005_relpose_only/checkpoint_final.pt`
- printed: `Trainable scope: relpose_only (model/colorizer/camera_rig frozen).`
- final train view metrics:
  - `camera_0006`: PSNR `19.0169`, SSIM `0.5805`
  - `camera_0014`: PSNR `19.3737`, SSIM `0.5373`
- final heldout `camera_0005`: PSNR `12.1514`, SSIM `0.0767`

The relpose-only follow-up is a negative result for this config: it improved the
train views slightly but degraded heldout from the joint checkpoint. Use the
joint checkpoint as the current artifact unless the objective or head-only loss
is changed.
