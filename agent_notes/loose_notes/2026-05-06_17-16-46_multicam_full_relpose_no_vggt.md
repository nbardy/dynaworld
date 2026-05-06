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

The head outputs one clip-level source-relative `camera_to_world` transform.
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

## Caveats / next checks

- The relpose head currently consumes the configured precomputed feature memory
  for the clip/sequence. It is not yet explicitly restricted to first-frame
  tokens. If the experiment needs "initial frame only" semantics, add a
  first-frame feature selection path in the precomputed feature adapter/cache.
- The full-pose output is bounded at `45 deg` and `1.0 * rig_radius` in the
  checked-in configs. That is a starting range, not tuned.
- No real V-JEPA training run was launched in this session; only a local
  RGB-pyramid 1-step smoke was run. The checked-in benchmark configs keep W&B
  enabled.
