# Step Result Builder

## Context

The trainer cleanup had already extracted scheduler, clip sampling, multicam
render-loss loops, and the mixed-step accumulator. One duplicated surface still
remained: every trainer branch assembled `StepResult` by hand.

That duplication is low-level but risky because `StepResult` is the payload that
drives scalar logging, media preview, source metadata, and later docs. The math
and backward paths should stay trainer-local, but detach policy and zero camera
loss defaults should not drift across base token-GS, known-camera, multicam, and
mixed same-view/heldout trainers.

## Change

- Added `runtime_types.build_step_result(...)`.
- The helper centralizes:
  - source path and sequence frame count from `SequenceData`
  - zero camera-loss defaults for paths without camera regularizers
  - detach policy for scalar tensors
  - bank-rate term detaching
  - aux-loss term detaching
- Updated base token-GS, known-camera, multicam, and mixed same-view/heldout
  trainer paths to call the helper.

This is intentionally payload-only. It does not change sampling, rendering,
loss math, backward strategy, optimizer stepping, W&B media selection, or
validation rendering.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/runtime_types.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train_mixed_same_heldout_implicit_dynamic.py
```

Result: passed.

Base token-GS runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_overfit_video_token_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_174952-bbbwz3dt`.

Normal multicam runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_175020-odhh2imp`.

Mixed same-view/heldout runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_175035-4qnp6etn`.

## Interpretation

These are plumbing checks. They prove the refactored result payload path runs
through real trainer call graphs. They do not prove convergence, renderer math,
background regularization quality, or baseline improvement.

The next evidence should be a longer W&B-enabled mixed trace with media and
separate same-view versus heldout-view curves. Only after that should this move
toward `BASELINES.md`.
