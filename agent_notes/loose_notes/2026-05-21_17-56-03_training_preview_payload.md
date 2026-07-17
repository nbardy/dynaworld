# Training Preview Payload

## Context

The trainer cleanup had already centralized result payload construction, but
the per-step image logging path still duplicated the same block in two places:

- base token-GS / multicam `Trainer.val_log(...)`
- relative-pose `MulticamRelativePoseImplicitTrainer.val_log(...)`

Both paths built `Render_GT_vs_Pred`, checked that `preview_features` existed
when `feature_pca_log` was on, converted features through PCA, and created the
same W&B image key. That is not training math, but it is a shared user-facing
payload contract and a common failure mode.

## Change

- Added `pipeline.validation_media.training_preview_payload(...)`.
- Base trainer `val_log(...)` now calls the helper.
- Relative-pose `val_log(...)` now calls the helper inside its existing
  temporary render-size context.

The trainers still own:

- scalar payload assembly
- log cadence
- validation-video payloads
- relative-pose render-size context
- W&B log dispatch

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/pipeline/validation_media.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py
```

Result: passed.

Base token-GS runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_overfit_video_token_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_175514-3nqmd2zg`.

Normal multicam runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_175542-6wm1w7v3`.

Relative-pose import/config suite:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_multicam_relative_pose_trainer.py -q
```

Result: `13 passed in 0.96s`.

## Interpretation

This is a log-path payload cleanup. It reduces duplicated trainer plumbing, but
it does not say anything about convergence, renderer correctness, or feature
background regularization quality. Those still need W&B media and loss-curve
evidence from longer runs.
