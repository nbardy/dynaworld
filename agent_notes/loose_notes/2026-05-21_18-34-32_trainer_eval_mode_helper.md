# Trainer Eval-Mode Helper

## Context

The modularization pass is trying to remove duplicated trainer mechanics without
claiming training quality from plumbing checks. After the lifecycle hook cleanup,
the active token-GS trainer family still had repeated initial-diagnostic model
mode handling:

- base `Trainer.initial_step_result(...)`
- `KnownCameraTrainer.initial_step_result(...)`
- `MulticamPrecomputedFeatureImplicitTrainer.initial_step_result(...)`

Each branch saved `was_training`, called `self.model.eval()`, ran its diagnostic
decode/loss path, then restored train mode in a `finally` block.

## Change

- Added `Trainer.model_eval_mode(...)`, a small context manager that switches
  `self.model` to eval mode and restores training mode only if the model was
  training on entry.
- Updated the base, known-camera, and multicam initial diagnostic paths to use
  the shared context manager.

The branch-specific work remains local: clip selection, known-camera handling,
multicam/camera-swap paths, decode, bank-rate terms, rig losses, and
`StepResult` construction were not moved.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile` passed for:
  - `src/train/train_video_token_implicit_dynamic.py`
  - `src/train/train_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_relative_pose_implicit_dynamic.py`
  - `src/train/train_mixed_same_heldout_implicit_dynamic.py`
- `git diff --check` passed on the touched trainer files.
- A lightweight runtime smoke passed. It verified:
  - train-mode models restore to train after the context,
  - eval-mode models remain eval after the context,
  - exception paths still restore correctly,
  - known-camera and multicam subclasses inherit the helper.

This is state-plumbing evidence only. It does not prove convergence, renderer
math, or visual quality.

## Next

The broad goal is still active. The next useful cleanup should be another
existing duplicated trainer boundary, not new unit-test surface. Actual training
claims still need W&B traces, media, benchmark rows, or overfit runs.
