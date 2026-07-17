# Trainer Initial Clip Helper

## Context

The trainer modularization pass is removing duplicated mechanics while keeping
math and data semantics explicit. After the step-context cleanup, the remaining
initial diagnostic paths still repeated the same first-window clip setup:

- base `Trainer.initial_step_result(...)`
- `KnownCameraTrainer.initial_step_result(...)`
- `MulticamPrecomputedFeatureImplicitTrainer.initial_step_result(...)`

Each path computed `train_frame_count`, built `torch.arange(0, clip_length)`,
and most then called `prepare_clip(...)`.

## Change

- Added `Trainer.initial_clip_indices(...)`.
- Added `Trainer.initial_clip_for_sequence(...)`.
- Updated base token-GS, known-camera, and multicam initial diagnostic paths to
  use the helpers.
- The multicam camera-swap initial path uses only `initial_clip_indices(...)`
  because its per-source decode path prepares clips later.

Branch-specific logic remains local: known-camera camera extraction, multicam
camera-swap sampling, decode paths, regularizers, and loss/result assembly were
not hidden behind the helper.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile` passed for:
  - `src/train/train_video_token_implicit_dynamic.py`
  - `src/train/train_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_relative_pose_implicit_dynamic.py`
  - `src/train/train_mixed_same_heldout_implicit_dynamic.py`
- `git diff --check` passed on the touched trainer files.
- A lightweight runtime smoke passed across:
  - `Trainer`
  - `KnownCameraTrainer`
  - `MulticamPrecomputedFeatureImplicitTrainer`

The smoke verified inherited first-window index construction, frame slicing,
and the existing `(1, F)` `prepare_clip(...)` time-batch shape.

This is initial-diagnostic plumbing evidence only. It does not prove
convergence, renderer math, or visual quality.

## Next

The broad modularization goal remains active. Future cleanup should keep this
pattern: extract repeated trainer mechanics only when the current tree shows the
duplication, and keep same-view versus heldout/novel-view semantics explicit.
