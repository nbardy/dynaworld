# Trainer Lifecycle Hook Cleanup

## Context

The active cleanup goal is to modularize the training code without turning the
trainers into a large abstract framework. After the earlier base-loop cleanup,
two variants still overrode `run(...)` only to wrap the shared loop:

- `PrecomputedFeatureImplicitTrainer.run(...)` printed feature-cache metadata
  before delegating to `super().run()`.
- `MulticamRelativePoseImplicitTrainer.run(...)` called `super().run()` and
  then saved an optional checkpoint.

Those are lifecycle concerns, not trainer math. Keeping them as whole-run
overrides made future run-loop changes easier to miss.

## Change

- Added `Trainer.training_preamble_messages(...) -> tuple[str, ...]`.
- Added `Trainer.after_training_complete(...) -> None`.
- Updated `Trainer.run(...)` to print preamble messages before the shared
  training header, run the shared loop, print the completion message, then call
  the post-success hook.
- Changed `PrecomputedFeatureImplicitTrainer` to return its feature-cache
  metadata through `training_preamble_messages(...)`.
- Changed `MulticamRelativePoseImplicitTrainer` to save its optional checkpoint
  through `after_training_complete(...)`.

This preserves the previous success-only checkpoint behavior and keeps the
precomputed feature-cache metadata before the training header.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile` passed for:
  - `src/train/train_video_token_implicit_dynamic.py`
  - `src/train/train_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
  - `src/train/train_multicam_relative_pose_implicit_dynamic.py`
- `git diff --check` passed on the touched trainer files.
- A lightweight lifecycle hook smoke passed. It exercised:
  - base `Trainer.run(...)` ordering,
  - precomputed feature preamble messages for disabled and enabled feature
    backends,
  - relative-pose post-success checkpoint hook dispatch.

This is trainer-plumbing evidence only. It does not prove convergence, visual
quality, or renderer math.

## Next

- Keep future checks in this lane as launch guards and runtime smokes, not a
  broad training-code unit-test suite.
- The next useful cleanup should either remove another real duplicated trainer
  boundary or move on to actual convergence/benchmark evidence for the mixed
  same-view plus heldout-view trainer.
