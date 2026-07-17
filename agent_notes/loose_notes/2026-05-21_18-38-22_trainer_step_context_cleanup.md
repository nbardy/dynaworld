# Trainer Step Context Cleanup

## Context

The active modularization goal is to keep shared trainer mechanics in one place
without moving convergence-critical math behind vague abstractions. The next
duplication was the optimizer-step envelope:

- base `Trainer.step(...)`
- `KnownCameraTrainer.step(...)`
- `MulticamPrecomputedFeatureImplicitTrainer.step(...)`
- `MixedSameHeldoutPrecomputedFeatureTrainer.step(...)`

Base, multicam, and mixed open-coded `reset_profile_timing`, `step_total`,
`zero_grad`, `optimizer_step`, and `finish_profile_timing`. Known-camera did
the same optimizer work without the profiling envelope.

## Change

- Added `Trainer.train_step_context(...)`.
  - Resets timing.
  - Profiles `step_total`.
  - Profiles `zero_grad`.
  - Calls `optimizer.zero_grad(set_to_none=True)`.
  - Finalizes timing after a successful step body.
- Added `Trainer.optimizer_step(...)`.
  - Profiles `optimizer_step`.
  - Calls `optimizer.step()`.
- Updated base token-GS, known-camera, multicam, and mixed same-view/heldout
  step paths to use those helpers.

Sampling, decode, backward strategy, camera-swap math, mixed loss aggregation,
and `StepResult` payload assembly remain branch-local.

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
  - `MixedSameHeldoutPrecomputedFeatureTrainer`

The smoke verified that the shared context clears existing gradients, performs
an optimizer update, and records `step_total`, `zero_grad`, and
`optimizer_step` timing terms.

This is step-plumbing evidence only. It does not prove convergence, renderer
math, or visual quality.

## Next

The broad goal remains active. Continue looking for duplicated trainer
mechanics that are already visible in the live tree. Do not grow a training-code
unit-test suite unless a check catches a real launch-time regression.
