# Multicam Trainer Module Split

## Goal

Continue reducing trainer-as-helper imports by moving the multicam base trainer
out of its CLI-named launcher module while preserving the old import surface.

## What changed

- Added `src/train/multicam_precomputed_trainer.py`.
- Moved `DATA_MULTICAM_DEFAULTS`, `CAMERA_RIG_DEFAULTS`,
  `TRAIN_MULTICAM_DEFAULTS`, `MulticamPrecomputedFeatureImplicitTrainer`, and
  multicam `run_training(...)` into that module.
- Replaced `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
  with a thin CLI/backcompat module that re-exports the public multicam trainer
  symbols and keeps the existing `run_config_arg(...)` CLI behavior.
- Updated mixed same-heldout and relative-pose trainers to subclass
  `MulticamPrecomputedFeatureImplicitTrainer` from the new owner module.
- Updated `trainer_registry.py` so `multicam_precomputed_feature_implicit_camera`
  resolves, instantiates, and runs through `multicam_precomputed_trainer`.
- Updated `tests/test_temporal_sampling.py` to test the base class from its new
  owner module.

## Validation

- `PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile`
  passed for the new module, the thin CLI wrapper, mixed same-heldout,
  relative-pose, trainer registry, and temporal sampling test module.
- Backcompat import smoke confirmed
  `train_multicam_precomputed_feature_implicit_dynamic.MulticamPrecomputedFeatureImplicitTrainer`
  is the same class object exported by `multicam_precomputed_trainer`.
- Registry smoke confirmed `multicam_precomputed_feature_implicit_camera` maps
  to module `multicam_precomputed_trainer` and class
  `MulticamPrecomputedFeatureImplicitTrainer`, and the mixed same-heldout smoke
  config still resolves to `MixedSameHeldoutPrecomputedFeatureTrainer`.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_trainer_registry.py tests/test_temporal_sampling.py tests/test_mixed_data_scheduler.py tests/test_mixed_same_heldout_trainer.py tests/test_multicam_relative_pose_trainer.py tests/test_train_cli.py -q`
  passed with `51 passed`.
- `git diff --check` passed for the touched multicam split paths.

## Remaining risk

The remaining large trainer-as-helper split is the Token-GS base trainer in
`train_video_token_implicit_dynamic.py`. That file still owns the base
`Trainer`, `KnownCameraTrainer`, and legacy class factory. Splitting it should
come with F=3, F=32, multicam, and registry smokes because many configs and
probes depend on that base surface.
