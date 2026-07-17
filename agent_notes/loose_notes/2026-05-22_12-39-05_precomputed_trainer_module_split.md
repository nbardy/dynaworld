# Precomputed Feature Trainer Module Split

## Goal

Reduce the live trainer-as-helper import chain without changing trainer
behavior. This is the first safe slice of the larger entrypoint cleanup: move a
base trainer out of a CLI-named module while preserving the old launcher.

## What changed

- Added `src/train/precomputed_feature_trainer.py`.
- Moved `FEATURE_OPTION_DEFAULTS`, `PrecomputedFeatureImplicitTrainer`, and
  precomputed-feature `run_training(...)` into that module.
- Replaced `src/train/train_precomputed_feature_implicit_dynamic.py` with a thin
  launcher/backcompat module that re-exports the public trainer symbols and
  keeps the existing `run_config_arg(...)` CLI behavior.
- Updated `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` to
  import `PrecomputedFeatureImplicitTrainer` from the new non-CLI owner module.
- Updated `trainer_registry.py` so the precomputed-feature, LTX feature, and
  WAN/VACE feature arches resolve, instantiate, and run through
  `precomputed_feature_trainer` instead of the CLI wrapper.

## Validation

- `PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile`
  passed for the new module, the thin CLI wrapper, multicam, mixed same-heldout,
  relative-pose, and trainer registry modules.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_trainer_registry.py tests/test_temporal_sampling.py tests/test_train_cli.py -q`
  passed with `27 passed`.
- A registry smoke confirmed `precomputed_feature_implicit_camera` now maps to
  module `precomputed_feature_trainer` and class
  `PrecomputedFeatureImplicitTrainer`.
- A backcompat import smoke confirmed
  `train_precomputed_feature_implicit_dynamic.PrecomputedFeatureImplicitTrainer`
  is the same class object exported by `precomputed_feature_trainer`.
- `git diff --check` passed for the touched trainer split paths.

## Remaining risk

The next real split is bigger: the Token-GS base trainer still lives in
`train_video_token_implicit_dynamic.py`, and mixed/relative-pose still subclass
the multicam trainer from its CLI-named module. Move those only with focused
registry tests plus real F=3, F=32, and multicam smoke coverage.
