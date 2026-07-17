# Token-GS Trainer Module Split

## Goal

Finish the live trainer-as-helper split by moving the base Token-GS trainer out
of the CLI-named launcher module while preserving the old import surface and
running real F=3/F=32/multicam smoke coverage.

## What changed

- Added `src/train/token_gs_trainer.py`.
- Moved `Trainer`, `KnownCameraTrainer`, Token-GS config defaults,
  `normalize_render_size_schedule(...)`, `resolve_config(...)`,
  `trainer_class_for_config(...)`, and Token-GS `run_training(...)` into that
  module.
- Replaced `src/train/train_video_token_implicit_dynamic.py` with a thin
  CLI/backcompat module that re-exports the public Token-GS symbols and keeps
  the existing `run_config_arg(...)` CLI behavior.
- Updated `src/train/precomputed_feature_trainer.py` to subclass
  `token_gs_trainer.Trainer` instead of importing the base class from the CLI
  wrapper.
- Updated `trainer_registry.py` so `tokengs`, `tokengs_video_implicit_camera`,
  and `tokengs_video_known_camera` resolve, instantiate, and run through
  `token_gs_trainer`.
- Updated `tests/test_temporal_sampling.py` to test the base trainer from its
  new owner module.

## Validation

- `PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile`
  passed for `token_gs_trainer.py`, the thin Token-GS CLI wrapper,
  precomputed/multicam/mixed/relative-pose trainer modules, trainer registry,
  and trainer tests touched by the split.
- Backcompat import smoke confirmed
  `train_video_token_implicit_dynamic.Trainer` and `KnownCameraTrainer` are the
  same class objects exported by `token_gs_trainer`.
- Registry smoke confirmed Token-GS arches map to module `token_gs_trainer`, and
  known-camera configs still select `KnownCameraTrainer`.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_trainer_registry.py tests/test_temporal_sampling.py tests/test_train_cli.py tests/test_config_factory_helpers.py tests/test_config_and_dataset_io.py -q`
  passed with `52 passed`.
- Real 1-step trainer smokes through `src/train/train.py` passed:
  - F=3 Token-GS smoke: `/tmp/dynaworld_tokengs_split_smoke.jsonc`, offline W&B
    run `wandb/offline-run-20260522_124623-1z1locoz`.
  - Mixed same-heldout/multicam smoke:
    `/tmp/dynaworld_mixed_split_smoke.jsonc`, offline W&B run
    `wandb/offline-run-20260522_124709-xa3eqago`.
  - F=32 Token-GS feature-splat smoke:
    `/tmp/dynaworld_f32_split_smoke.jsonc`, offline W&B run
    `wandb/offline-run-20260522_124810-uollyghf`.
- `git diff --check` passed.

## Remaining risk

This split clears the main CLI-named trainer-as-helper chain. The next cleanup
should be smaller: remove or shrink compatibility re-export surfaces only after
fresh import scans and smoke coverage, and continue avoiding a generic base
trainer abstraction.
