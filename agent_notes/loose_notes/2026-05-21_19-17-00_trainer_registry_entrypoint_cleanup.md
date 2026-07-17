# Trainer Registry Entrypoint Cleanup

## Goal

Continue the trainer modularization goal without touching training math: reduce
duplicated config dispatch logic and make `src/train/train.py` a thin
entrypoint surface.

## Change

- Added `src/train/trainer_registry.py` as the owner of:
  - `TrainerEntry`
  - `TRAINER_BY_ARCH`
  - top-level `arch` validation
  - `trainer_entry_for_arch(...)`
  - `trainer_entry_for_config(...)`
  - `load_config_and_entry(...)`
  - `run_config(...)`
- Slimmed `src/train/train.py` to CLI argument handling plus re-exports for
  the existing tests and scripts that dynamically load `train.py` by filename.

This is intentionally not a base trainer abstraction. It only moves the
entrypoint registry into a reusable module and removes the duplicate
unsupported-arch branch that lived in both lookup and runtime dispatch.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train.py src/train/trainer_registry.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_config_factory_helpers.py::test_train_router_accepts_star_uvt_video_overfit_config \
  tests/test_mixed_same_heldout_trainer.py::test_mixed_same_heldout_arch_dispatches_to_trainer \
  tests/test_multicam_relative_pose_trainer.py::test_multicam_relative_pose_arch_dispatches_to_trainer \
  tests/test_powerfoam_direct.py::test_powerfoam_direct_config_dispatches_to_trainer \
  -q
```

Result: `4 passed in 1.11s`.

## Next

The cleanup goal still has work left. The next useful slices are:

- keep collapsing duplicated trainer lifecycle boundaries only where two live
  trainers already share the same behavior contract
- avoid tests that assert helper internals; prefer dispatch/runtime smoke tests
  that catch real config or trainer boundary regressions
- do not add another trainer layer unless a concrete repeated behavior fork
  has already appeared in at least two active trainers
