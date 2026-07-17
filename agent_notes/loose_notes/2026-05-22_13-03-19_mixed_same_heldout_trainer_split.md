# Mixed Same-Heldout Trainer Module Split

## Context

After the Token-GS, precomputed-feature, and multicam trainer implementation
splits, the mixed same-view plus heldout-view bridge was still implemented in
the CLI-named `train_mixed_same_heldout_implicit_dynamic.py` file. That kept the
registry route and tests tied to a launcher module even though the trainer is a
real reusable implementation surface.

## Change

- Added `src/train/mixed_same_heldout_trainer.py`.
- Moved `TRAIN_MIXED_DEFAULTS`, `MixedBackwardResult`, `MixedStepAccumulator`,
  `MixedSameHeldoutPrecomputedFeatureTrainer`, and `run_training(...)` there.
- Replaced `src/train/train_mixed_same_heldout_implicit_dynamic.py` with a thin
  CLI/backcompat wrapper using `train_cli.run_config_arg(...)` and
  `run_config_or_path(...)`.
- Updated `trainer_registry.py` so
  `mixed_same_heldout_precomputed_feature_implicit_camera` routes to
  `mixed_same_heldout_trainer`.
- Updated `tests/test_mixed_same_heldout_trainer.py` to import the owner module
  directly and assert the registry route points there.

## Validation

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/train/mixed_same_heldout_trainer.py \
  src/train/train_mixed_same_heldout_implicit_dynamic.py \
  src/train/trainer_registry.py \
  tests/test_mixed_same_heldout_trainer.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_mixed_same_heldout_trainer.py tests/test_trainer_registry.py -q
# 12 passed
```

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from mixed_same_heldout_trainer import MixedSameHeldoutPrecomputedFeatureTrainer as Owner, run_training as owner_run
from train_mixed_same_heldout_implicit_dynamic import MixedSameHeldoutPrecomputedFeatureTrainer as Wrapper, run_training as wrapper_run
from trainer_registry import trainer_entry_for_arch
assert Owner is Wrapper
assert owner_run is wrapper_run
entry = trainer_entry_for_arch("mixed_same_heldout_precomputed_feature_implicit_camera")
assert entry.module == "mixed_same_heldout_trainer"
assert entry.trainer_class == "MixedSameHeldoutPrecomputedFeatureTrainer"
PY
```

```bash
PYTHONPATH=src/train WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  src/train/train.py /tmp/dynaworld_mixed_owner_split_smoke.jsonc
# passed; W&B offline run: wandb/offline-run-20260522_130229-5p9y6mrq
```

The smoke used the checked-in
`src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`
patched to one step and run through the real `src/train/train.py` registry path.

## Handoff

The mixed bridge is now in the same pattern as precomputed and multicam:
implementation owner module plus historical CLI wrapper. The next similar slice
would be `train_multicam_relative_pose_implicit_dynamic.py`, but that class is
larger and should be split only with the focused relative-pose tests plus a
runtime smoke.
