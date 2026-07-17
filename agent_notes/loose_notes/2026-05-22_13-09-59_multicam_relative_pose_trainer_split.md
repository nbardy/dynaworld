# Multicam Relative-Pose Trainer Module Split

## Context

The relative-pose multicam trainer had the same shape as the other trainer
splits already completed today: a real implementation in a CLI-named
`train_*.py` file with a small command-line footer. This made the registry and
tests treat a launcher module as the implementation owner.

## Change

- Added `src/train/multicam_relative_pose_trainer.py`.
- Moved `RELATIVE_POSE_TRAIN_DEFAULTS`, `FullRelativePosePrediction`,
  `MulticamRelativePoseImplicitTrainer`, multires normalization helpers,
  `first_frame_repeated_sequence(...)`, and `run_training(...)` there.
- Replaced `src/train/train_multicam_relative_pose_implicit_dynamic.py` with a
  thin CLI/backcompat wrapper.
- Updated `trainer_registry.py` so `multicam_relative_pose_implicit_camera`
  routes to `multicam_relative_pose_trainer`.
- Updated `tests/test_multicam_relative_pose_trainer.py` to import the owner
  module directly and patch `multicam_relative_pose_trainer.log_wandb_run_payload`
  for the disabled-W&B guard test.

## Validation

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/train/multicam_relative_pose_trainer.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train/trainer_registry.py \
  tests/test_multicam_relative_pose_trainer.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_multicam_relative_pose_trainer.py tests/test_trainer_registry.py -q
# 24 passed
```

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from multicam_relative_pose_trainer import MulticamRelativePoseImplicitTrainer as Owner, run_training as owner_run
from train_multicam_relative_pose_implicit_dynamic import MulticamRelativePoseImplicitTrainer as Wrapper, run_training as wrapper_run
from trainer_registry import trainer_entry_for_arch
assert Owner is Wrapper
assert owner_run is wrapper_run
entry = trainer_entry_for_arch("multicam_relative_pose_implicit_camera")
assert entry.module == "multicam_relative_pose_trainer"
assert entry.trainer_class == "MulticamRelativePoseImplicitTrainer"
PY
```

```bash
PYTHONPATH=src/train WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  src/train/train.py /tmp/dynaworld_relpose_owner_split_smoke.jsonc
# passed; W&B offline run: wandb/offline-run-20260522_130926-infb3j96
```

The runtime smoke used
`src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_128_16f_8192splats_goodset_train0006_0014_holdout0005.jsonc`
patched to one step, checkpoint save disabled, high image/video cadence, and a
temporary W&B run name. It still baked missing V-JEPA cache entries for the
source/query first-frame relpose sequences, then ran through
`src/train/train.py -> trainer_registry -> multicam_relative_pose_trainer`.

## Handoff

The Token-GS, precomputed-feature, multicam, mixed same-heldout, and
relative-pose trainers now share the owner-module plus thin CLI wrapper pattern.
The remaining `train_*.py` implementation owners are mostly distinct research
surfaces: PowerFoam Direct/Metal, Dynamic PowerFoam/Gauge Foam, and STAR UVT.
Those should only be split when there is a clear helper/model owner boundary and
focused runtime evidence.
