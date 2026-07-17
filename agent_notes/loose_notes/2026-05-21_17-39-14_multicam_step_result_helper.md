# Multicam StepResult Helper

## Context

After the rendered-view loss helper landed, the multicam trainer still repeated
the same `StepResult(...)` assembly in multiple branches:

- camera-swap initial/eval path
- normal multicam initial/eval path
- camera-swap training step
- normal multicam training step

The repeated fields were not experiment semantics; they were payload mechanics:
source path, sequence frame count, preview tensors, zero camera-loss payloads,
loss detaching, bank-rate detaching, and camera state assignment.

## Change

- Added `MulticamPrecomputedFeatureImplicitTrainer._step_result(...)`.
- Replaced the duplicated `StepResult(...)` constructors in multicam
  initial/eval, normal train, and camera-swap train/eval branches.
- Branch-specific math remains local. The helper only assembles the payload and
  applies the same detach/zero-field policy.

This is intentionally trainer-local for now. The mixed trainer has different
aux-loss payloads and weighted same/heldout aggregation, so forcing it into the
same helper would blur semantics.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train_mixed_same_heldout_implicit_dynamic.py
```

Result: passed.

Focused trainer/helper suite:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_multicam_video_data.py \
  tests/test_mixed_same_heldout_trainer.py \
  tests/test_mixed_data_scheduler.py \
  tests/test_rgb_recon_objective.py \
  tests/test_temporal_sampling.py \
  tests/test_pipeline_helpers.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_pipeline_diagnostics.py \
  tests/test_train_logging.py \
  tests/test_config_factory_helpers.py \
  tests/test_camera_swap_sampling.py \
  tests/test_multicam_relative_pose_trainer.py -q
```

Result: `88 passed in 0.99s`.

Normal multicam runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_173838-6ittgiyo`.

Camera-swap runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python - <<'PY'
from config_utils import load_config_file
from train_multicam_precomputed_feature_implicit_dynamic import main

cfg = load_config_file(
    "src/train_configs/"
    "local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc"
)
cfg["train"]["steps"] = 1
cfg["train"]["camera_swap_mode"] = "oracle_relative"
cfg["train"]["camera_swap_pairs_per_step"] = 2
cfg["train"]["camera_swap_include_self"] = True
cfg["train"]["camera_swap_include_cross"] = True
cfg["train"]["camera_swap_self_pair_probability"] = 0.5
cfg["logging"]["wandb_run_name"] = "multicam-step-result-helper-camera-swap-smoke"
main(cfg)
PY
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_173852-y5kp0ido`.

Checked-in mixed smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_173903-yockh84k`.

## Interpretation

This removes a payload-assembly fork, not a model-quality question. The next
larger cleanup target is mixed-step aggregation, but that should stay explicit
until a real W&B-enabled mixed run with media proves the loss names and payloads
are the right shape.
