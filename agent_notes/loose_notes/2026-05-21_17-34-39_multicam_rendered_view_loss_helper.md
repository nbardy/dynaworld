# Multicam Rendered-View Loss Helper

## Context

The previous cleanup put multicam train-view and heldout-view rendering through
`_recon_loss_for_views(...)`, but camera-swap still duplicated the lower-level
rendered-view block:

- alpha/background guard
- `recon_loss` profile section
- `rgb_objective.reconstruction_loss(...)`
- optional preview RGB and feature capture

That was a small but live behavior fork. Future changes to feature-background
rules or preview capture could land in train/heldout and miss camera-swap.

## Change

- Added
  `MulticamPrecomputedFeatureImplicitTrainer._rendered_view_recon_loss(...)`.
- `_recon_loss_for_views(...)` now calls it for train-view and heldout-view
  renders.
- `camera_swap_recon_loss(...)` now calls it after
  `render_camera_swap_pair(...)`.

Camera-swap still owns source grouping, source decode, relpose residuals,
relpose cycle loss, bank-rate aggregation, and pair sampling. The helper only
owns the common rendered-view loss and preview mechanics.

## Validation

Syntax/runtime-import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train_mixed_same_heldout_implicit_dynamic.py \
  src/train/mixed_data_scheduler.py
```

Result: passed.

Focused plumbing tests:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_mixed_same_heldout_trainer.py \
  tests/test_mixed_data_scheduler.py \
  tests/test_rgb_recon_objective.py -q
```

Result: `15 passed in 1.35s`.

Broader focused trainer/helper suite after docs:

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

Result: `88 passed in 1.34s`.

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
cfg["logging"]["wandb_run_name"] = "multicam-camera-swap-rendered-view-helper-smoke"
main(cfg)
PY
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_173425-bf4yc6h0`.

Checked-in mixed smoke rerun after this helper also passed:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_173547-6qpl53pz`.

## Interpretation

This is still a plumbing cleanup, not a convergence claim. The useful evidence
is that all three multicam rendered-view consumers now hit one helper, and the
camera-swap call graph still runs after the extraction.
