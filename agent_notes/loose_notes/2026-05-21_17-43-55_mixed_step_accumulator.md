# Mixed Step Accumulator

## Context

The mixed same-view plus heldout trainer had already moved schedule selection
into `mixed_data_scheduler.sample_mixed_step_batch(...)`, but `step(...)` still
had two parallel aggregation blocks:

- one for same-view recon
- one for heldout-view recon

Both blocks handled raw loss names, weighted recon loss, bank-rate totals,
preview selection, source metadata, and aux-loss payload assembly. That made the
step body harder to audit and gave future edits two places to keep in sync.

## Change

- Added a trainer-local `MixedBackwardResult`.
- Added a trainer-local `MixedStepAccumulator`.
- `_backward_same_view_batch(...)` and `_backward_novel_view_batch(...)` now
  return `MixedBackwardResult` instead of positional tuples.
- `step(...)` now adds those results to the accumulator and builds the final
  `StepResult` from the accumulator.

The helper is intentionally local to
`train_mixed_same_heldout_implicit_dynamic.py`. The shape is mixed-trainer
specific because it preserves `same_view_recon` and `heldout_view_recon` as
separate aux keys, while normal multicam and camera-swap branches have
different payload semantics.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_mixed_same_heldout_implicit_dynamic.py \
  tests/test_mixed_same_heldout_trainer.py
```

Result: passed.

Focused mixed tests:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_mixed_same_heldout_trainer.py \
  tests/test_mixed_data_scheduler.py -q
```

Result: `9 passed in 0.86s`.

Broader focused trainer/helper suite:

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

Result: `88 passed in 1.12s`.

Checked-in mixed smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_174347-hp1dtm6k`.

## Interpretation

This is a payload-aggregation cleanup, not a convergence claim. The important
constraint is preserved: `same_view_recon` and `heldout_view_recon` remain
separate names in the aux-loss payload.
