# Mixed Scheduler Source Boundary

## Context

`mixed_data_scheduler.py` already owned the public `MixedStepBatch` shape, view
sampling, and `both`/`alternate` schedule helper. The mixed trainer still
duplicated that branch locally: it called `scheduled_loss_kinds(...)`, then
manually sampled same-view and novel-view batches. That made the scheduler
boundary weaker than the docs claimed.

## Change

- `mixed_data_scheduler.sample_mixed_step_batch(...)` now accepts
  `same_view_sequence` as either a concrete `SequenceData` or a lazy
  `Callable[[], SequenceData]`.
- The scheduler resolves the callable only when a same-view step is actually
  scheduled.
- `MixedSameHeldoutPrecomputedFeatureTrainer.sample_mixed_step_batch(...)` now
  delegates to the scheduler function instead of reimplementing the schedule
  branch.

The reason for the callable is practical: in `alternate` mode, novel-view-only
steps should not load or prefetch a same-view sequence just to skip that loss.
This keeps lazy same-view manifest behavior intact while moving schedule
ownership to the shared module.

## Validation

Focused scheduler/trainer tests before the docs patch:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_mixed_data_scheduler.py \
  tests/test_mixed_same_heldout_trainer.py -q
```

Result: `9 passed in 1.24s`.

Broader focused trainer/helper suite after the docs patch:

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
  tests/test_config_factory_helpers.py -q
```

Result: `70 passed in 1.23s`.

Checked-in mixed smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_173114-em1oaiqp`.
