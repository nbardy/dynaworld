# Mixed Data Scheduler Boundary

Date: 2026-05-21 16:52

## Goal Context

The active goal is broad trainer modularization without a giant base trainer.
The code-organization roadmap names the mixed same-view plus heldout-view
scheduler as the next architectural bridge: keep the single-sequence scale path
and calibrated multicam heldout path distinct, but let one trainer eventually
consume both.

## Slice Implemented

Added `src/train/mixed_data_scheduler.py` with the first typed boundary for
that bridge:

- `SameViewBatch`: one same-camera sequence clip, named `same_view_recon`.
- `NovelViewBatch`: one multicam condition clip plus train/heldout view ids,
  named `heldout_view_recon`.
- `MixedStepBatch`: explicit container for one scheduler output.
- `scheduled_loss_kinds(...)`: supports `both` and `alternate` scheduling
  without hiding the loss family.
- `sample_view_indices(...)`: shared multicam view sampling with the existing
  `0/all views` convention.
- `sample_same_view_batch(...)`, `sample_novel_view_batch(...)`, and
  `sample_mixed_step_batch(...)`: small data-level helpers on top of
  `sample_clip_batch(...)`.

Wired `MulticamPrecomputedFeatureImplicitTrainer.sample_views()` to use the
shared `sample_view_indices(...)` helper. The trainer behavior is unchanged;
this just removes a duplicated sampler branch and makes future mixed training
consume the same view-sampling contract.

## What This Does Not Claim

This is not the full mixed trainer yet. No optimizer step currently consumes
`MixedStepBatch`, and there is no W&B/result row proving a combined run. The
remaining bridge is to implement the smoke trainer that alternates or jointly
runs same-view and heldout-view batches and logs both loss names separately.

## Validation

Passed:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/mixed_data_scheduler.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  tests/test_mixed_data_scheduler.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_mixed_data_scheduler.py \
  tests/test_temporal_sampling.py \
  tests/test_multicam_video_data.py \
  tests/test_pipeline_helpers.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_pipeline_diagnostics.py \
  tests/test_train_logging.py \
  tests/test_config_factory_helpers.py -q
```

Result: `60 passed in 1.26s`.

Also passed `git diff --check`.

## Next Refactor Sequence

1. Add a small mixed smoke trainer or trainer mode that consumes
   `MixedStepBatch`.
2. Keep `same_view_recon` and `heldout_view_recon` as separate scalar keys.
3. Run 1-step and 10-step smokes before promoting any real benchmark row.
