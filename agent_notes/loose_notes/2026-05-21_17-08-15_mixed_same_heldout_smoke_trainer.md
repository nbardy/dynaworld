# Mixed Same-Heldout Smoke Trainer

Date: 2026-05-21

## Goal

Make the broad same-view plus calibrated heldout-view training path a real
trainer boundary instead of another ad-hoc script. Keep same-view reconstruction
and heldout-view reconstruction separated in sampling, logs, and future
baseline rows.

## What changed

- Added `src/train/train_mixed_same_heldout_implicit_dynamic.py`.
- Added dispatch in `src/train/train.py` for
  `arch=mixed_same_heldout_precomputed_feature_implicit_camera`.
- Reused `src/train/mixed_data_scheduler.py` for typed `SameViewBatch`,
  `NovelViewBatch`, `MixedStepBatch`, explicit loss names, shared multicam view
  sampling, and `alternate`/`both` scheduling.
- Added `heldout_recon_loss(...)` to
  `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` so the
  mixed trainer can call the heldout render path without copying multicam
  internals.
- Added `loss_scale` to
  `src/train/train_video_token_implicit_dynamic.py::Trainer.recon_backward`
  so same-view loss can keep raw logged values while applying configured
  weighting to backward.
- Added checked-in smoke config:
  `src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`.

## Test stance

The pytest coverage added here is only plumbing protection for the refactor:
dispatch, batch names, scheduler behavior, and separate aux keys. It does not
prove training quality, convergence, renderer math, or novel-view correctness.
The meaningful gate for this lane is a real training run with separate
same-view and heldout-view loss traces, then a longer W&B run with media and
baseline rows.

## Validation

- Focused pytest:
  `PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest tests/test_mixed_same_heldout_trainer.py tests/test_mixed_data_scheduler.py tests/test_temporal_sampling.py tests/test_multicam_video_data.py tests/test_pipeline_helpers.py tests/test_sequence_data_single_frame.py tests/test_pipeline_diagnostics.py tests/test_train_logging.py tests/test_config_factory_helpers.py -q`
  passed with `63 passed in 1.03s`.
- Temporary 2-step mixed smoke passed offline on MPS:
  `wandb/offline-run-20260521_170331-x5rhxvzo`.
- Temporary 10-step mixed smoke passed offline on MPS:
  `wandb/offline-run-20260521_170523-5t2xn1n5`.
- Checked-in 10-step mixed smoke passed offline on MPS:
  `wandb/offline-run-20260521_170805-5g581zns`.

The checked-in 10-step trace alternated same-view and heldout-view optimizer
steps. Same-view visible loss moved roughly `0.5239 -> 0.4942`; heldout-view
visible loss moved roughly `0.6087 -> 0.5996`. That proves the path runs and
gives a tiny convergence trace; it is not enough to claim the math is solved.

## Remaining

- Run a longer W&B-enabled mixed trace with media enabled.
- Record separate same-view and heldout-view metrics in a result artifact before
  adding any row to `BASELINES.md`.
- Decide whether the mixed lane should train `both` losses every step or keep
  `alternate` scheduling for throughput and attribution.
- Audit whether the multicam F=32/colorize/random-background fixes are fully
  inherited before using this as the main feature-splatting training harness.
