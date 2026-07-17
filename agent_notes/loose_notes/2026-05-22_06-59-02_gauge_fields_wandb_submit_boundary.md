# Gauge Fields W&B Submit Boundary

## Context

Continued the trainer modularization goal by routing one more active research
trainer through shared logging primitives without changing its training math or
its custom config schema.

`research_experiments/gauge_fields/train.py` still had two direct
`wandb.log(...)` calls plus a direct `wandb.finish()` even though
`src/train/train_logging.py` already owns the common payload-submit and finish
guards.

## What Changed

- Imported `log_wandb_payload(...)` and `finish_wandb_run(...)` from
  `train_logging`.
- `wandb_log_training_logs(...)` now submits scalar rows through
  `log_wandb_payload(...)`.
- The final Gauge Fields media/metric payload now submits through
  `log_wandb_payload(...)`.
- The run cleanup block now calls `finish_wandb_run()`.

The Gauge Fields W&B init remains local. Its config uses `logging.log_to_wandb`
instead of the main trainer `logging.wandb_enabled` contract, so using
`init_wandb_run(...)` would hide a real schema difference rather than simplify
the code.

## Validation

- `py_compile` passed for `research_experiments/gauge_fields/train.py` and
  `src/train/train_logging.py`.
- Direct CLI `--help` passed with
  `PYTHONPATH=src/train:research_experiments/gauge_fields`.
- Focused pytest passed:
  `tests/test_gauge_incidence.py` and `tests/test_train_logging.py`.

## Remaining Work

The remaining Gauge Fields helper candidates are artifact I/O and run-summary
writers. Those should be routed only where the payload is a generic JSON/CSV
artifact. Keep renderer math, custom metrics, W&B init schema, and experiment
orchestration local.
