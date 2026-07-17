# Gauge Fields JSON Artifact Boundary

## Context

Continued the modularization goal by routing Gauge Fields JSON report writes
through the shared artifact primitive while preserving the research-lane helper
surface.

Gauge Fields already had `common.write_json(...)`, used by material-surfel
training, splat-baseline training, and cheat probes. That helper still
implemented local parent creation plus `json.dumps(..., sort_keys=True)`.
The two run-matrix launchers and summary script also had direct JSON report
writes.

## What Changed

- `research_experiments/gauge_fields/common.py` now implements
  `write_json(...)` by delegating to `train_artifacts.write_json(...)`.
- `run_deepview_3cam_holdout.py` and `run_deepview_incidence_matrix.py` now use
  Gauge `common.write_json(...)` for `wall_clock.json` reports.
- `summarize_runs.py` now uses Gauge `common.write_json(...)` for optional
  summary JSON output.

This keeps the Gauge-local API stable for experiment scripts while removing a
separate JSON writer implementation.

## Validation

- `py_compile` passed for Gauge common, both run-matrix launchers, the
  summarizer, Gauge train, Gauge splat baseline, Gauge cheat probe, and
  `train_artifacts.py`.
- Direct CLI `--help` passed for:
  - `run_deepview_incidence_matrix.py`
  - `run_deepview_3cam_holdout.py`
  - `summarize_runs.py`
- Focused pytest passed:
  `tests/test_gauge_incidence.py` and `tests/test_train_artifacts.py`.

## Remaining Work

Keep Gauge renderer math, run orchestration, W&B init schema, and custom metrics
local. Future cleanup candidates are only the generic boundaries still repeated
in live code: report readers, complete-table CSV/text writers, and device/sync
primitives where the contracts match.
