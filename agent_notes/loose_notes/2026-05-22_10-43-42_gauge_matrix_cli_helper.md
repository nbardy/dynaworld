# Gauge Matrix CLI Helper

## Context

Continuation of the trainer/code-organization cleanup. The Gauge Fields
DeepView run-matrix scripts had the same matrix CLI arguments, subprocess
command construction, wall-clock JSON artifact, and nonzero-exit handling, with
only the run list and default description/output root differing.

## Change

- Added `parse_gauge_matrix_args(...)` to
  `research_experiments/gauge_fields/common.py`.
- Added `run_gauge_matrix(...)` to the same module.
- Routed `run_deepview_3cam_holdout.py` and
  `run_deepview_incidence_matrix.py` through those helpers.
- Kept each script's `RUNS` list, description, and default output root local.

This keeps the experiment matrix definitions easy to read while removing the
duplicated run harness.

## Validation

- `rtk .venv/bin/python -m py_compile research_experiments/gauge_fields/common.py research_experiments/gauge_fields/run_deepview_3cam_holdout.py research_experiments/gauge_fields/run_deepview_incidence_matrix.py`
- `rtk uv run python research_experiments/gauge_fields/run_deepview_3cam_holdout.py --help`
- `rtk uv run python research_experiments/gauge_fields/run_deepview_incidence_matrix.py --help`

Both direct-script help commands succeeded and showed the shared
`--output-root`, `--steps`, `--device`, `--no-wandb`, and `--only` options.

## Handoff

No training matrix was launched. This only validates the CLI/import path and
shared harness shape. A real matrix run should still be treated as an
experiment and logged separately.
