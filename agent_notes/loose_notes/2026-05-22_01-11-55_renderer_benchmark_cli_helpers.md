# Renderer Benchmark CLI Helpers

## Context

Continuing the trainer/interface modularization goal, a live scan found
duplicated CLI helper blocks in the two reusable splat renderer benchmark
scripts:

- `src/benchmarks/splat_renderer_benchmark.py`
- `src/benchmarks/splat_renderer_accuracy.py`

Both had local copies of resolution parsing, comma-separated list parsing, torch
dtype lookup, project-relative output path resolution, and safe filename
normalization.

## What changed

- Added `src/benchmarks/renderer_benchmark_cli.py` with shared helpers:
  - `deep_merge(...)`
  - `normalize_resolution(...)`
  - `parse_csv_ints(...)`
  - `parse_csv_resolutions(...)`
  - `parse_csv_strings(...)`
  - `torch_dtype_from_name(...)`
  - `resolve_project_path(...)`
  - `safe_filename_part(...)`
- Updated `splat_renderer_benchmark.py` and `splat_renderer_accuracy.py` to use
  the shared helper module.
- Corrected the top `TODO/trainer_landscape_unification.md` TL;DR so it states
  the current helper-unification state instead of the original stale audit
  findings.
- Corrected the same doc's log-cadence/config-defaults rows so the remaining
  work table does not list already-landed cadence helpers as future work.
- Relabeled the original trainer inventory/duplication tables as historical
  audit context, because several referenced legacy files are no longer present
  in `src/train`.
- Kept renderer-specific behavior local:
  - output schemas remain in each script
  - the Taichi `"input"` precision sentinel remains handled by the benchmark
    path that needs it

## Validation

- `py_compile` passed for the new helper, both benchmark scripts, and
  `tests/test_renderer_benchmark_cli.py`.
- `PYTHONPATH=src/train:src/benchmarks uv run --with pytest python -m pytest
  tests/test_renderer_benchmark_cli.py -q` passed: `5 passed`.
- `splat_renderer_benchmark.py --help` passed.
- `splat_renderer_accuracy.py --help` passed.
- Targeted scan found no local `deep_merge`, `normalize_resolution`,
  `parse_csv_*`, `dtype_from_name`, `resolve_output_path`, or
  `safe_filename_part` definitions in the two benchmark scripts.

## Interpretation

This is not a model/trainer behavior change. It removes duplicated benchmark CLI
plumbing while preserving the CUDA-first benchmark device policy and keeping
renderer-specific configs/results local.
