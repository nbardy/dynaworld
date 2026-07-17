# Fast-Mac Benchmark Bootstrap Follow-Up

## Context

After the first renderer benchmark bootstrap cleanup, two reusable Fast-Mac
benchmark scripts still rebuilt repo roots locally:

- `src/benchmarks/fast_mac_project3d_benchmark.py`
- `src/benchmarks/fast_mac_v13_iteration_matrix.py`

They are benchmark/profiling entrypoints, not one-off shader internals, so they
fit the same helper-routing pattern as the earlier `src/benchmarks` cleanup.

## Changes

- Added `VENV_PYTHON` to `src/benchmarks/benchmark_bootstrap.py`.
- Routed `fast_mac_project3d_benchmark.py` through
  `benchmark_bootstrap.PROJECT_ROOT`, `TRAIN_ROOT`, and `ensure_sys_path(...)`
  for shared repo/train bootstrap. The script still owns its v5/v8/v9 variant
  list and build policy.
- Routed `fast_mac_v13_iteration_matrix.py` through
  `benchmark_bootstrap.ROOT` and `VENV_PYTHON` instead of rebuilding them from
  `Path(__file__)`.

## Validation

- `PYTHONPATH=src/benchmarks:src/train:. uv run python -m py_compile` passed
  for `benchmark_bootstrap.py`, `fast_mac_project3d_benchmark.py`, and
  `fast_mac_v13_iteration_matrix.py`.
- Import smoke for `fast_mac_v13_iteration_matrix` passed and reported the
  expected seven iteration specs.
- Import smoke for `fast_mac_project3d_benchmark` passed and printed the
  default case string.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This did not rerun the benchmarks. It only makes the reusable benchmark
entrypoints use the same project/train/subprocess bootstrap boundary as the
rest of the `src/benchmarks` cleanup.
