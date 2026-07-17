# WorldFoam Benchmark Bootstrap Cleanup

## Context

`src/benchmarks/world_foam_gate0_paired_benchmark.py` is a reusable benchmark
report script under `src/benchmarks`, but still rebuilt the Dynaworld root,
mutated `sys.path` for `research_experiments/world_foam_lane2`, and wrote its
optional JSON report with local parent-mkdir plus `write_text(...)`.

## Changes

- Routed WorldFoam lane path setup through `benchmark_bootstrap.ROOT` and
  `benchmark_bootstrap.ensure_sys_path(...)`.
- Routed optional `--out-json` report persistence through
  `train_artifacts.write_json(...)`.
- Left `_read_json(...)`, report construction, comparison rows, and benchmark
  math local because those are script-specific.

## Validation

- `PYTHONPATH=src/benchmarks:src/train:. uv run python -m py_compile` passed for
  `world_foam_gate0_paired_benchmark.py`, `benchmark_bootstrap.py`, and
  `train_artifacts.py`.
- Import smoke passed and resolved the expected `world_foam_lane2` path.
- A targeted grep found no remaining local `Path(__file__)`, `sys.path`,
  `write_text(...)`, or parent-mkdir output logic in the benchmark file.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This did not rerun the WorldFoam paired benchmark or validate any performance
claim. It only aligns the script with the shared benchmark bootstrap and
artifact-output boundaries.
