# Raw-Metal Benchmark Bootstrap Cleanup

## Context

`src/benchmarks/raw_metal_mlx_bridge.py` is a reusable backend bridge imported
by the renderer benchmark harness. It still rebuilt `BENCHMARK_DIR` and
`PROJECT_ROOT` locally and inserted the raw-Metal MLX path with a local
`sys.path` branch.

## Changes

- Routed the bridge through `benchmark_bootstrap.PROJECT_ROOT`.
- Replaced the local `sys.path` insertion with `benchmark_bootstrap.ensure_sys_path(...)`.
- Kept `RawMetalUnavailable`, MLX import handling, settings validation, and
  tensor conversion local because those are backend-specific behavior.

## Validation

- `PYTHONPATH=src/benchmarks:src/train:. uv run python -m py_compile` passed for
  `raw_metal_mlx_bridge.py`, the two renderer benchmark importers, and
  `benchmark_bootstrap.py`.
- Import smoke for `raw_metal_mlx_bridge` passed and resolved the expected
  `raw-metal-mlx-gsplat` vendored directory name.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This did not run MLX or renderer benchmarks. It only removes duplicated
benchmark/project root bootstrap from the raw-Metal bridge.
