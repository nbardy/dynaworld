# Depth DOF safe filename helper cleanup

## Context

The modularization goal is still active. A live benchmark-helper scan found
`src/benchmarks/depth_aware_dof_demo.py` carrying a local filename sanitizer
for comparison panel outputs while `renderer_benchmark_cli.py` already owned
safe benchmark filename parts.

## Changed

- `renderer_benchmark_cli.safe_filename_part(...)` now accepts
  `allow_dot: bool = True`.
- `depth_aware_dof_demo.py` now calls `safe_filename_part(case.path.stem,
  allow_dot=False)` directly when building comparison-panel filenames.
- The `allow_dot=False` path preserves the demo's previous behavior, where dots
  in the source stem were converted to underscores.
- `tests/test_renderer_benchmark_cli.py` now covers the no-dot variant.
- `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` record the
  expanded renderer benchmark helper boundary.

## Validation

- `py_compile` passed for:
  - `src/benchmarks/renderer_benchmark_cli.py`
  - `src/benchmarks/depth_aware_dof_demo.py`
  - `tests/test_renderer_benchmark_cli.py`
- `PYTHONPATH=src/benchmarks:src/train uv run --with pytest python -m pytest tests/test_renderer_benchmark_cli.py -q`
  passed: `8 passed in 0.59s`.
- `PYTHONPATH=src/benchmarks:src/train .venv/bin/python src/benchmarks/depth_aware_dof_demo.py --help`
  passed.

No depth-aware demo render was run; this was helper routing only.

## Handoff

Keep safe benchmark filename syntax in `renderer_benchmark_cli.py`. Keep demo
case loading, blur timing, panel construction, and output schema local to
`depth_aware_dof_demo.py`.
