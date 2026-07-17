# Temporal overlap benchmark CLI parser cleanup

## Context

The modularization pass is continuing as live-file helper routing, not a base
trainer rewrite. A scan for remaining generic CLI parser duplication found
`src/benchmarks/temporal_raster_overlap_profile.py` still carrying local
comma-separated int and float parsing while adjacent benchmark CLIs already use
`src/benchmarks/renderer_benchmark_cli.py`.

## Changed

- `renderer_benchmark_cli.py` now exposes `parse_csv_floats(...)` next to the
  existing int/string/resolution parsers.
- `temporal_raster_overlap_profile.py` now uses
  `parse_csv_ints(...)` / `parse_csv_floats(...)` for raw list parsing.
- The temporal profile keeps its own argparse validation for positive
  Gaussian counts and nonempty float lists because those are script-specific
  semantics, not generic CSV parsing.
- `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` now record
  that the temporal overlap benchmark is covered by the renderer benchmark CLI
  parser boundary.

## Validation

- `py_compile` passed for:
  - `src/benchmarks/renderer_benchmark_cli.py`
  - `src/benchmarks/temporal_raster_overlap_profile.py`
  - `tests/test_renderer_benchmark_cli.py`
  - `tests/test_temporal_raster_overlap_profile.py`
- `PYTHONPATH=src/benchmarks:src/train uv run --with pytest python -m pytest tests/test_renderer_benchmark_cli.py tests/test_temporal_raster_overlap_profile.py -q`
  passed: `9 passed in 2.74s`.
- `PYTHONPATH=src/benchmarks:src/train .venv/bin/python src/benchmarks/temporal_raster_overlap_profile.py --help`
  passed.
- A small multi-case JSON smoke passed:
  `--frames 2 --gaussians 8,16 --height 16 --width 16 --radius-px 1.0,2.0 --motion-px 0.5 --json`.

No renderer benchmark or trainer run was launched; this was a CLI/helper
boundary cleanup only.

## Handoff

Keep the generic benchmark CLI helper limited to syntax-level parsing, path
resolution, dtype lookup, merge, and image-save helpers. Keep benchmark meaning
and validation local to the owning script.
