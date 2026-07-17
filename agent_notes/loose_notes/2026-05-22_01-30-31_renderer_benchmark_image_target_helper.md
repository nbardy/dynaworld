# Renderer Benchmark Image Target Helper

## Context

Continuing the trainer/interface modularization goal, the two active splat
renderer benchmark CLIs still duplicated save-image target selection after the
parser/path/image helper cleanup:

- `src/benchmarks/splat_renderer_benchmark.py`
- `src/benchmarks/splat_renderer_accuracy.py`

Both scripts interpreted `save_images.largest_resolution_only`,
`save_images.largest_splat_count_only`, and `save_images.set_index` in the same
way, then checked each row against those targets.

## What Changed

- Added `ImageSaveTarget` to `src/benchmarks/renderer_benchmark_cli.py`.
- Added `resolve_image_save_target(...)`.
- Added `row_matches_image_save_target(...)`.
- Updated `splat_renderer_benchmark.py` to use the shared target resolver and
  row matcher with `required_status="ok"`.
- Updated `splat_renderer_accuracy.py` to use the same target resolver and row
  matcher for rows that have already reached the successful render path.
- Added focused coverage in `tests/test_renderer_benchmark_cli.py`.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Validation

- `py_compile` passed for:
  - `src/benchmarks/renderer_benchmark_cli.py`
  - `src/benchmarks/splat_renderer_benchmark.py`
  - `src/benchmarks/splat_renderer_accuracy.py`
  - `tests/test_renderer_benchmark_cli.py`
- Focused test passed:

```text
PYTHONPATH=src/train:src/benchmarks uv run --with pytest python -m pytest \
  tests/test_renderer_benchmark_cli.py -q
```

Result: `7 passed`.

- CLI smoke passed:

```text
PYTHONPATH=src/train:src/benchmarks .venv/bin/python src/benchmarks/splat_renderer_benchmark.py --help
PYTHONPATH=src/train:src/benchmarks .venv/bin/python src/benchmarks/splat_renderer_accuracy.py --help
```

- Targeted scan now finds `largest_resolution_only`,
  `largest_splat_count_only`, and save-image target-set construction only in
  `renderer_benchmark_cli.py`, except for the config defaults in the two
  benchmark scripts.

## Interpretation

This is still benchmark plumbing, not renderer behavior. The benchmark scripts
keep their row schemas, output filenames, and save-image directory defaults.
The shared helper now owns the common policy for deciding which cases are worth
preview images.
