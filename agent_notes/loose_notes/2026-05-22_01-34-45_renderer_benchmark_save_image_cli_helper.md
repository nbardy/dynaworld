# Renderer Benchmark Save-Image CLI Helper

## Context

Continuation of the trainer/benchmark interface cleanup. The previous slices
moved renderer benchmark parser/path/image-target/image-write helpers into
`src/benchmarks/renderer_benchmark_cli.py`, but
`splat_renderer_benchmark.py` and `splat_renderer_accuracy.py` still duplicated
the same `--save-images` / `--no-save-images` override mutation.

## Change

- Added `apply_save_image_cli_overrides(...)` to
  `src/benchmarks/renderer_benchmark_cli.py`.
- Routed both active splat renderer CLIs through that helper.
- Routed the timing benchmark renderer list and overlap-variant list through
  the shared `parse_csv_strings(...)` helper.
- Added a focused helper test in `tests/test_renderer_benchmark_cli.py`.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  so future agents do not rediscover this as an active duplicate.

## Validation

- `uv run python -m py_compile src/benchmarks/renderer_benchmark_cli.py src/benchmarks/splat_renderer_benchmark.py src/benchmarks/splat_renderer_accuracy.py tests/test_renderer_benchmark_cli.py`
- `PYTHONPATH=src/train:src/benchmarks uv run --with pytest python -m pytest tests/test_renderer_benchmark_cli.py -q`
  - `8 passed`
- `PYTHONPATH=src/train:src/benchmarks uv run python src/benchmarks/splat_renderer_benchmark.py --help`
- `PYTHONPATH=src/train:src/benchmarks uv run python src/benchmarks/splat_renderer_accuracy.py --help`
- Targeted scan confirmed the old duplicated `cfg.setdefault("save_images", {})`
  block now only exists in the shared helper.

The `uv` commands printed the existing parent-workspace warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` missing a `[project]`
table, but the commands completed successfully.
