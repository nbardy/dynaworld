# Renderer Benchmark Image Helper

## Context

Continuing the trainer/interface modularization goal, the two active splat
renderer benchmark CLIs already shared parser/path/dtype/deep-merge helpers via
`src/benchmarks/renderer_benchmark_cli.py`. A live scan still found duplicated
CHW tensor-to-PNG logic:

- `src/benchmarks/splat_renderer_benchmark.py`
- `src/benchmarks/splat_renderer_accuracy.py`

Both copies detached tensors, normalized `1`/`>3` channels to RGB, clamped and
converted to uint8 HWC, created parent directories, and called Pillow.

## What Changed

- Added `renderer_benchmark_cli.save_chw_image(...)`.
- Updated `splat_renderer_accuracy.py` to use it for reference/candidate/diff
  PNGs while keeping accuracy-specific filenames local.
- Updated `splat_renderer_benchmark.py` to use it for renderer preview PNGs
  while keeping benchmark-specific filename construction local.
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

Result: `6 passed`.

- CLI smoke passed:

```text
PYTHONPATH=src/train:src/benchmarks .venv/bin/python src/benchmarks/splat_renderer_benchmark.py --help
PYTHONPATH=src/train:src/benchmarks .venv/bin/python src/benchmarks/splat_renderer_accuracy.py --help
```

- Targeted scan now finds the tensor-to-PNG conversion only in
  `renderer_benchmark_cli.py`; the benchmark scripts retain only row-specific
  image filename wrappers.

## Interpretation

This is plumbing only. It does not change renderer math, image-selection
policy, or output naming. It makes the renderer benchmark helper boundary more
complete: parser/path/dtype/filename/image primitives are now shared, while
each benchmark script still owns its own rows and result semantics.
