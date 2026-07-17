# Fast-Mac Project3d Case Parser Cleanup

## Goal

Continue the trainer/benchmark interface cleanup by removing live one-off CLI
parsing where a shared benchmark helper already owns the generic behavior.

## Change

- Routed `src/benchmarks/fast_mac_project3d_benchmark.py` through
  `renderer_benchmark_cli.parse_csv_strings(...)` for comma-separated
  `--cases` tokenization.
- Kept project3d-specific validation local: each case is still parsed as
  `name:size:gaussians:batch`, and the benchmark still owns the `Case` schema,
  variant loading, build flags, timing, and gradient-parity behavior.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  so the benchmark-plumbing progress is documented without implying benchmark
  semantics moved into the generic parser module.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/benchmarks:src/train .venv/bin/python -m py_compile \
  src/benchmarks/fast_mac_project3d_benchmark.py src/benchmarks/renderer_benchmark_cli.py

PYTHONPATH=src/benchmarks:src/train uv run --with pytest python -m pytest \
  tests/test_renderer_benchmark_cli.py -q

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/benchmarks:src/train .venv/bin/python \
  src/benchmarks/fast_mac_project3d_benchmark.py --help
```

Results:

- `py_compile` passed.
- `tests/test_renderer_benchmark_cli.py` passed: `8 passed in 0.81s`.
- `fast_mac_project3d_benchmark.py --help` passed.
- Follow-up scan found comma splitting under reusable `src/benchmarks` CLIs only
  in `renderer_benchmark_cli.py`, the intended owner.

## Handoff

This is another helper-routing cleanup, not training evidence. The active
larger goal still needs W&B-enabled benchmark evidence for the mixed
same-view/heldout path and STAR/dynamic renderer decisions before any baseline
promotion.
