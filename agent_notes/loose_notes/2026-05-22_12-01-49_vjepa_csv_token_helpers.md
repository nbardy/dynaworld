# V-JEPA CSV Token Helpers

## Goal

Continue the modularization goal by routing one more live parser boundary
through an existing shared train/probe CLI helper without changing benchmark
semantics.

## Change

- Added `train_cli.parse_csv_strings(...)` next to the existing
  `parse_csv_ints(...)`.
- Changed `parse_csv_ints(...)` to reuse `parse_csv_strings(...)` for trimming
  and empty-item filtering.
- Updated `research_experiments/vjepa_performance/vjepa_benchmark_common.py` so
  `parse_positive_int_csv(...)` and `parse_nonempty_csv(...)` reuse the shared
  tokenizers.
- Kept V-JEPA-specific behavior local: positive-integer validation,
  nonempty-list validation, argparse error types, timing, seed setup, and
  benchmark config patching remain in `vjepa_benchmark_common.py`.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python -m py_compile \
  src/train/train_cli.py research_experiments/vjepa_performance/vjepa_benchmark_common.py

PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_train_cli.py -q

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python \
  research_experiments/vjepa_performance/benchmark_fast_mac_variants.py --help

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py --help
```

Additional direct parser check:

- `parse_positive_int_csv("1, 2,,3") == [1, 2, 3]`
- `parse_nonempty_csv("v5, v6,,v8") == ["v5", "v6", "v8"]`
- zero and empty-list inputs still raise `argparse.ArgumentTypeError`.

Results:

- `py_compile` passed.
- `tests/test_train_cli.py` passed: `8 passed in 0.04s`.
- Both V-JEPA help smokes passed.

## Handoff

This is parser/helper cleanup only. It does not change benchmark case
construction, trainer behavior, or convergence evidence.
