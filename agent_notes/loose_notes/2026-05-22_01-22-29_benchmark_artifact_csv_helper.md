# Benchmark Artifact CSV Helper

## Context

Continuing the trainer/interface modularization goal, the artifact helper layer
already owned parent-safe JSON, JSONL, text, resolved-config, and append-jsonl
writes. A live scan found remaining complete-table benchmark outputs still
hand-building parent directories plus CSV/JSONL writes.

This slice is deliberately narrow: it only routes complete in-memory benchmark
tables through shared artifact helpers. Row-at-a-time streaming logs and binary
images remain local to their owning scripts.

## What Changed

- Added `train_artifacts.write_csv(path, rows, *, fieldnames=None)`.
  - Creates parent directories.
  - Serializes `Path` and other config-like values through
    `serialize_config_value(...)`.
  - Infers sorted fieldnames when the caller does not provide an explicit
    schema.
- Updated `src/benchmarks/splat_renderer_benchmark.py`.
  - Optional JSONL result output now calls `write_jsonl(...)`.
  - Optional CSV result output now calls `write_csv(...)`.
- Updated `src/benchmarks/mac_renderer_stack_compare.py`.
  - Optional CSV output now calls `write_csv(...)` with the existing
    `BenchRow` dataclass field order.
- Updated `src/benchmarks/depth_aware_dof_demo.py`.
  - Summary JSON output now calls `write_json(...)`.
- Updated `tests/test_train_artifacts.py` for the CSV contract.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Validation

- `py_compile` passed for:
  - `src/train/train_artifacts.py`
  - `tests/test_train_artifacts.py`
  - `src/benchmarks/depth_aware_dof_demo.py`
  - `src/benchmarks/mac_renderer_stack_compare.py`
  - `src/benchmarks/splat_renderer_benchmark.py`
- Focused test passed:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_train_artifacts.py -q
```

Result: `6 passed`.

- CLI smoke passed:

```text
PYTHONPATH=src/train:src/benchmarks .venv/bin/python src/benchmarks/splat_renderer_benchmark.py --help
PYTHONPATH=src/train:src/benchmarks .venv/bin/python src/benchmarks/mac_renderer_stack_compare.py --help
```

- Targeted scan found no remaining local `csv.DictWriter`, optional benchmark
  JSONL/CSV parent-mkdir block, or depth-aware summary `write_text(json.dumps)`
  pattern in the three touched benchmark scripts.

## Interpretation

This is a plumbing cleanup, not a benchmark-result change. The benchmark scripts
still own their row schemas, result contents, images, and renderer-specific
logic. The shared layer now owns the repeated artifact write mechanics for
complete benchmark tables.
