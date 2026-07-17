# Renderer Benchmark CSV String Follow-Up

## Context

Continuation of the modular trainer/benchmark cleanup goal. The trainer-as-helper
import scan was clean except for intentional inheritance and structural tests.
The live low-risk duplication was in reusable benchmark CLIs:

- `mac_renderer_stack_compare.py` parsed `--renderers` with a local
  comma-separated string comprehension.
- `fast_mac_v13_iteration_matrix.py` parsed `--versions` the same way.

`src/benchmarks/renderer_benchmark_cli.py` already owns
`parse_csv_strings(...)`, so these benchmark CLIs should use that shared
boundary instead of reintroducing parser forks.

## Changes

- Routed `mac_renderer_stack_compare.py --renderers` through
  `renderer_benchmark_cli.parse_csv_strings(...)`.
- Routed `fast_mac_v13_iteration_matrix.py --versions` through the same helper.
- Restored the missing `Path` import in `mac_renderer_stack_compare.py`; the
  help smoke caught that stale import break after prior benchmark-bootstrap
  cleanup.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  src/benchmarks/renderer_benchmark_cli.py \
  src/benchmarks/fast_mac_v13_iteration_matrix.py \
  src/benchmarks/mac_renderer_stack_compare.py \
  tests/test_renderer_benchmark_cli.py
rtk env PYTHONPATH=src/train:src/benchmarks uv run --with pytest python -m pytest \
  tests/test_renderer_benchmark_cli.py -q
rtk env PYTHONPATH=src/train:src/benchmarks .venv/bin/python \
  src/benchmarks/fast_mac_v13_iteration_matrix.py --help
rtk env PYTHONPATH=src/train:src/benchmarks .venv/bin/python \
  src/benchmarks/mac_renderer_stack_compare.py --help
rtk git diff --check -- \
  src/benchmarks/fast_mac_v13_iteration_matrix.py \
  src/benchmarks/mac_renderer_stack_compare.py
```

Results: renderer benchmark CLI tests passed (`8 passed`), both help paths
loaded, `py_compile` passed, and `git diff --check` was clean. No benchmark run
was executed.
