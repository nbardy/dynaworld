# Benchmark Bootstrap Cleanup

## Context

The modularization pass has already moved benchmark output writing, device
sync, and renderer CLI parsing into shared helpers. A remaining live duplicate
was the train-root bootstrap in reusable `src/benchmarks` trainer/parity CLIs:
several scripts rebuilt the Dynaworld root, inserted `src/train` on `sys.path`,
then kept import-order `# noqa: E402` comments.

## Changes

- Added `src/benchmarks/benchmark_bootstrap.py` with shared `ROOT` and
  `TRAIN_ROOT`, inserting `src/train` once for direct benchmark script imports.
- Routed these scripts through the shared bootstrap:
  - `trainer_phase_benchmark.py`
  - `train_step_memory_benchmark.py`
  - `fixed_render_variant_parity.py`
  - `fixed_render_backward_mode_parity.py`
  - `camera_swap_variant_parity.py`
- Removed local `Path(__file__)` root discovery, local `sys.path` mutation, and
  stale import-order `# noqa: E402` comments from that cluster.
- Fixed a real stale import exposed by the import smoke:
  `train_step_memory_benchmark.py` imported `sync_device` from
  `trainer_phase_benchmark.py`, but that name did not exist. It now imports
  `sync_torch_device as sync_device` from `train_devices`.
- Left benchmark math, row schemas, CLI flags, W&B disabled defaults, and
  renderer/parity logic local.

## Validation

Commands run from the Dynaworld root:

```bash
rtk env PYTHONPATH=src/benchmarks:src/train:. uv run python -m py_compile \
  src/benchmarks/benchmark_bootstrap.py \
  src/benchmarks/trainer_phase_benchmark.py \
  src/benchmarks/train_step_memory_benchmark.py \
  src/benchmarks/camera_swap_variant_parity.py \
  src/benchmarks/fixed_render_backward_mode_parity.py \
  src/benchmarks/fixed_render_variant_parity.py
```

Result: exit 0.

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
bench_dir = Path('src/benchmarks').resolve()
sys.path.insert(0, str(bench_dir))
import benchmark_bootstrap
import trainer_phase_benchmark as phase
import train_step_memory_benchmark as memory
import fixed_render_variant_parity as fixed
import fixed_render_backward_mode_parity as backward
import camera_swap_variant_parity as camera_swap
print(benchmark_bootstrap.ROOT.name)
print(phase.TRAIN_STEP_PHASES[0])
print(memory._float_tensor(3))
print(fixed._diff_stats(None, None)['both_none'])
print(backward.FullChunkSpec.__name__)
print(camera_swap._top_diffs({}, limit=1))
PY
```

Output:

```text
dynaworld
sample
3.0
True
FullChunkSpec
[]
```

The first import smoke failed before the stale `sync_device` import was fixed:
`ImportError: cannot import name 'sync_device' from 'trainer_phase_benchmark'`.
The corrected import smoke above passed. The known `uv run` parent-project
warning about `/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking
`[project]` appeared before successful exits.

## State

This cleanup only centralizes the train-path bootstrap for reusable benchmark
CLIs. It does not rerun benchmarks or update baseline standings. Renderer
benchmarks with extra vendored Taichi paths and WorldFoam one-off benchmarks
still own their additional path setup unless they become shared trainer
benchmark entrypoints.
