# V-JEPA Performance Common Helper

## Goal

Continue the trainer/benchmark modularization goal by removing copied bootstrap,
seed, parsing, and timing helpers from the V-JEPA performance benchmark cluster
without changing benchmark behavior.

## What Changed

- Added `research_experiments/vjepa_performance/vjepa_benchmark_common.py`.
- The helper now owns:
  - Dynaworld repo-root and `src/train` path bootstrap.
  - Repo-root `chdir` for directly launched benchmark scripts.
  - Positive integer CSV parsing and non-empty string CSV parsing.
  - Deterministic seed setup for Python, Torch, and CUDA.
  - Device-synchronized timing fences through `train_devices.sync_torch_device`.
  - Numeric timing stats plus formatted timing summaries.
- Routed these scripts through the helper:
  - `benchmark_fast_mac_variants.py`
  - `benchmark_free_splats_throughput.py`
  - `benchmark_multicam_vjepa.py`
  - `compare_fast_mac_quality.py`
  - `profile_fast_mac_render_phases.py`

## Validation

- `py_compile` passed for the helper and all five V-JEPA performance scripts.
- Direct `--help` smokes passed for all five scripts.
- Search now shows no per-script `sys.path.insert(...)`, import-order
  `# noqa: E402`, local `timed(...)`, local random/Torch seed setup, or local
  positive-int CSV parser copies in the V-JEPA performance cluster.

## Current State

The V-JEPA performance scripts remain intentionally experiment-specific: they
still own their benchmark row schemas, trainer prep, stdout format, and timing
phase choices. The repeated operational shell around those experiments is now
one local helper instead of five small forks.

## Remaining Modularization Work

- Continue live-file scans for scripts importing full trainers only to reach a
  pure helper or config resolver.
- Keep deletion conservative. Old entrypoints should only be removed after a
  routed replacement exists and current `rg` proves no active script or test
  imports them.
- The next likely useful slice is another small cluster helper, not a base
  trainer class.
