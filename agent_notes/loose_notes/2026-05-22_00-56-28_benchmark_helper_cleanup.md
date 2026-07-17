# Benchmark Helper Cleanup

## Context

This was a continuation of the trainer/interface modularization pass. The
previous slice had already moved STAR feature-tube report artifacts through
shared helpers and rerun the checked-in mixed same-view/heldout smoke. A later
scan still found benchmark/probe scripts carrying repeated W&B finish and JSON
artifact boilerplate.

## What changed

- `train_logging.finish_wandb_run(...)` now accepts an optional run object and,
  when omitted, finishes the global active `wandb.run` if one exists.
- `src/benchmarks/trainer_phase_benchmark.py`,
  `src/benchmarks/fixed_render_variant_parity.py`,
  `src/benchmarks/camera_swap_variant_parity.py`,
  `src/benchmarks/fixed_render_backward_mode_parity.py`, and
  `src/benchmarks/train_step_memory_benchmark.py` now use shared W&B finish and
  `train_artifacts.write_json(...)` for optional JSON result artifacts.
- V-JEPA performance CLIs now use `finish_wandb_run(...)` instead of importing
  W&B only to call `wandb.finish()`:
  - `research_experiments/vjepa_performance/benchmark_free_splats_throughput.py`
  - `research_experiments/vjepa_performance/benchmark_multicam_vjepa.py`
  - `research_experiments/vjepa_performance/profile_fast_mac_render_phases.py`
  - `research_experiments/vjepa_performance/compare_fast_mac_quality.py`
- `profile_fast_mac_render_phases.py` also had a stale private import:
  `_make_v5_features_config` no longer exists. The profiler now calls the
  shared projected fast-mac rasterizer helpers instead, so projected RGB and
  feature probes follow the active `rgb_variant` / `feature_variant` dispatch.

I left `research_experiments/gauge_fields/train.py` alone. It is a separate
research CLI with its own long training surface, not a reusable benchmark helper
in the current trainer cleanup scope.

## Validation

- `py_compile` passed for the patched benchmark/probe scripts, `train_logging.py`,
  and `tests/test_train_logging.py`.
- `--help` passed for the five `src/benchmarks` CLIs touched in this slice.
- `--help` passed for the four V-JEPA performance CLIs.
- Targeted scan found no `wandb.finish()` or stale `_make_v5_features_config`
  references under `research_experiments/vjepa_performance`.
- Targeted scan found no direct JSON-output `write_text(...)` or local
  parent-mkdir pattern in the five patched `src/benchmarks` scripts.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest
  tests/test_train_logging.py tests/test_train_artifacts.py -q` passed:
  `16 passed`.
- `git diff --check` passed.
- Targeted trailing-whitespace scan passed.

## Current interpretation

The cleanup has real progress: the active trainer and reusable benchmark/probe
surfaces now share config dispatch, CLI boundaries, device sync/resolution,
artifact writes, W&B lifecycle helpers, scalar/media helpers, runtime payloads,
and the mixed same-view/heldout scheduler. The remaining work is not to invent
a new trainer framework; it is to remove smaller stale helper imports, keep
legacy surfaces out of active flows, and run real quality/benchmark contracts
before claiming the training lanes are solved.
