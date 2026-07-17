# Benchmark Compare And Fixed-Render Helpers

## Context

Continuation of the trainer modularization pass. After moving trainer
capability checks and memory sampling out of benchmark-specific modules, the
remaining local smell in this area was CLI-to-CLI helper ownership:
`fixed_render_backward_mode_parity.py` and `camera_swap_variant_parity.py`
imported private helpers from `fixed_render_variant_parity.py`.

## Changes

- Added `src/benchmarks/benchmark_compare.py` with shared benchmark helpers:
  - `seed_everything(...)`
  - `tensor_diff_stats(...)`
  - `grad_diff_stats(...)`
  - `max_tensor_diff(...)`
- Added `src/benchmarks/fixed_render_cases.py` for fixed-render case helpers:
  - `detach_gaussian_sequence(...)`
  - `prepare_heldout_fixed_render_case(...)`
- `fixed_render_variant_parity.py`,
  `fixed_render_backward_mode_parity.py`, and
  `camera_swap_variant_parity.py` now use those neutral helper modules instead
  of importing private functions from each other.
- `trainer_phase_benchmark.py` and `train_step_memory_benchmark.py` now use the
  shared `seed_everything(...)` helper for benchmark seed setup.

## Validation

- `py_compile` passed for:
  - `src/benchmarks/benchmark_compare.py`
  - `src/benchmarks/fixed_render_cases.py`
  - `src/benchmarks/trainer_phase_benchmark.py`
  - `src/benchmarks/train_step_memory_benchmark.py`
  - `src/benchmarks/fixed_render_variant_parity.py`
  - `src/benchmarks/fixed_render_backward_mode_parity.py`
  - `src/benchmarks/camera_swap_variant_parity.py`
  - `tests/test_benchmark_compare.py`
- Pytest passed:
  - `tests/test_benchmark_compare.py tests/test_benchmark_memory.py tests/test_trainer_capabilities.py tests/test_trainer_registry.py -q`
    -> `16 passed`
- CLI help smokes passed for:
  - `src/benchmarks/trainer_phase_benchmark.py`
  - `src/benchmarks/train_step_memory_benchmark.py`
  - `src/benchmarks/fixed_render_variant_parity.py`
  - `src/benchmarks/fixed_render_backward_mode_parity.py`
  - `src/benchmarks/camera_swap_variant_parity.py`

## Handoff

The fixed-render and camera-swap parity CLIs are still allowed to share the
core fixed-render graph primitives in `trainer_phase_benchmark.py`; that is the
next possible cleanup if these benchmark routes become more central. This slice
only removed private helper imports between parity CLIs and kept behavior
unchanged.
