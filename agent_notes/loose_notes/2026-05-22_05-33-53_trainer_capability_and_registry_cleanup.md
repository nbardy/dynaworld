# Trainer Capability And Registry Cleanup

## Context

Continuation of the trainer-landscape unification goal. The prior slice moved
class-based trainer construction into `trainer_registry` and started replacing
class-name checks with an API-surface capability predicate. This slice removed
the remaining benchmark-to-benchmark ownership problem around that predicate and
routed more performance probes through the shared instantiation boundary.

## Changes

- Added `src/train/trainer_capabilities.py` with:
  - `trainer_has_capabilities(...)`
  - `trainer_uses_multicam_phase(...)`
- `src/benchmarks/trainer_phase_benchmark.py`,
  `src/benchmarks/fixed_render_variant_parity.py`,
  `src/benchmarks/fixed_render_backward_mode_parity.py`, and
  `src/benchmarks/camera_swap_variant_parity.py` now import the multicam
  capability predicate from `trainer_capabilities.py`.
- Removed the local `trainer_for_config(...)` wrapper from
  `trainer_phase_benchmark.py`. The touched benchmark CLIs now instantiate
  directly through `trainer_registry.instantiate_trainer_for_config(...)`.
- `src/benchmarks/train_step_memory_benchmark.py` now also instantiates through
  `trainer_registry.instantiate_trainer_for_config(...)` instead of importing a
  wrapper from another benchmark module.
- `research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`
  now instantiates the dynamic gsplat trainer through
  `instantiate_trainer_for_config(...)`, and passes the patched config path to
  `run_config_dict(...)` for the STAR UVT branch.
- `research_experiments/vjepa_performance/benchmark_free_splats_throughput.py`,
  `profile_fast_mac_render_phases.py`, and `compare_fast_mac_quality.py` now
  instantiate through `instantiate_trainer_for_config(...)` instead of pulling a
  trainer class factory and immediately calling it.
- Added `src/benchmarks/benchmark_memory.py` for benchmark memory snapshots,
  cache clearing, sampled peak tracking, and `run_with_memory_sampling(...)`.
  `trainer_phase_benchmark.py` and `train_step_memory_benchmark.py` now import
  those generic runtime helpers from the utility module instead of from each
  other.

## Validation

- `py_compile` passed for all touched Python files in this slice.
- `pytest` passed:
  - `tests/test_trainer_capabilities.py tests/test_trainer_registry.py -q`
    -> `11 passed`
  - `tests/test_benchmark_memory.py tests/test_trainer_capabilities.py tests/test_trainer_registry.py -q`
    -> `13 passed`
- CLI help smokes passed for:
  - `src/benchmarks/trainer_phase_benchmark.py`
  - `src/benchmarks/fixed_render_variant_parity.py`
  - `src/benchmarks/fixed_render_backward_mode_parity.py`
  - `src/benchmarks/camera_swap_variant_parity.py`
  - `src/benchmarks/train_step_memory_benchmark.py`
  - `research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`
  - `research_experiments/vjepa_performance/benchmark_free_splats_throughput.py`
  - `research_experiments/vjepa_performance/profile_fast_mac_render_phases.py`
  - `research_experiments/vjepa_performance/compare_fast_mac_quality.py`

## Handoff

The capability predicate is now train-side shared code instead of a helper
owned by one benchmark. Remaining direct trainer imports in the current scan
are either structural inheritance, structural tests, PowerFoam diagnostics, or
the registry's own class-factory support; keep future cleanup live-file driven
and avoid moving experiment-specific train loops into a broad base framework.
