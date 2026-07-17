# Fixed-Render Graph Extraction

## Context

Continuation of the benchmark/trainer modularization pass. The previous slice
removed private helper imports between fixed-render parity CLIs, but
`trainer_phase_benchmark.py` still owned the reusable fixed-render graph
machinery. That left phase benchmarking as the implicit namespace for parity
scripts.

## Changes

- Added `src/benchmarks/fixed_render_graph.py` with the shared fixed-render
  graph surface:
  - `PhaseTimer`
  - `RasterGraph`
  - `FixedRenderChunk`
  - `FixedRenderCase`
  - `fast_mac_project_and_rasterize(...)`
  - `singlecam_sample_and_encode(...)`
  - `multicam_sample_and_encode(...)`
  - `iter_target_chunks(...)`
  - `detach_sequence_for_fixed_render(...)`
  - `clone_sequence_for_fixed_render(...)`
  - `background_for_chunk(...)`
  - `prepare_fixed_render_case(...)`
- `trainer_phase_benchmark.py` now imports and consumes those primitives, while
  keeping phase-specific loss construction, backward breakdowns, CLI args, and
  report payloads local.
- `fixed_render_cases.py`, `fixed_render_variant_parity.py`, and
  `fixed_render_backward_mode_parity.py` now import fixed-render graph
  primitives directly from `fixed_render_graph.py` instead of through
  `trainer_phase_benchmark.py`.
- Added `tests/test_fixed_render_graph.py` to cover background slicing and
  fixed-render sequence detach/clone semantics.

## Validation

- `py_compile` passed for:
  - `src/benchmarks/fixed_render_graph.py`
  - `src/benchmarks/fixed_render_cases.py`
  - `src/benchmarks/trainer_phase_benchmark.py`
  - `src/benchmarks/fixed_render_variant_parity.py`
  - `src/benchmarks/fixed_render_backward_mode_parity.py`
  - `src/benchmarks/camera_swap_variant_parity.py`
  - `tests/test_fixed_render_graph.py`
- Pytest passed:
  - `tests/test_fixed_render_graph.py tests/test_benchmark_compare.py tests/test_benchmark_memory.py tests/test_trainer_capabilities.py tests/test_trainer_registry.py -q`
    -> `20 passed`
- CLI help smokes passed for:
  - `src/benchmarks/trainer_phase_benchmark.py`
  - `src/benchmarks/fixed_render_variant_parity.py`
  - `src/benchmarks/fixed_render_backward_mode_parity.py`
  - `src/benchmarks/camera_swap_variant_parity.py`

## Handoff

The fixed-render parity scripts no longer depend on
`trainer_phase_benchmark.py` as their helper namespace. The remaining shared
surface is intentionally explicit: `fixed_render_graph.py` owns graph assembly
and project/raster splitting, while phase-specific benchmark orchestration and
reporting stay in each CLI.
