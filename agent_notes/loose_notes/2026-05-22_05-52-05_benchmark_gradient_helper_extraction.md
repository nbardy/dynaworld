# Benchmark Gradient Helper Extraction

## Goal

Continue the trainer/benchmark modularization pass by removing the last copied
gradient-snapshot helpers from the fixed-render and camera-swap parity CLIs.

## What Changed

- Added `src/benchmarks/benchmark_gradients.py` as the shared owner for:
  - `GaussianSequence` leaf-gradient snapshots across `xyz`, `scales`,
    `quats`, `opacities`, and `rgbs`.
  - Missing-gradient policy selection: keep missing grads as `None` or
    zero-fill them for chunk aggregation.
  - Named parameter gradient snapshots for one module or a prefixed module map.
- Routed fixed-render variant parity through the shared helper and reused
  `clone_sequence_for_fixed_render(..., freeze_colors=False)`.
- Routed fixed-render backward-mode parity through the shared helper for
  zero-filled sequence grads and colorizer grads.
- Routed camera-swap variant parity through `named_module_parameter_grads(...)`
  for model, colorizer, relative-pose head, and camera-rig parameters.
- Recorded the helper boundary in `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

- `py_compile` passed for the new helper, the three parity CLIs, and the helper
  test.
- Focused pytest passed:

```text
tests/test_benchmark_gradients.py
tests/test_fixed_render_graph.py
tests/test_benchmark_compare.py
tests/test_benchmark_memory.py
tests/test_trainer_capabilities.py
tests/test_trainer_registry.py
24 passed
```

- CLI help smokes passed for:
  - `src/benchmarks/fixed_render_variant_parity.py`
  - `src/benchmarks/fixed_render_backward_mode_parity.py`
  - `src/benchmarks/camera_swap_variant_parity.py`

## Current State

The benchmark helper layer now has clear ownership:

- `benchmark_compare.py`: seed setup and tensor/gradient diff summaries.
- `benchmark_memory.py`: memory snapshots and sampled peak tracking.
- `benchmark_gradients.py`: raw gradient snapshot capture.
- `fixed_render_cases.py`: heldout fixed-render case setup.
- `fixed_render_graph.py`: fixed-render graph construction, chunking,
  background slicing, and sequence detach/clone helpers.

This removes another dependency direction where parity CLIs imported private
helpers from each other or from `trainer_phase_benchmark.py`.

## What Is Still Left

- Continue shrinking trainer-specific benchmark code into small shared helpers
  when a second script needs the same behavior.
- Audit old benchmark/probe entrypoints for direct class imports and private
  helper imports after the current helper layer settles.
- Do not delete old training entrypoints yet. The unification target is shared
  config/launcher/helper boundaries first; deletion should wait until each old
  script has either a routed replacement or a documented reason to stay.
