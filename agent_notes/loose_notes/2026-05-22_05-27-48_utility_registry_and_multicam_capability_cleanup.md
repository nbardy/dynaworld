# Utility Registry And Multicam Capability Cleanup

## Context

Continuation of the trainer-landscape unification pass. Earlier slices moved
class-based trainer construction into `trainer_registry` and replaced several
benchmark imports of concrete precomputed/multicam trainers. This slice cleaned
up the remaining utility-style imports and one fragile benchmark capability
check.

## Changes

- `src/train/export_dynaworld_browser_bundle.py` no longer imports
  `PrecomputedFeatureImplicitTrainer` only to resolve config. `_load_model_input`
  already receives an arch-resolved config, so it now builds `VideoFeatureCache`
  from `resolved["features"]` directly.
- `src/train/visualize_camera_scene_diagnostic.py` no longer imports
  `MulticamRelativePoseImplicitTrainer` for the decoded multicam relative-pose
  diagnostic path. It now uses
  `trainer_registry.instantiate_trainer_for_config(...)`.
- `src/benchmarks/fixed_render_variant_parity.py` and
  `src/benchmarks/fixed_render_backward_mode_parity.py` no longer check
  `trainer.__class__.__name__ == "MulticamPrecomputedFeatureImplicitTrainer"`.
  They use `trainer_phase_benchmark.trainer_uses_multicam_phase(...)`, which is
  a capability check tied to the trainer API surface rather than a concrete
  class name.
- `fixed_render_variant_parity.py` and
  `fixed_render_backward_mode_parity.py` now pass the config path into
  `trainer_for_config(...)`, matching the updated registry-backed boundary.

## Validation

- `rg` found no remaining local imports of
  `PrecomputedFeatureImplicitTrainer` or `MulticamRelativePoseImplicitTrainer`
  in the two utility scripts touched here.
- `py_compile` passed for:
  - `src/train/export_dynaworld_browser_bundle.py`
  - `src/train/visualize_camera_scene_diagnostic.py`
  - `src/benchmarks/fixed_render_variant_parity.py`
  - `src/benchmarks/fixed_render_backward_mode_parity.py`
  - `src/benchmarks/trainer_phase_benchmark.py`
- Pytest passed:
  - `tests/test_visualize_camera_scene_diagnostic.py tests/test_trainer_registry.py -q`
    -> `11 passed`
- CLI help smokes passed:
  - `src/train/export_dynaworld_browser_bundle.py --help`
  - `src/train/visualize_camera_scene_diagnostic.py --help`
  - `src/benchmarks/fixed_render_variant_parity.py --help`
  - `src/benchmarks/fixed_render_backward_mode_parity.py --help`

## Handoff

The shared registry/config boundary is now covering the main diagnostic,
benchmark, export, and V-JEPA performance entrypoints found in the current
scan. Remaining trainer-module imports should be reviewed case by case:
structural tests that exercise class methods or object shells are acceptable;
new generic helper lookups should be routed through a shared module or
`trainer_registry`.
