# Trainer Registry Resolution Boundary

## Context

The active cleanup goal is to make trainer code modular through small shared
boundaries. A remaining smell was that diagnostics and profiling scripts
imported `train_video_token_implicit_dynamic.py` as a helper namespace just to
reach `resolve_config(...)` or `trainer_class_for_config(...)`.

That import pattern makes the base trainer file act like a utility module and
keeps unrelated probes coupled to one trainer implementation.

## Change

- Extended `src/train/trainer_registry.py` so each `TrainerEntry` records its
  config resolver.
- Added `resolve_config_for_arch(config, config_path)` as the shared arch-aware
  config-resolution boundary.
- Added `trainer_class_for_config(config, config_path)` as the registry-owned
  legacy Token-GS class-factory boundary for benchmark/probe callers.
- Routed these callers through the registry instead of direct base-trainer
  imports:
  - `src/train/probe_colorize_init.py`
  - `src/train/probe_colorize_matrix.py`
  - `src/train/probe_init_diagnostics.py`
  - `src/train/export_dynaworld_browser_bundle.py`
  - `src/train/visualize_camera_scene_diagnostic.py`
  - `research_experiments/vjepa_performance/*.py`
  - `research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`
  - `tests/test_config_factory_helpers.py`
- Added focused registry tests for module-level resolvers, classmethod
  resolvers, and the legacy Token-GS class factory.

## What remains direct by design

`train_precomputed_feature_implicit_dynamic.py` still imports the base
`Trainer` class because it subclasses it. `tests/test_temporal_sampling.py`
imports `Trainer` directly because it tests that class. STAR feature-tube
experiment scripts still import `run_training` from the STAR feature-overfit
trainer when they intentionally launch that exact training path.

## Validation

Validation passed:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/trainer_registry.py \
  src/train/probe_colorize_init.py \
  src/train/probe_colorize_matrix.py \
  src/train/probe_init_diagnostics.py \
  src/train/export_dynaworld_browser_bundle.py \
  src/train/visualize_camera_scene_diagnostic.py \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  research_experiments/vjepa_performance/profile_fast_mac_render_phases.py \
  research_experiments/vjepa_performance/compare_fast_mac_quality.py \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  tests/test_trainer_registry.py \
  tests/test_config_factory_helpers.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_trainer_registry.py \
  tests/test_config_factory_helpers.py \
  tests/test_train_cli.py -q
```

The focused pytest slice passed with 25 tests.
