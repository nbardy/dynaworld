# W&B Default-Mode Helper

## Goal

Continue the modularization pass by centralizing benchmark/probe W&B environment
defaults without changing whether callers can override those defaults.

## What Changed

- Added `train_logging.set_default_wandb_mode(mode="disabled", silent=None)`.
- The helper uses `os.environ.setdefault(...)`, preserving any caller-provided
  `WANDB_MODE` or `WANDB_SILENT`.
- Routed these entrypoints through the helper:
  - `src/benchmarks/trainer_phase_benchmark.py`
  - `src/benchmarks/train_step_memory_benchmark.py`
  - `src/train/visualize_camera_scene_diagnostic.py`
  - `research_experiments/vjepa_performance/benchmark_free_splats_throughput.py`
  - `research_experiments/vjepa_performance/benchmark_multicam_vjepa.py`
  - `research_experiments/vjepa_performance/profile_fast_mac_render_phases.py`
  - `research_experiments/vjepa_performance/compare_fast_mac_quality.py`
- Preserved the previous distinction:
  - general benchmark/diagnostic scripts still default `WANDB_SILENT=true`
  - V-JEPA performance scripts still only default `WANDB_MODE=disabled`

## Validation

- `py_compile` passed for the shared logging module, logging tests, and all
  routed scripts.
- Focused pytest passed:

```text
tests/test_train_logging.py tests/test_train_cli.py
18 passed
```

- Direct `--help` smokes passed for all seven routed scripts.
- Search over the routed files now shows no local `os.environ.setdefault(...)`
  W&B writes outside `train_logging.py` and its tests.

## Current State

`train_logging.py` owns W&B run init, finish, cadence checks, scalar payloads,
generic payload submit, row-output logging, and benchmark/probe default W&B
environment setup. Entry points still choose whether they want silent W&B
output; the helper just owns the repeated environment mutation pattern.

## Remaining Work

- Keep W&B media object construction in `wandb_media.py`.
- Continue looking for active train/benchmark scripts that import large trainer
  modules only to reach pure helpers.
- Deletion should remain conservative until every old entrypoint has an active
  replacement or documented reason to stay.
