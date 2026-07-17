# PowerFoam Metal Config Boundary

## Context

`train_powerfoam_metal.py` owned two separate concerns:

- pure config/default normalization (`DATA_DEFAULTS`, `MODEL_DEFAULTS`,
  `RENDER_DEFAULTS`, `TRAIN_DEFAULTS`, `LOSS_DEFAULTS`, `LOGGING_DEFAULTS`,
  feature-mode sets, LR group specs, and `resolve_config(...)`)
- the full Metal trainer/runtime, including the Metal rasterizer imports

That made diagnostics and point-cloud builders import the full trainer just to
resolve a config. It also kept the config contract embedded in a large runtime
file.

## Changes

- Added `src/train/powerfoam_metal_config.py` for pure PowerFoam Metal config
  normalization.
- Moved the PowerFoam Metal defaults, feature-mode sets, LR group specs, and
  `resolve_config(...)` into that module.
- Updated `train_powerfoam_metal.py` to import/re-export those names so old
  tests and callers that import from the trainer still work.
- Routed Dynamic Foam diagnostics and point-cloud builders that call
  `resolve_config(...)` through `powerfoam_metal_config`.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for
  `powerfoam_metal_config.py`, `train_powerfoam_metal.py`, and the touched
  Dynamic Foam diagnostic/point-cloud scripts.
- Import/config equivalence smoke passed:
  `train_powerfoam_metal.resolve_config is powerfoam_metal_config.resolve_config`,
  and both produced identical resolved config for
  `local_mac_powerfoam_metal_video_64_smoke.jsonc`.
- Focused PowerFoam tests passed:
  `tests/test_powerfoam_direct.py::test_powerfoam_metal_lr_schedule_uses_official_cosine_shape_and_warmups`
  and
  `tests/test_powerfoam_direct.py::test_powerfoam_metal_resample_schedule_matches_official_geometric_growth`.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is config-boundary cleanup only. It does not change PowerFoam training,
Metal kernels, rasterizer behavior, or benchmark results.
