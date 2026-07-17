# STAR RGB Grid Probe Adapter Boundary

## Goal

Continue STAR UVT trainer modularization by stopping the target-grid RGB probe
from acting as the public namespace for grid RGB adapter helpers and loss math.

## Change

- Added public grid-RGB helpers to `src/train/star_uvt_feature_targets.py`:
  `FEATURE_TARGET_GRID_ADAPTERS`, `adapt_rgb_to_grid(...)`,
  `upsample_grid_rgb(...)`, and `mean_rgb_grid_loss(...)`.
- Updated `src/train/train_star_uvt_feature_rgb_probe.py` to consume those
  public helpers instead of aliasing private `_adapt_rgb_to_grid` /
  `_upsample_grid_rgb` names and keeping a local `_mean_loss(...)`.
- Updated `src/train/star_uvt_feature_config.py` to import the adapter set from
  the feature-target module instead of defining its own copy.
- Updated `src/train/star_uvt_rendered_feature_probe_objective.py` to reuse the
  same adapter set for rendered-feature target-grid sparse-pixel validation.
- Added `src/train/star_uvt_feature_rgb_probe_config.py` for target-grid RGB
  probe config validation: required sections/keys, adapter validation, positive
  step/LR checks, and target-grid materialization requirements.
- Updated `src/train/train_star_uvt_feature_rgb_probe.py` to import
  `resolve_config(...)` from that config module and keep run orchestration
  local.
- Updated `tests/test_star_uvt_feature_rgb_probe.py` so adapter tests import
  the feature-target helper module directly and config tests import the probe
  config module directly.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Why This Boundary

The RGB target-grid probe and rendered-feature sparse probe are both consumers
of the same grid RGB adapter vocabulary. Keeping the adapter set and RGB-grid
MSE in the feature-target module prevents probes, config validation, and tests
from treating a trainer file as a helper module.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/star_uvt_feature_targets.py src/train/star_uvt_feature_config.py src/train/star_uvt_rendered_feature_probe_objective.py src/train/train_star_uvt_feature_rgb_probe.py tests/test_star_uvt_feature_rgb_probe.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_rgb_probe.py tests/test_star_uvt_feature_target_adapter.py -q` passed: `43 passed`.
- `rtk .venv/bin/python -m py_compile src/train/star_uvt_feature_rgb_probe_config.py src/train/train_star_uvt_feature_rgb_probe.py tests/test_star_uvt_feature_rgb_probe.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_rgb_probe.py tests/test_trainer_registry.py -q` passed: `17 passed`.
