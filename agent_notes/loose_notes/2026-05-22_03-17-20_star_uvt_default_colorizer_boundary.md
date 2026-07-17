# STAR UVT Default Colorizer Boundary

## Context

STAR UVT already had `star_uvt_colorizers.build_feature_colorizer(...)` for
config-driven feature colorizer construction, but the dense feature-tube model
and the autograd overfit benchmark still repeated the default
`FeatureToColor(feature_dim, hidden_dim=None, activation=sigmoid, pre_norm=true,
weight_init=kaiming, weight_init_gain=4.0)` constructor locally.

That was small duplication, but it was exactly the kind of repeated experiment
default that drifts when a better feature-colorizer default is found.

## Changes

- Added `DEFAULT_FEATURE_COLORIZE_CFG` and
  `build_default_feature_colorizer(...)` to `src/train/star_uvt_colorizers.py`.
- Routed `src/train/star_uvt_feature_tube_model.py` through the shared default
  helper.
- Routed
  `research_experiments/star_uvt_feature_tubes/feature_autograd_overfit_benchmark.py`
  through the same helper while preserving its MPS device placement.
- Added a focused contract test for the default STAR feature colorizer settings.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for
  `star_uvt_colorizers.py`, `star_uvt_feature_tube_model.py`, and
  `feature_autograd_overfit_benchmark.py`.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest
  tests/test_star_uvt_colorizers.py tests/test_star_uvt_models.py -q` passed
  with 5 tests.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is a plumbing cleanup only. It does not change STAR UVT quality, renderer
timing, alpha-background behavior, or the selected overfit strategy.
