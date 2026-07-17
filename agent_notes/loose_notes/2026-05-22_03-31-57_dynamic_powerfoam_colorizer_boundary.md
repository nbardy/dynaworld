# Dynamic PowerFoam Colorizer Boundary

## Context

After Token-GS and STAR UVT colorizer construction were routed through their
own helper modules, Dynamic PowerFoam still kept a special
`FeatureToColor` path inside `train_dynamic_powerfoam_metal.py`.

That path is not the same contract as Token-GS:

- it is only active for `model.dynamic_mode == "token_rbf_features"`
- it has `colorize.init_rgb_identity`
- identity init requires a single 1x1 Conv2d colorizer with no hidden layer and
  no pre-norm

So the right cleanup was a PowerFoam-specific colorizer module, not forcing this
into `model_factories.build_colorizer(...)`.

## Changes

- Added `src/train/powerfoam_colorizers.py`.
- Moved Dynamic PowerFoam colorize defaults, RGB identity initialization, and
  token-feature-mode colorizer construction there.
- Kept `train_dynamic_powerfoam_metal.build_colorizer(...)` as a compatibility
  wrapper around the new helper.
- Updated the Dynamic PowerFoam tests to import the identity init from the new
  module and added a focused test for the builder's feature-mode gate plus RGB
  identity weights.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for
  `powerfoam_colorizers.py`, `train_dynamic_powerfoam_metal.py`, and
  `tests/test_dynamic_powerfoam_metal.py`.
- Focused tests passed:
  `test_feature_colorizer_identity_and_background_composition`,
  `test_dynamic_powerfoam_colorizer_builder_gates_on_feature_mode`, and
  `test_token_dynamic_powerfoam_features_mps_raster_backward_smoke`.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is a colorizer-boundary cleanup only. It does not change Dynamic PowerFoam
training behavior, MPS kernels, background strategy, or quality metrics.
