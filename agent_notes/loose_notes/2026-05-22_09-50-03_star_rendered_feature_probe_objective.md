# STAR Rendered-Feature Probe Objective Boundary

## Goal

Continue trainer modularization in the STAR UVT rendered-feature RGB probe by
removing inline sparse sampling and sparse RGB loss formulas from the trainer
file.

## Change

- Added `src/train/star_uvt_rendered_feature_probe_objective.py`.
- Moved the rendered-feature probe's target-grid pixel-id selection,
  stratified-grid pixel-id selection, target RGB gather, sparse RGB
  composition, and sparse RGB local feature/alpha VJP entrypoint into that
  module.
- The new helper delegates to existing shared contracts:
  `star_uvt_sparse_grid.py`, `star_uvt_sparse_visual_sampling.py`, and
  `star_uvt_sparse_visual_losses.py`.
- Updated `src/train/train_star_uvt_rendered_feature_rgb_probe.py` so it keeps
  config validation, model/colorizer setup, render orchestration, optimizer
  stepping, checkpointing, W&B logging, and row output local.
- Updated `tests/test_star_uvt_feature_rgb_probe.py` so objective-helper tests
  import the helper module directly instead of treating the trainer as the
  helper namespace.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Why This Boundary

The rendered-feature RGB probe had reimplemented stratified-grid pixel sampling
and black-background sparse RGB loss even though the STAR feature overfit path
already had sparse visual sampling/loss modules. The probe still needs its own
policy for choosing target-grid versus stratified-grid pixels, but the math and
local VJP contract should not drift from the shared STAR sparse-visual helpers.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/train_star_uvt_rendered_feature_rgb_probe.py src/train/star_uvt_rendered_feature_probe_objective.py tests/test_star_uvt_feature_rgb_probe.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_rgb_probe.py -q` passed: `8 passed`.
