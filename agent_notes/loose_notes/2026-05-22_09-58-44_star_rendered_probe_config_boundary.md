# STAR Rendered-Feature Probe Config Boundary

## Goal

Continue STAR UVT trainer modularization by moving rendered-feature RGB probe
config validation out of the trainer file.

## Change

- Added `src/train/star_uvt_rendered_feature_probe_config.py`.
- Moved rendered-feature RGB probe required section/key checks, STAR
  data/colorize/output/logging validation, positive step/LR checks,
  trainable-scope defaults, resume requirements, sparse pixel-source and
  grid-adapter validation, sample-grid shape/bounds checks, frame-chunk checks,
  and feature-dim checks into that module.
- Updated `src/train/train_star_uvt_rendered_feature_rgb_probe.py` so it
  imports `resolve_config(...)` from the config module and keeps render,
  optimizer, checkpoint, W&B, and result-row orchestration local.
- Added focused config assertions to `tests/test_star_uvt_feature_rgb_probe.py`
  for rendered-probe defaults, required resume checkpoint, and stratified-grid
  sample bounds.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Why This Boundary

The rendered-feature RGB probe had already moved sparse objective math into
`star_uvt_rendered_feature_probe_objective.py`, but its trainer still owned a
large config-validation block. Moving config normalization beside the objective
and feature-target helper modules keeps the trainer file focused on execution
policy and prevents tests from importing trainer code just to validate config
contracts.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/star_uvt_rendered_feature_probe_config.py src/train/train_star_uvt_rendered_feature_rgb_probe.py tests/test_star_uvt_feature_rgb_probe.py src/train/star_uvt_feature_rgb_probe_config.py src/train/train_star_uvt_feature_rgb_probe.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_rgb_probe.py tests/test_trainer_registry.py -q` passed: `19 passed`.
