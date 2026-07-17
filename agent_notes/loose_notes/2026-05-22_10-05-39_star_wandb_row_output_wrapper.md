# STAR W&B Row Output Wrapper

## Goal

Continue trainer modularization by removing repeated STAR-local
`_log_wandb_outputs(...)` wrappers from RGB STAR, feature STAR, target-grid RGB
probe, and rendered-feature RGB probe trainers.

## Change

- Added `log_star_uvt_row_outputs(...)` to `src/train/star_uvt_outputs.py`.
- The helper wraps `train_logging.log_wandb_row_outputs(...)` with STAR's
  standard contact-sheet and side-by-side media keys.
- Feature overfit still passes its extra RGB-probe contact-sheet and video keys
  as an explicit override.
- Updated these trainers to call the shared helper directly:
  `train_star_uvt_video_overfit.py`,
  `train_star_uvt_feature_overfit.py`,
  `train_star_uvt_feature_rgb_probe.py`, and
  `train_star_uvt_rendered_feature_rgb_probe.py`.
- Added focused tests in `tests/test_star_uvt_outputs.py` for default media
  keys and the feature-overfit RGB-probe override.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Why This Boundary

The lower-level scalar flattening/media attachment was already centralized in
`train_logging.log_wandb_row_outputs(...)`, but each STAR trainer still carried
a local wrapper that repeated the same default media key tuples. This helper
keeps the STAR-specific row-output convention close to STAR media/output
helpers while preserving each trainer's metric prefix and row schema.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/star_uvt_outputs.py src/train/train_star_uvt_video_overfit.py src/train/train_star_uvt_feature_rgb_probe.py src/train/train_star_uvt_rendered_feature_rgb_probe.py src/train/train_star_uvt_feature_overfit.py tests/test_star_uvt_outputs.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_outputs.py tests/test_star_uvt_config_keys.py tests/test_star_uvt_feature_rgb_probe.py tests/test_trainer_registry.py -q` passed: `30 passed`.
