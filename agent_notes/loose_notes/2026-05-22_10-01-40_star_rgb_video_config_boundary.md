# STAR RGB Video Config Boundary

## Goal

Continue STAR UVT trainer modularization by moving RGB STAR video-overfit
config validation out of `train_star_uvt_video_overfit.py`.

## Change

- Added `src/train/star_uvt_video_overfit_config.py`.
- Moved RGB STAR video required section/key validation for data, train, UVT,
  per-frame, output, and logging sections into that module.
- Updated `src/train/train_star_uvt_video_overfit.py` to import
  `resolve_config(...)` from the config module. The trainer keeps the external
  `run_video_fit_comparison(...)` bridge, result assertions, W&B media logging,
  row output, and CLI entrypoint local.
- Added focused tests in `tests/test_star_uvt_config_keys.py` for resolving a
  checked-in STAR RGB video config and for requiring a per-frame key.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Why This Boundary

The STAR RGB video trainer had the same trainer-owned config validation pattern
that was already removed from the target-grid RGB and rendered-feature RGB
probe trainers. Moving it into a config module keeps STAR trainer files aligned:
config modules validate shape, while trainer files handle execution policy.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/star_uvt_video_overfit_config.py src/train/train_star_uvt_video_overfit.py tests/test_star_uvt_config_keys.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_config_keys.py tests/test_trainer_registry.py -q` passed: `14 passed`.
- The first focused pytest attempt failed because the test assumed the checked-in RGB STAR config used `direct_atomic`; the current file uses `metal_tile`. The assertion was corrected to verify resolver preservation of the config value rather than a stale benchmark detail.
