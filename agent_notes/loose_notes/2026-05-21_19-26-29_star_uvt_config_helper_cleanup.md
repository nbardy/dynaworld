# STAR UVT Config Helper Cleanup

## Goal

Continue the trainer modularization cleanup by centralizing repeated STAR UVT
config helper mechanics without changing train/probe behavior.

## Change

- Added `config_utils.require_config_keys(...)` beside
  `require_config_sections(...)` and `path_or_none(...)`.
- Rewired required-key validation in:
  - `src/train/train_star_uvt_video_overfit.py`
  - `src/train/train_star_uvt_feature_overfit.py`
  - `src/train/train_star_uvt_feature_rgb_probe.py`
  - `src/train/train_star_uvt_rendered_feature_rgb_probe.py`
- Removed local `_path_or_none(...)` helper copies from the STAR UVT scripts and
  used `config_utils.path_or_none(...)`.
- Removed the rendered-feature RGB probe's import of `_path_or_none` from the
  feature overfit trainer, which was an unnecessary trainer-to-trainer helper
  dependency.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/config_utils.py \
  src/train/train_star_uvt_video_overfit.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_config_and_dataset_io.py \
  tests/test_config_factory_helpers.py::test_train_router_accepts_star_uvt_video_overfit_config \
  tests/test_star_uvt_feature_rgb_probe.py \
  tests/test_star_uvt_feature_target_adapter.py::test_rgb_probe_config_requires_target_grid_materialization \
  tests/test_star_uvt_background_cheat_diagnostic.py \
  -q
```

Result: `20 passed in 1.23s`.

## Notes

This is a small config-contract cleanup. It does not make config schemas fully
typed yet, but it removes another repeated local helper pattern and one
unnecessary helper import across trainer scripts.
