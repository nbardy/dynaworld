# STAR UVT W&B Row Logging Helper

## Goal

Continue the trainer modularization cleanup by removing duplicated logging
payload mechanics from the STAR UVT train/probe scripts without changing
training math or result schemas.

## Change

- Added shared helpers to `src/train/train_logging.py`:
  - `flatten_scalar_metrics(...)`
  - `flattened_scalar_payload(...)`
  - `add_existing_wandb_media(...)`
  - `log_wandb_row_outputs(...)`
- Rewired the local `_log_wandb_outputs(...)` wrappers in:
  - `src/train/train_star_uvt_video_overfit.py`
  - `src/train/train_star_uvt_feature_overfit.py`
  - `src/train/train_star_uvt_feature_rgb_probe.py`
  - `src/train/train_star_uvt_rendered_feature_rgb_probe.py`

Each STAR UVT script still owns its metric prefix and which configured output
paths become W&B media. The shared helper only owns the repeated mechanics:
recursive numeric metric flattening, bool-skip behavior, existing-file checks,
and `wandb.log(...)`.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_logging.py \
  src/train/train_star_uvt_video_overfit.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_logging.py \
  tests/test_config_factory_helpers.py::test_train_router_accepts_star_uvt_video_overfit_config \
  tests/test_star_uvt_feature_rgb_probe.py \
  -q
```

Result: `15 passed in 1.27s`.

## Notes

This is a real shared logging contract, not a trainer framework. It protects the
STAR UVT row-output pattern while leaving experiment-specific payload names and
media choices explicit at the caller.
