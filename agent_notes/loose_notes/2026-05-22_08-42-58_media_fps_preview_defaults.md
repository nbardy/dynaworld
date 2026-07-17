# Media FPS and Preview Defaults

## Context

The live duplicate scan after the metric-payload cleanup showed the same small
media policy repeated across Direct PowerFoam, shared PowerFoam eval artifacts,
Dynamic PowerFoam Metal, and Dynamic Gauge Foam:

- `float(cfg.get("video_fps", 4.0))`
- `make_preview_image(..., caption=f"step {step}: GT | render")`

Those are not branch-specific trainer semantics. They are media defaults and
caption policy, so they belong with the existing media helpers.

## Changes

- `src/train/video_io.py`
  - Added `video_fps_from_config(cfg, default=4.0)`.
- `src/train/wandb_media.py`
  - Added `make_step_preview_image(target, render, step)`.
- Updated the four PowerFoam/Gauge eval paths to use those helpers.
- `tests/test_video_io.py`
  - Added default/coercion coverage for `video_fps_from_config(...)`.
- `tests/test_wandb_media.py`
  - Added caption coverage for `make_step_preview_image(...)`.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/video_io.py src/train/wandb_media.py tests/test_video_io.py tests/test_wandb_media.py src/train/train_powerfoam_direct.py src/train/powerfoam_eval_artifacts.py src/train/train_dynamic_powerfoam_metal.py src/train/train_dynamic_gauge_foam.py`
- `rtk sh -lc 'PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_video_io.py tests/test_wandb_media.py tests/test_train_logging.py tests/test_powerfoam_direct.py tests/test_dynamic_gauge_foam.py tests/test_dynamic_powerfoam_metal.py -q'`
  - `102 passed, 1 skipped`

## Notes

This keeps video cadence and payload ownership local. It only centralizes the
small shared media-policy literals.
