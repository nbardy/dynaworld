# RGB Alpha Eval File Artifact Helpers

## Context

After centralizing the W&B RGB+alpha validation video payload, the adjacent
file-artifact block was still repeated in four eval paths:

- Direct PowerFoam
- shared PowerFoam eval artifacts
- Dynamic PowerFoam Metal
- Dynamic Gauge Foam

Each path built a target/render/alpha preview triptych and saved render plus
side-by-side MP4s with the same filenames. Direct/shared PowerFoam eval also
had heldout variants with `heldout_` filename prefixes.

## Changes

- `src/train/video_io.py`
  - Added `alpha_to_rgb_video(...)`.
  - Added `rgb_alpha_preview(...)`.
  - Added `save_rgb_alpha_preview(...)`.
  - Added `save_render_side_by_side_videos(...)`.
- `src/train/wandb_media.py`
  - Reuses `video_io.alpha_to_rgb_video(...)` instead of owning a separate
    alpha expansion implementation.
- Updated the four PowerFoam/Gauge eval paths to use the shared file-artifact
  helpers.
- Added `tests/test_video_io.py` for preview triptych layout and stable
  render/side-by-side filenames.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/video_io.py src/train/wandb_media.py tests/test_video_io.py tests/test_wandb_media.py src/train/train_powerfoam_direct.py src/train/powerfoam_eval_artifacts.py src/train/train_dynamic_powerfoam_metal.py src/train/train_dynamic_gauge_foam.py`
- `rtk sh -lc 'PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_video_io.py tests/test_wandb_media.py tests/test_powerfoam_direct.py tests/test_dynamic_gauge_foam.py tests/test_dynamic_powerfoam_metal.py -q'`
  - `83 passed, 1 skipped`
- `rtk rg -n "alphas\\[0\\]\\.unsqueeze|torch\\.cat\\(\\[targets\\[0\\]|torch\\.cat\\(\\[targets\\.cpu\\(\\), renders\\]|heldout_side_by_side|side_by_side = torch\\.cat" src/train/train_powerfoam_direct.py src/train/powerfoam_eval_artifacts.py src/train/train_dynamic_powerfoam_metal.py src/train/train_dynamic_gauge_foam.py`
  - No remaining hits.
- `rtk git diff --check -- src/train/video_io.py src/train/wandb_media.py tests/test_video_io.py tests/test_wandb_media.py src/train/train_powerfoam_direct.py src/train/powerfoam_eval_artifacts.py src/train/train_dynamic_powerfoam_metal.py src/train/train_dynamic_gauge_foam.py`

## Notes

This keeps filenames and `should_log_video(...)` cadence unchanged. It only
moves the artifact assembly pattern into `video_io.py`, which already owns
PNG/MP4 serialization.
