# RGB Alpha W&B Video Payload Helper

## Context

The live duplicate scan showed four active eval paths open-coding the same W&B
video payload:

- `src/train/train_powerfoam_direct.py`
- `src/train/powerfoam_eval_artifacts.py`
- `src/train/train_dynamic_powerfoam_metal.py`
- `src/train/train_dynamic_gauge_foam.py`

Each path called `build_validation_video_payload(...)`, then added `GT_Video`,
then expanded `[T,H,W]` alpha masks into `[T,3,H,W]` for `Alpha_Video`.

## Changes

- `src/train/wandb_media.py`
  - Added `alpha_to_rgb_video(...)`.
  - Added `build_rgb_alpha_validation_video_payload(...)`, which returns:
    - `Render_Video`
    - `Render_GT_Video`
    - `GT_Video`
    - `Alpha_Video`
- Updated the four PowerFoam/Gauge eval paths to call the shared helper.
- Kept branch-specific scalar payloads local.
- Kept Dynamic Gauge Foam's `Depth_Video` local because it is not part of the
  RGB+alpha common contract.
- Added focused coverage in `tests/test_wandb_media.py`.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/wandb_media.py tests/test_wandb_media.py src/train/train_powerfoam_direct.py src/train/powerfoam_eval_artifacts.py src/train/train_dynamic_powerfoam_metal.py src/train/train_dynamic_gauge_foam.py`
- `rtk sh -lc 'PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_wandb_media.py tests/test_powerfoam_direct.py tests/test_dynamic_gauge_foam.py tests/test_dynamic_powerfoam_metal.py -q'`
  - `81 passed, 1 skipped`
- `rtk rg -n "payload\\.update\\(build_validation_video_payload\\(renders|GT_Video\\\"\\] = make_wandb_video\\(targets|Alpha_Video\\\"\\] = make_wandb_video\\(alphas\\.unsqueeze" src/train`
  - No remaining hits.
- `rtk git diff --check -- src/train/wandb_media.py tests/test_wandb_media.py src/train/train_powerfoam_direct.py src/train/powerfoam_eval_artifacts.py src/train/train_dynamic_powerfoam_metal.py src/train/train_dynamic_gauge_foam.py`

## Notes

This is a payload-construction cleanup only. It does not change video cadence,
file artifact saving, scalar metric names, or training behavior.
