# Config-Aware Log Cadence Wrappers

## What changed

- Base Token-GS `should_log_scalars(...)`, `should_log_images(...)`, and
  `should_log_videos(...)` now call:
  - `train_logging.should_log_scalar(...)`
  - `train_logging.should_log_image(...)`
  - `train_logging.should_log_video(...)`
- The trainer no longer rebuilds the shared `should_log_step(...)` argument
  bundle from `self.logging_cfg` / `self.train_cfg` in three local methods.
- The wrapper methods stay in place because relative-pose and other branches
  call `Trainer.log_gate_flags(...)` and may still need named override points.

## Boundary kept local

The base trainer still passes `log_step_zero=bool(logging.log_initial_media)`
for image/video cadence. That is a trainer policy choice, while the interval,
last-step, and config-key lookup live in `train_logging`.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_logging.py

rtk sh -lc 'PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_logging.py tests/test_train_cli.py \
  tests/test_multicam_relative_pose_trainer.py -q'

rtk git diff --check -- src/train/train_video_token_implicit_dynamic.py
```

The focused test set passed: `35 passed`.
