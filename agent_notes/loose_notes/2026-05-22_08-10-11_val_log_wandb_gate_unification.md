# Val Log W&B Gate Unification

## What changed

- Added `Trainer.log_gate_flags(step)` in
  `src/train/train_video_token_implicit_dynamic.py` to share the
  scalar/image/video cadence tuple used by `val_log(...)`.
- Base Token-GS `val_log(...)` now uses that helper and submits through
  `train_logging.log_wandb_run_payload(self.wandb_run, ...)`.
- `MulticamRelativePoseImplicitTrainer.val_log(...)` now uses the inherited
  cadence helper and the same explicit-run submit helper.
- The relative-pose override now exits early when `self.wandb_run is None`,
  matching the base trainer and avoiding accidental global `wandb.log(...)`
  calls when a config disables W&B.

## Boundary kept local

The relative-pose override still owns render-size contexts:

- scalar/image payloads are built at the result render size
- validation videos are built at `base_render_size`

That is branch-specific behavior, so this cleanup only unified the cadence and
W&B submit envelope.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py

rtk sh -lc 'PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_logging.py tests/test_train_cli.py tests/test_multicam_relative_pose_trainer.py -q'

rtk git diff --check -- \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py
```

The focused test set passed: `34 passed`.
