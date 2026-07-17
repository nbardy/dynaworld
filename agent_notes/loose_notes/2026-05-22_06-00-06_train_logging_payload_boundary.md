# Train Logging Payload Boundary

## Goal

Continue the trainer modularization pass by moving the last generic W&B
payload-submit call out of core trainer files and into the shared logging
module.

## What Changed

- Added `train_logging.log_wandb_payload(payload, step=None)`.
- Updated `train_logging.log_wandb_row_outputs(...)` to call the new helper.
- Updated `train_video_token_implicit_dynamic.py` to use the helper in
  `Trainer.val_log(...)` and removed its direct `wandb` import.
- Updated `train_multicam_relative_pose_implicit_dynamic.py` to use the helper
  in its `val_log(...)` override and removed its direct `wandb` import.
- Added a focused `tests/test_train_logging.py` check that the helper forwards
  payloads and explicit steps to `wandb.log(...)`.

## Validation

- `py_compile` passed for:
  - `src/train/train_logging.py`
  - `src/train/train_video_token_implicit_dynamic.py`
  - `src/train/train_multicam_relative_pose_implicit_dynamic.py`
- Import smoke passed for:
  - `train_logging.log_wandb_payload`
  - `train_video_token_implicit_dynamic.Trainer`
  - `train_multicam_relative_pose_implicit_dynamic.MulticamRelativePoseImplicitTrainer`
- Focused pytest passed:

```text
tests/test_train_logging.py tests/test_train_cli.py
16 passed
```

## Current State

`train_logging.py` now owns W&B setup, finish, cadence checks, base scalar
payload construction, row-output logging, and the generic payload submit call.
Trainer files still own payload assembly where the metrics/media differ by
family.

## Remaining Modularization Work

- Continue scanning for trainer files that import heavy trainer modules only to
  reach pure helpers or config resolution.
- Keep W&B media construction in `wandb_media.py` and validation media helpers;
  do not collapse media naming/payload policy into this generic log call.
- Do not delete trainer entrypoints yet. The current work is reducing duplicate
  boundaries while preserving active run scripts.
