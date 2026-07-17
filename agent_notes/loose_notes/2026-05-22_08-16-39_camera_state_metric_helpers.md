# Camera-State Metric Helper Cleanup

## What changed

- Added `pipeline.diagnostics.camera_state_summary_metrics(...)` for the raw
  camera fov/radius/rotation/translation summary scalars.
- Added `pipeline.diagnostics.camera_state_payload(...)` for W&B-style payload
  keys with caller-selected prefixes.
- `train_logging.scalar_payload(...)` now uses the shared payload helper for
  `Camera/FOVDegrees`, `Camera/Radius`,
  `Camera/RotationDeltaMeanDegrees`, and `Camera/TranslationDeltaMean`.
- Base Token-GS progress messages now use the shared summary helper.
- Full-sequence eval payloads now use the same helper with
  `key_prefix="Camera/Eval"`, preserving existing keys such as
  `Camera/EvalFOVDegrees`.

## Why

The same camera-state scalar math was duplicated in train-step logging,
progress messages, and eval payload assembly. Keeping the math in
`pipeline.diagnostics` makes metric changes harder to drift while leaving each
caller's payload key names intact.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  src/train/pipeline/diagnostics.py \
  src/train/train_logging.py \
  src/train/train_video_token_implicit_dynamic.py \
  tests/test_pipeline_diagnostics.py

rtk sh -lc 'PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_pipeline_diagnostics.py tests/test_train_logging.py \
  tests/test_train_cli.py tests/test_multicam_relative_pose_trainer.py -q'

rtk git diff --check -- \
  src/train/pipeline/diagnostics.py \
  src/train/train_logging.py \
  src/train/train_video_token_implicit_dynamic.py \
  tests/test_pipeline_diagnostics.py
```

The focused test set passed: `39 passed`.
