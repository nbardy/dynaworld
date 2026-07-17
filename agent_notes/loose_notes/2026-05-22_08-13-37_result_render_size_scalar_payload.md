# Result Render-Size Scalar Payload Cleanup

## What changed

- `Trainer.scalar_payload(...)` now writes `RenderSize` from
  `result.render_size` when the training branch attaches one, otherwise it
  falls back to the trainer's active `self.render_size`.
- `MulticamRelativePoseImplicitTrainer.scalar_payload(...)` no longer writes
  the generic `RenderSize` and `Render/BaseSize` keys itself. It keeps only
  relative-pose and multires-specific scalars.
- Added a focused regression test that constructs the relative-pose scalar path
  without running the full trainer and verifies a multires result reports the
  result render size while preserving the base render size.

## Why

The relative-pose trainer already had to annotate `StepResult.render_size` for
multires training. Having the base scalar payload honor that annotation keeps
the generic render-size metric in one place and prevents future multires
branches from repeating the same override.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  tests/test_multicam_relative_pose_trainer.py

rtk sh -lc 'PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_logging.py tests/test_train_cli.py tests/test_multicam_relative_pose_trainer.py -q'

rtk git diff --check -- \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  tests/test_multicam_relative_pose_trainer.py
```

The focused test set passed: `35 passed`.
