# STAR UVT Visibility Support Cleanup

## Context

Continued the STAR UVT helper-boundary cleanup under the broader trainer
modularization goal. After feature target loading, rendering, sparse-grid,
colorizer, checkpoint, sparse visual sampling, and sparse visual loss helpers
were split out, the remaining visibility proxy and support-birth-split geometry
still lived inside `train_star_uvt_feature_overfit.py` and tests imported those
helpers from the trainer.

## Change

- Added `src/train/star_uvt_visibility_support.py`.
- Moved visibility-proxy target sampling and loss math into the new module.
- Moved support-birth-split target-point selection, sample-grid selection, line
  fitting, tube grouping/counting, support offsets, tube selection, and tube
  reallocation into the new module.
- Kept `train_star_uvt_feature_overfit.py` responsible for config validation,
  alpha pre-sampling orchestration, model construction, and train-loop timing.
- Updated `tests/test_star_uvt_feature_target_adapter.py` to import the
  visibility/support helpers directly from `star_uvt_visibility_support.py`.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_visibility_support.py \
  src/train/train_star_uvt_feature_overfit.py \
  tests/test_star_uvt_feature_target_adapter.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_sparse_grid.py \
  tests/test_star_uvt_checkpoints.py -q
```

Passed: 39 tests.

## Notes

This is a boundary cleanup, not a new training result. It removes another
reason for probe/profile/test code to import the feature overfit trainer as a
utility module. The broad trainer modularization goal remains active because
the feature overfit trainer still owns the main loop and several feature-target
VJP paths.
