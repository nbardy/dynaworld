# STAR UVT Schedule Helper Cleanup

## Context

Continued the trainer modularization goal after splitting STAR UVT feature
targets, rendering, sparse-grid, colorizer, checkpoint, sparse-visual, and
visibility/support helpers out of `train_star_uvt_feature_overfit.py`.

The next leftover trainer-as-utility pattern was schedule policy:
feature-target weight schedules and optimizer LR schedules were defined in the
overfit trainer, but tests and profile/benchmark scripts imported them as
standalone helper contracts.

## Change

- Added `src/train/star_uvt_schedules.py`.
- Moved `FeatureTargetWeightStage`, `OptimizerLrStage`,
  `_feature_target_enabled(...)`, `_rgb_loss_weight(...)`,
  `_feature_target_weight_schedule(...)`,
  `_feature_target_weights_for_step(...)`,
  `_feature_target_weight_schedule_json(...)`, `_optimizer_lr_schedule(...)`,
  `_optimizer_lr_for_step(...)`, and `_optimizer_lr_schedule_json(...)` out of
  `train_star_uvt_feature_overfit.py`.
- Rewired the feature overfit trainer to import the schedule contract.
- Rewired `tests/test_star_uvt_feature_target_adapter.py` to import schedules
  from `star_uvt_schedules.py` and the colorizer-init checkpoint loader from
  `star_uvt_common.py`.
- Rewired STAR profile/benchmark scripts that used schedule helpers to import
  them from `star_uvt_schedules.py` instead of from the overfit trainer.
- While touching those profile scripts, replaced a few trainer re-export
  imports with the existing direct modules for checkpoint load, runtime sync,
  target-grid slicing, feature-target adapters, and sparse-grid helpers.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_schedules.py \
  src/train/train_star_uvt_feature_overfit.py \
  tests/test_star_uvt_feature_target_adapter.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_target_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_step_benchmark.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_common.py \
  tests/test_star_uvt_sparse_grid.py \
  tests/test_star_uvt_checkpoints.py -q
```

First run caught a real import mistake: `star_uvt_common` exports
`load_colorizer_init_checkpoint`, not `_load_colorizer_init_checkpoint`. After
fixing the test to alias the canonical export, the same focused gate passed:
42 tests.

## Notes

This is still a cleanup slice, not a new training result. It makes the trainer
less of a utility barrel and leaves remaining trainer-only contracts clearer:
the main loop, feature-target VJP variants, tile stats, and a few profile-only
loss helpers.
