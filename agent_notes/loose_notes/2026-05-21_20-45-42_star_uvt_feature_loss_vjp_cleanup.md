# STAR UVT Feature Loss VJP Cleanup

## Context

Continued the active trainer modularization goal. After moving STAR UVT
feature-target loading, rendering, sparse-grid, colorizer, checkpoint,
sparse-visual, visibility/support, and schedule helpers out of
`train_star_uvt_feature_overfit.py`, the remaining trainer-as-utility cluster
was feature-target loss and VJP math.

These helpers are not trainer lifecycle policy. Tests and profiling scripts use
them as reusable mechanics, so leaving them inside the overfit trainer kept the
trainer as a utility barrel.

## Change

- Added `src/train/star_uvt_feature_losses.py`.
- Moved VJP result records, feature-target loss math, RGB-probe analytic grid
  VJP, trainable colorizer grid gradients, dense target-grid VJP, sparse
  target-grid forward/VJP, batched sparse target-grid VJP, and sparse image VJP
  packing into the new module.
- Rewired `train_star_uvt_feature_overfit.py` to import those mechanics from
  `star_uvt_feature_losses.py`.
- Rewired `tests/test_star_uvt_feature_target_adapter.py` and the touched STAR
  profiling scripts to import feature-loss/VJP helpers from the dedicated
  module or existing direct modules instead of through the overfit trainer.
- Kept the trainer responsible for deciding which feature-target path runs in
  the warm loop.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_feature_losses.py \
  src/train/train_star_uvt_feature_overfit.py \
  tests/test_star_uvt_feature_target_adapter.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_target_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_sparse_forward_profile.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_common.py \
  tests/test_star_uvt_sparse_grid.py \
  tests/test_star_uvt_checkpoints.py -q
```

Passed: 42 tests.

## Notes

This is a code organization change, not a new training result. It reduces the
trainer's helper-export surface while preserving the existing VJP behavior and
test contract. The broad modularization goal remains active because the trainer
still owns the main STAR UVT overfit loop, tile stats, config policy, and
several profile-facing runtime exports.
