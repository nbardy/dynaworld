# STAR UVT Sparse-Visual Sampling Helper Cleanup

## Context

Continuation of the trainer modularization goal. After checkpoint and colorizer
helpers moved out of the STAR UVT feature overfit trainer, the next small
non-policy block was sparse-visual sampling:

- sparse-visual pixel-source and VJP-mode enums
- stratified grid / patch grid / phased patch-grid pixel selection
- local frame-id selection for render chunks
- patch phase cycling
- sparse-visual loss sample-count math

These are pure helper contracts. The overfit trainer uses them, the
sparse-visual VJP profiler uses them, and tests cover them. They did not need
to live inside the 5k-line trainer.

## Changes

- Added `src/train/star_uvt_sparse_visual_sampling.py`.
- Rewired `src/train/train_star_uvt_feature_overfit.py` to import sparse-visual
  sampling constants/helpers from the new module.
- Rewired `tests/test_star_uvt_feature_target_adapter.py` to import
  sparse-visual sampling helpers from the new module.
- Rewired
  `research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py`
  to import sparse-visual sampling helpers directly from the new module.
- While touching that profiler import block, also routed generic helpers through
  their existing shared modules:
  - `config_utils.path_or_none`
  - `star_uvt_checkpoints.load_star_training_checkpoint`
  - `star_uvt_common.load_colorizer_init_checkpoint`
  - `star_uvt_common.load_training_sequence`
  - `star_uvt_runtime.resolve_device`
  - `star_uvt_runtime.sync_device`
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_sparse_visual_sampling.py \
  src/train/train_star_uvt_feature_overfit.py \
  tests/test_star_uvt_feature_target_adapter.py \
  research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_sparse_grid.py \
  tests/test_star_uvt_checkpoints.py -q
```

Passed: `39 passed in 1.82s`.

## Remaining Work

The sparse-visual loss/VJP math itself still lives in
`train_star_uvt_feature_overfit.py`; it is larger and more coupled to colorizer
gradient accumulation. It is a plausible next extraction, but it should move as
one coherent loss module with the existing tests, not as a half-wrapper.
