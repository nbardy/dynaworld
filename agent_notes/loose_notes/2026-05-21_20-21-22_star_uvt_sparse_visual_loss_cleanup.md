# STAR UVT Sparse-Visual Loss Helper Cleanup

## Context

Continuation of the trainer modularization goal. The previous slice split
sparse-visual sampling out of `train_star_uvt_feature_overfit.py`. The next
coherent block was sparse-visual loss and VJP math:

- sparse RGB composition and target-background composition
- autograd sparse visual RGB loss
- manual hidden64 and manual linear colorizer VJPs
- GELU derivative helpers
- target-area cell helpers
- sparse alpha and black-hole losses
- native target-area sparse visual backward-mode mapping

These helpers are used by the STAR UVT feature overfit trainer, the
sparse-visual VJP profiler, and the sparse visual tests. They are not trainer
policy and do not need to live inside the main overfit script.

## Changes

- Added `src/train/star_uvt_sparse_visual_losses.py`.
- Rewired `src/train/train_star_uvt_feature_overfit.py` to import sparse-visual
  loss/VJP helpers from the new module.
- Rewired `tests/test_star_uvt_feature_target_adapter.py` so sparse-visual
  loss tests import from `star_uvt_sparse_visual_losses.py`.
- Rewired
  `research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py`
  so the profiler imports hidden/linear colorizer VJP helpers and target-loss
  helpers from the new module.
- Kept `_trainable_colorizer_grid_loss_and_grid_grad(...)` in the overfit
  trainer for now. It is adjacent colorizer/grid loss code, but not part of the
  sparse-visual VJP contract.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_sparse_visual_losses.py \
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

Passed: `39 passed in 1.72s`.

## Remaining Work

The main overfit trainer still owns visibility proxy and support-birth-split
geometry helpers, plus `_trainable_colorizer_grid_loss_and_grid_grad(...)`.
Those can move in later slices if they have clear module boundaries and tests.
