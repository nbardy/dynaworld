# STAR UVT Render Mode Boundary

Date: 2026-05-21 21:12:54 +07

## Goal

Continue the trainer modularization work by removing another set of
feature-overfit trainer exports and duplicated profiling logic. The target was
the STAR UVT feature render-mode contract: allowed mode names, mode order,
backward-mode dispatch, effective-mode reporting, and fallback reporting.

## What Changed

- Expanded `src/train/star_uvt_render_modes.py` from a one-constant module into
  the owner for:
  - `FEATURE_RENDER_MODE_ORDER`
  - `FEATURE_RENDER_MODES`
  - `FEATURE_RENDER_BACKWARD_MODES`
  - `backward_mode_for_feature_render_mode(...)`
  - `effective_feature_render_mode_for_report(...)`
  - `feature_render_mode_fallback_required(...)`
- Rewired `src/train/star_uvt_feature_config.py` to validate
  `feature_uvt.render_mode` against `FEATURE_RENDER_MODES` from the shared
  module.
- Rewired `src/train/train_star_uvt_feature_overfit.py` to use the shared
  helper for `kernel_backward_mode`, `effective_render_mode`, and
  `mode_fallback_required` row metadata.
- Preserved the current trainer dispatch quirk explicitly:
  `cap_plain_gradcache=False` keeps plain `feature_direct_gradcache` routed to
  `gradcache` in the trainer even though report/fallback metadata treats
  gradcache modes as cap-limited.
- Rewired STAR profiling/matrix scripts to stop carrying their own
  `MODE_TO_BACKWARD` / `_mode_to_backward` / gradcache-cap logic:
  - `research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py`
  - `research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py`
  - `research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py`
- Fixed a runtime CLI regression found by the smoke: `train_star_uvt_feature_overfit.py`
  used `sys.argv` in `main()` but no longer imported `sys`.
- Added `tests/test_star_uvt_render_modes.py` to pin the shared mode contract.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_render_modes.py \
  src/train/star_uvt_feature_config.py \
  src/train/train_star_uvt_feature_overfit.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py \
  tests/test_star_uvt_render_modes.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_render_modes.py \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_common.py \
  tests/test_star_uvt_sparse_grid.py \
  tests/test_star_uvt_checkpoints.py -q
```

Passed: `51 passed in 1.37s`.

Runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py /tmp/star_uvt_render_modes_smoke_2step.jsonc
```

Passed with trainer row `pass=true`, loss `0.186020497 -> 0.119176909`, zero
tile overflow, `requested_render_mode=feature_direct_atomic`,
`kernel_backward_mode=direct_atomic`, `effective_render_mode=feature_direct_atomic`,
and `mode_fallback_required=false`.

## Remaining Cleanup

This removes one more namespace dependency on the feature-overfit trainer. The
broader modularization goal is still open: there are still trainer-local warm
paths and orchestration blocks that should only be extracted when a shared
contract is obvious and runtime-smokable.
