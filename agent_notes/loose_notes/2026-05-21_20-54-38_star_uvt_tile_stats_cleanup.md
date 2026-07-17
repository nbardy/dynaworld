# STAR UVT Tile Stats Cleanup

Date: 2026-05-21 20:54:38 Asia/Ho_Chi_Minh

## Goal

Continue trainer modularization by removing another trainer-as-utility import
pattern from STAR UVT profiling scripts.

## Change

- Added `src/train/star_uvt_tile_stats.py` for tile-load summary metrics:
  active tile count, clipped/raw refs, overflow/unstable counts, percentiles,
  and fixedbin eligibility.
- Added `src/train/star_uvt_render_modes.py` for the shared
  `FEATURE_GRADCACHE_CAP` constant.
- Moved the feature-overfit RGB-probe checkpoint loader wrapper into
  `src/train/star_uvt_checkpoints.py`.
- Updated `train_star_uvt_feature_overfit.py` to import those helpers instead
  of defining them inline.
- Updated STAR UVT diagnostics/profilers to import tile stats, runtime helpers,
  checkpoint loading, sequence loading, and gradcache cap from the explicit
  helper modules where applicable:
  - `star_uvt_feature1_wholegraph_profile.py`
  - `star_uvt_targetgrid_vjp_bridge_profile.py`
  - `star_uvt_logit_handoff_rgb_vjp_profile.py`
  - `firstclass_backward_breakdown.py`
  - `dense_alpha_failure_diagnostic.py`
  - `alpha_only_visibility_profile.py`

## Validation

Compile:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_render_modes.py \
  src/train/star_uvt_tile_stats.py \
  src/train/star_uvt_checkpoints.py \
  src/train/train_star_uvt_feature_overfit.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py
```

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_common.py \
  tests/test_star_uvt_sparse_grid.py \
  tests/test_star_uvt_checkpoints.py \
  -q
```

Result: `42 passed`.

## Remaining Import Shape

Remaining imports from `train_star_uvt_feature_overfit.py` are now narrower:

- true entrypoints: `run_training`
- trainer policy/config: `resolve_config`

Those can be addressed in later slices, but this step removes pure statistics,
runtime, checkpoint, sequence-loading, and RGB-probe checkpoint-loader exports
from the trainer boundary.
