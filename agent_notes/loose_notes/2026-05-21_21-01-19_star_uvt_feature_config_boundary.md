# STAR UVT Feature Config Boundary

Date: 2026-05-21 21:01:19 Asia/Ho_Chi_Minh

## Goal

Continue trainer modularization by moving STAR UVT feature-overfit config
normalization out of the training harness. Several diagnostics and tests only
needed `resolve_config`, but importing it from `train_star_uvt_feature_overfit`
made the trainer module the implicit config utility surface.

## Change

- Added `src/train/star_uvt_feature_config.py`.
- Moved feature-overfit config constants and `resolve_config(...)` into that
  module:
  - required section/key tables
  - feature render modes
  - feature-target validation
  - sparse-visual validation
  - dense-alpha validation
  - visibility-proxy validation
  - support-birth-split validation
- Updated `train_star_uvt_feature_overfit.py` to import `resolve_config` from
  the config module.
- Updated diagnostics/profilers/tests that only needed config normalization to
  import from `star_uvt_feature_config`:
  - `star_uvt_feature1_wholegraph_profile.py`
  - `star_uvt_logit_handoff_rgb_vjp_profile.py`
  - `firstclass_backward_breakdown.py`
  - `sparse_visual_loss_vjp_profile.py`
  - `dense_alpha_failure_diagnostic.py`
  - `tests/test_star_uvt_feature_target_adapter.py`

## Validation

Compile:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_feature_config.py \
  src/train/star_uvt_render_modes.py \
  src/train/star_uvt_tile_stats.py \
  src/train/star_uvt_checkpoints.py \
  src/train/train_star_uvt_feature_overfit.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py \
  research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py \
  tests/test_star_uvt_feature_target_adapter.py
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

## Remaining Shape

The STAR UVT profiling surface still imports `run_training` from
`train_star_uvt_feature_overfit.py` where it genuinely launches the trainer.
Config-only users now avoid the trainer module.
