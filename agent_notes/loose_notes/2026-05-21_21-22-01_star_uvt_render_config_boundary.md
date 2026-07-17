# STAR UVT Render Config Boundary

Date: 2026-05-21 21:22:02 +07

## Goal

Continue the modularization pass by removing repeated STAR UVT feature render
config construction from the feature overfit trainer and active
feature-overfit diagnostics/profilers.

## What Changed

- Added `src/train/star_uvt_render_configs.py`.
- The new helper owns:
  - `feature_tube_render_config_from_cfg(cfg)`
  - `uvt_render_config_from_cfg(cfg, feature_config=None)`
  - `star_uvt_render_configs_from_cfg(cfg)`
- Rewired these files to call the shared helper instead of repeating the same
  `data.max_frames`, `data.target_size`, `feature_uvt.feature_dim`,
  `alpha_threshold`, `max_alpha`, `tile_t`, and `tile_capacity` mapping:
  - `src/train/train_star_uvt_feature_overfit.py`
  - `research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py`
  - `research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py`
  - `research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py`
  - `research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py`
  - `research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py`
- Added `tests/test_star_uvt_render_configs.py` to pin the feature and UVT
  config fields built from a minimal config dict.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_render_configs.py \
  src/train/train_star_uvt_feature_overfit.py \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  tests/test_star_uvt_render_configs.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_render_modes.py \
  tests/test_star_uvt_colorizers.py \
  tests/test_star_uvt_checkpoints.py -q
```

Passed: `14 passed in 1.09s`.

Runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py /tmp/star_uvt_render_configs_smoke_2step.jsonc
```

Passed with trainer row `pass=true`, loss `0.186020497 -> 0.119176909`, zero
tile overflow, `requested_render_mode=feature_direct_atomic`,
`kernel_backward_mode=direct_atomic`, `effective_render_mode=feature_direct_atomic`,
and `mode_fallback_required=false`.

## Remaining Cleanup

This removes another repeated config boundary from STAR UVT feature scripts.
The broad modularization goal remains open; future slices should keep targeting
contracts that are shared by multiple trainer/profiler paths and can be covered
by compile checks plus a focused runtime smoke.
