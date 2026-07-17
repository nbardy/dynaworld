# Gauged UVT Real Active-Set Distribution

## Context

The Q2 active-set strata report proved a local synthetic support/culling split.
That left an obvious question: do real compiled projective interval atlases
expose active-set topology in a measurable, bounded way, or were we only
proving a toy q-family split?

## What Changed

Added a real active-set distribution report:

```text
research_experiments/star_uvt_feature_tubes/projective_real_active_set_distribution_report.py
tests/test_star_uvt_projective_real_active_set_distribution_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_active_set_distribution/summary.json
```

It reads three existing checked-in high-motion trained artifacts:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json
```

For each, it verifies the underlying trained-high-motion report and extracts
the `trained_checkpoint` rows over `4,8,16` frames.

## Evidence

Saved report metrics:

```text
artifact_count = 3
row_count = 9
all_underlying_verifiers_pass = true
all_source_videos_exist = true
all_fallback_free = true
max_cells_per_active_set_group = 3
max_active_set_group_to_dense_tile_pair_ratio = 0.04009499860296172
max_final_active_set_group_to_dense_tile_pair_ratio = 0.0372484856924041
max_cell_to_active_set_group_ratio = 1.3214953271028038
```

The top-level goal-progress audit now has:

```text
real_video_active_set_distribution
```

It proves 22 progress rows and keeps `full_goal_completion` open.

## Tests Run

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_active_set_distribution_report.py -q

8 passed in 0.99s
```

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_real_active_set_distribution_report.py -q

37 passed in 1.11s
```

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_camera_family_2d_metal_lowering_report.py \
  tests/test_star_uvt_projective_camera_family_2d_metal_chain_rule_report.py \
  tests/test_star_uvt_projective_camera_family_2d_materialized_batch_report.py \
  tests/test_star_uvt_projective_camera_family_2d_native_eval_report.py \
  tests/test_star_uvt_projective_camera_family_2d_native_interval_forward_report.py \
  tests/test_star_uvt_projective_camera_family_2d_native_interval_backward_report.py \
  tests/test_star_uvt_projective_camera_family_2d_tile_order_reuse_report.py \
  tests/test_star_uvt_projective_camera_family_2d_tile_order_strata_report.py \
  tests/test_star_uvt_projective_camera_family_2d_active_set_strata_report.py \
  tests/test_star_uvt_projective_real_active_set_distribution_report.py -q

106 passed in 8.30s
```

## Remaining Gap

This upgrades the active-set story from synthetic-only to checked-in
high-motion real-video evidence. It still does not prove broad real-scene
quality acceptance or a full compiled-adjoint trainer replacement. The goal
remains active.
