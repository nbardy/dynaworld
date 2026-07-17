# Gauged UVT Active-Set Strata Progress

## Context

The previous Q2 metadata work proved two local cases:

- stable tile/order topology across all sampled q-pairs
- depth-order topology splitting into two strata across q

The remaining metadata gap in the goal-progress audit still mentioned active
sets changing across q. That is a different topology event: support/culling can
change which primitives are present before visibility order even matters.

## What Changed

Added a focused active-set split-strata report:

```text
research_experiments/star_uvt_feature_tubes/projective_camera_family_2d_active_set_strata_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_active_set_strata/summary.json
tests/test_star_uvt_projective_camera_family_2d_active_set_strata_report.py
```

The report deliberately constructs a 5x5 Q2 family where support/culling
topology changes over q:

```text
q_phase < 0:                 active set (0, 1), order (0, 1)
q_phase >= 0 and q_height < 0: active set (1, 2), order (2, 1)
q_phase >= 0 and q_height >= 0: active set (0, 2), order (2, 0)
```

It reuses the existing tile/order topology compressor, but with
`ACTIVE_TRACE_COUNT = 3`, so localized primitive ids remain meaningful while
materialized q-pair ids still differ.

## Evidence

Saved active-set artifact metrics:

```text
q_pair_count = 25
shared_topology_group_count = 3
active_set_stratum_count = 3
materialized_tile_order_metadata_growth = 25.0
shared_tile_order_metadata_growth = 3.0
shared_to_materialized_tile_order_metadata_ratio = 0.19692307692307692
expanded_topology_matches_materialized = true
all_active_set_strata_depth_order_stable = true
min_active_set_union_depth_order_gap = 0.2630399994850159
```

The top-level goal-progress audit now imports and verifies this artifact:

```text
local_camera_family_2d_active_set_strata
```

The regenerated audit proves 21 progress rows and still keeps
`full_goal_completion` open.

## Tests Run

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_camera_family_2d_active_set_strata_report.py -q

8 passed in 0.60s
```

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_camera_family_2d_active_set_strata_report.py -q

36 passed in 1.22s
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
  tests/test_star_uvt_projective_camera_family_2d_active_set_strata_report.py -q

97 passed in 7.50s
```

## Remaining Gap

This closes the local synthetic active-set metadata case. It does not prove
real-scene active-set distributions, broad real-scene quality acceptance, or a
full compiled-adjoint trainer replacement. The goal should remain active.
