# Gauged UVT Q2 Tile/Order Strata

## Context

The stable-topology tile/order reuse report proved that identical sampled-Q
tile/order records can be compressed to one local topology plus q-index
applicability. This pass adds the next metadata case: depth order changes
across q, but only into a small number of strata.

## Current Model

The compiler target is:

```text
{topology_group_j, q_region_or_indices_j, depth_certificate_j}
```

not one cell per q-pair. For this smoke, q applicability is still represented
as sampled q indices, but the important scaling invariant is:

```text
shared metadata growth ~= number_of_order_strata
materialized metadata growth ~= number_of_sampled_q_pairs
```

## New Artifact

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_strata/summary.json
```

Saved metrics:

```text
q_axis_count = 5
q_pair_count = 25
materialized_cell_count = 25
shared_topology_group_count = 2
order_stratum_count = 2
materialized_tile_order_metadata_growth = 25.0
shared_tile_order_metadata_growth = 2.0
shared_to_materialized_tile_order_metadata_ratio = 0.15692307692307692
expanded_topology_matches_materialized = true
all_strata_depth_order_stable = true
min_stratum_union_depth_order_gap = 0.33200000002980246
```

The top-level goal-progress audit now proves twenty rows and includes:

```text
local_camera_family_2d_tile_order_strata
```

## What This Proves

This proves that q-family metadata does not have to fall back to one tile/order
record per sampled q-pair when depth order changes. If the arrangement has two
stable order strata, the metadata can grow with two strata.

## What Remains Open

This is still not broad q-family metadata completion. The remaining metadata
gap is active-set splits and mixed support/order strata in real compiled
camera-family atlases, ideally with continuous q-region certificates rather
than sampled q-index applicability.

## Verification

Focused strata tests:

```text
8 passed in 0.73s
```

Goal-progress plus strata tests:

```text
35 passed in 0.91s
```

The saved strata artifact verifies by CLI, and the regenerated goal-progress
artifact includes it.
