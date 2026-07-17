# Gauged UVT Q2 Tile/Order Metadata Reuse

## Context

After native Q2 family interval forward and backward were verified, the
remaining local Q2 gap shifted from coefficient consumption to metadata: the
prototype still materialized one tile/order cell per sampled q-pair. This pass
adds a focused stable-topology report that asks whether the sampled-Q
tile/order records can be represented by one shared topology record plus q-index
applicability.

## Current Model

For a stable q-family stratum, the compiler can separate:

```text
topology = (tile_u, tile_v, local_start, local_stop, local_ids, local_order)
applicability = {q_index}
certificate = union_q depth_interval(local_id, q)
```

The materialized prototype stores:

```text
(q_index, global_start, global_stop, global_ids, global_order, depth_intervals_q)
```

once per sampled q-pair. The shared encoding stores the local topology once,
stores q indices as applicability, and replaces per-q depth intervals with
conservative family-union intervals. It is valid only while the union intervals
preserve the stored order.

## New Artifact

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_reuse/summary.json
```

Saved metrics:

```text
q_axis_count = 5
q_pair_count = 25
materialized_cell_count = 25
shared_topology_group_count = 1
materialized_tile_order_metadata_growth = 25.0
shared_tile_order_metadata_growth = 1.0
shared_to_materialized_tile_order_metadata_ratio = 0.11692307692307692
expanded_topology_matches_materialized = true
stable_union_depth_order = true
min_union_depth_order_gap = 0.6033999919891357
```

The top-level goal-progress audit now proves nineteen rows and includes:

```text
local_camera_family_2d_tile_order_reuse
```

## What This Does And Does Not Prove

Proves:

- stable q-family tile/order records can be compressed below sampled-Q storage;
- one shared topology record can expand back to the materialized local topology;
- conservative union depth intervals can certify the same order over the whole
  sampled q-family in this smoke.

Does not prove:

- split-strata compression when active sets differ across q;
- q-continuous interval ranges instead of sampled q-index applicability;
- broad real-scene camera-family metadata behavior.

## Verification

Focused verifier:

```text
8 passed in 0.80s
```

Goal-progress plus tile/order tests:

```text
34 passed in 0.96s
```

## Next Branch

The next metadata branch should deliberately create a q-family with two order
strata. The desired compiler output is not one topology group but a small set:

```text
{topology_group_j, q_region_j, depth_certificate_j}
```

The falsification metric is whether group count grows with actual arrangement
complexity instead of with sampled q-pair count.
