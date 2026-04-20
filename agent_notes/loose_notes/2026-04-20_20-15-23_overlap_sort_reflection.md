# Overlap Sort Reflection

## Context

We added overlap-key pressure diagnostics after discussing Speedy-Splat,
SnugBox, AccuTile, pruning, and the radix-sort bottleneck. The motivating
question was whether our Mac/Taichi sorting problems could be fixed by tighter
tile intersection instead of a new sort/raster architecture.

Files touched in this work:

- `src/train/renderers/overlap_metrics.py`
- `src/benchmarks/splat_renderer_benchmark.py`
- `src/train/debug_metrics.py`
- `src/benchmark_configs/splat_renderer_*.jsonc`
- `agent_notes/key_learnings.md`

Prior state:

- We had renderer timings, but mostly by `G` splat count.
- We did not measure `K`, the actual Gaussian-tile key count.
- Without `K`, the phrase "sorting bottleneck" was underspecified.

## Current Model

Current belief:
    On random splat benchmarks, Taichi's current OBB tile query is not sloppy
    enough for AccuTile/exact intersection to be the main speed unlock.

Confidence:
    Medium for random benchmarks, low-to-medium for trained Dynaworld outputs.

Evidence:
    The exact conic/tile test reduced Taichi OBB key count by only about 8-10%
    in the checked random cases:

    - 64x64, G=1024: Taichi/exact = 1.090x
    - 128x128, G=65536: Taichi/exact = 1.093x
    - 1024x1024, G=65536: Taichi/exact = 1.099x

Inference:
    If random splats are representative, speeding up Taichi by 2x+ probably
    requires reducing kernel boundaries, avoiding global-memory sort stages, or
    fusing tile-local sort/raster, not merely replacing the OBB tile test with
    exact AccuTile.

Could be wrong if:
    Real trained outputs create large, transparent floaters whose projected
    support is much worse than our random splats. In that case, pruning or
    better exact tile tests could be first-order.

## Definitions

Let:

- `G` = number of projected splats.
- `K` = number of Gaussian-tile overlap keys materialized for sorting/raster.
- `K/G` = duplication factor, or average number of tiles touched per splat.
- `tile_count` = number of screen tiles.
- `splats_per_tile` = occupancy distribution after binning.

Two configurations can have the same `G` and very different cost if their
`K/G` or tile occupancy tails differ.

The right diagnostic question is not:

```text
How many splats do we render?
```

It is:

```text
How many tile-local work items do we actually create, and how skewed are they?
```

## Backtrack: Custom Rect Ratio

Initial impression:
    The custom tiled renderer looked about 2.8x sloppier than exact conic
    intersection.

Status:
    Weakened.

Evidence:
    That comparison mixed tile sizes. Custom used 8px tiles, while Taichi/exact
    used 16px tiles. After adding `exact_conic_custom_tile`, the same-tile
    comparison for 64x64/G=512 was:

    - custom_rect K = 4757
    - exact_conic_custom_tile K = 3688
    - custom/exact = 1.290x

Replacement model:
    Custom rectangular bounds are still sloppier, but in the checked case the
    same-tile penalty was around 29%, not 180%. Tile size must be part of every
    `K` comparison.

Decision implication:
    Keep exact conic stats tied to the renderer tile size before judging culling
    quality.

## Branches

### Branch A: Sort Is Dominated By Key Count

Hypothesis:
    Runtime is bad because `K` is too large.

Why it might be true:
    Standard 3DGS sort cost tracks materialized Gaussian-tile overlaps, and
    large floaters can multiply `K`.

What would make it false:
    `K/G` stays low, e.g. around 3-5, and exact intersection only reduces keys
    by single-digit or low-double-digit percentages.

Cheap test:
    Run benchmark/training diagnostics and inspect:

    - `total_overlap_keys`
    - `duplication_factor`
    - `max_tiles_per_splat`
    - `large_splat_count`

If supported:
    Prioritize pruning, projected-radius constraints, opacity/scale controls,
    and exact AccuTile/SnugBox.

If invalidated:
    Prioritize fused tile-local sort/raster and lower kernel launch/global-memory
    overhead.

Current status:
    Weakened for random splats; unresolved for trained Dynaworld outputs.

### Branch B: Sort Is Dominated By Kernel Structure

Hypothesis:
    Runtime is bad because the current Mac path stages bin/sort/raster through
    separate kernels and global memory, not because it creates too many keys.

Why it might be true:
    The Taichi OBB query is close to exact on random tests, and previous
    Taichi-side bucket/global sorts were slower than the Torch/MPS sort
    reference.

What would make it false:
    Real trained outputs show `K/G` or `max_tiles_per_splat` much larger than
    random tests, especially due to floaters.

Cheap test:
    Enable `logging.with_metrics.renderer` during training and watch
    `TileDiag/ExactConic_*` and `TileDiag/CustomRect*`.

If supported:
    Move speed work toward raw Metal/MLX or a native extension with fused
    threadgroup-memory tile-local sort and raster.

If invalidated:
    Fix scene/splat generation first.

Current status:
    Favored for random splats.

### Branch C: Training Produces Pathological Floaters

Hypothesis:
    Random splats are too clean, while trained dynamic-token outputs may produce
    large semi-transparent splats that dominate sort pressure.

Why it might be true:
    Training models can exploit blurry or transparent large supports as cheap
    loss reducers, especially early in optimization.

What would make it false:
    Training diagnostics show `K/G` similar to random benchmarks and no large
    tail in `max_tiles_per_splat`.

Cheap test:
    Run a short training config with renderer metrics enabled:

    ```jsonc
    "logging": {
      "with_metrics": {
        "renderer": true,
        "every": 25,
        "print_summary": true,
        "wandb": true
      }
    }
    ```

If supported:
    Add pruning/floater controls before new sort architecture:

    - projected tile-span penalty
    - opacity-weighted support penalty
    - max projected radius clamp or schedule
    - periodic low-contribution pruning

If invalidated:
    Do not spend time on pruning as a renderer-speed fix yet.

Current status:
    Open.

## Falsification Tests

1. Random benchmark K/G sweep

Command shape:

```bash
uv run python src/benchmarks/splat_renderer_benchmark.py \
  --config src/benchmark_configs/splat_renderer_64k_throughput.jsonc \
  --renderers taichi \
  --resolutions 128,512,1024 \
  --splat-counts 65536,131072,262144 \
  --sets-per-case 1 \
  --warmup-iters 1 \
  --timed-iters 1 \
  --forward-only \
  --no-save-images
```

Supports key-count hypothesis if:
    `K/G` rises sharply with resolution/count or large-splat tails appear.

Weakens key-count hypothesis if:
    `K/G` remains around 3-5 and exact/OBB ratios stay near 1.1x.

2. Real training K/G probe

Use:
    `logging.with_metrics.renderer=true`.

Supports floater hypothesis if:
    `TileDiag/ExactConic_max_tiles_per_splat_max` or
    `TileDiag/ExactConic_duplication_factor_mean` is much larger than random
    probes.

Weakens floater hypothesis if:
    trained outputs match random K/G and tails.

3. Same-tile culling ablation

Compare only queries with equal tile size:

- custom rect vs exact with custom tile size
- Taichi OBB vs exact with Taichi tile size

Supports AccuTile work if:
    same-tile exact saves a large fraction of keys, say >25-30%, on target data.

Weakens AccuTile work if:
    same-tile exact saves only ~10%.

## Decision Implications

Near-term:
    Use the new telemetry before changing sort architecture. Run it on trained
    outputs, not just random splats.

If trained `K/G` is low:
    The next meaningful speed work is fused tile-local sort+raster in raw Metal
    or another lower-level path. Taichi-side separated kernels are unlikely to
    become the fast path by small tweaks.

If trained `K/G` is high:
    Prioritize pruning and projected-support control. In that branch,
    Speedy-Splat/FastGS-style ideas matter before shader fusion.

If same-tile exact saves >25% keys:
    Port exact conic/AccuTile-style intersection into the mapper.

If same-tile exact saves ~10% keys:
    Treat exact intersection as cleanup/quality of implementation, not the main
    speed project.

## Open Questions

- What is `K/G` for actual decoded Dynaworld frames at steps 0, 25, 100, and
  late training?
- Are high tile occupancies caused by many small splats in the same screen
  region or by a few large floaters?
- Does tile occupancy skew correlate better with wall time than total `K`?
- What is the overhead of computing overlap metrics during training at 64k+
  splats, and should it be sampled on a subset of frames/splats?
- Should the benchmark add optional histograms/artifact dumps for the worst
  splats by tile count?

## Bottom Line

The work went well because it replaced a vague bottleneck theory with a
measurable quantity. The surprising result is that random Taichi splats are not
creating wildly excessive keys. That does not make Speedy-Splat irrelevant, but
it changes the order of operations: measure real trained `K/G` first, then choose
between pruning/culling work and fused Metal architecture work.
