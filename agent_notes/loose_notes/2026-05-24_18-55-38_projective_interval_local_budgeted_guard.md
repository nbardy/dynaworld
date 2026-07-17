# Projective Interval Local-Budgeted Guard

## Context

The active objective is still the same: fast 2D rasters across time from 4D
spacetime primitives, sharing projection/support/binning/visibility/backward
work across frames. The previous fixed guard proved that widening compiled
support can eliminate stale support rebins, but only with cap256. The first
global budgeted guard avoided cap128 overflow by scalar bisection, but remained
churny and slow.

## Change

Added a third support guard policy:

```text
projective_interval.support_guard_policy = "local_budgeted"
```

Implementation:

- `src/train/star_uvt_projective_interval_backend.py`
  - validates `fixed | budgeted | local_budgeted`
  - compiles full target guard first
  - detects overflowing packed tile coordinates
  - compiles base support
  - replaces only overflowing target tiles with base-support cells
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py`
  - threads the same policy through trainer-state refresh
  - applies the same local target/base tile mixing after optimizer movement
- `tests/test_star_uvt_projective_uvt_producer.py`
  - adds a headroom fixture proving local-budgeted preserves guard cells in a
    non-overflow tile while downgrading crowded tiles to fit capacity
- `research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py`
  - accepts `local_budgeted`
  - records tile overflow and max tile count in future markdown summaries

## Evidence

Focused config/producer:

```text
25 passed in 21.64s
```

Focused projective plus interval-gated trainer suite:

```text
113 passed in 13.77s
```

Artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_local_budgeted_cap128_explicit/summary.md
```

Rows:

```text
cadence:
    pass True
    end_loss 0.08477679640054703
    tile_capacity 128
    tile_overflow_sum 0
    max_tile_count 70
    support_rebins 4
    no_first_step_ms 4025.4261

measured:
    pass True
    end_loss 0.08477679640054703
    tile_capacity 128
    tile_overflow_sum 0
    max_tile_count 70
    rebuilds 1
    support_rebins 7
    no_first_step_ms 2468.3305
```

## Interpretation

`local_budgeted` is the right cap-safety structure and is much cheaper than
global bisection. It preserves guarded support where tile headroom exists and
keeps cap128 packable.

It does not solve ordinary tube-motion churn. The measured row still rebins
support on every live update (`7/7`). That means the stale regions are probably
inside the same crowded tiles that were downgraded to base support. Tile-level
replacement is too coarse.

## Next

The next guard should allocate headroom per trace/cell within crowded tiles:

```text
choose guard_i,C such that
    base_count(C) + count({i with expanded support entering C}) <= tile_capacity
```

Traces that cannot receive guard headroom should split/refit, use a smaller
motion-dependent guard, or fall back locally. This keeps the mathematical goal
intact: preserve reusable UVT traces where the camera bundle is stable, and pay
per-frame/per-sample costs only in the true high-complexity regions.
