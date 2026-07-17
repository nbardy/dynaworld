# Projective Interval Trace-Budgeted Guard

## Context

The active objective remains fast 2D rasters across time from 4D spacetime
primitives, with clean derivatives and maximum reuse of projection, support,
binning, visibility, and backward work across frames.

The previous `local_budgeted` guard was cap-safe and cheap, but it downgraded
whole overflowing tiles to base support. That preserved capacity but left
support rebins at `7/7` in measured mode.

## Change

Added:

```text
projective_interval.support_guard_policy = "trace_budgeted"
```

Behavior:

1. Compile target guard support.
2. Detect overflowing packed tiles.
3. Compile base support.
4. In overflowing tiles, keep base-active trace ids.
5. Spend remaining tile slots on deterministic extra guarded trace ids.
6. If that still overflows, fall back to `local_budgeted`, then global bisection.

Touched paths:

- `src/train/star_uvt_projective_interval_backend.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py`
- `tests/test_star_uvt_render_configs.py`
- `tests/test_star_uvt_projective_uvt_producer.py`
- `research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py`

## Verification

Focused config/producer:

```text
26 passed in 14.89s
```

Focused projective plus interval-gated trainer suite:

```text
114 passed in 15.82s
```

Artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_rerun/summary.md
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
    no_first_step_ms 7326.1793

measured:
    pass True
    end_loss 0.08477679640054703
    tile_capacity 128
    tile_overflow_sum 0
    max_tile_count 70
    rebuilds 1
    support_rebins 7
    no_first_step_ms 2460.0213
```

## Interpretation

Trace-budgeted allocation works mechanically and is cap-safe. The synthetic
test proves it spends crowded-tile headroom instead of downgrading the whole
tile to base support.

The real cap128 benchmark still rebins support on every measured live update.
This weakens the hypothesis that slot allocation alone is the missing piece.
The next model should use support-event-root margins and optimizer displacement:

```text
guard_i,C >= predicted_update_displacement_i before next refresh
```

subject to tile capacity. Fixed guard2 at cap256 already proves that enough
slack can eliminate churn; cap128 budget policies prove safety but not enough
slack for this trajectory.

## Next

Instrument the refresh path to report the nearest missing support boundary/root
margin for stale traces, then choose guard allocation by the ratio:

```text
predicted_update_displacement_i / root_margin_i,C
```

That will tell whether cap128 needs smarter allocation, smaller optimizer
steps, split/refit, or whether the honest solution for this row is cap256.
