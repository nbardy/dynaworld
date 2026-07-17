# Projective Interval Support Overshoot Debounce

## Context

The active objective is still fast 2D rasters across time from 4D spacetime
primitives, with clean derivatives and maximum sharing of projection, support,
binning, visibility, memory bandwidth, and backward work across frames.

The previous trace-budgeted cap128 artifact was cap-safe but still rebinned
support on every measured live update. The missing evidence was how far traces
actually crossed their compiled support tile boundary.

## Change

Added support-boundary overshoot telemetry:

```text
projective_trace_cell_atlas_support_margin_report(...)
projective_interval_cache_last_support_max_overshoot_px
projective_interval_cache_max_support_max_overshoot_px
```

Added an exact-by-default debounce:

```text
projective_interval.support_stale_overshoot_epsilon
```

When coverage is stale only because support entered a neighboring tile by less
than this epsilon, refresh can skip support rebinning. Invalid active samples
still force stale refresh.

## Tests

Focused targeted gate:

```text
28 passed in 8.06s
```

Full focused projective plus interval-gated suite:

```text
115 passed in 22.20s
```

## Artifacts

Trace-budgeted margin, no debounce:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_margin/summary.md
```

Result: measured support rebins remain `7/7`, but max support overshoot is only
`0.0912px`.

Debounce sweep:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_eps0125/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_eps025/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_eps05/summary.md
```

Measured rows:

```text
epsilon 0.125:
    support_rebins 3/7
    max_overshoot 0.1690px
    final_loss 0.08477679640054703

epsilon 0.25:
    support_rebins 1/7
    max_overshoot 0.2986px
    final_loss 0.08477679640054703

epsilon 0.5:
    support_rebins 0/7
    max_overshoot 0.4932px
    final_loss 0.08477679640054703
    tile_overflow_sum 0
    max_tile_count 70
    no_first_step_ms 1277.4629
```

## Interpretation

The current cap128 churn was mostly a subpixel tile-boundary debounce problem,
not a multi-pixel support-guard failure. This is a nicer failure mode: the
compiler already has almost enough support margin, and exact tile-membership
checking was forcing rebuilds for tiny slivers.

Do not promote `0.5px` globally yet. It is an approximation. It needs visual
and error stress on wide-FOV, near-camera, fast-motion, and occlusion-heavy
cases. But for this smoke it recovers the desired measured-cache behavior:
one rebuild, zero support rebins, zero overflow, identical loss.

## Next

Run a harder synthetic trajectory or orbit-window case with the same debounce
sweep. The falsification condition is visible/metric regression at equal loss
or a larger overshoot distribution that requires a tolerance too large to
justify.
