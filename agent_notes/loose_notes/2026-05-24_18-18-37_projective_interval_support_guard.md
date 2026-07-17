# Projective Interval Support Guard

## Context

The active STAR UVT / Gauged UVT Trace Atlas goal is still:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The previous measured cache artifact proved that a live atlas can avoid full
compatible-atlas rebuilds, but it still repaired support metadata on every live
update:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.md

cadence:  rebuilds=4, live_updates=4, stale_refreshes=4, support_rebins=4
measured: rebuilds=1, live_updates=7, stale_refreshes=7, support_rebins=7
```

## Current Model

The old implementation used one `uv_padding` for two different jobs:

```text
1. required live support checked for correctness
2. conservative support compiled into tile-time metadata
```

That makes the atlas hypersensitive to small optimizer moves: if the current
required support crosses a tile boundary, the compiler rebins; after rebinning
with the same exact padding, the next small move can stale it again.

The implemented split is:

```text
coverage check support = uv_padding
compiled chart support = uv_padding + support_guard_padding
```

In bundle language, `support_guard_padding` is a local chart margin. It is not a
fallback and not a weaker correctness test. It stores a larger coordinate
neighborhood for the same pulled-back trace, while refresh still verifies
coverage with the actual required footprint.

## Code Changes

- `src/train/star_uvt_projective_interval_backend.py`
  - adds `feature_uvt.projective_interval.support_guard_padding`
  - validates it as non-negative
  - exposes `support_uv_padding = uv_padding + support_guard_padding`
  - compiles atlas support using `support_uv_padding`
  - passes base `uv_padding` and guarded `support_uv_padding` into
    `ProjectiveCellIntervalTrainerState`

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py`
  - already carries `support_uv_padding`
  - refresh checks coverage with `uv_padding`
  - support rebin uses `support_uv_padding`

- `tests/test_star_uvt_projective_uvt_producer.py`
  - adds a CPU gate showing a `+9px` live support move stays covered and does
    not rebin when guarded support was compiled.

- `research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py`
  - records `support_guard_padding`
  - optionally sets `tile_capacity`
  - sets `STAR_UVT_TILE_CAPACITY` when a larger tile cap is requested, so the
    Python metadata and Metal kernel agree.

## Evidence

Syntax and focused suites:

```text
py_compile passed for backend, producer test, config test, benchmark, and harness file
tests/test_star_uvt_render_configs.py tests/test_star_uvt_projective_uvt_producer.py
    23 passed in 21.25s

focused projective interval bundle
    114 passed in 17.81s
```

The first guard attempts were negative and important:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard8/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2/summary.md

support_guard_padding=8 at cap128: packed projective interval tile capacity overflow
support_guard_padding=2 at cap128: packed projective interval tile capacity overflow
```

The budgeted guard artifact passes:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_cap256/summary.md

cadence:
    pass=true
    end_loss=0.08477679640054703
    rebuilds=4
    live_updates=4
    stale_refreshes=0
    support_rebins=0
    no_first_step_ms=7468.6249
    max_tile_count=70/256

measured:
    pass=true
    end_loss=0.08477679640054703
    rebuilds=1
    live_updates=7
    stale_refreshes=0
    support_rebins=0
    no_first_step_ms=2496.7392
    max_tile_count=70/256
```

Interpretation:

```text
support guard solves the churn in this route
but only after spending tile-capacity headroom
```

## Branches

Hypothesis A:
    Fixed support guards are the right operational chart margin.
Why it might be true:
    The cap256 artifact removes stale refreshes/rebins without changing loss.
What would make it false:
    Larger scenes overflow or slow down so much that churn was cheaper.
Cheap test:
    Sweep guard in `{0.5,1,2,4}` and cap in `{128,192,256}` while recording
    max/p95 tile count, stale_refreshes, support_rebins, and no-first-step time.

Hypothesis B:
    Guards should be adaptive per cell/primitive, not a global scalar.
Why it might be true:
    Guard2 already overflows cap128 globally even though the final cap256
    max_tile_count is only 70/256 after trainer support pruning/reporting.
What would make it false:
    A small global guard clears churn across real rows with acceptable cap128 or
    cap192 occupancy.
Cheap test:
    Add a diagnostic that reports which tile-time cells are responsible for
    guard-induced overflow and whether local guard clipping would avoid it.

Hypothesis C:
    The real next win is not guards but local support charts/root intervals.
Why it might be true:
    A global guard widens every support list; support-boundary roots already
    know when each tile actually needs the primitive.
What would make it false:
    Root interval splitting costs more metadata than a guarded stable cell.
Cheap test:
    Compare fixed guard versus root-aware time-local guard on the same
    cache-policy benchmark.

## Decision Implications

The implementation should keep `support_guard_padding`, but it should not become
a large default. The next gate is adaptive/budget-aware support guarding:

```text
choose guard <= available tile headroom
or split/refit the offending chart
or keep base support and accept measured refresh
```

This is the same philosophical answer as the fiber-bundle theory: rich gauges
reduce fallback and churn only when their coordinate neighborhoods respect the
packed memory budget.

## Follow-up: Budgeted Global Guard

The next implementation added:

```text
projective_interval.support_guard_policy = fixed | budgeted
projective_interval.support_guard_bisect_steps
```

`fixed` keeps the previous behavior. `budgeted` treats
`support_guard_padding` as a maximum and bisects downward until packed interval
tile bins fit the configured `feature_uvt.tile_capacity`. The same budgeted
path is used when trainer-state refresh rebins stale support.

New gates:

```text
tests/test_star_uvt_projective_uvt_producer.py::test_projective_interval_budgeted_support_guard_respects_tile_capacity
tests/test_star_uvt_render_configs.py
tests/test_star_uvt_projective_uvt_producer.py
focused projective interval bundle
```

Results:

```text
tests/test_star_uvt_render_configs.py tests/test_star_uvt_projective_uvt_producer.py
    24 passed in 18.16s

focused projective interval bundle
    115 passed in 27.69s

fresh projective plus interval-gated rerun after report/handoff updates
    112 passed in 10.12s
```

Saved artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_budgeted_cap128/summary.md
```

Measured row:

```text
pass=true
end_loss=0.08477679640054703
tile_capacity=128
tile_overflow_sum=0
rebuilds=1
live_updates=7
staleness_checks=7
stale_refreshes=6
support_rebins=6
no_first_step_ms=6107.7410
```

Cadence row timed out at `240s`.

Interpretation:

```text
budgeted global guard prevents hard cap128 overflow
but does not create enough margin to eliminate churn at cap128
and repeated global search is too expensive for cadence rebuilds
```

Updated decision implication:

```text
next guard policy should be local/headroom-aware per tile/trace,
or should split/refit only the local offenders.
```
