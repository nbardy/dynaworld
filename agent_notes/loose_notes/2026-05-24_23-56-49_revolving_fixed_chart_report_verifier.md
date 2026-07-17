# 2026-05-24 23:56:49 - Revolving Fixed-Chart Report Verifier

## Context

Continuation of the Gauged UVT Trace Atlas goal. The cache-policy,
tail-alpha, anisotropic-tail, and trained high-motion scaling artifacts already
had saved-report verifiers. The revolving-camera lane still had a measured
artifact but no executable report contract:

```text
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
```

That artifact is important because it addresses the user's core objection to
"charts": a revolving camera should be carried by the projective/screen-fiber
gauge, with local domains serving as event-certified gauge regions. The saved
report shows exactly that on a synthetic elevated orbit.

## Current Model

For the orbit fixture:

```text
fixed_chart:
    temporal chart count per tube = fixed
    segment/trace/payload count should not grow with frame count

per_frame:
    one segment per frame
    segment/trace/payload count grows with frame count
```

The executable claim is narrower than "all video rendering is sublinear":

```text
unavoidable pixel samples grow with frames
world-side projection/support/binning/atlas topology is reused across frames
backward still reaches the same shared orbit trace parameters
```

## What Changed

Added to:

```text
research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py
```

New exports:

```text
verify_orbit_fixed_chart_scaling_report(report) -> list[str]
assert_orbit_fixed_chart_scaling_report(report)
--verify-report <summary.json>
```

Added focused tests:

```text
tests/test_star_uvt_projective_orbit_fixed_chart_scaling_benchmark.py
```

## Contract

The verifier checks:

```text
topology:
    frame_counts strictly increase
    one fixed_chart and one per_frame row per frame count
    fixed_chart segment_count, trace_count, and payload bytes are constant
    per_frame segment_count, trace_count, and payload bytes strictly grow

support:
    fallback_fraction = 0 for every row
    fixed_chart interval ratios are non-increasing
    final fixed_chart interval ratio falls by at least 65%
    fixed_chart interval entries grow slower than dense samples and stay <2x

timing:
    final fixed/per-frame CPU compile ratio < 0.5
    final fixed/per-frame forward ratio < 0.5 when Metal timings are present
    final fixed/per-frame backward ratio < 0.5 when Metal timings are present

derivatives:
    fixed-chart autograd reaches ma, opacity, color
    fixed-chart autograd reaches q_uv and temporal q_uvt terms
    direct backward reaches spatial_precision_uv
```

Mutation tests reject:

- fixed trace count growth
- fallback
- zero `q_uv` orbit metric gradient
- lost fixed/per-frame forward timing win

## Verification

Focused verifier:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_orbit_fixed_chart_scaling_benchmark.py -q
```

Result:

```text
6 passed in 15.60s
```

Saved artifact CLI:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
```

Result:

```text
verified outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
```

Underlying orbit suite plus verifier:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_orbit_fixed_chart_scaling_benchmark.py \
  tests/test_star_uvt_projective_orbit_windows.py -q
```

Result:

```text
20 passed in 179.84s
```

## Decision Implication

This strengthens the answer to "do fibers/gauges handle revolving cameras?"
The current evidence says: yes, for the synthetic elevated orbit fixture, the
screen-fiber metric is carried in `q_uvt`, the fixed chart topology is reused
as frame samples densify, and Metal/autograd still reaches the shared orbit
parameters. The next falsification step is to move this from synthetic orbit
to real high-motion camera views or a richer WorldFoam/instance scene.
