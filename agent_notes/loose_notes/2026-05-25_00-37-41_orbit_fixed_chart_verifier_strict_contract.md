# Orbit Fixed-Chart Verifier Strict Contract

## Context

The active Gauged UVT goal needs the revolving-camera / screen-fiber lane to
prove more than "a chart exists." The claim is that a known camera orbit can be
compiled into reusable sensor-time traces whose world-side topology and payload
stay fixed while frame count grows, with useful forward/backward reuse and
nonzero gradients through the gauge terms.

The existing `projective_orbit_fixed_chart_scaling_benchmark.py` verifier
already checked the important topology and timing ideas: constant fixed
segment/trace/payload counts, growing per-frame replay counts, zero fallback,
sublinear interval entries, CPU/Metal timing ratios, and fixed-chart autograd
into `ma`, opacity, color, `q_uv`, and temporal `q_uvt`.

The remaining weakness was stale or internally inconsistent report data:
summary fields were checked only by a selected subset, and rows could drift in
ways that still preserved the headline.

## Current Model

For this artifact, the fixed-chart route is a certificate if every frame-count
row satisfies:

```text
interval_ratio = interval_trace_entries / dense_trace_samples
cpu_compile_ms = project_ms + atlas_build_ms
fallback_fraction = 0
if Metal ran:
    mps_atlas_build_ms > 0
    forward_ms > 0
    backward_ms > 0
    grad_coeff_abs_sum > 0
    grad_opacity_abs_sum > 0
    grad_color_abs_sum > 0
    grad_spatial_precision_uv_abs_sum > 0
```

And the per-frame replay route is a true replay baseline only if:

```text
interval_trace_entries = dense_trace_samples
```

The summary is derived evidence:

```text
summary == summarize(sorted fixed rows + sorted per-frame rows)
```

for every current summary key, with `all_fixed_chart_fallback_zero` and
`all_fixed_chart_autograd_q_uv_nonzero` both true.

## What Changed

File:
    `research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py`

The verifier now rejects:

- nonpositive or missing `iterations` and negative `warmup`
- row interval ratios that do not match row counts
- per-frame rows that do not replay all dense trace samples
- nonpositive `project_ms`, `atlas_build_ms`, or `mps_atlas_build_ms`
- `cpu_compile_ms` that is not `project_ms + atlas_build_ms`
- missing direct Metal gradients into coeffs, opacity, color, or spatial precision
- stale summary fields for any current summary key

File:
    `tests/test_star_uvt_projective_orbit_fixed_chart_scaling_benchmark.py`

The fixture now includes direct Metal gradient fields and `mps_atlas_build_ms`.
Mutation tests cover:

- missing direct Metal opacity gradient
- inconsistent interval ratio
- inconsistent CPU compile sum
- stale summary field

## Verification

Syntax:

```text
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py \
  tests/test_star_uvt_projective_orbit_fixed_chart_scaling_benchmark.py
```

Focused verifier:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_orbit_fixed_chart_scaling_benchmark.py -q

10 passed in 28.79s
```

Saved artifact CLI:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json

verified outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
```

Paired orbit-window plus fixed-chart verifier suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_orbit_fixed_chart_scaling_benchmark.py -q

24 passed in 126.68s
```

## Decision Implications

This does not prove arbitrary full-orbit production quality. It does strengthen
the specific revolving-camera evidence lane: fixed screen-fiber charts can hold
world-side topology/payload constant across frame densification, while the
saved report is now protected against stale summaries and row-level timing or
gradient inconsistencies.

Next useful gate:

1. Bring the same strict row/summary discipline to the next real high-motion
   revolving-camera benchmark.
2. Keep the "charts vs fibers" language precise: the gauge/fiber math defines
   the trace representation, while the verifier certifies the accepted gauge
   domains where support, order, memory, and gradients remain valid.
