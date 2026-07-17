# Gauged UVT Goal Contract And Chart Fit

Date: 2026-05-24 01:17:46

## Context

The user asked to work overnight on the spacetime-to-camera UVT renderer idea
and to start with:

```text
what is this? what is the goal?
```

They also re-emphasized the key objective:

```text
4D spacetime primitives that allow fast rasterization into 2D viewport rasters
across time, with clean derivatives, shared compute/memory bandwidth, and
backward-pass reuse so non-pixel cost grows sublinearly with frame count.
```

The existing Gauged UVT Trace Atlas folder already had ten theory subfolders,
the compact `GOAL_META_KEY_MATH.md`, and a Gate A projective trace Metal probe.

## Work Done

Added:

```text
research_notes/gauged_uvt_trace_atlas/00_WHAT_IS_THIS_GOAL.md
```

This is now the first-read contract. It defines the project as a
camera-program compiler, not a video cache:

```text
world spacetime primitive -> UVT sensor-time trace
```

It states the performance target explicitly:

```text
output pixels still cost O(F H W)
world-side project/bin/support/visibility/backward replay should grow
sublinearly with F when camera traces are coherent
```

It also records derivative, memory-bandwidth, revolving-camera, visibility, and
WorldFoam conditions.

Updated:

```text
research_notes/gauged_uvt_trace_atlas/README.md
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
research_notes/gauged_uvt_trace_atlas/03_projective_rational_traces/README.md
research_notes/gauged_uvt_trace_atlas/09_metal_acceptance_plan/README.md
```

These now route future agents through the first-read contract and record that
Gate B has started as a compiler helper.

## Code Work

Extended:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

with:

```text
ProjectiveTraceFit
ProjectiveTraceWindow
fit_projective_trace_polynomial(coeffs, times, degree)
eval_projective_trace_polynomial_fit(fit, times)
split_projective_trace_windows(coeffs, times, degree, thresholds)
```

The helper samples the projective/rational trace and fits local polynomial
charts for:

```text
[u(t), v(t), h_z(t)]
```

It reports:

```text
residual_max_uv
residual_rms_uv
residual_max_depth
denominator_min_abs
valid_fraction
valid_count
```

This is intentionally a compiler-side certificate helper, not a hot renderer
kernel yet. It answers: "can this orbit window be represented by an affine or
quadratic local UVT chart, or should the compiler change gauge/split?"

`split_projective_trace_windows(...)` is the first concrete atlas-window
constructor: it accepts a long interval if the chart residual/denominator/valid
sample thresholds hold, splits when they do not, and marks tiny unresolved
windows as fallback candidates.

Updated the package export in:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
```

and extended:

```text
tests/test_star_uvt_projective_trace.py
```

with tests for:

- exact affine screen traces
- affine residual rejecting curved traces while quadratic accepts them
- denominator-boundary underconstrained windows reporting invalid residuals

## Tests

Focused projective trace suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_trace.py -q
```

Result:

```text
10 passed in 1.06s
```

Renderer import/regression smoke:

```text
PYTHONPATH=src/train uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/uvt_pair_benchmark.py \
  --scenes single_static
```

Key result:

```text
max_rgb_error = 5.960464477539063e-08
mean_rgb_error = 1.1123878485008731e-09
forward_wall_clock_ms = 74.40933299949393
overflow_tile_count = 0
unstable_tile_fraction = 0.0
```

## Current Model

The refined model is:

```text
Gauged UVT is a ray-bundle trace atlas.
Projective/rational gauges are the rich math for revolving cameras.
Residual certificates are not the theory; they are the chart-validity proof.
Fallback is only the guardrail when gauge/atlas choices stop being economical.
```

## Next Useful Gates

1. Measure chart count versus frame count on synthetic circular/revolving
   camera paths.
2. Add support-bound tests: sampled rational traces must stay inside compiled
   tile-time bounds.
3. Add depth monotonicity/order sidecars so visibility strata can be measured.
4. Only then wire rational/projective charts into binning/rendering.
