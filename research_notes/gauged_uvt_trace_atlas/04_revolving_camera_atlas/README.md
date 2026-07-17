# 04 - Revolving Camera Gauge-Domain Atlas

A full revolve is not one UVT chart in the weak "fit patch" sense. It is an
event-certified gauge-domain atlas:

```text
{ C_a, chi_a, certificates_a, transition_ab }_a
```

over sensor-time. A domain can cover an orbit window when the chosen gauge keeps:

```text
projected trace complexity low
denominator safely away from zero
visibility order locally stratified
memory bounded
```

The domain is not just a speed trick. It is the region where the compiler can
certify support, active tiles, depth/order, interval gates, and backward support.

## Orbit Window Coordinates

For camera angle `theta(t)`, use charts centered at:

```text
theta_a
```

with local time:

```text
tau_a = theta(t) - theta_a
```

or a normalized window coordinate:

```text
r_a = (theta(t) - theta_a) / Delta_theta_a
```

The chart does not need to make the whole orbit linear. It only needs a
low-order rational expression over its window.

## Transition Maps

On overlap:

```text
C_a cap C_b != empty
```

transition maps relate fiber coordinates:

```text
z_b = h_ab(y, z_a)
```

and trace coordinates:

```text
alpha_{i,b}(y) = alpha_{i,a}(y)
z_hat_{i,b}(y) = h_ab(y, z_hat_{i,a}(y))
```

Depth order is preserved if:

```text
partial h_ab / partial z_a > 0
```

Otherwise the overlap must be treated as a visibility/gauge boundary.

## Revolve-Specific Boundaries

- object behind camera
- projection through image infinity
- primitive crossing the near plane
- disocclusion edge
- active-set change as the object turns around
- depth-order swap in foreground/background layers

These are geometric boundaries, not bugs.

## Practical Compiler Rule

For each orbit window, try gauges in this order:

```text
projective denominator
inverse depth
ordinary depth
object-local / foam-local
```

Then choose between:

```text
raise rational order
split orbit window
split primitive support
fallback to local per-frame renderer
```

The first three are mathematical refinement. The last is only a safety rail.

## Verified Fixed-Chart Scaling Contract

The current measured revolving-camera artifact is:

```text
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
```

Its verifier lives in:

```text
research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py
```

The contract now treats the orbit atlas as a report-level certificate, not just
as a timing table. It checks:

```text
fixed charts:
    segment_count, trace_count, and payload bytes stay constant through 32 frames
    fallback_fraction stays 0
    interval entries grow slower than dense samples

per-frame replay:
    segment_count, trace_count, and payload bytes grow with frame count
    interval entries equal dense samples

row consistency:
    interval_ratio = interval_trace_entries / dense_trace_samples
    cpu_compile_ms = project_ms + atlas_build_ms
    current summary fields match recomputed summary fields

derivatives:
    direct Metal backward reaches coeffs, opacity, color, and spatial precision
    fixed-chart autograd reaches ma, opacity, color, q_uv, and temporal q_uvt
```

Verification:

```text
focused fixed-chart verifier: 10 passed in 28.79s
saved artifact CLI: verified
orbit-window plus fixed-chart verifier suite: 24 passed in 126.68s
```
