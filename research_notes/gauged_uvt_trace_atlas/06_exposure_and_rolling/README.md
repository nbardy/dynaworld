# 06 - Exposure And Rolling Shutter

Finite exposure and rolling shutter live naturally on the same base:

```text
B = Omega x T
```

For frame `k`:

```text
I_k(u,v) = integral w_k(u,v,tau) I(u,v,tau) d tau
```

Global shutter:

```text
w_k(u,v,tau) = w_k(tau)
```

Rolling shutter:

```text
t_sensor(u,v,tau) = t0 + r(v) + tau
```

or, for scan direction `d`:

```text
t_sensor(u,v,tau) = t0 + r(dot((u,v), d)) + tau
```

The camera map is:

```text
Gamma(u,v,tau,z)
```

with row/time coupling inside `Gamma`, not as a postprocess.

## Gauge Consequence

Rolling shutter makes time and image coordinates inseparable:

```text
partial Gamma / partial v
```

contains both spatial ray change and sensor-time change. A charted bundle
handles this because `y=(u,v,tau)` is the base, and gauges are allowed to mix
all base coordinates.

## Integration Order

Visibility is nonlinear. Do not integrate primitive opacity over exposure and
then composite unless order is proven stable. Preferred:

```text
for tau samples / quadrature:
    evaluate visible composite I(u,v,tau)
integrate I over tau
```

Analytic exposure integration is valid only inside a chart/stratum with stable
or safely commuted visibility.

## Current Evidence

The focused quadrature report now lives at:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.md
```

It verifies the operational contract:

```text
frame = integral_tau Composite(TraceAtlas(u,v,tau)) d tau
```

Finite-exposure midpoint samples lower into a sample-indexed interval atlas and
match the CPU/Torch direct sensor-time oracle exactly on the focused scene.
Rolling shutter lowers per-row schedules into one shared unique-time schedule
plus `row_weights[Q,H]`; the saved artifact records `7` unique sample times
for `8` row samples (`0.875` ratio) with exact rowwise/batched CPU parity.

On this machine the available Metal paths also match the CPU oracle:

```text
finite interval Metal max error:       5.96e-8
rolling row-weighted Metal max error:  2.98e-8
finite mixed fallback max error:       2.98e-8
rolling mixed fallback max error:      2.98e-8
```

The mixed cases intentionally mark visibility-ambiguous tile/sample cells as
`visibility_ambiguous_depth`, render non-fallback regions through interval
Metal, patch fallback regions with live-depth reference ordering, then apply
the exposure or row-weight accumulation.

The matching backward report is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.md
```

It verifies the adjoint rule:

```text
dL/d sample_image[q,row] = weight[q,row] * dL/d final_image[row]
```

For finite exposure, `weight[q,row]` is just the scalar quadrature weight. For
rolling shutter, it is the row-weight matrix produced by the shared unique-time
lowering. The report then calls one interval-cell VJP on those sample adjoints.
On the saved MPS run, finite and rolling Metal gradients match Torch autograd
on the lowered interval atlas with max absolute error `1.43e-6` and max
relative error `6.38e-7`. The strict verifier recomputes the rolling reuse
ratio, requires positive sample image/adjoint support, checks nonzero
coeff/opacity/color reference gradients, validates Metal aggregate errors
against their subrows, and recomputes the summary; focused tests pass
`11 passed in 25.19s`, and the saved artifact verifies by CLI.

The mixed fallback backward report is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.md
```

It locks the ambiguous-visibility case. Marked `visibility_ambiguous_depth`
tile/sample regions are not detached oracle patches: non-fallback regions use
the interval Metal autograd wrapper, fallback regions use live-depth Torch
reference compositing on the same trace tensors, and the patched sample
adjoints are accumulated with the same exposure or rolling row weights. The
saved MPS artifact has both finite and rolling mixed backward cases active,
fallback fraction `0.5`, max output error `5.96e-8`, max gradient absolute
error `2.15e-6`, max gradient relative error `7.41e-7`, and rolling row-time
reuse `11/12`.

## Tests

- fast horizontal pan with rolling shutter
- orbiting camera with row-dependent time
- finite exposure through an occlusion swap

Report:

```text
analytic_vs_sampled_error
visibility_stratum_crossings_per_frame
chart_count
fallback_fraction
```
