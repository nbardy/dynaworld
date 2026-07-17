# 03 - Projective Rational Traces

Current affine UVT uses:

```text
u(t) ~= u0 + u1 t
v(t) ~= v0 + v1 t
depth(t) ~= z0 + z1 t
```

A revolving camera should instead start in homogeneous coordinates:

```text
h(t) = K(t) [R(t)|T(t)] X(t)
h(t) = (h_u(t), h_v(t), h_z(t))
```

Then:

```text
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
depth_gauge(t) = h_z(t)
```

If `K`, `R`, `T`, and object motion are locally polynomial or fitted by a low
order basis, then `h_u,h_v,h_z` are polynomial and the screen trace is rational.

## Quadratic Homogeneous Probe

The first implementation slice stores:

```text
coeffs_i =
  [u0,u1,u2, v0,v1,v2, z0,z1,z2]
```

and evaluates:

```text
h_u(t) = u0 + u1 t + u2 t^2
h_v(t) = v0 + v1 t + v2 t^2
h_z(t) = z0 + z1 t + z2 t^2
screen_i(t) = (h_u/h_z, h_v/h_z)
```

The output includes a denominator sign/validity channel:

```text
[u, v, h_z, valid_sign]
```

where:

```text
valid_sign = 1   if h_z > eps
valid_sign = -1  if h_z < -eps
valid_sign = 0   if |h_z| <= eps
```

This is not yet rendering. It is the GPU primitive needed before binned rational
trace support.

## Polynomial Chart Fit Probe

The next implementation slice is a compiler-side residual helper:

```text
fit_projective_trace_polynomial(coeffs, times, degree)
```

It samples the rational trace and fits:

```text
[u(t), v(t), h_z(t)] ~= sum_k a_k ((t - t_c) / t_s)^k
```

for `degree = 1` or `degree = 2`. It returns:

```text
poly_coeffs
residual_max_uv
residual_rms_uv
residual_max_depth
denominator_min_abs
denominator_has_root
valid_fraction
valid_count
```

This does not replace the projective gauge. It is a certificate for deciding
whether a chosen orbit window can be consumed by the existing affine/quadratic
UVT machinery, or whether the compiler should choose a different gauge/window.

The first atlas-window splitter wraps this certificate:

```text
split_projective_trace_windows(coeffs, times, degree, thresholds)
```

It recursively splits a time interval until the local polynomial chart satisfies
residual, denominator, and valid-sample thresholds, or the interval is too small
and must be marked as an unresolved compiler fallback candidate.

As of the continuous-denominator correction, denominator validity is checked
over the whole chart domain, not only at frame samples. The compiler normalizes
the queried interval, evaluates the quadratic at its endpoints and stationary
point, and records both:

```text
denominator_has_root  = zero or numerical root-boundary event in the interval
denominator_min_abs   = continuous min_t |h_z(t)| in the interval
```

Thus a root is a `denominator_boundary`, even when every sampled frame is
valid, and a root-free but near-zero denominator still fails a requested chart
margin.

## Pixel-Varying Conditional Depth Probe

The first richer-depth metadata contract is now explicit. A cell trace still
stores the center trajectory:

```text
u_c(t), v_c(t), z_c(t)
```

and may additionally carry a screen-fiber depth plane:

```text
z(u,v,t) =
  z_c(t)
  + z_u(t) * (u - u_c(t))
  + z_v(t) * (v - v_c(t))
```

with time-polynomial slopes:

```text
depth_affine_uv[N,6] =
  [zu0, zu1, zu2, zv0, zv1, zv2]

z_u(t) = zu0 + zu1 t + zu2 t^2
z_v(t) = zv0 + zv1 t + zv2 t^2
```

This is the practical "UVT screen fiber" object: over a local projective gauge,
each screen coordinate has a conditional depth section, not only a scalar
center depth. `eval_projective_trace_cell_depth_at_uv_torch(...)` evaluates it
for compiler-side certificates; atlas transforms, support-event rebinning,
quadrature lowering, and detached CPU conversion preserve it.

Important boundary: current interval Metal visibility/sorting still consumes
the existing scalar/cell depth metadata. The depth-plane field is now a tested
compiler contract, not yet a claim that production Metal sorting is
pixel-varying.

## Failure Modes

- `h_z = 0` is a projective chart boundary.
- Large `|d/dt (h_u/h_z)|` near zero denominator requires a different chart.
- A rational center alone does not solve footprint covariance; the footprint
  still needs local sigma-point or Schur-complement support fitting.

## Acceptance

Metal and Torch evaluation must match within `1e-5` on:

- affine case (`z1=z2=0`)
- mild orbit-like denominator variation
- near-boundary invalid samples

The chart-fit probe must also prove:

- affine screen traces fit with near-zero residual
- quadratic screen traces are rejected by affine residual and accepted by
  quadratic residual
- underconstrained denominator-boundary windows report invalid residuals
- the window splitter accepts a whole affine interval, splits a curved interval
  into accepted local charts, and marks denominator-boundary windows unresolved
- a root-free quadratic with an unsafe between-sample vertex fails the continuous
  denominator-margin gate
- a small raw-time quadratic coefficient still detects roots after interval
  normalization
- a stereographic yaw-orbit trace has chart counts that grow sublinearly with
  frame density and increase with orbit-span complexity at fixed frame count
