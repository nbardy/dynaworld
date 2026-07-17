# 00 - Bundle Foundations

Working claim:

```text
STAR UVT should be viewed as a local chart of a camera-ray bundle, not as the
global representation itself.
```

Let:

```text
B = Omega x T
y = (u, v, tau) in B
```

For each sensor-time point `y`, the camera defines a ray fiber:

```text
F_y = pi^{-1}(y)
```

The total camera-ray space is:

```text
pi: E_Gamma -> B
```

A local chart/gauge over `C_a subset B` is:

```text
chi_a: pi^{-1}(C_a) -> C_a x D_a
```

with local coordinate:

```text
(y, z_a) = (u, v, tau, z_a)
```

The camera program is a smooth map:

```text
Gamma: E_Gamma -> M
```

where:

```text
M = R^3 x R
```

for ordinary world spacetime.

Given a world primitive:

```text
rho_i: M -> R_+
c_i: M -> R^k
```

the camera pulls it back to the ray bundle:

```text
tilde_rho_i = Gamma^* rho_i
tilde_rho_i(y, z_a) = rho_i(Gamma_a(y, z_a))
```

The UVT trace is the fiber pushforward:

```text
bar_rho_i = pi_* tilde_rho_i
bar_rho_i(y) = integral_{F_y} rho_i(Gamma(y, z)) dmu_y(z)
```

So the compact invariant equation is:

```text
bar_rho_i = pi_* Gamma^* rho_i
```

The current screen-time Gaussian is a local approximation to `bar_rho_i` in one
gauge. This matters because a revolving camera can be globally hard while being
locally simple in multiple gauges.

## Invariants

- `bar_rho_i(y)` is coordinate-invariant if the fiber measure transforms with
  the chart Jacobian.
- Depth order is invariant under monotone fiber coordinate changes:

```text
z_b = h_ab(y, z_a),   partial h_ab / partial z_a > 0
```

- A chart boundary is not a failure. It is the place where the current gauge no
  longer provides a low-complexity coordinate expression.

## Falsification Test

Compile the same static primitive under two monotone depth gauges and compare:

```text
pi_* Gamma^* rho
```

as a function of `(u,v,tau)`. Differences beyond quadrature tolerance indicate
that the implementation forgot the measure/Jacobian or used a non-monotone
depth transform without marking a chart boundary.

## Implemented Gauge-Invariance Probe

The falsification test above now has a saved revolving-camera artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.md
```

The script:

```text
research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_invariance_report.py
```

integrates the same spacetime Gaussian through an orbiting pinhole camera in
two fiber gauges:

```text
z = ordinary camera-forward depth
r = log(z)
```

The invariant comparison is:

```text
integral rho(Gamma(y,z)) dz
==
integral rho(Gamma(y,exp(r))) exp(r) dr
```

where `exp(r)` is the fiber-measure Jacobian. The saved artifact reports
`max_rel_error = 3.50e-13` across five sensor-time samples, while deliberately
omitting the Jacobian produces at least `0.600` relative error. It also verifies
that the monotone log-depth gauge preserves depth order and that an
orientation-reversing `-log(z)` gauge flips order and must be treated as a
visibility/gauge boundary.

Verification:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_invariance_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json
```

This is a math-contract gate, not a renderer-speed gate. Its job is to protect
the statement that STAR UVT is one local gauge expression of the camera-ray
bundle atlas. The verifier now also recomputes the report summary from the
rows/order certificate and rejects stale row errors, bad near/far/sample
metadata, lost missing-Jacobian control, and non-monotone gauge derivatives.

## Implemented Gauge-Derivative Probe

The derivative side of the same contract now has a saved artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.md
```

The script:

```text
research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_gradient_report.py
```

uses the same revolving-camera depth/log-depth gauges, but differentiates a
weighted trace objective with respect to primitive parameters:

```text
mean
log_precision
log_amplitude
```

The saved result reports:

```text
max_gradient_rel_error with Jacobian: 2.33e-12
min gradient relative error without Jacobian: 0.592
finite-difference check for mean[0]: 1.42e-10 relative error
```

Verification:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_gradient_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json
```

This protects the "clean derivatives" part of the active objective: a valid
fiber gauge must preserve not only the trace value, but the primitive adjoint
that training/backward will use. The verifier now checks row gradient norms,
finite-difference internal consistency, missing-Jacobian value/gradient
controls, and stale summary fields.

Current focused value+gradient verifier status:

```text
21 passed in 6.45s
```
