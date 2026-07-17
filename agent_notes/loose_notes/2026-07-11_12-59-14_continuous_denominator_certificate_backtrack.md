# Continuous Denominator Certificate Backtrack

Date: 2026-07-11 12:59:14

## Context

A review of the Gauged UVT orbit chart-count work found that the previous
denominator certificate checked roots continuously but measured the denominator
margin only at frame samples. It also classified linear versus quadratic depth
polynomials with an absolute raw-time coefficient threshold.

Relevant implementation:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/
torch_gsplat_bridge_star_uvt/projective_trace.py
```

## Backtrack: Sampled Margin Was Not A Chart Certificate

Status:

```text
invalidated
```

The old rule accepted when sampled values satisfied:

```text
min_j |h_z(t_j)| >= epsilon_den
```

This does not imply the continuous condition:

```text
min_{t in [t_min, t_max]} |h_z(t)| >= epsilon_den.
```

Counterexample:

```text
h_z(t) = (t - 0.5)^2 + 1e-5
t_samples = {0, 1}
epsilon_den = 1e-3
```

At both samples, `h_z = 0.25001`, but at `t=0.5`, `h_z = 1e-5`. The old
splitter accepted this window despite a projection value near `1e5` at the
between-sample vertex.

## Replacement Certificate

For raw-time denominator:

```text
p(t) = c + b t + a t^2,
```

normalize the queried interval:

```text
t = t_c + s_t s,
s in [s_min, s_max].
```

Then:

```text
p(t_c + s_t s) = C + B s + A s^2,
C = c + b t_c + a t_c^2,
B = s_t (b + 2 a t_c),
A = a s_t^2.
```

For a quadratic on a closed interval, its range and minimum absolute value are
fully determined by the endpoints and the stationary candidate:

```text
s_star = -B / (2 A),  when A != 0 and s_star in [s_min, s_max].

V = {p(s_min), p(s_max)} union {p(s_star) if eligible}.
```

The compiler now uses:

```text
denominator_has_root = min(V) <= tol_root and max(V) >= -tol_root
denominator_min_abs  = 0                         if denominator_has_root
                       min_{v in V} |v|          otherwise.
```

`tol_root` is floating-point roundoff scaled by normalized coefficient magnitude.
It is a conservative numerical root-boundary event, not a symbolic theorem.

## Second Backtrack: Raw Coefficient Epsilon Was Scale Fragile

Status:

```text
invalidated
```

The old helper treated `|a| <= eps` as nonquadratic. But:

```text
h_z(t) = 1e-7 t^2 - 1e-5
t in [-20, 20]
```

has roots at `t = +/-10` even though its raw quadratic coefficient is below
the evaluator epsilon. The normalized interval range test detects the sign
change without branching on an arbitrary raw coefficient scale.

## Stored-Certificate Contract

The splitter tests a half-open sample window together with its next sample as
the continuous right boundary. Before this correction, that domain test could
reject a window while `window.fit.denominator_has_root` described only the
smaller sampled fit interval and remained false.

The emitted `ProjectiveTraceWindow.fit` now stores the same domain root and
minimum-margin certificate that decided acceptance. Visibility sidecars inherit
the correct event state.

## Healthy-Gauge Clarification

The prior statement that another homogeneous component can be nonzero remains
correct for a local directional atlas in projective space. It is not sufficient
to keep a trace finite in the existing pinhole sensor coordinates:

```text
u = h_u / h_z,
v = h_v / h_z.
```

At `h_z=0`, an x-chart or y-chart can represent the direction, but it does not
by itself produce finite `(u,v)` on the current raster base. The compiler must
either:

```text
1. transition to a projective sensor atlas with chart-aware rasterization, or
2. classify the event as horizon/offscreen/frustum topology and keep fallback
   local.
```

Current code implements only the second prerequisite: it detects and rejects
the unsafe `h_z` domain. Candidate gauge selection and raster transitions remain
planned work, not completed functionality.

## Schur Pushforward Correction

The prior healthy-anchor note gave the Schur precision and conditional mean but
omitted the marginal amplitude. With `delta = m_i - x0`,
`g = J^T Lambda_i delta`, scalar `z`, `H_zz > 0`, locally constant measure
factor `J_0`, and an untruncated local fiber:

```text
bar_rho_i(y)
  ~= J_0 a_i sqrt(2 pi / H_zz) exp[-1/2 q_y(delta_y)],

q_y(delta_y)
  = delta_y^T S delta_y
    - 2 (g_y - H_yz H_zz^-1 g_z)^T delta_y
    + delta^T Lambda_i delta - g_z^T H_zz^-1 g_z.
```

The amplitude factor is required before `S` can be treated as a value-preserving
projected opacity model. With clipping or a varying gauge Jacobian, it becomes a
local approximation that must carry residual/support evidence or use quadrature.

## Falsification Tests Added

```text
1. Root-free near-zero vertex between samples fails the margin gate.
2. Small raw quadratic coefficient with roots on a large time interval fails.
3. A split-window boundary event is stored in the emitted fit certificate.
4. Fixed-frame orbit span increases chart count, while denser sampling does not.
```

Focused verification after the patch:

```text
28 passed in 18.40s
```

## Decision Implications

The valid current claim is:

```text
The compiler has a continuous quadratic denominator safety certificate for its
single supplied h_z chart.
```

The following remain separate future work:

```text
candidate gauge/anchor selection
projective sensor-chart transitions
pixel-varying footprint support under those transitions
full gauge-aware Metal lowering
```

## Repository Boundary

The current `star_uvt_v0` directory contains the projective Python compiler,
Metal kernels, bindings, renderer helpers, and many unrelated experimental
changes in one dirty 2.2 GB variant. Do not make a partial commit that adds only
`projective_trace.py`: the Metal path depends on sibling binding/kernel changes.

The correct preservation task is separate from this certificate fix:

```text
fork the complete intended variant from a known baseline
copy only the intended projective implementation surface
build the fork
run the focused projective and Metal parity gates
commit the submodule first, then the parent pointer
```

No staging or commit was performed in this correction pass because the current
dirty variant cannot be safely partitioned without first identifying which of
its large C++/Metal diffs belong to the projective atlas versus other research
lanes.
