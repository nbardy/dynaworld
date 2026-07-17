# Healthy Gauge Anchor Math To Algorithm

Date: 2026-07-08 00:06:39

## Context

The user asked whether the project can anchor camera space around healthy
coordinate offsets / gauges instead of accepting projective denominator zeros
as renderer failures, and asked to show the equations, proof spine, and where
those equations lead to algorithm and math changes.

Relevant current docs:

```text
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md
research_notes/gauged_uvt_trace_atlas/CODE_IMPLEMENTATION_PLAN.md
research_notes/gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
```

## Current Model

The robust formulation is:

```text
Do not represent the world tube by one fragile screen chart.
Represent it in the camera-ray bundle, then choose local healthy gauge domains.
```

A gauge domain is not just a fit window. It certifies:

```text
projection denominator regularity
trace residual / approximation error
conservative support bounds
tile-time active set
conditional depth or ray-fiber interval
stable order / commutable order / fallback
forward-backward support agreement
```

The algorithmic principle is:

```text
try gauge/anchor changes before fallback;
split on real events;
fallback only where no cheap valid domain exists.
```

## 1. Bundle Object Before Coordinates

Sensor-time base:

```text
B = Omega x T
y = (u, v, tau)
```

Camera-ray bundle:

```text
pi: E_Gamma -> B
F_y = pi^{-1}(y)
```

Camera program:

```text
Gamma: E_Gamma -> M
M = R^3 x R
```

Invariant trace:

```text
Trace_Gamma[w_i] = pi_* Gamma^* w_i
```

For density:

```text
bar_rho_i(y)
    = integral_{F_y} rho_i(Gamma(e)) dmu_y(e)
```

Local gauge:

```text
chi_a: E_Gamma | C_a -> C_a x D_a
chi_a(e) = (y, z_a)
Gamma_a(y,z_a) = Gamma(chi_a^{-1}(y,z_a))
```

In gauge `a`:

```text
bar_rho_i^a(y)
    =
    integral_{D_a}
        rho_i(Gamma_a(y,z_a)) J_a(y,z_a) dz_a
```

Proof of coordinate invariance:

Let another gauge use:

```text
z_b = phi_y(z_a)
```

with:

```text
Gamma_b(y,z_b) = Gamma_a(y,z_a)
J_b(y,z_b) |partial z_b / partial z_a| = J_a(y,z_a)
```

Then:

```text
Trace^b(y)
 = integral rho(Gamma_b(y,z_b)) J_b(y,z_b) dz_b
 = integral rho(Gamma_a(y,z_a)) J_a(y,z_a) dz_a
 = Trace^a(y)
```

Algorithm change:

```text
Every gauge change must carry a fiber-measure Jacobian.
Tests must include missing-Jacobian controls that fail.
```

## 2. Projective Denominator Is A Chart Event

Homogeneous camera trace:

```text
h(t) = (h_u(t), h_v(t), h_z(t))
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
```

Bad event for this chart:

```text
h_z(t) = 0
```

This is not an invariant failure of the ray. It is a failure of one screen
coordinate chart.

Healthy-chart idea:

```text
choose denominator d_k(t) from a gauge/chart family
such that |d_k(t)| >= epsilon_den on C_k
```

Examples:

```text
perspective z-chart:       u = x/z, v = y/z
x-chart / y-chart:         ratios against another nonzero component
unit ray chart:            direction on S^2 with local tangent chart
homogeneous trace chart:   keep [h_u:h_v:h_z] until raster evaluation
orbit chart:               r = tan((theta - theta0) / 2)
```

Proof sketch:

For any nonzero homogeneous point:

```text
h = (h_x,h_y,h_z) != 0
```

at least one component has:

```text
|h_k| = max(|h_x|, |h_y|, |h_z|) > 0
```

So a local chart using denominator `h_k` is valid around that point by
continuity. No single chart covers all projective directions, but an atlas of
such charts covers the visible domain.

Algorithm change:

```text
for each candidate time/ray interval:
    evaluate candidate gauges/denominators
    choose one with largest denominator margin and lowest residual/support cost
    split when no candidate keeps margin above threshold
```

Domain certificate:

```text
denominator_margin(C) = min_{t in C} |d_k(t)|
valid iff denominator_margin(C) >= epsilon_den
```

## 3. Anchor Offsets And Centered Coordinates

For numerical stability and low polynomial degree, center local coordinates:

```text
tau = (t - t0) / Delta_t
xi  = (u - u0) / Delta_u
eta = (v - v0) / Delta_v
```

In camera-family form:

```text
q = (camera parameter offsets)
B_q(q) = local basis in q
h_i(q,t) = sum_b B_q,b(q) h_{i,b}(t)
```

Healthy anchor:

```text
a = (t0, camera pose center, depth gauge, denominator chart)
```

Cost:

```text
cost(C,a)
    =
    w_r residual(C,a)
  + w_d / denominator_margin(C,a)
  + w_s support_area(C,a)
  + w_o order_uncertainty(C,a)
  + w_b backward_replay_risk(C,a)
```

Algorithm change:

```text
pick local anchors that minimize cost subject to hard validity gates.
```

This is the math version of "anchor camera space around healthy coordinate
offsets."

## 4. Gaussian World Tubes: Fiber Pushforward

Spacetime Gaussian:

```text
rho_i(x) = a_i exp[-1/2 (x - m_i)^T Lambda_i (x - m_i)]
```

Local affine gauge:

```text
Gamma_a(y,z) ~= x0 + J [delta_y, delta_z]^T
```

Pulled-back precision:

```text
H = J^T Lambda_i J
  = [ H_yy  H_yz
      H_zy  H_zz ]
```

Marginalize fiber coordinate:

```text
S = H_yy - H_yz H_zz^{-1} H_zy
```

Conditional depth:

```text
z_hat_i(y) = z0 + H_zz^{-1}(g_z - H_zy delta_y)
Var(z | y) = H_zz^{-1}
```

Proof sketch:

Complete the square in `z` for the joint Gaussian over `(y,z)`. The Schur
complement is the precision of the marginal over `y`; the conditional mean
comes from solving the quadratic optimum in `z` for fixed `y`.

Algorithm change:

```text
store UVT footprint precision S
store conditional depth z_hat and depth variance/interval
use S for support bounds and tile-time binning
use z_hat / depth interval for visibility sidecars
```

## 5. Support Bounds Become Certificates

For a local footprint with center `mu_y(t)` and precision/covariance `Sigma_y`,
a conservative ellipsoid support is:

```text
(y - mu_y)^T S (y - mu_y) <= r_alpha^2
```

For tile-time binning, compile a rectangle/time interval:

```text
u_min(C), u_max(C), v_min(C), v_max(C), t_start, t_stop
```

The certificate must prove:

```text
all sampled/analytic trace support inside compiled bounds
```

Algorithm change:

```text
bound_projective_trace_windows
bin_projective_trace_support_bounds
assemble_projective_trace_tile_time_atlas
```

Fallback condition:

```text
if support bound explodes or crosses too many tiles:
    split, choose a better gauge, or mark tile-local fallback
```

## 6. Visibility: Do Not Let Pushforward Lie

Pure UVT marginal:

```text
alpha_i(u,v,t)
```

is not enough for all visibility. Keep a lifted depth/order sidecar:

```text
z_hat_i(y), sigma_z,i(y)
```

or interval:

```text
D_i(y) = [z_i^-(y), z_i^+(y)]
```

Pairwise depth predicates:

```text
Delta_ij(y) = z_i(y) - z_j(y)

Delta_ij^-(y) = z_i^-(y) - z_j^+(y)
Delta_ij^+(y) = z_i^+(y) - z_j^-(y)
```

Order certificate:

```text
Delta_ij^+(y) < 0  => i in front of j
Delta_ij^-(y) > 0  => j in front of i
otherwise          => split, commute, or fallback
```

Swap bound:

```text
|Delta I_ij(y)| <= alpha_i(y) alpha_j(y) |c_i(y) - c_j(y)|
```

Algorithm change:

```text
build local support-overlap graph
certify order only for overlapping pairs
store total order, partial order, commutation residual, or fallback flag
```

## 7. WorldFoam: Retain The Fiber Instead

WorldFoam keeps lifted ray-fiber opacity:

```text
lambda_l(y,z) dz = Gamma_l(y,.)^* dmu
eta_l(y,z) dz    = Gamma_l(y,.)^* dnu
```

Gauge transformation:

```text
lambda'(y,z') dz' = lambda(y,z) dz
eta'(y,z') dz'    = eta(y,z) dz
```

Optical depth:

```text
tau_y(z) = integral_{z_front}^{z} lambda_y(s) ds
T_y(z)   = exp(-tau_y(z))
I(y)     = integral T_y(z) eta_y(z) dz + T_back I_bg(y)
```

Algorithm change:

```text
do not sort primitive centers;
compile cell/ray event words and transmittance prefixes along z.
```

## 8. Visibility Monoid And Product Integral

Optical transfer element:

```text
g = (beta, m)
```

where:

```text
beta = residual transmittance
m    = premultiplied visible color against black
```

Composition:

```text
g1 otimes g2
    =
    (beta1 beta2, m1 + beta1 m2)
```

Associativity proof:

```text
(a otimes b) otimes c
  =
  (beta_a beta_b beta_c,
   m_a + beta_a m_b + beta_a beta_b m_c)

a otimes (b otimes c)
  =
  (beta_a beta_b beta_c,
   m_a + beta_a m_b + beta_a beta_b m_c)
```

So:

```text
([0,1] x R^C, otimes)
```

is a visibility monoid.

Matrix form:

```text
M(g) =
    [ beta I_C   m ]
    [ 0          1 ]
```

Then:

```text
M(g1) M(g2) = M(g1 otimes g2)
```

Continuous transfer:

```text
A_y(z) =
    [ -lambda_y(z) I_C   eta_y(z) ]
    [ 0                  0        ]

M_y = P exp int A_y(z) dz
```

Algorithm change:

```text
WorldFoam renderer = scan / product-integral over compiled cell-path events.
```

## 9. Cell-Path Atlas

Power cell:

```text
pow_i(x) = ||x - p_i||^2 - r_i^2
```

Cell ownership:

```text
B_i = { x : ||x - p_i|| <= r_i
            and pow_i(x) <= pow_j(x) for neighbors j }
```

Ray:

```text
x(s) = o + s d
```

Radical face:

```text
n_ij = p_j - p_i
h_ij = 0.5 (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)
x dot n_ij <= h_ij
```

Crossing:

```text
s_face = (h_ij - o dot n_ij) / (d dot n_ij)
```

The denominator:

```text
d dot n_ij
```

is a tangency/event hazard, analogous to projective `h_z = 0`.

Compiled cell word:

```text
w_y = (i_1, ell_1, i_2, ell_2, ..., i_R, ell_R)
```

For constant density/color in cell `i`:

```text
g_i(ell) =
    ( exp(-sigma_i ell),
      (1 - exp(-sigma_i ell)) c_i )
```

Evaluation:

```text
G(y) = otimes_{r=1}^{R(y)} g_{i_r}(ell_r(y))
I(y) = m(y) + beta(y) I_bg(y)
```

Algorithm change:

```text
owner-run cutwalk -> compiled event word -> monoid scan -> image.
```

## 10. Same-Representation Replay Proof

Per-frame replay:

```text
w_y = (i_1, ell_1, ..., i_R, ell_R)
G_replay(y) = otimes_r g_{i_r}(ell_r(y))
```

Compiled atlas:

```text
w_hat_y = w_y
ell_hat_r(y) = ell_r(y)
G_compiled(y) = otimes_r g_{i_r}(ell_hat_r(y))
```

If:

```text
w_hat_y = w_y
ell_hat_r = ell_r
```

then:

```text
G_compiled(y) = G_replay(y)
I_compiled(y) = I_replay(y)
```

Approximate error:

```text
|I_compiled - I_replay|
    <= C sum_r epsilon_r
     + epsilon_support
     + epsilon_fallback
```

Algorithm change:

```text
first prove compiler equals same-representation replay,
then compare to STAR/GS baselines.
```

## 11. Prefix/Suffix VJP

For:

```text
h = a otimes b
beta_h = beta_a beta_b
m_h = m_a + beta_a m_b
```

reverse-mode:

```text
bar m_a    += bar m_h
bar m_b    += beta_a bar m_h
bar beta_a += beta_b bar beta_h + dot(bar m_h, m_b)
bar beta_b += beta_a bar beta_h
```

Final decode:

```text
I = m + beta I_bg
bar m = bar I
bar beta = dot(bar I, I_bg)
```

For segment `k`:

```text
G = P_k^- otimes g_k otimes S_k^+
P_k^- = (B_k^-, M_k^-)
S_k^+ = (B_k^+, M_k^+)
I_k^+ = M_k^+ + B_k^+ I_bg
```

Then:

```text
delta I = B_k^- (delta m_k + delta beta_k I_k^+)
```

so:

```text
bar m_k    = B_k^- bar I
bar beta_k = B_k^- dot(bar I, I_k^+)
```

For constant-density segment:

```text
tau = sigma ell
beta = exp(-tau)
m = (1 - beta)c
```

adjoints:

```text
bar tau = beta (dot(bar m, c) - bar beta)
bar sigma = ell bar tau
bar ell = sigma bar tau
bar c = (1 - beta) bar m
```

Algorithm change:

```text
store/recompute front transmittance prefix and behind-radiance suffix
for exact fixed-topology WorldFoam VJP.
```

## 12. Topology Boundary Terms

Within a fixed topology/domain:

```text
cell word, active intervals, event partition, fallback mask fixed
```

VJP is exact for continuous parameters inside the domain.

If a boundary moves:

```text
I(theta) = integral_{a(theta)}^{b(theta)} f(z,theta) dz
```

then:

```text
dI/dtheta =
    integral_a^b partial_theta f(z,theta) dz
    + f(b,theta) b'(theta)
    - f(a,theta) a'(theta)
```

Algorithm change:

```text
current direct VJP claim = fixed-topology exactness.
moving topology needs refresh, split, smoothing, fallback, or future boundary calculus.
```

## 13. Event-Complexity Scaling

Let:

```text
F = number of frames / samples
R = average per-frame structural replay work
E(F) = certified camera-path event records
```

Per-frame replay:

```text
W_replay(F) = O(F R)
```

Compiled structural payload:

```text
W_compiled(F) = O(E(F) + basis_payload)
```

Sublinear structural win when:

```text
E(F) = o(F R)
```

Pixels remain:

```text
O(F H W)
```

Algorithm change:

```text
measure payload, trace count, interval entries, fallback fraction,
support rebins, forward/backward timings versus per-frame replay.
```

## Algorithm Summary

World Tubes route:

```text
1. Input world primitives and known camera program Gamma.
2. Propose gauge anchors:
       depth/log-depth/inverse-depth/projective/orbit/camera-family gauges.
3. For each primitive and candidate domain:
       fit/evaluate homogeneous trace;
       compute denominator margin, residual, support, depth/order sidecars.
4. Choose healthy gauge domain or split at event.
5. Compile support bounds into tile-time cells.
6. Build visibility gauge atlas over support-overlap pairs.
7. Mark order-certified, commutable, or fallback cells.
8. Lower accepted intervals into packed Metal atlas.
9. Render slices/exposure/rolling samples.
10. Backward through compiled direct VJP where topology is fixed;
    fallback/reference VJP where not.
```

WorldFoam route:

```text
1. Pull cell optical measures onto the camera-ray bundle.
2. Keep lambda(y,z), eta(y,z) over the depth fiber.
3. Compile cell entry/exit/topology event words over sensor-time domains.
4. Evaluate front-to-back optical transfer with the visibility monoid.
5. Backward with prefix/suffix VJP inside fixed topology.
6. Refresh/split/fallback when cell topology or event words change.
```

## Falsification Tests

Gauge anchors:

```text
ordinary-depth/log-depth integration matches only with Jacobian;
candidate gauge domains reject hidden denominator roots;
anchored gauge count grows with event complexity, not frame density.
```

World Tubes:

```text
Schur footprint matches dense fiber integration;
support bounds contain sampled/analytic support;
visibility swap error follows alpha_i alpha_j |c_i-c_j|;
compiled atlas image/loss/gradient matches per-frame replay on fixed domains.
```

WorldFoam:

```text
compiled cell word equals per-frame cutwalk word;
monoid scan equals sorted alpha compositing for atomic splats;
prefix/suffix VJP matches finite differences for beta, color, sigma, length;
event-density death curve reports E(F)/(F R), fallback, memory, and speedup.
```

## Decision Implications

The user's instinct is right:

```text
anchor camera space around healthy coordinates before calling fallback.
```

But the correct invariant is not any one anchor. The invariant is the bundle
trace / optical transfer object. Anchors are local coordinate choices whose
validity is certified by margins, residuals, support, visibility, and backward
agreement.

The math changes the implementation posture from:

```text
fit a tube, clamp bad denominators, fallback when ugly
```

to:

```text
compile an atlas of healthy gauge domains;
represent denominator/support/order/topology changes as events;
lower only certified domains to fast packed kernels;
prove replay equivalence and fixed-topology VJP.
```
