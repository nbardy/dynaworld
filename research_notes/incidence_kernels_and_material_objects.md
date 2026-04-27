# Incidence Kernels And Material Objects

Date: 2026-04-27

This note answers a narrow design question:

```text
Should the next world primitive use projected conics, exact ray-Gaussian line
integrals, slab intersections, or full volume integration?
```

Short answer:

```text
Yes, we can ablate all four, but not as four unrelated models.
```

They should share one world-state interface and differ only in the
ray-event incidence law:

```text
world event e_i(t)
camera ray ell = (o, d)
kappa(ell, e_i(t)) -> optical interaction
renderer -> RGB / alpha / depth / X-map / diagnostics
```

The best current concrete object is not "a gauge" by itself. Gauge is the
discipline around coordinate freedom and certificates. The concrete object is:

```text
a typed transported event measure with constrained ray-event incidence
```

The first implementable child of that object is:

```text
transported rank-adaptive metric events
```

with:

```text
e_i = (q_i, G_i, rho_i, c_i)
x_i(t) = Phi_t(q_i)
Sigma_i(t) = D Phi_t(q_i) G_i D Phi_t(q_i)^T
kappa(ell, e_i(t)) = constrained geometric interaction between ray ell
                     and world support (x_i(t), Sigma_i(t), rho_i)
```

The question is which `kappa` and renderer are the right first law.

Dialed-in current answer:

```text
rendering ray:
    finite origin-direction ray (o, d, s_near, s_far)

diagnostic ray:
    Pluecker line (d, m = o x d)

first exact kappa:
    mass-normalized ray-Gaussian line integral

first fast kappa:
    projected conic optical-depth approximation

first structural diagnostic:
    Pluecker witness spectrum from sampled ray/event contributions

claim boundary:
    this may still be a constrained Gaussian/splat family; the novelty is the
    transport, incidence constraints, held-out-camera selector, and diagnostics,
    not the use of Gaussian basis functions alone
```

---

## 1. Current Evidence

Current held-out DeepView camera result:

| representation | held-out PSNR |
| --- | ---: |
| `free_dynamic_3dgs` | 9.7392 |
| `screen_disk` | 9.6479 |
| `rank_adaptive_metric` | 9.5662 |
| `oriented_slab` | 9.3344 |

This means:

```text
source-view PSNR is not selecting geometry
world-derived support has not yet won
screen_disk remains the internal control
free_dynamic_3dgs remains the external baseline
rank_adaptive_metric is a hypothesis, not a result
oriented_slab is demoted as the next primitive
```

The next useful experiment is not "invent a grand object." It is:

```text
same event state, multiple incidence/render laws, same train/held-out split,
same reporting discipline
```

---

## 2. Shared Object Interface

Let:

```text
q_i             canonical material/event coordinate
x_i(t)          world position at time t
Phi_t           transport map from canonical event space to world space
G_i             canonical local PSD metric
Sigma_i(t)      world support tensor
rho_i           density / mass / optical strength
c_i             color or material appearance
ell = (o, d)    camera ray with origin o and unit direction d
kappa           ray-event incidence law
```

The shared event is:

```text
e_i(t) = (x_i(t), Sigma_i(t), rho_i, c_i, phase_i)
```

where:

```text
x_i(t) = Phi_t(q_i)
```

and for metric events:

```text
Sigma_i(t) = D Phi_t(q_i) G_i D Phi_t(q_i)^T + eps I_3
```

The camera only supplies rays:

```text
ell_{c,t,p} = (o_c, d_{c,t,p})
```

The renderer should factor as:

```text
event state -> constrained incidence -> alpha/color contribution
```

not:

```text
ray id -> learned RGB
```

This is what "kappa constrained by world geometry" means.

### Formal Constraint: Admissible Kappa

A `kappa` is admissible for this research lane only if it behaves like a
world interaction law, not a ray-cache score.

For an event `e` and ray `ell`, write:

```text
kappa(ell, e) >= 0
```

and:

```text
alpha(ell, e) = 1 - exp(-kappa(ell, e))
```

The admissibility conditions are:

```text
K1. World-state dependence:
    kappa may depend on ray geometry and event state.
    It may not depend on pixel id, camera id, training-frame id, or arbitrary
    learned ray embeddings except through the ray and event state.

K2. SE(3) invariance:
    Applying the same rigid transform to the ray and the event must not change
    optical depth.

K3. Nonnegativity and monotonic mass:
    kappa >= 0, and increasing event optical mass/density with fixed support
    cannot decrease kappa.

K4. Finite interval semantics:
    for rendering, kappa is evaluated over a camera ray segment
    (s_near, s_far), even if a whole-line approximation is used as a control.

K5. Boundedness under support clamps:
    if eigenvalues, density/mass, and ray interval length are bounded, kappa
    must be bounded.

K6. Low-rank appearance:
    RGB should live on event appearance/material parameters, not in kappa.

K7. Differentiability almost everywhere:
    the law should support gradient-based training except at controlled
    boundaries such as slab entry/exit or tile culling.
```

These rules are deliberately stricter than "can render pixels." They are there
to rule out a hidden light-field.

### Ray-Cache Counterexample

If `kappa` is arbitrary, the model can memorize the training rays.

Construction for `K` observed rays:

```text
events:
    e_1, ..., e_K

kappa(ell_k, e_j):
    very large if j = k
    zero otherwise

appearance:
    c_j = observed RGB on ray ell_j
```

Then training RGB can be fit with:

```text
I_hat(ell_k) = c_k
```

with no shared world geometry. Therefore a useful `kappa` must forbid at least
one of:

```text
one event per ray
arbitrary ray-conditioned support
arbitrary ray-conditioned RGB
camera-id or pixel-id dependence
```

The current v1 choice forbids arbitrary ray dependence by making `kappa` a
closed-form function of:

```text
ray segment (o, d, s0, s1)
event position mu
event covariance Sigma
event mass/density
```

---

## 2.25. Ray Representation: Pluecker Is A Diagnostic, Not A Requirement

The renderer does not need Pluecker coordinates. The simplest rendering ray is:

```text
ell = (o, d, s_near, s_far)
```

where:

```text
o       ray origin in world coordinates
d       unit direction in world coordinates
s       depth/ray parameter
r(s)    o + s d
```

This is the best representation for `kappa` because all four laws need a ray
path through world space.

Pluecker coordinates are best for **line diagnostics**:

```text
ell_P = (d, m)
m = o x d
||d|| = 1
d dot m = 0
```

They remove the arbitrary choice of origin along the same line. If:

```text
o' = o + lambda d
```

then:

```text
m' = o' x d
   = (o + lambda d) x d
   = o x d
   = m
```

So `(d,m)` represents the oriented line, not a particular point on that line.

### Incidence Proof

Let `[d]_x` be the skew matrix such that:

```text
[d]_x x = d x x
```

A point `x` lies on the oriented line `(d,m)` iff:

```text
x = o + s d
```

Then:

```text
[d]_x x + m
    = d x (o + s d) + o x d
    = d x o + s d x d + o x d
    = d x o + o x d
    = 0
```

because:

```text
d x o = - o x d
```

Thus the incidence residual is:

```text
r_inc(x, ell_P) = [d]_x x + m
```

and for unit `d`:

```text
||r_inc|| = Euclidean distance from x to the line
```

Proof:

```text
r_inc = d x (x - o)
||d x (x-o)||^2 = ||x-o||^2 - (d^T(x-o))^2
```

which is exactly squared perpendicular distance to the line.

### Witness Matrix

For weighted rays assigned to event `i`, define:

```text
E_i(x) = sum_k w_ki ||[d_k]_x x + m_k||^2
```

The normal equations are:

```text
W_i x = b_i
```

with:

```text
W_i = sum_k w_ki (I - d_k d_k^T)
b_i = - sum_k w_ki [d_k]_x^T m_k
```

because:

```text
[d]_x^T [d]_x = I - d d^T
```

The eigenvalues of `W_i` are the witness spectrum:

```text
lambda_1 >= lambda_2 >= lambda_3 >= 0
```

One ray gives:

```text
W = I - d d^T
eigenvalues = {1, 1, 0}
```

So:

```text
lambda_3 ~= 0
```

means depth along the ray is not witnessed.

### Which Ray Type To Use

| ray type | use | warning |
| --- | --- | --- |
| origin-direction segment `(o,d,s0,s1)` | rendering, ray-Gaussian, slab, volume | has origin-gauge along the line, but that is fine for finite camera rays |
| Pluecker `(d,m)` | line incidence, concurrence, witness rank | loses finite near/far segment unless stored separately |
| camera pixel ray `(K,C,p)` | data input and projection | not a world representation by itself |
| ray cone/differential | anti-aliasing, pixel footprint, finite aperture | later renderer quality feature |
| two-plane ray parameterization | light-field baselines | dangerous as core state because it invites ray caches |
| projective homogeneous line | equivalent line geometry to Pluecker | unnecessary unless using projective algebra heavily |

Recommendation:

```text
Use (o,d,s0,s1) for kappa/rendering.
Use Pluecker (d,m) for witness/concurrence diagnostics.
Add ray cones later for anti-aliasing and aperture/DoF.
```

---

## 2.5. Kappa Solved: Make It Optical Depth

The cleanest definition is:

```text
kappa(ell, e_i(t)) = tau_i(ell, t)
```

where `tau_i` is the optical depth contributed by event `i` along ray `ell`.
Then:

```text
alpha_i(ell, t) = 1 - exp(-kappa(ell, e_i(t)))
```

This choice matters. `kappa` should not output RGB. It should not be an
arbitrary learned score. It should be the geometric or physical interaction
between a ray and a world event.

The renderer then has three separate pieces:

```text
kappa(ell, e_i)       optical interaction / alpha precursor
s(ell, e_i)           ordering statistic, usually depth or closest-ray point
a(e_i, ell)           appearance, ideally event-attached and low view-dependence
```

For event compositing:

```text
sort events by s(ell, e_i)
alpha_i = 1 - exp(-kappa_i)
w_i = T_i alpha_i
I_hat = sum_i w_i a_i + T_final bg
```

For true volume compositing:

```text
kappa is per ray interval, not per whole event
T(s) = exp(- accumulated optical depth before s)
```

### Canonical Kappa Definitions

Use these as the first precise menu:

| mode | kappa | ordering `s` | first status |
| --- | --- | --- | --- |
| projected conic | `tau_i exp(-0.5 y^T Sigma2^-1 y)` | projected depth `z_i` | current fast approximation |
| ray-Gaussian, peak density | `rho_i integral exp(-0.5 Mahalanobis(x(s))) ds` | closest point `s_star` | diagnostic ablation |
| ray-Gaussian, mass normalized | `m_i / Z(Sigma_i) * integral exp(-0.5 Mahalanobis(x(s))) ds` | closest point `s_star` | preferred v1 exact kappa |
| slab | `mu_i K_tan / (abs(n dot d)+eps)` | plane hit depth | surface/cloth branch |
| volume | `integral_interval sigma_t(o+s d) ds` | sample interval order | fog/smoke branch |
| ray transform | hit opacity plus `ell_out -> ell_in` map | surface hit depth | water/glass/specular future branch |

### Why Mass-Normalized Ray-Gaussian Is The Preferred Exact Kappa

If we use peak density:

```text
sigma_i(x) = rho_i exp(-0.5 (x-mu_i)^T Sigma_i^-1 (x-mu_i))
```

then increasing `Sigma_i` increases total mass:

```text
total_mass_i = rho_i (2 pi)^(3/2) sqrt(det Sigma_i)
```

This encourages metric inflation unless regularized.

For material-like events, prefer:

```text
sigma_i(x) =
    m_i / ((2 pi)^(3/2) sqrt(det Sigma_i))
    exp(-0.5 (x-mu_i)^T Sigma_i^-1 (x-mu_i))
```

Now `m_i` is total optical mass, and changing the support shape redistributes
the same mass rather than creating free density.

This is not always right. Smoke sources can create/destroy mass. But for the
first rank-adaptive metric test, mass-normalized `kappa` is the cleanest way to
reduce the opacity/metric-inflation cheat.

### The Current Answer

For the next ablation, solve `kappa` as:

```text
primary exact kappa:
    mass-normalized ray-Gaussian line integral

control kappa:
    projected conic optical-depth approximation

surface branch:
    slab optical-depth intersection

volume branch:
    interval optical depth from a density field
```

Do not implement a free learned `kappa_theta(ell, e)` yet. That is exactly the
ray-cache failure mode.

### Does This Need Ground-Truth Depth?

No. The first useful ablation does not require ground-truth depth.

Training can still be RGB-only:

```text
L_train = L_rgb(source cameras)
```

and selection should use held-out cameras:

```text
score = PSNR/L1/LPIPS(held-out cameras)
```

The geometry diagnostics are self-supervised certificates, not labels:

```text
X-map occupancy
X-map temporal/flow consistency
Pluecker witness spectrum
view-stress consistency
metric spectrum and volume
projected-conic validity ratio
```

Ground-truth depth would help disambiguate and debug, but requiring it would
change the research problem. The point of the current lane is:

```text
RGB trains the model.
held-out cameras and structural diagnostics select the representation.
```

Depth, if available from synthetic data or a trusted sensor, should be used as
an audit channel:

```text
L_depth_eval = |D_hat - D_gt|
```

not as a requirement for the v1 representation search.

### Is This Just Splats?

It can still be a splat family if the event state is Gaussian.

Projected 3D Gaussian splatting is roughly:

```text
event:
    (mu_i, Sigma_i, alpha_i, c_i)

screen contribution:
    alpha_i exp(-0.5 y^T Sigma_screen_i^-1 y)
```

The ray-Gaussian version is:

```text
event:
    (mu_i, Sigma_i, m_i, c_i)

ray contribution:
    kappa_i(ell) = integral_ray sigma_i(o + s d) ds
    alpha_i(ell) = 1 - exp(-kappa_i(ell))
```

So yes, the first concrete child is close to a **volumetric Gaussian splat
law**. That is acceptable as an experiment. The important differences are:

```text
1. support is world-derived, not chosen as an independent screen footprint
2. optical strength is mass/density, not arbitrary alpha painted onto pixels
3. motion is transported through Phi_t and optionally D Phi_t
4. held-out cameras, not source PSNR, select the model
5. witness/X-map/cheat diagnostics expose ray-cache explanations
```

If those constraints do not improve held-out/view-stress behavior, then this
branch is splat-compatible but not splat-replacing. The doc should then demote
it rather than rename it.

---

## 3. The Four Candidate Incidence Laws

### Summary

| law | status | best for | main risk |
| --- | --- | --- | --- |
| projected conic approximation | implement_now / control | fast splat-like local support | still too close to splats |
| true ray-Gaussian line integral | implement_next | metric events, volumetric splat basis | slower, harder compositing |
| slab intersection | baseline_only / demote | opaque surfaces, cloth sheets | surface bias |
| full volume integration | defer / special phase | fog, smoke, participating media | expensive and ambiguous |

The key point:

```text
projected conic and ray-Gaussian line integral can share the same event state
```

That makes them a clean ablation.

---

## 4. Law A: Projected Conic Approximation

### Object

```text
e_i(t) = (x_i(t), Sigma_i(t), rho_i, c_i)
```

where:

```text
x_i(t) in R^3
Sigma_i(t) in S_+^3
rho_i >= 0
c_i in [0,1]^3
```

### Projection

Let camera projection be:

```text
u_i = pi(x_i)
```

and let:

```text
J_pi_i = d pi / d x evaluated at x_i
```

Then:

```text
Sigma_i_screen = J_pi_i Sigma_i J_pi_i^T + sigma_floor^2 I_2
```

For a pixel `p`:

```text
y = p - u_i
K_i(p) = exp(-0.5 y^T Sigma_i_screen^{-1} y)
alpha_i(p) = 1 - exp(-rho_i K_i(p))
```

Then alpha-composite in depth order:

```text
T_i(p) = product_{j before i} (1 - alpha_j(p))
w_i(p) = T_i(p) alpha_i(p)
I_hat(p) = sum_i w_i(p) c_i + (1 - sum_i w_i(p)) c_bg
```

### Derivation: Why This Is The Pushforward Approximation

Claim:

```text
If a local world support is Gaussian and projection is locally linear, the
screen support covariance is J_pi Sigma J_pi^T.
```

Assumptions:

```text
x = mu + delta
delta ~ N(0, Sigma)
pi is differentiable near mu
```

First-order projection:

```text
pi(x) ~= pi(mu) + J_pi delta
```

Therefore:

```text
Cov[pi(x)] ~= Cov[J_pi delta]
            = J_pi Cov[delta] J_pi^T
            = J_pi Sigma J_pi^T
```

This proves that projected conics are the first-order screen pullback of a
world metric.

### Approximation Bound

Projected conics are accurate only when the event is small relative to depth and
projection curvature.

Taylor expand projection around `mu`:

```text
pi(mu + delta)
  = pi(mu) + J delta + 0.5 H[delta, delta] + O(||delta||^3)
```

The projected-conic renderer keeps only:

```text
J delta
```

Therefore the neglected image displacement is approximately:

```text
e_proj(delta) = 0.5 H[delta, delta] + O(||delta||^3)
```

For pinhole projection, the second derivative scale is roughly:

```text
||H|| = O(f / z^2)
```

If:

```text
R_world = sqrt(lambda_max(Sigma))
```

then a practical dimensionless validity ratio is:

```text
eta_proj = R_world / z
```

and the second-order pixel error scales like:

```text
||e_proj|| = O(f eta_proj^2)
```

So projected conics are safest when:

```text
R_world << z
```

They become suspect for:

```text
large blobs
near-camera events
wide-FOV edges
strongly anisotropic supports spanning depth
```

Diagnostic:

```text
log eta_proj quantiles and compare projected_conic vs ray_gaussian_line
when eta_proj is large.
```

### What It Forbids

If `Sigma_i` is truly world-space and query-camera independent, it forbids:

```text
choosing an independent circular support per query view
```

### What It Still Permits

It still permits:

```text
metric inflation
opacity-density tradeoffs
approximate billboard behavior under weak view changes
source-view fitting without true multi-ray support
```

### Pseudocode

```python
def projected_conic_event(event, camera, pixel):
    x = event.position_world
    Sigma3 = event.Sigma_world
    rho = event.rho

    u, z, Jpi = project_with_jacobian(camera, x)
    Sigma2 = Jpi @ Sigma3 @ Jpi.T + sigma_floor**2 * I2
    Sigma2 = clamp_spd_2x2(Sigma2)

    y = pixel - u
    K = exp(-0.5 * y.T @ inv(Sigma2) @ y)
    alpha = 1.0 - exp(-rho * K)
    return z, alpha, event.color
```

### Scaling

Naive:

```text
O(N H W)
```

Production:

```text
project event -> ellipse bounding box -> assign tiles -> sort by depth
-> rasterize local pixels
```

Expected:

```text
O(total projected tile coverage + tile sorting)
```

### Rigid / Water / Fog / Cloth

| regime | behavior |
| --- | --- |
| rigid body | good if events share rigid transform and support is stable |
| water | poor for clear/refractive water; okay for foam/spray as blobs |
| fog | only approximate, acts like local particles not true participating medium |
| cloth | plausible if metrics become thin rank-2 patches and transport is smooth |

---

## 5. Law B: True Ray-Gaussian Line Integral

This is the most important theoretical next law for rank-adaptive metric
events.

Instead of projecting a 3D Gaussian to a 2D conic, compute how much a camera ray
passes through the 3D Gaussian support.

### Object

```text
e_i = (mu_i, Sigma_i, rho_i, c_i)
```

where:

```text
mu_i = x_i(t)
A_i = Sigma_i^{-1}
ray r(s) = o + s d, ||d|| = 1
```

Define density:

```text
sigma_i(x) = rho_i exp(-0.5 (x - mu_i)^T A_i (x - mu_i))
```

The optical depth contribution along a ray segment `[s0, s1]` is:

```text
tau_i(ell) = integral_{s0}^{s1} sigma_i(o + s d) ds
```

Then:

```text
alpha_i(ell) = 1 - exp(-tau_i(ell))
```

### Peak-Density Version

This version treats `rho_i` as peak density:

```text
sigma_i(x) = rho_i exp(-0.5 (x - mu_i)^T A_i (x - mu_i))
```

It is useful as an ablation, but it lets total optical mass grow with support
volume:

```text
M_i = rho_i (2 pi)^(3/2) sqrt(det Sigma_i)
```

That makes metric inflation easier.

### Mass-Normalized Version

This version treats `m_i` as total optical mass:

```text
sigma_i(x) =
    m_i / ((2 pi)^(3/2) sqrt(det Sigma_i))
    exp(-0.5 (x - mu_i)^T A_i (x - mu_i))
```

Then the finite-segment optical depth is:

```text
tau_i_mass =
m_i / ((2 pi)^(3/2) sqrt(det Sigma_i))
* exp(-0.5 (c - b^2/a))
* sqrt(pi/(2a))
* [
    erf(sqrt(a/2) (s1 + b/a))
    -
    erf(sqrt(a/2) (s0 + b/a))
  ]
```

For the whole line:

```text
tau_i_mass_whole_line =
m_i / (2 pi sqrt(det Sigma_i) sqrt(a))
* exp(-0.5 (c - b^2/a))
```

This should be the default exact `kappa` for transported metric events because
it makes support shape and optical mass more identifiable.

### Closed-Form Derivation

Let:

```text
v = o - mu
a = d^T A d
b = d^T A v
c = v^T A v
```

Then:

```text
(o + s d - mu)^T A (o + s d - mu)
    = (v + s d)^T A (v + s d)
    = a s^2 + 2 b s + c
```

Complete the square:

```text
a s^2 + 2 b s + c
    = a (s + b/a)^2 + c - b^2/a
```

So:

```text
tau_i = rho_i exp(-0.5 (c - b^2/a))
        integral_{s0}^{s1} exp(-0.5 a (s + b/a)^2) ds
```

Let:

```text
u = sqrt(a/2) (s + b/a)
ds = sqrt(2/a) du
```

Then:

```text
tau_i =
rho_i exp(-0.5 (c - b^2/a)) sqrt(pi/(2a))
[
  erf(sqrt(a/2) (s1 + b/a))
  -
  erf(sqrt(a/2) (s0 + b/a))
]
```

For the whole line:

```text
tau_i_whole_line =
rho_i sqrt(2 pi / a) exp(-0.5 (c - b^2/a))
```

The closest point in the Mahalanobis metric occurs at:

```text
s_star = -b/a
```

and the squared Mahalanobis ray distance is:

```text
d_M^2 = c - b^2/a
```

### Bounds And Invariance

#### Nonnegativity

For SPD `A`, we have:

```text
a = d^T A d > 0
```

and:

```text
d_M^2 = c - b^2/a >= 0
```

Proof:

```text
d_M^2 = min_s (v + s d)^T A (v + s d)
```

because the minimizing point is:

```text
s_star = -b/a
```

An SPD quadratic is nonnegative, so the minimum is nonnegative.

#### Segment Bound

The finite segment optical depth is bounded by the whole-line optical depth:

```text
0 <= tau_segment <= tau_whole_line
```

because the density is nonnegative and `[s0,s1]` is a subset of the full line.

This gives a safe candidate-culling upper bound:

```text
if tau_whole_line < tau_min:
    skip event for this ray
```

#### Isotropic Closed Form

For:

```text
Sigma = sigma^2 I
```

mass-normalized whole-line optical depth becomes:

```text
tau_line =
m / (2 pi sigma^2)
exp(-r_perp^2 / (2 sigma^2))
```

where:

```text
r_perp = Euclidean distance from mu to ray
```

This is a useful unit test.

#### Eigenvalue Bounds

Suppose:

```text
sigma_min^2 I <= Sigma <= sigma_max^2 I
```

Then:

```text
1/sigma_max^2 <= a <= 1/sigma_min^2
```

and if `r_perp` is Euclidean distance from `mu` to the ray:

```text
r_perp^2 / sigma_max^2 <= d_M^2 <= r_perp^2 / sigma_min^2
```

So ray-event incidence decays at least as:

```text
exp(-r_perp^2 / (2 sigma_max^2))
```

up to the determinant/`a` prefactor.

For mass-normalized whole-line optical depth:

```text
tau_line <=
m sigma_max / (2 pi sigma_min^3)
exp(-r_perp^2 / (2 sigma_max^2))
```

This is not tight, but it is useful for conservative ray/event culling.

#### Rigid Invariance

Apply a rigid transform:

```text
mu' = R mu + t
o' = R o + t
d' = R d
Sigma' = R Sigma R^T
```

Then:

```text
A' = R A R^T
v' = o' - mu' = R(o-mu) = R v
```

Therefore:

```text
a' = d'^T A' d' = d^T A d = a
b' = d'^T A' v' = d^T A v = b
c' = v'^T A' v' = v^T A v = c
```

So:

```text
tau'(ell', e') = tau(ell, e)
```

This proves ray-Gaussian `kappa` is rigid invariant.

### Why This Is More Geometric Than Projected Conics

Projected conic:

```text
approximate the projection of 3D support into the image plane
```

Ray-Gaussian line integral:

```text
evaluate the physical ray path through the world support
```

It is less dependent on local first-order projection and more directly
ray-event incidence.

### Compositing

There are two options.

Approximate event compositing:

```text
sort events by s_star
alpha_i = 1 - exp(-tau_i)
front-to-back composite
```

True volume compositing:

```text
sample ray intervals or split events into ordered interval contributions
T(s) = exp(- integral_{near}^{s} sigma(u) du)
C = integral T(s) sigma(s) c(s) ds
```

The first is closer to 3DGS. The second is physically better but more expensive.

### Pseudocode

```python
def ray_gaussian_tau_peak(ray_o, ray_d, mu, Sigma, rho, s0, s1):
    A = inv_spd(Sigma)
    v = ray_o - mu

    a = dot(ray_d, A @ ray_d).clamp_min(eps)
    b = dot(ray_d, A @ v)
    c = dot(v, A @ v)

    dist2 = (c - b * b / a).clamp_min(0.0)
    prefactor = rho * exp(-0.5 * dist2) * sqrt(pi / (2.0 * a))

    u1 = sqrt(a / 2.0) * (s1 + b / a)
    u0 = sqrt(a / 2.0) * (s0 + b / a)
    tau = prefactor * (erf(u1) - erf(u0))

    s_star = -b / a
    return tau, s_star, dist2


def ray_gaussian_tau_mass(ray_o, ray_d, mu, Sigma, mass, s0, s1):
    A = inv_spd(Sigma)
    v = ray_o - mu

    a = dot(ray_d, A @ ray_d).clamp_min(eps)
    b = dot(ray_d, A @ v)
    c = dot(v, A @ v)

    dist2 = (c - b * b / a).clamp_min(0.0)
    det = det_spd(Sigma).clamp_min(eps)
    norm = mass / (((2.0 * pi) ** 1.5) * sqrt(det))
    prefactor = norm * exp(-0.5 * dist2) * sqrt(pi / (2.0 * a))

    u1 = sqrt(a / 2.0) * (s1 + b / a)
    u0 = sqrt(a / 2.0) * (s0 + b / a)
    tau = prefactor * (erf(u1) - erf(u0))

    s_star = -b / a
    return tau, s_star, dist2


def render_ray_gaussian_events(events, ray):
    hits = []
    for e in candidate_events(ray):
        tau, s_star, dist2 = ray_gaussian_tau_mass(
            ray.o, ray.d, e.mu, e.Sigma, e.mass, near, far
        )
        if tau > tau_min:
            hits.append((s_star, tau, e.color, e.id))

    hits.sort(key=lambda h: h[0])
    T = 1.0
    C = 0.0
    A = 0.0
    X = 0.0
    for s_star, tau, color, event_id in hits:
        alpha = 1.0 - exp(-tau)
        w = T * alpha
        C += w * color
        A += w
        X += w * canonical_coord(event_id)
        T *= 1.0 - alpha
        if T < trans_cutoff:
            break
    return C + T * bg, A, X / (A + eps)
```

### What It Forbids

It forbids:

```text
screen-only support
per-view footprint selection
arbitrary ray-conditioned incidence
```

if `Sigma`, `mu`, and `rho` are world-state parameters and `tau` is computed by
the closed-form line integral.

### What It Still Permits

It still permits:

```text
many Gaussian blobs forming a source-view shell
density/color memorization in 3D
opacity-density tradeoffs
one-event-per-ray if event count is unconstrained
```

### Rigid / Water / Fog / Cloth

| regime | behavior |
| --- | --- |
| rigid body | good if events are attached to rigid transform or deformation graph |
| water | good for spray/foam/turbidity; insufficient for clear refraction |
| fog | good sparse basis for participating density |
| cloth | can approximate thin sheets with low-rank metrics, but exact surface/slab may be sharper |

### Why This Is A Strong Ablation

It uses the same event state as rank-adaptive metric:

```text
(x_i, Sigma_i, rho_i, c_i)
```

but changes only:

```text
projected conic rasterization -> ray-Gaussian incidence integral
```

If this wins held-out/view stress, the issue was the renderer law, not the
world event state.

---

## 6. Law C: Slab Intersection

This is the surface-biased branch.

### Object

```text
e_i = (x_i, e1_i, e2_i, n_i, r1_i, r2_i, h_i, rho_i, c_i)
```

where:

```text
e1_i, e2_i, n_i form an orthonormal frame
r1_i, r2_i are tangent support radii
h_i is thickness
```

### Plane Intersection

Ray:

```text
r(s) = o + s d
```

Plane through `x_i` with normal `n_i`:

```text
n_i^T (r(s) - x_i) = 0
```

Solve:

```text
s_star = n_i^T (x_i - o) / (n_i^T d)
```

If:

```text
abs(n_i^T d) < eps
```

the ray is grazing and the slab optical path becomes unstable.

At intersection:

```text
y = r(s_star) - x_i
u = e1_i^T y
v = e2_i^T y
```

Tangent kernel:

```text
K_tan = exp(-0.5 (u^2/r1_i^2 + v^2/r2_i^2))
```

Optical path through a thin slab:

```text
path_length ~= h_i / (abs(n_i^T d) + eps)
tau_i = rho_i path_length K_tan
alpha_i = 1 - exp(-tau_i)
```

For a cleaner conserved-surface-mass version, use surface optical mass `m_i`
instead of peak tangent density:

```text
mu_i = m_i / (2 pi r1_i r2_i)
tau_i = mu_i K_tan / (abs(n_i^T d) + eps)
```

where:

```text
mu_i        peak surface density after area normalization
m_i         total optical mass of the local patch
```

The volume-thickness form and area-normalized form can be related by:

```text
m_i ~= rho_i h_i 2 pi r1_i r2_i
```

For the first slab branch, prefer area-normalized `m_i` if the goal is fair
support-size comparison. Prefer `rho_i h_i` if the goal is matching the current
oriented-slab implementation.

### Derivation: Angle-Aware Thickness

For a slab of physical thickness `h` measured along normal `n`, a ray crossing
at angle `theta` relative to the normal travels:

```text
path_length = h / |cos(theta)|
```

Since:

```text
cos(theta) = n^T d
```

we get:

```text
path_length = h / |n^T d|
```

This is why slab opacity should increase at grazing angles. A screen disk does
not have this geometry.

### Grazing Bound

The infinite-plane slab formula is singular at grazing angles:

```text
|n^T d| -> 0
```

But a real finite patch has finite tangent extent. If:

```text
r_eff = max(r1, r2)
sin(theta) = sqrt(1 - (n^T d)^2)
```

then an upper bound on distance through the tangent footprint is roughly:

```text
path_side <= 2 r_eff / (sin(theta) + eps)
```

A safer slab path length is:

```text
path_length =
min(
  h / (abs(n^T d) + eps),
  2 r_eff / (sqrt(1 - (n^T d)^2) + eps)
)
```

This avoids unbounded grazing opacity from tiny surface patches.

Diagnostic:

```text
track fraction of slab hits where grazing cap is active
```

### Pseudocode

```python
def slab_event(ray_o, ray_d, e):
    denom = dot(e.normal, ray_d)
    if abs(denom) < grazing_eps:
        return None

    s = dot(e.normal, e.center - ray_o) / denom
    if s < near or s > far:
        return None

    y = ray_o + s * ray_d - e.center
    u = dot(e.e1, y)
    v = dot(e.e2, y)

    K = exp(-0.5 * ((u / e.r1)**2 + (v / e.r2)**2))
    tau = e.rho * e.thickness * K / (abs(denom) + eps)
    alpha = 1.0 - exp(-tau)
    return s, alpha, e.color
```

### What It Forbids

It forbids:

```text
thick volumetric blobs for opaque surfaces
camera-facing circular support unless the slab frame rotates that way
```

### What It Still Permits

It still permits:

```text
wrong surface hallucination
depth-placement ambiguity
texture-card behavior
failure on fog/water/speculars
```

### Rigid / Water / Fog / Cloth

| regime | behavior |
| --- | --- |
| rigid body | excellent for solid opaque surfaces if normals are right |
| water | good only for water surface sheet, not refraction/caustics/volume |
| fog | wrong object |
| cloth | good match if the cloth is a thin sheet with smooth deformation |

### Current Status

The first held-out result demotes this as the next primitive:

```text
oriented_slab held-out PSNR = 9.3344
screen_disk held-out PSNR = 9.6479
gap = -0.3135
```

It remains useful as:

```text
surface-only baseline
cloth/sheet branch
proof that surface bias alone is not automatically better
```

---

## 7. Law D: Full Volume Integration

This is the participating-media branch.

### Object

World density:

```text
sigma_t(x) >= 0
c_t(x, d) in RGB or radiance features
```

Ray:

```text
r(s) = o + s d
```

Continuous volume rendering:

```text
T(s) = exp(- integral_{near}^{s} sigma_t(r(u)) du)
C(ell) = integral_{near}^{far} T(s) sigma_t(r(s)) c_t(r(s), d) ds
A(ell) = 1 - T(far)
```

Discrete form:

```text
sigma_m = sigma_t(r(s_m))
alpha_m = 1 - exp(-sigma_m Delta s_m)
T_m = product_{k<m} (1 - alpha_k)
C = sum_m T_m alpha_m c_m
A = sum_m T_m alpha_m
```

### Stability And Quadrature Bounds

For nonnegative density:

```text
sigma_m >= 0
```

we always have:

```text
0 <= alpha_m <= 1
0 <= T_m <= 1
0 <= A <= 1
```

because:

```text
alpha_m = 1 - exp(-sigma_m Delta s_m)
```

This makes discrete volume compositing numerically stable if `sigma_m` and
`Delta s_m` are finite.

For midpoint quadrature, if:

```text
f(s) = T(s) sigma(r(s)) c(r(s))
```

is twice differentiable on `[near, far]`, the global integration error scales
as:

```text
O((far-near) Delta s^2 max_s ||f''(s)||)
```

In practice, choose step size so that:

```text
sigma_max Delta s << 1
```

in high-density regions, or use adaptive stepping / occupancy grids.

For RGB-only training, too-large steps can create false geometry differences
between volume and splat laws. Report step size and samples per ray in any
comparison.

### Parameterizations

A "world-space density field" can be:

| parameterization | lookup table? | comment |
| --- | --- | --- |
| dense voxel grid | yes | literal 3D table |
| sparse voxel grid | yes | table only where occupied |
| hash grid | yes, compressed | Instant-NGP style |
| triplane | yes, low-rank 2D tables | fast but can leak view priors |
| MLP | no table, function | continuous but slower |
| Gaussian/splat basis | sparse continuous table | adaptive density basis |
| transported particles/events | sparse moving basis | our closest path |

Splats already act like a sparse adaptive 3D lookup table:

```text
sigma(x) = sum_i rho_i exp(-0.5 (x - mu_i)^T Sigma_i^{-1} (x - mu_i))
```

That is why splats work. The issue is not representational power. The issue is
that the learned density basis can choose bad 3D explanations from source-view
RGB.

### Dynamics For Smoke/Fog

For participating media, persistent opaque material identity is often wrong.
Density evolves by advection, sources, and dissipation:

```text
partial_t sigma + div(sigma v) = s - kappa_decay sigma
```

where:

```text
v              velocity field
s              source term
kappa_decay    dissipation
```

This is different from persistent material points:

```text
x_i(t) = Phi_t(q_i)
rho_i constant
```

Fog/smoke often needs:

```text
density phase
velocity field
source/dissipation
high ray entropy allowed
```

### Pseudocode

```python
def volume_render(ray, density_field, color_field, samples):
    T = 1.0
    C = 0.0
    A = 0.0
    for s0, s1 in samples:
        smid = 0.5 * (s0 + s1)
        ds = s1 - s0
        x = ray.o + smid * ray.d

        sigma = softplus(density_field(x))
        color = sigmoid(color_field(x, ray.d))
        alpha = 1.0 - exp(-sigma * ds)

        w = T * alpha
        C += w * color
        A += w
        T *= 1.0 - alpha
        if T < trans_cutoff:
            break

    return C + T * bg, A
```

### What It Forbids

If the density field is world-space, it forbids:

```text
screen-only alpha footprints
per-frame 2D texture cards as the only state
```

### What It Still Permits

It still permits:

```text
floaters
foggy opacity shells
view-dependent density if camera leaks into field
dense memorization if the grid/hash is too expressive
```

### Rigid / Water / Fog / Cloth

| regime | behavior |
| --- | --- |
| rigid body | can represent it, but inefficient and blurry without surface bias |
| water | good for murky water, spray, foam; not enough for clear refractive surface |
| fog | best match |
| cloth | inefficient; should be a surface/slab/mesh, not volume |

---

## 7.5. Law E: Ray-Transform Surface For Clear Water And Glass

This is not part of the first four-way ablation, but it is required if the
question includes clear water, glass, mirrors, or strong speculars.

Diffuse surface, splat, and volume laws assume a ray receives light from a local
event. A refractive/specular surface instead maps the outgoing camera ray to a
different incoming ray.

### Object

```text
e_i = (x_i, n_i, r1_i, r2_i, h_i, eta_i, attenuation_i, roughness_i)
```

where:

```text
eta_i             relative index of refraction
n_i               surface normal
roughness_i       width of reflected/refracted ray cone
attenuation_i     absorption/tint
```

### Incidence

First compute a surface hit using the slab or surface intersection law:

```text
s_star = n_i^T (x_i - o) / (n_i^T d)
K_tan = exp(-0.5 (u^2/r1_i^2 + v^2/r2_i^2))
kappa_hit = tau_i K_tan
```

Then compute a ray transform.

Reflection:

```text
d_reflect = d - 2 (d^T n) n
```

Refraction, with `d` pointing into the surface:

```text
eta = eta_out / eta_in
cos_i = -n^T d
k = 1 - eta^2 (1 - cos_i^2)
```

If:

```text
k < 0
```

there is total internal reflection. Otherwise:

```text
d_refract = eta d + (eta cos_i - sqrt(k)) n
```

### Rendering Form

The event contribution is not just:

```text
alpha * color
```

It is:

```text
alpha_hit * sample_radiance(transformed_ray)
```

or a mixture:

```text
L_out(ell) =
    F(ell,n) L_reflect(ell_reflect)
  + (1-F(ell,n)) attenuation L_refract(ell_refract)
```

where `F` is a Fresnel term.

### Status

This is a future branch, not the first ablation.

Use it when:

```text
water/glass/specular scenes dominate
diffuse event laws fail specifically on reflective/refractive regions
```

Do not fake this with an unconstrained view-dependent color MLP unless it is
explicitly labeled as a costly residual, because that becomes a light-field
cache.

---

## 8. Multiple Objects, Not One Object Zoo

We probably do need more than many instances of one object. But we should avoid
a hand-labeled zoo.

A practical world asset can be:

```text
W_t = {
  event_measure_t,
  transport_fields_t,
  visibility_laws,
  material_laws,
  diagnostics
}
```

### Object 1: Material/Event Measure

Discrete events:

```text
nu_t = sum_i m_i delta_{e_i(t)}
```

Events can have metric support:

```text
e_i(t) = (x_i(t), Sigma_i(t), rho_i, c_i)
```

This covers:

```text
points
splats
metric blobs
thin patches if Sigma is rank-2-like
particles
```

### Object 2: Transport Field

Rigid:

```text
x_i(t) = R_b(t) x_i^0 + b_b(t)
```

Low-rank current harness:

```text
x_i(t) = x_i^0 + sum_l gamma_{t,l} B_{i,l}
```

Deformation graph:

```text
Phi_t(q_i) = sum_{k in N(i)} w_{ik}
             [R_k(t)(q_i - c_k) + c_k + b_k(t)]
```

Fluid:

```text
partial_t sigma + div(sigma v) = s - kappa sigma
```

The same material event should not be forced to explain fog and rigid metal
with the same transport law.

### Object 3: Visibility Law

At minimum:

```text
surface/event law
metric density law
volume law
ray-transform law
```

But implementation should stage them:

```text
stage 1: projected conic and ray-Gaussian metric events
stage 2: slab/surface only for surface benchmarks
stage 3: volume law for fog/smoke scenes
stage 4: ray-transform law for water/glass/speculars
```

### Object 4: Material Appearance

Current:

```text
c_i = persistent baked RGB
```

Better:

```text
albedo_i
normal_i
roughness_i
emission_i
view-dependent residual_i(d) with high cost
```

Do not add this before geometry tests pass unless the data requires it.

---

## 9. Regime Walkthrough

### Rigid Bodies

Best object:

```text
surface/metric events attached to shared rigid transforms
```

Equations:

```text
x_i(t) = R_b(t) q_i + b_b(t)
Sigma_i(t) = R_b(t) G_i R_b(t)^T
```

Recommended renderer laws:

```text
projected conic for fast ablation
ray-Gaussian for exact density support
slab/mesh for opaque surface branch
```

Failure if:

```text
each event has independent motion
support inflates instead of sharing rigid body motion
color is baked from source view and fails novel view
```

Missing diagnostic:

```text
body-level shared-transform residual
multi-ray witness rank
view-stress edge stability
```

### Cloth

Best object:

```text
deformable surface sheet
```

Candidate event form:

```text
q = (u, v) in cloth material coordinates
x(u, v, t) in R^3
normal n(u, v, t)
thin support h
```

Local metric:

```text
G_i has lambda_1 ~= lambda_2 >> lambda_3
```

Useful regularizers:

```text
stretch = ||partial_u x||, ||partial_v x||
bending = ||partial_uu x|| + ||partial_vv x||
normal_smoothness
```

Recommended renderer laws:

```text
slab intersection for explicit surface tests
rank-adaptive metric with rank-2 spectrum
projected conic for fast approximation
```

Failure if:

```text
cloth folds create self-occlusion and the renderer lacks correct visibility
metric events detach from a coherent sheet
```

Need something else?

```text
Probably yes for high-quality cloth: a mesh/sheet topology or strong gluing
between neighboring material coordinates.
```

### Fog / Smoke

Best object:

```text
participating density field
```

Equations:

```text
C(ell) = integral T(s) sigma_t(r(s)) c_t(r(s)) ds
partial_t sigma + div(sigma v) = source - decay
```

Recommended renderer laws:

```text
full volume integration
ray-Gaussian particles as sparse density basis
```

Bad fits:

```text
slabs
opaque surface events
low ray-entropy penalties
```

Can splats handle it?

```text
Yes, if treated as volumetric Gaussian density particles. But free splats can
also form source-view opacity shells unless constrained by held-out views,
advection, and density regularization.
```

Need something else?

```text
Yes: source/dissipation dynamics and volume-specific evaluation. Persistent
material opacity is not enough.
```

### Water

Water is not one regime.

#### Clear water surface

Best object:

```text
surface with refractive/specular ray-transform law
```

Equations:

Reflection:

```text
d_reflect = d - 2 (d^T n) n
```

Refraction using Snell:

```text
eta = eta_air / eta_water
cos_i = -n^T d
k = 1 - eta^2 (1 - cos_i^2)
d_refract = eta d + (eta cos_i - sqrt(k)) n
```

This is not standard diffuse splatting. The event maps an outgoing camera ray
to another incoming ray.

#### Foam / spray / bubbles

Best object:

```text
volumetric particles or ray-Gaussian density events
```

#### Murky water

Best object:

```text
volume absorption/scattering field
```

Need something else?

```text
Yes. Water needs at least two phases:
surface ray transform + volumetric foam/turbidity.
```

A splat field can fake water appearance in source views. It does not naturally
encode refraction unless the renderer has ray-transform events or a learned
view-dependent residual, which is dangerous as a cache.

---

## 10. Ablation Plan

We can ablate all four laws if we keep the world state stable.

### Shared State

Use:

```text
event_state = (x_i(t), Sigma_i(t), rho_i, c_i, q_i)
```

for:

```text
projected_conic
ray_gaussian_line_integral
```

Use a restricted surface state for:

```text
slab_intersection
```

Use density grid/particles for:

```text
volume_integration
```

### Separate Fairness Axes

Do not claim one matched experiment matches all. Run these separately:

```text
same primitive/event count
same active parameter count
same initial coverage budget
same optimizer steps
same wall-clock budget
same source PSNR band
```

### Minimal Matrix

| run | object | kappa/render law | purpose |
| --- | --- | --- | --- |
| A | transported point | screen disk | internal control |
| B | transported metric | projected conic | current world-support approximation |
| C | transported metric | ray-Gaussian line integral | exact incidence test |
| D | transported slab | slab intersection | surface branch |
| E | density particles/grid | volume integration | fog/smoke branch |
| F | free dynamic 3DGS | existing renderer | external baseline |

## 10.5. Per-Law Implementation Plans

### Plan A: Projected Conic

Status:

```text
control / keep
```

Goal:

```text
Make the current projected-conic path the stable control for every future law.
```

Implementation:

```text
1. Keep current support_mode=screen_disk and rank_adaptive_metric projected
   conic path.
2. Add explicit incidence_mode="projected_conic" even if it initially aliases
   the current support renderer.
3. Log the exact kappa convention:
       kappa = tau_i exp(-0.5 y^T Sigma2^-1 y)
4. Add mass/peak parameter labels:
       raw_tau_peak or raw_mass
5. Add coverage-budget matching utilities.
```

Proof obligation:

```text
Show that projected conic is the first-order pushforward of world covariance.
Already derived above.
```

Diagnostics:

```text
projection coverage
screen eigenvalue clamp fraction
metric spectrum if world metric is used
view-stress delta
held-out PSNR
```

Kill criterion:

```text
Do not kill as a control. Kill only as a final primitive claim.
```

### Plan B: Ray-Gaussian Line Integral

Status:

```text
implement_next
```

Goal:

```text
Test whether exact world ray-event incidence beats projected conic
using the same metric event state.
```

Implementation phases:

```text
Phase B0: scalar function tests
    implement ray_gaussian_tau_peak
    implement ray_gaussian_tau_mass
    test against numeric quadrature on random SPD matrices

Phase B1: no-grad diagnostic renderer
    render a small 32px/64px frame by ray-event loops
    compare projected_conic vs ray_gaussian_line on same checkpoint

Phase B2: trainable small renderer
    use mass-normalized kappa
    dense ray/event loop at small N/H/W
    train only short smoke configs

Phase B3: fair ablation
    same event count
    same active parameter count table
    same coverage or source-PSNR band
    held-out DeepView comparison

Phase B4: acceleration only if signal exists
    tile/ray candidate culling
    spatial hash or projected bounds
```

Unit tests:

```text
1. Isotropic Gaussian:
       Sigma = sigma^2 I
       tau_whole_line should equal
       mass / (2 pi sigma^2) * exp(-dist_perp^2 / (2 sigma^2))

2. Numeric quadrature:
       closed_form_tau ~= sum_s sigma(o+s d) Delta s

3. Rigid equivariance:
       transform mu, Sigma, ray by same rigid transform
       tau should remain unchanged

4. Mass conservation:
       integrate sigma over 3D grid approximately equals mass
```

Expected first failure:

```text
too slow
candidate selection too broad
mass-normalized events become too transparent at large Sigma
event sorting by s_star is only approximate for broad Gaussians
```

Decision rule:

```text
Continue if held-out/view-stress improves or cheat probes become more
detectable at acceptable cost.

Demote if it only improves source RGB or is slower with no structural gain.
```

### Plan C: Slab Intersection

Status:

```text
baseline_only / surface-specialist
```

Goal:

```text
Keep an explicit surface law for cloth/opaque surfaces, but stop treating it as
the general next primitive.
```

Implementation:

```text
1. Reuse oriented_slab state.
2. Add incidence_mode="slab_intersection" only for surface tests.
3. Compare peak thickness form versus area-normalized surface mass.
4. Run on synthetic planes, rigid boxes, and cloth sheets before real clips.
```

Synthetic tests:

```text
fronto-parallel plane
tilted plane
grazing-angle plane
folded cloth sheet with known mesh
```

Diagnostics:

```text
grazing alpha histogram
normal consistency
surface depth error on synthetic scenes
held-out PSNR on surface-only scenes
```

Kill criterion:

```text
If it keeps losing to screen_disk and rank_adaptive_metric on surface-heavy
held-out scenes, remove it from near-term research and keep only as reference.
```

### Plan D: Full Volume Integration

Status:

```text
defer until fog/smoke data or synthetic volume tests
```

Goal:

```text
Handle participating media honestly instead of forcing fog/smoke into surfaces
or opaque splats.
```

Implementation choices:

```text
D1. Sparse Gaussian density basis
    sigma(x) = sum_i mass_i N(x; mu_i, Sigma_i)
    use ray-Gaussian kappa as primitive interval/event contribution

D2. Sparse voxel/hash density
    sigma = lookup(grid, x)
    use standard ray marching

D3. Transported density particles
    particles advect with velocity
    density can dissipate or be sourced
```

Start with:

```text
D1 sparse Gaussian density basis
```

because it shares math with ray-Gaussian metric events.

Needed dynamics:

```text
partial_t sigma + div(sigma v) = source - decay sigma
```

Diagnostics:

```text
ray termination entropy
density mass over time
source/dissipation magnitude
held-out view consistency through translucent regions
```

Kill criterion:

```text
If the scene is opaque-rigid/cloth and volume only creates foggy floaters,
do not use volume as default. Keep it phase-specific.
```

### Plan E: Ray-Transform Surface

Status:

```text
defer / water-glass-specialist
```

Goal:

```text
Handle clear water, glass, and mirror-like objects without letting the model
learn arbitrary view-dependent RGB.
```

Implementation stages:

```text
E0. Diagnostic only:
    detect where diffuse/splat laws fail at high view-dependence

E1. Synthetic reflection plane:
    known mirror plane, known reflected texture

E2. Refractive water plane:
    known eta, simple background, Snell ray transform

E3. Rough water:
    distribution of transformed rays instead of one ray
```

Renderer:

```text
surface hit gives alpha_hit
ray transform maps ell_out to ell_in
sample source/environment/world along ell_in
```

Kill criterion:

```text
If implementation becomes a free view-dependent color model, kill or label as
residual, not world geometry.
```

### Metrics

Prediction:

```text
train PSNR/L1
held-out camera PSNR/L1
view-stress consistency
```

Geometry/identity:

```text
X-map occupancy
X-map flow consistency
Pluecker witness spectrum
weak witness fraction
concurrence residual
```

Support health:

```text
coverage budget
metric eigenvalue ratios
metric log-volume
opacity-density tradeoff
termination entropy
```

Cost:

```text
sec/step
render-only time
memory
events touched per pixel/tile
```

Cheat probes:

```text
depth slide
metric inflation with density compensation
opacity split
X-map shuffle
time coefficient perturbation
wrong-world swap
```

---

## 11. What Is Ready?

Ready now:

```text
projected conic approximation
screen_disk control
oriented slab baseline
rank-adaptive metric state
held-out DeepView lane
free_dynamic_3dGS baseline
summary tooling
```

Ready to formulate and implement next:

```text
ray-Gaussian line integral kappa for metric events
Pluecker witness diagnostic from sampled ray/event contributions
coverage-matched support-law ablation
```

Solved enough to implement:

```text
kappa is optical depth
alpha = 1 - exp(-kappa)
rendering ray is origin-direction segment (o,d,s0,s1)
Pluecker ray is diagnostic line coordinate (d,m)

projected_conic:
    kappa = tau_i exp(-0.5 y^T Sigma2^-1 y)

ray_gaussian_line, preferred:
    kappa = mass-normalized closed-form Gaussian line integral

slab:
    kappa = area-normalized tangent kernel divided by abs(n dot d)

volume:
    kappa = interval optical depth integral

ray_transform:
    kappa_hit gates a deterministic or rough reflection/refraction map
```

Not ready as final representation:

```text
event-measure incidence as a broad framework
phase-conditioned visibility zoo
water/glass ray-transform renderer
full scalable tiled/volume renderer
```

---

## 12. What Is Missing?

### Missing 1: Exact Kappa Decision

This is now mostly solved at the theory level.

The core formula is not the abstract event measure. It is the concrete:

```text
kappa(ell, e_i)
```

For metric events, the strongest candidate is:

```text
kappa = mass-normalized true ray-Gaussian line integral
```

because it is world-geometric, closed form, and shares parameters with current
rank-adaptive metric elements.

What remains missing is empirical selection:

```text
projected_conic vs ray_gaussian_line
peak-density vs mass-normalized line integral
event compositing by s_star vs interval volume compositing
```

### Missing 2: Contribution Weights For Witness Metrics

For each element:

```text
W_i = sum_k w_ki (I - d_k d_k^T)
```

We need practical `w_ki`.

Approximate first version:

```text
sample pixels
compute top-k event alpha weights for those pixels
accumulate ray directions into W_i
```

Exact later version:

```text
renderer emits per-tile top contributing event ids/weights
```

Ray representation for this metric:

```text
Use Pluecker (d,m) because witness rank is a line-concurrence property.
Do not use Pluecker for finite-segment alpha integration unless near/far
segment bounds are carried separately.
```

### Missing 3: Motion Laws

Current:

```text
x_i(t) = x_i^0 + sum_l gamma[t,l] B[i,l]
```

This is a useful bottleneck but not enough for every material.

Likely needed:

```text
rigid group transforms
deformation graph
cloth sheet constraints
fluid advection
```

### Missing 4: Phase/Material Law

Current:

```text
persistent baked RGB + alpha
```

Needed later:

```text
surface event
volume density
ray transform
BRDF/albedo/shading split
```

Do not add all now. Add when a benchmark demands it.

### Missing 5: Scalable Renderer

Current pure Torch harness is enough to falsify ideas at 128px.

Final path likely needs:

```text
tiled projected conic rasterizer
or ray/event acceleration for line integrals
or sparse volume ray marcher
```

---

## 13. Is This Useful?

Yes, but only if we keep the claim narrow.

Not ready to claim:

```text
we replaced splats
we solved novel-view 3D
rank-adaptive metrics are better
event measures are the answer
```

Ready to claim:

```text
we have a clean ablation interface for ray-event incidence laws
the next best concrete object is transported rank-adaptive metric events
the next best kappa is probably the true ray-Gaussian line integral
screen_disk and free_dynamic_3dGS remain the controls
slab is demoted unless surface-only data revives it
volume/ray-transform phases are real but should be staged
```

---

## 14. Recommended Next Implementation

Implement:

```text
support_mode = rank_adaptive_metric
incidence_mode = projected_conic | ray_gaussian_line_peak | ray_gaussian_line_mass
```

Keep event state identical:

```text
x_i(t), Sigma_i(t), rho_i, c_i
```

Only swap:

```text
how kappa(ray, event) is computed
```

Use:

```text
ray_gaussian_line_mass
```

as the preferred exact-incidence candidate.

### Minimal Function Contract

Use an explicit `kappa` API so the ablation is a law swap, not a model rewrite:

```python
def compute_event_state(params, frame_index):
    """
    Returns:
        mu:      [N, 3] world positions
        Sigma:   [N, 3, 3] world SPD supports
        mass:    [N] nonnegative optical mass or density parameter
        color:   [N, 3] event-attached RGB/baked appearance
    """

def kappa_projected_conic(rays, pixels, camera, event_state):
    """
    Fast approximation.
    Returns sparse or dense entries:
        event_id, ray_id, optical_depth, order_s
    """

def kappa_ray_gaussian_line(rays, event_state, mass_normalized=True):
    """
    Exact v1 incidence.
    rays carry (origin, direction, near, far).
    Returns:
        event_id, ray_id, optical_depth, order_s
    """

def composite(entries, color, background):
    """
    Sort entries by order_s per ray, then alpha-composite with:
        alpha = 1 - exp(-optical_depth)
    """
```

`kappa_*` functions are not allowed to read:

```text
training image RGB
pixel id as an embedding
camera id as an embedding
frame id except through transported event state
```

They may read:

```text
ray origin/direction/near/far
camera projection for projected-conic approximation
event state
global numerical constants and clamps
```

### Exact Ray-Gaussian Helper

For each event/ray pair:

```python
def gaussian_line_tau(o, d, s0, s1, mu, Sigma, mass, eps=1e-6):
    A = inv_spd(Sigma)
    v = o - mu

    a = d.T @ A @ d
    b = d.T @ A @ v
    c = v.T @ A @ v

    a = clamp_min(a, eps)
    d2 = c - (b * b) / a

    norm = mass / (((2*pi) ** 1.5) * sqrt(det(Sigma)))
    erf_hi = erf(sqrt(a / 2) * (s1 + b / a))
    erf_lo = erf(sqrt(a / 2) * (s0 + b / a))

    tau = norm * exp(-0.5 * d2) * sqrt(pi / (2 * a)) * (erf_hi - erf_lo)
    return clamp_min(tau, 0.0)
```

Whole-line approximation:

```text
s0 = -infinity
s1 = +infinity

tau_whole =
mass / (2 pi sqrt(det Sigma) sqrt(a))
* exp(-0.5 (c - b^2/a))
```

Use finite segments for rendering when possible. Use whole-line only as a speed
or sanity-control variant.

Add metrics:

```text
metric eigenvalue ratios
metric log-volume
approx Pluecker witness spectrum
ray-Gaussian closed-form vs numeric quadrature error
rigid invariance error for kappa
projected-conic validity ratio eta_proj = sqrt(lambda_max(Sigma)) / z
termination entropy
held-out view stress
```

Decision rule:

```text
If ray_gaussian_line beats projected_conic or screen_disk on held-out/view
stress at acceptable cost, continue.

If it only improves source RGB or collapses into inflated density, kill or
demote rank-adaptive metric events.
```

This is the shortest path from theory to evidence.
