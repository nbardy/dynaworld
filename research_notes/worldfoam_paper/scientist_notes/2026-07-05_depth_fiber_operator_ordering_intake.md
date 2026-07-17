# Scientist Dump Intake: Depth Fiber And Operator Ordering

Date: 2026-07-05

Source:

```text
/Users/nicholasbardy/.codex/attachments/fd8d76a2-220a-4046-a0c4-5001bd849ab1/pasted-text.txt
```

Purpose: critique the scientist dump, keep the depth-fiber details rolling, and
record the best ideas in a form that updates both paper lanes without confusing
World Tubes and WorldFoam.

## Verdict

The dump is high-value. It is not a new renderer implementation plan by itself,
but it sharpens the mathematical spine and gives a clean reviewer-facing proof
set.

Best durable idea:

```text
Known camera program
+ ray-bundle gauge
+ delayed or early depth pushforward
+ event-certified atlas
= reusable dynamic rendering calculus.
```

Most important distinction:

```text
World Tubes:
    push / marginalize primitives to UVT,
    then repair visibility with depth/order certificates.

WorldFoam:
    do not marginalize depth before visibility;
    keep sigma(u,v,t,z),
    make visibility a prefix integral.
```

This distinction is mathematically real. It is not just branding.

## What To Keep

### K1. Camera-ray trace operator

Keep the invariant operator:

```text
Trace_Gamma[w] = pi_* Gamma^* w
```

Expanded:

```text
B = Omega x T
y = (u,v,t)
pi: E_Gamma -> B
F_y = pi^{-1}(y)
Gamma: E_Gamma -> M
```

For a density-like world primitive:

```text
bar_rho_i(y)
    = integral_{F_y} rho_i(Gamma(e)) dmu_y(e)
```

This is the coordinate-free definition of a camera-path compiler. It separates:

```text
Gamma^*: world -> camera-ray bundle
pi_*:    ray-fiber integration / summarization
```

### K2. Gauge-invariant fiber integration

Keep the theorem:

```text
pi_* Gamma^* w
```

is invariant under smooth depth/fiber coordinate changes if the fiber measure
uses the correct Jacobian.

Local gauge:

```text
chi_a(e) = (y,z_a)
Gamma_a(y,z_a) = Gamma(chi_a^{-1}(y,z_a))
dmu_y(e) = J_a(y,z_a) dz_a
```

Trace:

```text
bar_rho_i^a(y)
    =
    integral rho_i(Gamma_a(y,z_a)) J_a(y,z_a) dz_a
```

If `z_b = phi_y(z_a)`, then:

```text
J_a(y,z_a) dz_a = J_b(y,z_b) dz_b
```

This is the defense for ordinary depth, log depth, inverse depth, projective
depth, orbit gauges, and rolling-shutter gauges as coordinate choices rather
than separate renderers.

### K3. Schur-complement World Tubes derivation

Keep this as the algebraic jewel of World Tubes, not WorldFoam.

World spacetime Gaussian:

```text
rho_i(x) = a_i exp[-1/2 (x - m_i)^T Lambda_i (x - m_i)]
```

Local camera gauge:

```text
Gamma_l(y,z) ~= x0 + J eta
eta = [delta_y, delta_z]^T
```

Pulled-back precision:

```text
H = J^T Lambda_i J

H = [H_yy H_yz
     H_zy H_zz]
```

Marginalizing depth:

```text
S = H_yy - H_yz H_zz^{-1} H_zy
```

Conditional depth:

```text
z_hat_i(y) = z0 + H_zz^{-1}(g_z - H_zy (y - y0))
Var(z | y) = H_zz^{-1}
```

Paper claim:

```text
Under an affine camera-gauge approximation, the ray-fiber marginal of a
spacetime Gaussian is a sensor-time Gaussian whose precision is the Schur
complement of the pulled-back precision with respect to the fiber coordinate.
```

### K4. Visibility/marginalization non-commutation

This is the deepest shared lesson.

```text
render_after_marginalizing_z
    !=
marginalize_after_rendering_along_z
```

Two translucent layers:

```text
I_12 = alpha_1 c_1 + (1 - alpha_1) alpha_2 c_2
I_21 = alpha_2 c_2 + (1 - alpha_2) alpha_1 c_1
```

Difference:

```text
I_12 - I_21 = alpha_1 alpha_2 (c_1 - c_2)
```

So a UVT-only marginal that stores total opacity and average/premultiplied
color cannot determine every translucent rendering. Depth/order information is
not optional.

Implication:

```text
World Tubes needs a visibility gauge atlas after early pushforward.
WorldFoam keeps the lifted fiber because visibility lives on that fiber.
```

### K5. WorldFoam lifted opacity pullback

Keep this as WorldFoam's central equation:

```text
rho_{j,l}(y,z)
    =
    1_{Gamma_l(y,z) in F_j}
    sigma_j(Gamma_l(y,z))
    J_l(y,z)
```

Aggregate:

```text
sigma_l(y,z) = sum_j rho_{j,l}(y,z)
q_l(y,z) = sum_j rho_{j,l}(y,z) c_j(Gamma_l(y,z))
c_l(y,z) = q_l(y,z) / sigma_l(y,z)
```

This is not a UVT footprint. It is a lifted field over `(u,v,t,z)`.

### K6. Transmittance-prefix rendering

Keep:

```text
tau_l(y,z) = integral_{z_front}^{z} sigma_l(y,s) ds
T_l(y,z) = exp(-tau_l(y,z))
I(y) = integral T_l(y,z) sigma_l(y,z) c_l(y,z) dz
alpha(y) = 1 - exp(- integral sigma_l(y,z) dz)
```

Beer-Lambert is standard. The contribution is not "we discovered
transmittance." The contribution is the camera-gauged, bounded-cell, event
compiled representation that reuses the expensive ray/cell/visibility/backward
structure across time.

### K7. Prefix/suffix adjoint

Keep this because it directly constrains the GPU implementation.

For:

```text
I(y) = integral T(z) sigma(z) c(z) dz
```

variation:

```text
delta I(y)
    =
    integral T(z) sigma(z) delta c(z) dz
    +
    integral T(z) [c(z) - I_behind(y,z)] delta sigma(z) dz
```

Therefore backward needs:

```text
front prefix T
behind suffix I_behind
local basis derivatives
fixed-topology endpoint derivatives or refresh/fallback discipline
```

This explains why "just store alpha" is not enough.

### K8. Power-cell event equations

Keep the inherited PowerFoam/Radiant-Foam geometry as WorldFoam's event
calculus.

Power distance:

```text
pow_i(x) = ||x - p_i||^2 - r_i^2
```

Bounded power cell:

```text
B_i = { x : ||x - p_i|| <= r_i
            and pow_i(x) <= pow_j(x) for neighbors j }
```

Radical face:

```text
n_ij = p_j - p_i
h_ij = 0.5 (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)
x dot n_ij <= h_ij
```

Ray crossing:

```text
x(s) = o + s d
s_face = (h_ij - o dot n_ij) / (d dot n_ij)
```

This gives WorldFoam sparse event records rather than dense UVTZ memory.

### K9. Same-representation replay equivalence

Keep as the first proof baseline.

Claim:

```text
On certified atlas domains, compiled WorldFoam evaluation equals per-frame
WorldFoam replay up to declared basis, quadrature, support, and fallback error.
```

This isolates compiler speedup from representation changes. It also avoids the
mistake of using STAR/GS quality comparisons as the first theorem-level proof.

### K10. Fixed-topology VJP boundary

Keep the limitation as a theorem boundary.

For a moving interval:

```text
I(theta) = integral_{a(theta)}^{b(theta)} f(z,theta) dz
```

Leibniz:

```text
dI/dtheta
    =
    integral_a^b partial_theta f(z,theta) dz
    + f(b,theta) b'(theta)
    - f(a,theta) a'(theta)
```

If an implementation ignores endpoint movement or topology changes, then the
honest claim is:

```text
direct VJP is exact for fixed compiled topology;
boundary/topology gradients are handled by refresh, split, smoothing, fallback,
or future boundary calculus.
```

### K11. Event-complexity scaling

Keep the asymptotic story:

```text
F = number of frames / temporal samples
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

If:

```text
E(F) = o(F R)
```

then structural work is sublinear relative to replay.

Precise wording:

```text
Output pixels/samples remain linear. The claim is sublinear growth of dominant
world-side structural work and, in regimes where that dominates, sublinear
measured wall time over the tested frame range.
```

## What Not To Overclaim

### D1. Do not call the ingredients individually new

Classical ingredients:

```text
Schur complement
change of variables
Beer-Lambert transmittance
power diagrams
Leibniz boundary rule
front-to-back compositing gradients
```

Novel-looking synthesis:

```text
camera-ray bundle traces as pi_* Gamma^* world primitives
gauge-invariant camera-path compilation with explicit fiber Jacobians
Schur-complement UVT traces for dynamic Gaussian world tubes
visibility/marginalization non-commutation as the split between papers
WorldFoam delayed fiber rendering R_z Gamma^* matter
event-certified atlases whose complexity tracks camera-path events
prefix/suffix direct VJP over compiled transmittance records
```

### D2. Do not sell dense sigma(u,v,t,z)

Dense UVTZ is the wrong memory shape.

Preferred object:

```text
sparse cell-ray events
interval endpoints
layer summaries
local basis coefficients
prefix summaries / scalar tapes
fallback masks
```

### D3. Do not force Schur complement onto foam

Schur complement is for local Gaussian marginalization. Foam is bounded-cell
event geometry plus retained-fiber transmittance.

### D4. Do not hide topology nondifferentiability

Topology changes are real:

```text
adjacency changes
endpoint winner changes
support interval birth/death
depth-layer partition changes
fallback mask changes
```

The paper should state the fixed-topology VJP boundary and then measure
refresh/fallback rates.

## Paper Theorem Set To Promote

World Tubes paper:

```text
1. Gauge-invariant trace.
2. Local Gaussian Schur-complement UVT marginal.
3. Visibility/marginalization non-commutation.
4. Pairwise commutation bound for unresolved order.
5. Event/gauge-domain scaling for known camera programs.
```

WorldFoam paper:

```text
1. Gauge-invariant lifted opacity pullback.
2. Transmittance-prefix rendering over retained depth fiber.
3. Prefix/suffix adjoint.
4. Power-cell ray event intervals.
5. Same-representation replay equivalence.
6. Fixed-topology VJP boundary.
7. Event-complexity scaling.
8. Optical-depth/radiance/support/quadrature error decomposition.
```

## Reviewer Attack Map

Likely reviewer attack:

```text
"Is this just volume rendering?"
```

Answer:

```text
The transmittance equation is standard. The contribution is a camera-compiled,
sparse bounded-cell event atlas that reuses ray/cell/visibility/adjoint
structure over a known camera program.
```

Likely reviewer attack:

```text
"Where is sublinear if pixels are linear?"
```

Answer:

```text
Output samples are linear. The claimed sublinear term is structural world-side
work: intersection, visibility/prefix metadata, cache payload, and backward
replay. End-to-end sublinear wall time is a measured regime claim, not an
absolute law.
```

Likely reviewer attack:

```text
"Topology is nondifferentiable."
```

Answer:

```text
Yes. The direct VJP is fixed-topology. Topology changes trigger refresh,
split, smoothing, fallback, or future boundary calculus. We measure topology
churn as a first-class failure mode.
```

Likely reviewer attack:

```text
"Quality is worse than Gaussian splatting."
```

Answer:

```text
Current claim boundary is theory/prototype/speed-scale until public quality
gates clear. Same-representation replay is the first proof baseline; STAR/GS
comparisons are contextual until quality parity is real.
```

## Figure And Ablation Ideas To Keep

Figures:

```text
1. World Tubes vs WorldFoam operator ordering.
2. Camera-ray bundle with base B and depth fiber F_y.
3. Gauge-change Jacobian diagram.
4. Schur complement: 4D Gaussian -> (u,v,t,z) -> UVT footprint.
5. Non-commutation counterexample with two translucent layers.
6. Bounded power cell pulled into ray-fiber intervals.
7. Transmittance prefix and suffix adjoint along z.
8. Foam atlas data structure.
9. Same-representation replay equivalence ladder.
10. Event-density death curve.
```

Ablations:

```text
ordinary depth vs log depth vs inverse/projective depth with Jacobian checks
dense UVTZ grid vs sparse event atlas
compiled prefix vs per-frame foam replay
fixed-topology VJP vs finite-difference reference
endpoint tape vs recompute vs scalar prefix tape
crossing translucent slabs: sorted splat vs visibility atlas vs foam prefix
event-density sweep: smooth orbit to near-camera chaotic support
```

## Immediate Lane Updates

This intake creates or points to:

```text
research_notes/gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
```

Those files are the durable homes for the depth-fiber/operator-ordering math.
Future scientist dumps should be triaged into either:

```text
scientist_notes/      raw critique and idea extraction
proofs/               cleaned theorem/proof objects
experiment_designs/   runnable experiment specs
figures/              figure/table plans
```
