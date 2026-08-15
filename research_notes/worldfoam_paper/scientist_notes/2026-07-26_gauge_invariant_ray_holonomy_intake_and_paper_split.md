# Gauge-Invariant Ray Holonomy Intake And Paper Split

Date: 2026-07-26

Status: normalized research intake plus project-level paper decision.

Source:

```text
/Users/nicholasbardy/.codex/attachments/c3f6b522-fd32-4797-941d-8fc2ed5722e2/pasted-text.txt
```

Related canonical notes:

```text
research_notes/renderer_lane_taxonomy.md
research_notes/gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
research_notes/worldfoam_paper/WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
```

## Executive Decision

The proposal is strong, but it should not be folded wholesale into the current
World Tubes / STAR UVT paper.

The correct paper split is:

```text
Shared theorem layer:
    gauged camera-ray bundle
    coordinate-invariant fiber pullback
    event-certified camera domains

Paper A — World Tubes, implemented by STAR UVT:
    dynamic Gaussian-compatible representation
    early depth pushforward into UVT traces
    conditional depth/order certificates
    visibility strata and bounded fallback
    compiled interval forward and direct adjoint

Paper B — WorldFoam / gauge-invariant ray-holonomy renderer:
    retained ray-depth fiber
    path-ordered optical transfer
    noncommutative visibility algebra
    fiber-resolved trace records
    convex-potential or bounded-cell matter
    certified support/discriminant compiler
```

The attachment does not invalidate STAR UVT. It identifies STAR UVT's precise
approximation boundary and supplies a cleaner retained-depth alternative.
The noncommutation theorem explains why STAR needs visibility certificates and
fallbacks; it does not show that early pushforward is a poor engineering choice.

The phrase **gauge-invariant ray-holonomy renderer for moving cameras** is an
excellent Paper B description. It should currently be treated as a technical
subtitle or thesis sentence, not a new implementation-family name. The canonical
method name remains WorldFoam until evidence justifies renaming the lane.

## What Is Genuinely New Relative To The Existing Notes

Much of the attachment matches the existing WorldFoam optical-transfer plan:

```text
already present:
    retained depth fiber
    visibility monoid / affine transfer matrices
    product integral or path-ordered exponential
    commutator visibility theorem
    prefix/suffix VJP
    camera-gauge Jacobian invariance
    event-word / cell-path compilation
```

The strongest new or materially sharpened content is:

```text
1. The terminology correction:
   camera program changes the measurement; gauge changes its coordinates.

2. A renderer-level headline:
   gauge-invariant ray holonomy under moving cameras.

3. A candidate primitive family:
   self-normalized strongly convex polynomial spacetime potentials.

4. A one-interval ray-support theorem for those atoms.

5. Algebraically exact single-atom optical depth and endpoint-free first
   derivatives.

6. A discriminant-certified adaptive sensor-time compiler for the retained
   fiber polynomial.

7. A Duhamel-form exact transfer derivative and explicit approximation-error
   bound.

8. A clearer conditional complexity statement in terms of genuine chart,
   support, discriminant, and tile-incidence complexity rather than requested
   frame count.
```

The primitive proposal is not yet part of the tested WorldFoam implementation.
It is a research branch within Paper B, not a reason to reset the existing
bounded-cell/owner-run system.

## 1. Camera Programs, Gauges, And Ray Fibers

The attachment usefully corrects the loose phrase "cameras are gauges."

```text
camera program C:
    defines the measurement bundle and changes the observation

gauge:
    selects coordinates or a local trivialization of that bundle

ray:
    one-dimensional fiber over a sensor-time sample
```

Let:

```text
pi_C: E_C -> B_C
B_C = Omega x J
y = (u,v,tau)
F_y = pi_C^{-1}(y)
Gamma_C: E_C -> M
```

Here `J` is sensor or shutter time, and `Gamma_C` maps a point on a camera-ray
fiber into world spacetime.

A local gauge identifies:

```text
E_C | U ~= U x D
(y,s) -> Gamma_C(y,s)
```

Another valid oriented fiber coordinate may be:

```text
s' = phi_y(s)
partial_s phi_y > 0
```

The physical render must be invariant to this coordinate choice. Ordinary
depth, inverse depth, projective depth, log depth, and half-angle orbit
coordinates may therefore be selected for conditioning and algebraic degree,
provided their fiber-measure Jacobians are included.

A 360-degree orbit does not require one global UVT chart. It requires a finite
set of valid gauges whose transitions preserve the ray transfer. This retains
the existing project rule:

```text
chart, certify, and fall back; do not demand one global slab.
```

## 2. Why A Depth-Marginalized Summary Cannot Be Universally Exact

For a thin colored layer with opacity `alpha_i` and color `c_i`, define the
affine transfer matrix:

```text
T_i =
    [ 1 - alpha_i    alpha_i c_i ]
    [ 0              1           ]
```

For two layers:

```text
T_i T_j - T_j T_i =
    [ 0    alpha_i alpha_j (c_i - c_j) ]
    [ 0    0                              ]
```

Therefore:

```text
T_i T_j = T_j T_i
iff
alpha_i alpha_j (c_i - c_j) = 0.
```

This is the exact algebra behind:

```text
|Delta I_ij| <= alpha_i alpha_j |c_i - c_j|.
```

No order-independent set of per-primitive `(alpha, color, representative
depth)` summaries can reproduce arbitrary colored overlap exactly. The honest
options are:

```text
retain a depth-resolved field along the ray
retain ordered jump operators for surface-like atoms
accept and certify a bounded commutation approximation
```

This is the clean theorem separating Paper A and Paper B.

## 3. Vertical Transfer Connection And Ray Holonomy

Let atom `i` carry extinction `sigma_i >= 0` and emitted radiance density
`j_i in R^3`. Define the augmented optical generator:

```text
A_i(x,t,omega) =
    [ -sigma_i I_3    j_i ]
    [ 0               0   ]
```

Pulled onto a camera-ray fiber:

```text
calA_i,C =
    A_i(Gamma_C(y,s), omega_C(y)) d ell_C(y,s)
```

where `d ell_C` is physical ray length. The scene generator adds locally:

```text
calA_C = sum_i calA_i,C.
```

The full ray transfer is:

```text
H_C(y) = P exp integral_{F_y} calA_C
```

where `P exp` is the path-ordered exponential. Acting on augmented background
radiance:

```text
Lbar_bg = [L_bg, 1]^T
L_C(y) = project_RGB(H_C(y) Lbar_bg).
```

This is ordinary emission-absorption rendering written as parallel transport
along the ray fiber. "Holonomy" here means the ordered transfer accumulated
along that one-dimensional connection. It must not be confused with the older
cell-graph loop-holonomy diagnostic.

### Gauge invariance

Under `s' = phi_y(s)`:

```text
calA'_C(y,s') = calA_C(y,s) ds/ds'
calA'_C ds' = calA_C ds
H'_C(y) = H_C(y).
```

The rendering operator is invariant; only its local coordinates change.

### Moving overlap

At one ray location:

```text
sigma_total(s) = sum_i sigma_i(s)
j_total(s) = sum_i j_i(s).
```

Overlapping generators add. Disjoint supports factor in fiber order. When a
moving camera changes which atom is in front, the fiber generator changes
continuously instead of requiring a global primitive-order metadata swap.

The revised trace target is therefore:

```text
retain a compact representation of A_i(y,s),
not merely integral A_i(y,s) ds.
```

## 4. Determinant Atom: Useful Baseline, Not Preferred Object

The proposed unconstrained determinant density was:

```text
sigma(x,t) = det(L(x,t))^m 1_{L(x,t) >= 0}.
```

It has attractive ray algebra, but training can produce empty or unbounded
slices, redundant spatial directions, ill-conditioned analytic centers, and
structural fixes with privileged interior witnesses.

A capped version can be formed with:

```text
z(x,t) = U(t)x + b(t)

B(z) =
    [ 1    z^T ]
    [ z    I_3 ]

L(x,t) = B(z) direct_sum
         [C_0(t) + sum_{k=1}^3 z_k C_k(t)].
```

Because:

```text
B(z) >= 0 iff |z| <= 1,
```

invertible `U(t)` makes the support bounded. If `C_0(t) > 0`, `z=0` is an
interior point, every time slice is nonempty and full-dimensional, and
`log det B(z) = log(1 - |z|^2)` supplies strict concavity and a unique ridge.

The drawback is the privileged witness:

```text
x_witness(t) = -U(t)^{-1} b(t).
```

This is close to storing a reference position. The capped determinant atom is
therefore a strong matched baseline, not the preferred intrinsic object.

## 5. Self-Normalized Convex-Potential Spacetime Atom

Let a global spacetime polynomial satisfy uniform spatial strong convexity:

```text
q_theta: R^3 x I -> R
nabla_x^2 q_theta(x,t) >= lambda I_3
```

Define the derived minimum and ridge:

```text
mu_theta(t) = min_x q_theta(x,t)
r_theta(t) = argmin_x q_theta(x,t)
D_theta(x,t) = q_theta(x,t) - mu_theta(t).
```

Define density:

```text
sigma_theta(x,t)
    = alpha_theta(t) (1 - D_theta(x,t))_+^p.
```

The atom stores one global field. Its center/ridge is derived rather than
stored.

### Degrees of freedom and scalar gauge

For spatial degree `d_x` and temporal degree `d_t`:

```text
q(x,t) =
    sum_{|beta| <= d_x, 0 <= j <= d_t}
        theta_{beta,j} x^beta t^j.
```

The dense raw coefficient count is:

```text
(d_t + 1) choose(d_x + 3, 3).
```

Adding any time-only `c(t)` does not change `D`. Quotienting that scalar gauge
removes `d_t + 1` coefficients:

```text
K_q = (d_t + 1) [choose(d_x + 3, 3) - 1].
```

Opacity and radiance marks add separate finite basis dimensions.

### Structurally safe parameterization

One sufficient parameterization is:

```text
q(x,t) =
    1/2 x^T G(t) x + ell(t)^T x
    + sum_r w_r(t) (a_r(t)^T x + b_r(t))^(2 d_r)

G(t) = lambda_0 I + F(t)^T F(t)
w_r(t) = epsilon + p_r(t)^2.
```

Its Hessian is:

```text
nabla_x^2 q =
    G(t)
    + sum_r w_r(t) 2 d_r (2 d_r - 1)
      (a_r^T x + b_r)^(2 d_r - 2) a_r a_r^T
    >= lambda_0 I.
```

Temporal functions can use global polynomial or Chebyshev bases. SOS-convex
certificates are a broader optional branch; they are semidefinite-checkable but
only sufficient in general dimensions/degrees.

Reference from the intake:

```text
https://arxiv.org/abs/1111.4587
```

## 6. Guaranteed Slice Topology And Ridge Conditioning

Strong convexity gives one unique minimizer and:

```text
q(x,t) - q(r(t),t) >= lambda/2 |x-r(t)|^2.
```

Thus:

```text
S_t = {x : D(x,t) <= 1}
S_t subset B(r(t), sqrt(2/lambda)).
```

Every time slice is nonempty, compact, convex, and connected. It cannot acquire
a hole, disconnected component, or second ridge while the certificate holds.

The boundary `D=1` is regular: `nabla_x D=0` would imply `x=r(t)`, where
`D=0`, a contradiction.

At the ridge:

```text
H_q(t) = nabla_x^2 q(r(t),t) >= lambda I.
phi = -log sigma
nabla_x^2 phi(r(t),t) = p H_q(t).
```

So the intrinsic scale tensor has a certified lower spectral bound
`p lambda I`. An upper bound can be obtained from coefficient bounds and a
certified enclosure of `r(t)`.

## 7. Derived Kinematics And Clean Normalization Gradients

Differentiating `nabla_x q(r(t),t)=0` yields:

```text
dot r(t) = -H_q(t)^{-1} q_xt(r(t),t).
```

The acceleration is:

```text
ddot r = -H_q^{-1}
    [q_xtt + 2 q_xxt dot r + q_xxx[dot r,dot r]].
```

For nonzero velocity:

```text
kappa_vec =
    (I - vhat vhat^T) ddot r / |dot r|^2.
```

Orientation and scale come from eigenvectors/eigenvalues of `p H_q(t)`.
Position, velocity, quaternion, and covariance are not independent stored
parameters.

The envelope theorem gives:

```text
partial_theta mu(t) = partial_theta q(r(t),t)

partial_theta D(x,t) =
    partial_theta q(x,t) - partial_theta q(r(t),t).
```

There is no `partial_theta r` term because the spatial gradient vanishes at the
minimizer. This is a valuable optimization property.

## 8. One-Interval Ray Support And Exact Single-Atom Optical Depth

For a straight ray at physical time `t_y`:

```text
x(s) = o_y + s d_y
h_y(s) = 1 + mu(t_y) - q(o_y + s d_y, t_y).
```

Strong convexity implies:

```text
d^2/ds^2 q(o_y+s d_y,t_y) >= lambda |d_y|^2.
```

Therefore `h_y` is strictly concave, and `{s : h_y(s)>0}` is empty or exactly
one interval `[a(y),b(y)]`.

If `q` has spatial degree `d_x`, then `h_y` is a degree-`d_x` polynomial. The
single-atom optical depth is:

```text
tau_i(y) =
    alpha_i(t_y) integral_a^b h_y(s)^p ds.
```

Because `h_y^p` is polynomial, this integral is algebraically exact. Endpoints
may be exact algebraic numbers or certified isolating intervals.

For any parameter:

```text
partial_theta tau_i =
    integral_a^b partial_theta [alpha_i h_i^p] ds.
```

Endpoint terms vanish because `h_i(a)=h_i(b)=0`.

Near a generic support tangency:

```text
h(s,delta) ~= delta - c(s-s_0)^2
tau(delta) ~ C delta_+^(p + 1/2).
```

The integrated opacity is `C^p` through generic support birth. With `p>=2`,
first and second derivatives are continuous through silhouettes.

## 9. Fiber-Resolved Sensor-Time Trace Record

Instead of storing only:

```text
alpha_i(y), c_i(y), zhat_i(y), sigma_z,i(y),
```

retain the ray polynomial:

```text
h_i(y,s) = sum_{k=0}^{d_x} a_ik(y) s^k.
```

On a certified sensor-time chart `C_l`, a record may contain:

```text
R_i,l = {
    chart C_l,
    support endpoints a_i(y), b_i(y),
    fiber coefficients a_i0(y), ..., a_i,d_x(y),
    radiance c_i(y,s),
    approximation/certificate error epsilon_i,l
}.
```

Coefficient functions over sensor-time can themselves use low-degree
polynomial or Chebyshev expansions.

The compiled camera object becomes:

```text
K_C = {C_l, active-set A_l, records R_l, events E_l}_{l=1}^L.
```

There is no mandatory global total-order table. At render time:

```text
A_total(y,s) = sum_{i in active(C_l,y)} A_i(y,s).
```

Then solve the small transfer ODE along the fiber. For average active atom count
`hbar`, either integrate the near-far interval with positive-part densities or
sort only the `2 hbar` local support endpoints and integrate polynomial
segments. The local `O(hbar log hbar)` work is per-pixel materialization, not
an `O(NT)` world-to-camera ordering table.

## 10. Certified Adaptive Event Compiler

For atom `i` in a camera gauge:

```text
h_i(y,s) = 1 + mu_i(t_y) - q_i(Gamma_C(y,s)).
```

The support discriminant is:

```text
Delta_i = {
    y : exists s,
        h_i(y,s)=0 and partial_s h_i(y,s)=0
}.
```

It must be combined with:

```text
camera-chart denominator zeros
near/far-plane crossings
frustum and tile tangencies
rolling-shutter chart boundaries
```

Away from the discriminant, the two support endpoints are smooth by the
implicit-function theorem. More generally, semialgebraic families are locally
trivial away from generalized critical values.

A compiler cell is accepted only when interval certificates prove:

```text
|partial_s h_i| >= gamma_s > 0 at each endpoint
camera/projective denominators remain nonzero
the pulled camera map remains regular as required
coefficient approximation error <= requested tolerance
```

When a certificate fails:

```text
split the sensor-time chart
switch projective gauge
or mark the small event cell for live root solving
```

Parametric Krawczyk methods are a candidate for certifying root branches and
solution paths. The systems analogy is a kinetic data structure: maintain
certificates and update only when they fail, rather than replaying all
world-to-camera transformations at every requested time.

During training, atlas records can carry parameter-space trust regions.
Coefficient updates remain differentiable while topology certificates hold;
only dirty records crossing a margin require recompilation.

Conditional termination claim:

```text
If polynomial degree, camera-chart complexity, root separation, and distance
to the discriminant are bounded, adaptive compilation terminates with cost
controlled by genuine discriminant patches rather than requested frame count.
```

References from the intake:

```text
https://link.springer.com/article/10.1007/s00229-020-01227-w
https://arxiv.org/abs/2402.07053
https://www.sigmod.org/publications/dblp/db/journals/jal/jal31.html
```

## 11. Exact Transfer Differentiation And Certified Approximation

Let:

```text
U(s_1,s_0) = P exp integral_{s_0}^{s_1} A(s) ds.
```

The Duhamel formula gives:

```text
partial_theta U(s_1,s_0)
    =
    integral_{s_0}^{s_1}
        U(s_1,r) partial_theta A(r) U(r,s_0) dr.
```

This directly supports:

```text
one forward transfer sweep
one reverse adjoint sweep
reduction into canonical spacetime-atom parameters
```

Camera parameters enter through:

```text
partial_eta A =
    D A(Gamma_C) partial_eta Gamma_C
    + ray-measure/Jacobian terms.
```

Pose, rolling-shutter, and exposure gradients can therefore reuse the same
compiled fiber coefficients.

Arbitrary colored overlap will not generally have an elementary closed form,
but its error can be certified. If:

```text
||A - A_tilde||_infinity <= delta
||A||, ||A_tilde|| <= M
interval length = L,
```

then variation of constants yields a bound of the form:

```text
||U - U_tilde|| <= L delta exp(2 M L).
```

Analogous coefficient bounds control VJP error.

Claim boundary:

```text
single-atom optical depth             algebraically exact
isolated constant-color transfer      exact
arbitrary overlapping colored transfer certified numerical ODE
reverse mode                           same coefficients and support cells
```

## 12. Conditional Complexity

Let:

```text
K       world-atom representation dimension
K_C     camera-chart/program complexity
E       chart/support/discriminant event complexity
B_P     persistent atom-tile incidence count
K_tr    bounded fiber-trace record dimension
hbar    average active atoms after culling
q_f/q_r certified forward/reverse transfer integration cost
```

Then:

```text
M_scene = O(NK)

M_compiled =
    O(NK + (N+E)K_tr + B_P)
```

under bounded charts per atom.

For sorted queried times:

```text
G_temporal =
    C_compile(N,K,K_C,E,P) + O(T+E).
```

For arbitrary query order:

```text
G_temporal = C_compile + O(T log E).
```

There is no required `N x T` transformed primitive array or `N x T` global
order table.

Full forward and reverse remain:

```text
F = G_temporal + O(PT hbar q_f)

R = G_temporal + O(PT hbar q_r) + O(NK).
```

These are conditional complexity statements. They are useful only if real
camera paths keep chart count, event density, root conditioning, and persistent
tile incidence manageable.

## 13. What The Construction Claims And Does Not Claim

Proved by the mathematical construction, assuming the stated strong-convexity
and regularity hypotheses:

```text
one global finite-dimensional spacetime field
finite K independent of requested T
unique smooth derived ridge
compact, connected, smooth time slices
derived motion, curvature, orientation, and scale
at most one support interval per straight ray
algebraically exact single-atom optical depth
endpoint-free first derivatives
no primitive-level global depth order for volumetric overlap
fiber-coordinate gauge-invariant rendering
noncommutation no-go theorem for order-free colored alpha summaries
```

Certified rather than elementary closed form:

```text
algebraic minimizer branch r(t)
ray entry and exit roots
nonlinear camera-chart approximation
arbitrary colored overlap
finite exposure and rolling shutter
reverse/VJP error
```

Conjectural or empirical:

```text
degree-4 or degree-6 potentials have enough scene capacity
GPU fiber integration is cheap enough
orbit scenes keep E, chart count, and tile incidence small
incremental recompilation remains rare during aggressive training
reduced global visibility bookkeeping repays local polynomial integration
```

## 14. Unavoidable Failure Regime

No representation turns arbitrary camera and visibility data into constant
complexity.

If `T` poses are independent:

```text
K_C >= 6T
```

locally, because `SE(3)^T` has dimension `6T`.

If visibility alternates at every queried sample:

```text
E >= T - 1.
```

A smooth 360-degree orbit is different: it can have a low-dimensional camera
program. Its cost becomes large only when the scene induces genuinely many
support or occlusion events. The compiler should expose that intrinsic
complexity instead of hiding it in an `N x T` replay.

## 15. Proposed Paper-B Formulation

The attachment's preferred pair is:

```text
Self-Normalized Convex-Potential Spacetime Atoms

Gauge-Covariant Ray-Holonomy Compilation
```

with:

```text
D_i(x,t) = q_i(x,t) - min_y q_i(y,t)

sigma_i(x,t) = alpha_i(t) (1 - D_i(x,t))_+^p

H_C(y) =
    P exp integral_{F_y}
        Gamma_C^* [
            sum_i
            [ -sigma_i I    j_i ]
            [ 0             0   ]
        ] d ell.
```

Candidate contributions:

```text
1. Intrinsic kinematics from self-normalized strongly convex spacetime
   potentials.

2. Gauge-invariant ray-holonomy rendering for moving cameras.

3. Exact one-interval support and optical-depth transforms.

4. A noncommutativity theorem proving the limitation of depth-marginalized
   alpha traces under arbitrary colored overlap.

5. A certified sensor-time compiler whose world-side cost follows chart and
   discriminant complexity rather than requested sampling density.

6. Shared forward coefficients and multi-frame adjoints.
```

This is materially stronger than the determinant-atom proposal. It also
upgrades a working camera-path compiler idea rather than discarding the
existing camera-gauge foundation.

## 16. Relationship To STAR UVT Paper A

Paper A should incorporate only the parts that clarify its own claim:

```text
include:
    camera program != gauge
    ray-coordinate invariance as shared setup
    the commutator/no-go theorem as the limit of early pushforward
    an explicit approximation boundary
    WorldFoam/retained-fiber method as a principled alternative

do not absorb into Paper A:
    convex-potential atoms
    path-ordered transfer as STAR's default renderer
    retained fiber-polynomial records
    discriminant root compiler for those atoms
    transfer-ODE VJP as though it were implemented by STAR
```

Trying to absorb the second list would weaken Paper A by changing its
representation, comparator, renderer semantics, and evidence requirements. It
would turn a finishable Gaussian compiler paper into a new method without
native quality/performance evidence.

The elegant integration point is a shared diagram:

```text
world matter
    |
camera-ray pullback in certified gauge domains
    |
    +-- early fiber pushforward --> World Tubes / STAR UVT
    |       UVT trace + conditional depth/order certificates
    |
    +-- retained fiber transfer --> WorldFoam / ray holonomy
            optical generator + ordered transfer scan
```

## 17. Critical Risks And Backtracks

### Branch A: Convex-potential atoms are the right Paper-B primitive

Hypothesis:
    Strong convexity gives an unusually clean topology, ridge, support, and
    differentiation contract.

What would make it false:
    Degree-4/6 atoms need too many components, cannot express thin or
    multi-lobed matter economically, or train much worse than Gaussian/cell
    baselines.

Cheap test:
    Fit one moving anisotropic synthetic object and one two-object occlusion
    clip at matched parameter counts; compare support error, heldout rays,
    conditioning, and optimizer stability.

If supported:
    Promote the atom family into Paper B.

If invalidated:
    Keep ray-holonomy compilation over existing bounded cells or Gaussian
    densities; the renderer theorem survives.

### Branch B: The holonomy formulation is explanatory, not computational

Hypothesis:
    The path-ordered exponential gives the right theorem but the event monoid
    remains the only useful implementation.

What would make it false:
    A low-degree segment ODE kernel beats the owner-run/event scan at matched
    image error and memory.

Cheap test:
    Compare constant owner-runs, adaptive quadrature, and low-degree
    fiber-polynomial segments on the existing cell-path fixture.

If supported:
    Use holonomy in the paper theorem and keep discrete event elements in code.

If invalidated:
    Promote a native polynomial transfer kernel.

### Branch C: STAR's certificates already make retained depth unnecessary in
the practical regime

Hypothesis:
    Real Gaussian scenes rarely incur enough high-opacity, high-color-contrast
    overlaps for retained-fiber rendering to repay its cost.

What would make it false:
    Measured commutator energy or fallback rate predicts a material heldout
    quality gap that WorldFoam closes.

Cheap test:
    Log `T_before alpha_i alpha_j ||c_i-c_j||`, order crossings, and fallback
    pixels on the same public rows used for the paper matrix.

If supported:
    Keep WorldFoam as theory/prototype and finish STAR first.

If invalidated:
    Reopen native WorldFoam integration on the failing rows.

### Branch D: Certified compilation does not amortize during training

Hypothesis:
    Parameter trust regions keep most topology records valid for many steps.

What would make it false:
    Aggressive updates dirty a large fraction of records every step, making
    root certification more expensive than replay.

Cheap test:
    Instrument dirty-record fraction, certificate margin quantiles, and
    recompilation cost on a tiny trained scene.

If supported:
    Develop incremental compilation.

If invalidated:
    Compile only for inference/frozen-camera export; use live reference
    rendering during training.

## 18. Minimum Falsification Ladder

Do not begin with a full renderer rewrite.

```text
Gate 1 — terminology and algebra
    verify gauge reparameterization with physical ray-length Jacobian
    verify two-layer commutator and ordinary alpha equivalence

Gate 2 — primitive capacity
    convex-potential atom vs capped determinant vs Gaussian/cell baseline
    matched parameter count on synthetic moving support

Gate 3 — root/support contract
    one-interval support
    certified endpoints
    tangency smoothness
    finite-difference normalization and endpoint-free gradients

Gate 4 — transfer implementation
    event monoid vs polynomial-segment ODE
    matched error, memory, forward, and backward

Gate 5 — camera-path compiler
    orbit plus rolling shutter
    chart/discriminant count
    dirty-record fraction
    break-even frame count

Gate 6 — decisive overlap row
    colored volumes whose depth relation changes under camera motion
    compare STAR certificates/fallback with retained-fiber reference

Gate 7 — paper breadth
    only after prior gates: public multi-scene heldout quality and native
    performance
```

## 19. Current Confidence

High:

```text
the ray-coordinate gauge-invariance statement
the noncommutativity/no-go theorem
the separation between camera program and gauge
the distinction between STAR early pushforward and WorldFoam retained transfer
the value of the holonomy formulation as Paper-B mathematics
```

Medium:

```text
self-normalized convex potentials are a competitive primitive family
the discriminant compiler can remain sublinear in requested frame sampling
the phrase "gauge-invariant ray-holonomy renderer" will improve Paper-B framing
```

Low until measured:

```text
native polynomial transfer beats the discrete event scan
incremental certification stays cheap during training
WorldFoam closes a real public heldout gap against World Tubes
the added mathematical elegance yields a stronger systems result
```

## Final Recommendation

Finish the current World Tubes / STAR UVT evidence chain without importing the
new primitive or retained-transfer implementation.

In parallel only at the documentation level, preserve this as the strengthened
WorldFoam Paper-B thesis:

```text
WorldFoam is a gauge-invariant ray-holonomy renderer for moving cameras:
it compiles intrinsic spacetime matter through certified camera gauges into
fiber-resolved optical transfer, then evaluates a path-ordered visibility
operator with shared adjoints.
```

The first future code gate should not be another broad shader fork. It should
be the smallest matched experiment that can falsify one of two load-bearing
claims:

```text
convex-potential atoms buy useful capacity and conditioning
or
retained ray transfer fixes a measured STAR overlap/fallback failure.
```

