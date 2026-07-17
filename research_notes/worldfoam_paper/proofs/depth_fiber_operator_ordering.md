# Depth Fiber And Operator Ordering Proof Notes

Date: 2026-07-05

Status: curated proof scaffold for the WorldFoam paper lane.

This is the cleaned theorem set behind the current WorldFoam framing. It
preserves the shared camera-ray bundle math with World Tubes, but states the
operator-ordering difference that makes WorldFoam a separate paper.

## Definitions

Sensor-time base:

```text
B = Omega x T
y = (u,v,t)
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

Local gauge:

```text
chi_l: E_Gamma | C_l -> C_l x Z_l
chi_l(e) = (y,z)
Gamma_l(y,z) = Gamma(chi_l^{-1}(y,z))
dmu_y(e) = J_l(y,z) dz
```

`z` is the ray-depth/fiber coordinate. `J_l` is the induced fiber-measure
Jacobian.

## Theorem 1: Gauge-Invariant Ray-Fiber Trace

Claim:

```text
pi_* Gamma^* w
```

is independent of the chosen smooth fiber coordinate when the induced fiber
measure is transformed correctly.

Proof sketch:

In gauge `a`:

```text
Trace^a(y) =
    integral rho(Gamma_a(y,z_a)) J_a(y,z_a) dz_a
```

In gauge `b`, let:

```text
z_b = phi_y(z_a)
```

Then:

```text
Gamma_b(y,z_b) = Gamma_a(y,z_a)
J_b(y,z_b) |partial z_b / partial z_a| = J_a(y,z_a)
```

Therefore:

```text
Trace^b(y)
= integral rho(Gamma_b(y,z_b)) J_b(y,z_b) dz_b
= integral rho(Gamma_a(y,z_a)) J_a(y,z_a) dz_a
= Trace^a(y)
```

Diagnostic:

```text
ordinary-depth and log-depth integration should match only when the Jacobian is
included; the missing-Jacobian control should fail.
```

## Proposition 2: World Tubes Gaussian Fiber Marginal

This is included here only as contrast.

For a spacetime Gaussian:

```text
rho_i(x) = a_i exp[-1/2 (x - m_i)^T Lambda_i (x - m_i)]
```

under local affine gauge:

```text
Gamma_l(y,z) ~= x0 + J eta
eta = [delta_y, delta_z]^T
```

define:

```text
H = J^T Lambda_i J

H = [H_yy H_yz
     H_zy H_zz]
```

Then marginalizing `z` yields UVT precision:

```text
S = H_yy - H_yz H_zz^{-1} H_zy
```

and conditional depth:

```text
z_hat_i(y) = z0 + H_zz^{-1}(g_z - H_zy delta_y)
Var(z | y) = H_zz^{-1}
```

Interpretation:

```text
World Tubes gets a compact UVT footprint because Gaussian atoms are closed
under marginalization.
```

WorldFoam should not pretend this is its main mechanism.

## Proposition 3: Visibility Does Not Commute With Depth Marginalization

Claim:

```text
render_after_marginalizing_z
    !=
marginalize_after_rendering_along_z
```

in general.

Counterexample:

Two translucent layers with opacity/color `(alpha_1,c_1)` and `(alpha_2,c_2)`.

Order 1 then 2:

```text
I_12 = alpha_1 c_1 + (1 - alpha_1) alpha_2 c_2
```

Order 2 then 1:

```text
I_21 = alpha_2 c_2 + (1 - alpha_2) alpha_1 c_1
```

Difference:

```text
I_12 - I_21 = alpha_1 alpha_2 (c_1 - c_2)
```

Therefore, a UVT-only marginal that stores total opacity and premultiplied
color cannot determine every translucent front-to-back rendering.

Implication:

```text
World Tubes:
    early pushforward is fast, but visibility needs conditional depth/order
    certificates and fallbacks.

WorldFoam:
    keep sigma(y,z) and perform visibility as transmittance along the fiber.
```

## Definition 4: Lifted WorldFoam Opacity

For bounded world cells `F_j` with density `sigma_j` and color `c_j`:

```text
rho_{j,l}(y,z)
    =
    1_{Gamma_l(y,z) in F_j}
    sigma_j(Gamma_l(y,z))
    J_l(y,z)
```

Aggregate lifted opacity:

```text
sigma_l(y,z) = sum_j rho_{j,l}(y,z)
```

Color numerator:

```text
q_l(y,z) = sum_j rho_{j,l}(y,z) c_j(Gamma_l(y,z))
```

Local color, where `sigma_l > 0`:

```text
c_l(y,z) = q_l(y,z) / sigma_l(y,z)
```

This retained-fiber object is the WorldFoam state. It should be represented
sparsely as events, intervals, bases, and prefix summaries, not as a dense
UVTZ grid.

## Proposition 5: Transmittance-Prefix Rendering

Given lifted opacity and color:

```text
tau_l(y,z) = integral_{z_front}^{z} sigma_l(y,s) ds
T_l(y,z) = exp(-tau_l(y,z))
I(y) = integral T_l(y,z) sigma_l(y,z) c_l(y,z) dz
```

Alpha:

```text
alpha(y) = 1 - exp(- integral sigma_l(y,z) dz)
```

Interpretation:

```text
WorldFoam replaces discrete primitive order with cumulative optical depth on
the camera-ray fiber.
```

Boundary:

```text
Beer-Lambert is classical. The research contribution is the sparse,
camera-compiled bounded-cell event atlas and its forward/backward reuse over a
known camera program.
```

## Proposition 5.5: Visibility Monoid And Optical Transfer Elements

Define an optical transfer element:

```text
g = (beta, m)
```

where:

```text
beta = residual transmittance
m    = visible color contribution against black background
```

Composition is front-over:

```text
(beta_1,m_1) otimes (beta_2,m_2)
    =
    (beta_1 beta_2, m_1 + beta_1 m_2)
```

Identity:

```text
e = (1,0)
```

Decode:

```text
decode((beta,m), B) = m + beta B
```

Associativity:

```text
(a otimes b) otimes c = a otimes (b otimes c)
```

because both sides equal:

```text
(beta_a beta_b beta_c,
 m_a + beta_a m_b + beta_a beta_b m_c)
```

Continuous foam is the product integral of infinitesimal elements. A
camera-compiled WorldFoam atlas approximates that product by certified event
elements:

```text
G(y) = otimes_k (beta_k(y), m_k(y))
I(y) = decode(G(y), I_bg(y))
```

For a standard alpha splat:

```text
g_i(y) = (1 - alpha_i(y), alpha_i(y) c_i(y))
```

so sorted alpha compositing is the atomic-measure case of optical transfer.

Safe claim:

```text
WorldFoam generalizes splat compositing at the ray-transfer equation level.
```

Unsafe claim:

```text
WorldFoam is automatically better than splatting as a trainable renderer.
```

## Proposition 5.6: Optical-Transfer Commutator Criterion

For continuous generators:

```text
A_i =
    [ -lambda_i I_C   eta_i ]
    [ 0               0     ]

eta_i = lambda_i c_i
```

the nonzero commutator term is:

```text
[A_1,A_2] color-column
    =
    lambda_1 lambda_2 (c_1 - c_2)
```

For discrete alpha elements:

```text
g_i = (1 - alpha_i, alpha_i c_i)
g_j = (1 - alpha_j, alpha_j c_j)
```

the order-swap color difference is:

```text
Delta m = alpha_i alpha_j (c_i - c_j)
```

Interpretation:

```text
visibility/order error is opacity overlap times color contrast.
```

This should drive interval splitting and compression diagnostics before any
arbitrary depth-layer-count heuristic.

## Proposition 6: Prefix/Suffix VJP

For:

```text
I(y) = integral T(z) sigma(z) c(z) dz
T(z) = exp(- integral_{front}^{z} sigma(s) ds)
```

the first variation is:

```text
delta I(y)
    =
    integral T(z) sigma(z) delta c(z) dz
    +
    integral T(z) [c(z) - I_behind(y,z)] delta sigma(z) dz
```

where:

```text
I_behind(y,z) = radiance accumulated behind depth z
```

Implementation implication:

```text
backward needs front transmittance prefix and behind-radiance suffix, or a
compact way to recompute them.
```

## Proposition 7: Power-Cell Ray Event Structure

Power distance:

```text
pow_i(x) = ||x - p_i||^2 - r_i^2
```

Bounded cell:

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

These endpoints define sparse event records for the foam atlas.

## Theorem 8: Same-Representation Replay Equivalence

Claim:

```text
On certified domains, compiled WorldFoam evaluation equals per-frame WorldFoam
replay up to declared basis, quadrature, support, and fallback error.
```

Proof sketch:

Per-frame replay and compiled evaluation use the same lifted functions:

```text
sigma_l(y,z)
q_l(y,z)
tau_l(y,z)
T_l(y,z)
I(y)
```

Compilation changes when structural work is performed, not which mathematical
function is evaluated. If the atlas stores exact active intervals, exact basis
coefficients, exact prefix scans, and exact fallback decisions, equality holds.
Approximation error is introduced only by the declared basis/quadrature/support
approximations.

This is the first baseline to beat before comparing against STAR/GS quality.

## Theorem 9: Fixed-Topology VJP Boundary

Within a certified atlas domain, hold fixed:

```text
cell topology
active intervals
event partition
depth-layer partition
fallback mask
```

Then the direct VJP is exact for the compiled renderer with respect to
continuous parameters inside that fixed topology.

Boundary case:

```text
I(theta) = integral_{a(theta)}^{b(theta)} f(z,theta) dz
```

requires Leibniz terms:

```text
dI/dtheta
    =
    integral_a^b partial_theta f(z,theta) dz
    + f(b,theta) b'(theta)
    - f(a,theta) a'(theta)
```

Therefore, if endpoint/topology motion is not differentiated, the claim must be:

```text
fixed-topology exactness, with topology handled by refresh, split, smoothing,
fallback, or future boundary calculus.
```

## Proposition 10: Event-Complexity Scaling

Let:

```text
F = number of output frames or temporal samples
R = average per-frame structural replay work
E(F) = number of certified camera-path event records
```

Per-frame replay:

```text
W_replay(F) = O(F R)
```

Compiled structural payload:

```text
W_compiled(F) = O(E(F) + basis_payload)
```

When:

```text
E(F) = o(F R)
```

the structural part of rendering/training grows sublinearly relative to
per-frame replay.

Honest wording:

```text
Pixel/sample evaluation remains linear. Sublinear frame scaling refers to the
world-side structural work and measured regimes where that work dominates.
```

## Error Decomposition

For approximate WorldFoam:

```text
|I - I_hat|
    <= E_tau
     + E_radiance_basis
     + E_support
     + E_quadrature
     + E_fallback
```

If:

```text
|tau_hat(y,z) - tau(y,z)| <= epsilon_tau
```

then since `T = exp(-tau)` is 1-Lipschitz for nonnegative optical depth in the
relevant direction:

```text
|T_hat - T| <= epsilon_tau
```

This makes optical-depth residual a natural refinement signal.

## Falsification Tests

1. Gauge Jacobian test:

```text
ordinary-depth and log-depth WorldFoam pullback match with Jacobian, fail
without it.
```

2. Non-commutation test:

```text
two translucent slabs with matched UVT marginals but reversed order produce
different images by alpha_1 alpha_2 (c_1 - c_2).
```

3. Replay equivalence test:

```text
compiled WorldFoam atlas equals per-frame WorldFoam replay on fixed certified
intervals.
```

4. Prefix/suffix VJP test:

```text
finite-difference density/color coefficients against direct VJP on a fixed
topology scene.
```

5. Alpha-equivalence and commutator tests:

```text
sorted alpha compositing equals the monoid scan of atomic elements; measured
swap error follows alpha_i alpha_j (c_i - c_j).
```

6. Monoid VJP test:

```text
finite-difference beta, m, DeltaTau, color, sigma, and owner-run length against
the direct monoid-scan VJP.
```

7. Event-density death curve:

```text
sweep camera orbit speed, near-camera support, cell density, and opacity depth;
plot E(F)/(F R), memory, fallback rate, and speedup.
```

## Math Appendix Extensions Added 2026-07-06

The polished math appendix lives in:

```text
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
```

The appendix does not replace the proof scaffold above. It promotes the
implementation-facing version of the same algebra and adds several branches
that must stay behind tests.

Ready to promote into the paper:

```text
1. Cell-path word rasterization:
   a camera ray induces a certified front-to-back cell/event word w_y, and
   WorldFoam evaluates w_y in the visibility monoid.

2. Compiled atlas definition:
   K_Gamma = {C_l, w_l, Phi_l, S_l, P_l, E_l}, where Phi_l maps each run/event
   to a monoid element and S/P/E hold certificates, scan metadata, and fallback.

3. Cell-path replay theorem:
   if compiled and per-frame replay emit the same word and run lengths, their
   monoid products and rendered images match.

4. Owner-run VJP:
   for tau = sigma ell, beta = exp(-tau), m = (1 - beta)c, the direct VJP gives
   bar tau = B^- beta dot(bar I, c - I^+), then
   bar sigma = ell bar tau and bar ell = sigma bar tau.
```

Branches to keep behind finite-difference or correlation tests:

```text
1. Segment Hessian / second-order optical-depth structure:
   elegant and potentially useful for preconditioning, but not paper-ready
   until numerical Hessian checks pass.

2. Interface flux:
   dI/ds = T [lambda_A(c_A - I_>) - lambda_B(c_B - I_>)] must pass moving-face,
   center/radius, and sphere-endpoint finite differences.

3. Flux witness score:
   W_ij = int |Phi_ij(y)| dy should be tested against heldout-free residual,
   source leave-one-camera-out error, topology churn, and traversal instability.

4. Gauge-covariant feature transfer:
   A_tilde = H^{-1} A H - H^{-1} partial_z H is real connection math, but only
   relevant once the renderer uses changing feature bases.

5. Universal ray-space transfer:
   useful north-star construction, not a first-paper claim.
```

Immediate proof tests:

```text
cell-path replay equivalence
cell-path VJP finite differences for beta/m/DeltaTau/sigma/color/run length
commutator prediction on crossing translucent slabs
interface flux finite differences after monoid/cell-path VJP passes
flux witness correlation before using witness metrics as regularizers
```
