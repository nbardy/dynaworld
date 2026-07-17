# WorldFoam Math Appendix

Date: 2026-07-06

Status: polished appendix scaffold for the WorldFoam paper lane. This is a
paper-math document, not an implementation proof or a public-quality claim.

Source:

```text
/Users/nicholasbardy/.codex/attachments/14488d44-0430-4b9c-ade0-45f1077c1a93/pasted-text.txt
```

Related files:

```text
research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md
research_notes/worldfoam_paper/WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
```

## Appendix Claim

The strongest appendix identity is:

```text
WorldFoam is gauge-covariant optical transfer factored through a compiled
cell-path atlas.
```

This replaces the weaker summary:

```text
WorldFoam is sigma(u,v,t,z), then tau, then T, then I.
```

The weaker summary is physically correct, but the stronger one names the
objects that make the renderer useful:

```text
ray-fiber optical density
visibility monoid
optical transfer matrix
cell-path/event word
same-representation replay theorem
prefix/suffix VJP
commutator visibility criterion
```

Claim boundary:

```text
This appendix supports a theory/prototype renderer paper. It does not prove
public dynamic-NVS quality, official CUDA/Warp parity, or full topology-
differentiable training.
```

## A. Notation

Sensor-time base:

```text
B = Omega x T
y = (u,v,t)
```

Camera-ray bundle:

```text
pi: E_Gamma -> B
pi^{-1}(y) = F_y
```

Camera map:

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

`z` is a ray-fiber coordinate. It may be ordinary depth, inverse depth, log
depth, projective depth, or another valid local coordinate. The Jacobian
`J_l` is part of the gauge contract.

## B. Optical Matter And Ray-Fiber Pullback

Represent world matter by optical measures:

```text
W_theta = (mu_theta, nu_theta)
```

where:

```text
mu_theta >= 0   optical-depth / extinction measure
nu_theta        color-weighted optical measure
c = dnu / dmu   color where the derivative exists
```

For ordinary bounded foam cells:

```text
dmu(x) = sum_j 1_{x in F_j} sigma_j(x) dx
dnu(x) = sum_j 1_{x in F_j} sigma_j(x) c_j(x) dx
```

This notation also includes:

```text
foam density       absolutely continuous optical depth
surface sheets     surface-supported optical depth
splats             atomic optical-depth events
```

Pull the measures through the camera gauge. Define ray-fiber extinction and
color-weighted density by:

```text
lambda_l(y,z) dz = Gamma_l(y,.)^* dmu
eta_l(y,z) dz    = Gamma_l(y,.)^* dnu
```

For absolutely continuous bounded cells:

```text
lambda_l(y,z)
    =
    sum_j 1_{Gamma_l(y,z) in F_j}
          sigma_j(Gamma_l(y,z))
          J_l(y,z)

eta_l(y,z)
    =
    sum_j 1_{Gamma_l(y,z) in F_j}
          sigma_j(Gamma_l(y,z))
          c_j(Gamma_l(y,z))
          J_l(y,z)
```

Use `sigma` for world-local density and `lambda` for pulled ray-fiber
extinction. The rasterizer should be written in `lambda` and `eta`.

## C. Gauge Invariance

Let:

```text
z' = phi_y(z)
partial z' / partial z > 0
```

Optical depth is a measure along the ray:

```text
lambda'(y,z') dz' = lambda(y,z) dz
eta'(y,z') dz'    = eta(y,z) dz
```

Therefore:

```text
int_{z_a}^{z_b} lambda(y,z) dz
    =
int_{z'_a}^{z'_b} lambda'(y,z') dz'

int_{z_a}^{z_b} eta(y,z) dz
    =
int_{z'_a}^{z'_b} eta'(y,z') dz'
```

So the pulled optical one-form is invariant:

```text
lambda(y,z) dz = lambda'(y,z') dz'
```

Interpretation:

```text
ordinary depth, log depth, inverse depth, orbit depth, and projective depth
are gauge choices, not different renderers.
```

Diagnostic:

```text
ordinary-depth and log-depth pullbacks must match with the Jacobian and diverge
without it.
```

## D. Visibility Monoid

Define an optical transfer element:

```text
g = (beta, m)
```

where:

```text
beta in [0,1]     residual transmittance
m in R^C          premultiplied visible color against black
```

Composition for a front element followed by a rear element:

```text
g1 otimes g2
    =
    (beta1 beta2, m1 + beta1 m2)
```

Identity:

```text
e = (1,0)
```

Decode against background:

```text
decode((beta,m), I_bg) = m + beta I_bg
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

Thus:

```text
([0,1] x R^C, otimes)
```

is a visibility monoid. A ray is a word evaluated in this monoid.

## E. Optical Transfer Matrix And Product Integral

Represent:

```text
g = (beta,m)
```

as an affine matrix:

```text
M(g) =
    [ beta I_C   m ]
    [ 0          1 ]
```

Then:

```text
M(g1) M(g2) = M(g1 otimes g2)
```

For infinitesimal ray slice `dz`:

```text
beta(z,dz) = exp(-lambda(z) dz)
m(z,dz)    = (1 - exp(-lambda(z) dz)) c(z)
           ~= eta(z) dz
```

The infinitesimal transfer generator is:

```text
A_y(z) =
    [ -lambda_y(z) I_C   eta_y(z) ]
    [ 0                  0        ]
```

and the gauge-invariant one-form is:

```text
A_y(z) dz
```

Ray transfer:

```text
M_y = P exp int_{Z_y} A_y(z) dz
```

Rendering:

```text
[ I(y) ]     [ M_y ] [ I_bg(y) ]
[ 1    ]  =          [ 1       ]
```

Equivalent scalar form:

```text
tau_y(z) = int_{z_front}^{z} lambda_y(s) ds
T_y(z)   = exp(-tau_y(z))
I(y)     = int T_y(z) eta_y(z) dz + T_back I_bg(y)
```

The product-integral form is preferred in the appendix because it exposes
algebra, scan structure, VJP, compression, and noncommutativity.

## F. Splats As Optical Atoms

A splat with opacity `alpha_i(y)`, color `c_i(y)`, and depth `z_i(y)`
corresponds to an optical-depth atom:

```text
DeltaTau_i(y) = -log(1 - alpha_i(y))
```

Then:

```text
beta_i = 1 - alpha_i = exp(-DeltaTau_i)
m_i    = alpha_i c_i
g_i    = (1 - alpha_i, alpha_i c_i)
```

Sorted alpha compositing is:

```text
G(y) = otimes_{i in depth order} (1 - alpha_i(y), alpha_i(y)c_i(y))
```

Safe claim:

```text
3DGS alpha compositing is the atomic optical-measure case of WorldFoam
transfer.
```

Unsafe claim:

```text
WorldFoam is automatically better than splatting as a trainable renderer.
```

Equation-level containment is not quality, speed, memory, or stability
dominance.

## G. Cell-Path Rasterization

For bounded power cells:

```text
pow_i(x) = ||x - p_i||^2 - r_i^2
```

Cell `i` owns a point when it is inside support and wins the local power
comparison:

```text
B_i = { x : ||x - p_i|| <= r_i and pow_i(x) <= pow_j(x) for neighbors j }
```

A ray through the cell complex induces a front-to-back cell word:

```text
w_y = (i_1, ell_1, i_2, ell_2, ..., i_R, ell_R)
```

where `i_r` is the owner cell and `ell_r` is the physical ray length inside
that owner run.

For constant density and color in cell `i`:

```text
g_i(ell)
    =
    ( exp(-sigma_i ell),
      (1 - exp(-sigma_i ell)) c_i )
```

Then:

```text
G(y) = otimes_{r=1}^{R(y)} g_{i_r}(ell_r(y))
```

Key statement:

```text
A camera ray induces a word in the cell complex; WorldFoam evaluates that word
in the visibility monoid.
```

This is the implementation-facing form of owner-run cutwalk.

## H. Compiled Atlas Definition

A polished WorldFoam atlas can be written:

```text
K_Gamma = { C_l, w_l, Phi_l, S_l, P_l, E_l }_{l=1}^{L}
```

where:

```text
C_l       gauge/event domain in sensor-time base B
w_l(y)   certified cell word or event word
Phi_l    maps each run/event to a visibility-monoid element
S_l       support/topology/error certificates
P_l       prefix/suffix or scan metadata
E_l       fallback metadata
```

Evaluation on accepted domain `C_l`:

```text
G(y) = otimes_{r in w_l(y)} Phi_l(r,y)
I(y) = m(y) + beta(y) I_bg(y), where G(y) = (beta(y),m(y))
```

This is the paper-ready renderer definition:

```text
gauge-covariant optical transfer factored through a compiled cell-path atlas.
```

## I. Same-Representation Replay Theorem

Let per-frame replay compute the exact cell word:

```text
w_y = (i_1, ell_1, ..., i_R, ell_R)
```

Let the compiled atlas return the same word and lengths on a certified domain:

```text
w_hat_y = w_y
ell_hat_r(y) = ell_r(y)
```

Replay transfer:

```text
G_replay(y) = otimes_r g_{i_r}(ell_r(y))
```

Compiled transfer:

```text
G_compiled(y) = otimes_r g_{i_r}(ell_hat_r(y))
```

If the word and lengths match:

```text
G_compiled(y) = G_replay(y)
I_compiled(y) = I_replay(y)
```

Approximate version:

```text
|I_compiled - I_replay|
    <= C sum_r epsilon_r
     + epsilon_support
     + epsilon_fallback
```

This is the first proof baseline. External STAR/GS rows position the result,
but same-representation replay proves the compiler.

## J. Monoid VJP And Prefix/Suffix Form

For:

```text
h = a otimes b
beta_h = beta_a beta_b
m_h = m_a + beta_a m_b
```

reverse-mode gives:

```text
bar m_a    += bar m_h
bar m_b    += beta_a bar m_h
bar beta_a += beta_b bar beta_h + dot(bar m_h, m_b)
bar beta_b += beta_a bar beta_h
```

For the final decode:

```text
I = m + beta I_bg
bar m = bar I
bar beta = dot(bar I, I_bg)
```

For a segment `k`:

```text
G(y) = P_k^- otimes g_k otimes S_k^+
P_k^- = (B_k^-, M_k^-)
S_k^+ = (B_k^+, M_k^+)
I_k^+ = M_k^+ + B_k^+ I_bg
```

A perturbation gives:

```text
delta I = B_k^- (delta m_k + delta beta_k I_k^+)
```

Therefore:

```text
bar m_k    = B_k^- bar I
bar beta_k = B_k^- dot(bar I, I_k^+)
```

This is the discrete form of front transmittance and behind radiance.

## K. Constant-Density Segment Derivatives

For:

```text
tau = sigma ell
beta = exp(-tau)
m = (1 - beta)c
```

first derivatives:

```text
d beta / d tau = -beta
d m / d tau    = beta c
d m / d c      = (1 - beta) I_C
```

adjoint:

```text
bar tau = beta (dot(bar m, c) - bar beta)
```

Substitute prefix/suffix adjoints:

```text
bar tau = B^- beta dot(bar I, c - I^+)
```

For physical parameters:

```text
bar sigma = ell   bar tau
bar ell   = sigma bar tau
bar c     = (1 - beta) bar m
```

This is the first finite-difference target for owner-run cell-path VJP.

## L. Commutator Visibility Theorem

For optical generators:

```text
A_i =
    [ -lambda_i I_C   lambda_i c_i ]
    [ 0               0            ]
```

the commutator has color column:

```text
[A_1,A_2] color-column = lambda_1 lambda_2 (c_1 - c_2)
```

Interpretation:

```text
visibility noncommutativity = opacity overlap x color contrast
```

Consequences:

```text
c_1 = c_2                      -> order does not matter
lambda_1 = 0 or lambda_2 = 0    -> order does not matter
large opacity and color contrast -> order matters
```

For discrete alpha elements:

```text
Delta m = alpha_i alpha_j (c_i - c_j)
|Delta I| <= T_before alpha_i alpha_j ||c_i - c_j||
```

This recovers the World Tubes swap-bound intuition as a finite optical-transfer
commutator.

## M. Compression And Optical-Depth Basis Branches

These are promising but should remain behind tests.

First-order interval collapse:

```text
Lambda = int_a^b lambda(z) dz
Q      = int_a^b lambda(z)c(z) dz
beta   = exp(-Lambda)
c_bar  = Q / Lambda
m      = (1 - exp(-Lambda)) c_bar
```

Second-order commutator moment:

```text
K = 1/2 int_a^b int_a^{z1}
        lambda(z1) lambda(z2) (c(z1) - c(z2)) dz2 dz1
```

Commutator-energy split rule:

```text
C([a,b]) < epsilon   -> collapse interval
C([a,b]) >= epsilon  -> split interval
```

This is intellectually strong, but it must beat simple adaptive transmittance
or replay-error splitting at equal memory before becoming mainline.

Optical-depth basis:

```text
s = tau(z) - tau(a)
ds = lambda(z) dz
m = int_0^{DeltaTau} exp(-s) c(s) ds
```

If:

```text
c(s) = sum_{n=0}^{p} c_n s^n
```

then:

```text
m = sum_n c_n M_n(DeltaTau)
M_n(a) = int_0^a exp(-s) s^n ds
```

For integer `n`:

```text
M_n(a) = n! (1 - exp(-a) sum_{q=0}^{n} a^q / q!)
```

This is a plausible compact segment basis, but not a blocker for first
cell-path tests.

## N. Geometry Branches Behind Finite-Difference Gates

### N1. Interface flux

For an interface at depth `s`, moving it by `ds` replaces material `B` with
material `A`. The image derivative is:

```text
dI/ds =
    T(s) [
      lambda_A (c_A - I_>(s))
      - lambda_B (c_B - I_>(s))
    ]
```

This is elegant and useful, but it must pass finite differences before paper
promotion.

Required gates:

```text
moving plane face
moving support-sphere endpoint
moving center/radius in a bounded power-cell pair
near-parallel face denominator fallback
fixed-topology guard
```

### N2. Radical-face crossing

For face `(i|j)`:

```text
n_ij = p_j - p_i
h_ij = 0.5 (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)
s_ij = (h_ij - dot(o,n_ij)) / dot(d,n_ij)
```

The denominator `dot(d,n_ij)` is the tangency hazard. Near zero, topology
should split or fallback rather than trusting geometry gradients.

### N3. Sphere support endpoint

Endpoint equation:

```text
||o + s d - p_i||^2 - r_i^2 = 0
```

Implicit derivative:

```text
ds = - d_theta F / F_s
```

This is the entry/exit support-boundary analogue of radical-face crossing.

### N4. Interface-flux witness score

Define face flux:

```text
Phi_ij(y)
    =
    dot(bar I(y),
        T_y(s_ij)
        [lambda_i(c_i - I_>) - lambda_j(c_j - I_>)])
```

Then:

```text
W_ij = int_{B_ij} |Phi_ij(y)| dy
```

Interpretation:

```text
the face is witnessed if moving it would change the training objective.
```

This is a stronger topology diagnostic than "ray hit the face." It should be
tested against heldout-free validation, source leave-one-camera-out residuals,
topology churn, and traversal instability.

## O. Speculative Branches To Keep Out Of The First Claim

### O1. Segment Hessian

The discrete statement:

```text
d^2 I / d tau_i d tau_j = -D_max(i,j)
```

and the Gauss-Newton approximation:

```text
H_ij^GN = dot(D_i, D_j)
```

are elegant and may help preconditioning. They should not appear as a paper
claim until checked numerically.

### O2. Gauge-covariant feature transfer

If the optical state changes local feature basis:

```text
s = H(z) s_tilde
```

then generator transforms like:

```text
A_tilde = H^{-1} A H - H^{-1} partial_z H
```

This is real gauge-connection math, but it is only relevant once WorldFoam uses
learned feature bases, spherical harmonics, latent radiance charts, or
cell-local color frames.

### O3. Universal ray-space transfer

A universal ray-space optical-transfer field is beautiful:

```text
I_Gamma(y) = I_R(kappa_Gamma(y))
```

where a camera program maps sensor samples into ray space. This is long-horizon
and should not lead the first paper.

## P. Minimum Validation Suite

Before promoting beyond theory/prototype:

```text
1. Gauge Jacobian:
   ordinary-depth and log-depth pullbacks match with Jacobian and diverge
   without it.

2. Alpha equivalence:
   sorted alpha compositing equals monoid scan of atomic elements.

3. Cell-path replay:
   per-frame owner-run word and compiled atlas word produce identical images
   under fixed certified intervals.

4. Cell-path VJP:
   finite-difference beta, m, DeltaTau, sigma, color, and run length.

5. Commutator prediction:
   swap/order error follows opacity overlap times color contrast.

6. Interface flux:
   finite-difference moving face, sphere endpoint, center, and radius.

7. Witness score:
   test whether flux-weighted witnessed faces predict heldout-free residuals,
   topology churn, or source leave-one-camera-out failure.

8. Compression bakeoff:
   commutator-energy splitting versus simple adaptive splitting at equal
   error and memory.
```

If these fail, the math remains a useful explanatory appendix, but WorldFoam
stays a side-paper theory/prototype lane rather than the main arXiv push.
