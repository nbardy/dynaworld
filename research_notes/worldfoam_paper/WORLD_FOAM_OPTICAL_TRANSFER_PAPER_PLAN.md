# WorldFoam Optical-Transfer Paper Plan

Date: 2026-07-05

Source note:

```text
research_notes/world_foam_reformulation.md
```

Triage note:

```text
research_notes/worldfoam_paper/scientist_notes/2026-07-05_optical_transfer_reformulation_intake.md
```

Polished appendix:

```text
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
```

Purpose: lift the genuinely useful parts of the scientist reformulation into a
paper plan, while quarantining the elegant-but-unproven parts. This should be
the route for the WorldFoam paper until experiments prove otherwise.

## Verdict

The reformulation is genuinely interesting, but uneven.

Promote now:

```text
visibility monoid
optical transfer event element
transfer matrix / product integral
commutator visibility theorem
monoid scan VJP
owner-run event rasterization
cell-path atlas / certified event word
event closure vs Schur closure
```

Develop behind tests:

```text
interface flux adjoint
power-face and sphere endpoint derivatives
optical-depth coordinate basis
Magnus / commutator compression
```

Do not promote as solved:

```text
public quality parity
full topology-differentiable training
Magnus Foam as the main implementation
"strictly generalizes splatting" beyond the ray-equation level
```

The best new identity is:

```text
WorldFoam is a camera-compiled optical-transfer algebra over ray fibers.
```

This is stronger than:

```text
WorldFoam is a lifted sigma(u,v,t,z) transmittance field.
```

The older statement is physically correct. The new statement is
renderer-complete: it names the algebra that makes scans, compression,
commutation, and backward passes explicit.

## Revised Paper Thesis

Current thesis:

```text
WorldFoam keeps the depth fiber alive and renders by transmittance prefixes.
```

Better thesis:

```text
WorldFoam compiles bounded world matter through a known camera program into a
depth-ordered sequence of optical transfer elements. Rendering is an
associative visibility-monoid scan over the ray fiber; splat alpha compositing
is the atomic case, continuous foam is the dense case, and event intervals are
the sparse camera-compiled case.
```

Safe claim boundary:

```text
The paper is a camera-program compiler / renderer paper. It is not yet a claim
that foam beats mature Gaussian splatting on public dynamic novel-view quality.
```

## Core Objects To Introduce

### 1. Camera-Gauged Optical Matter

World matter is represented by optical measures:

```text
mu  = optical-depth measure
nu  = color-weighted optical measure
c   = dnu / dmu where defined
```

This is a useful unifying notation for:

```text
continuous foam density
surface-like sheets
atomic splat-like opacity events
```

But this is not the main contribution. It is setup.

### 2. Ray-Fiber Optical Density

Given camera gauge `Gamma_l(y,z)` and fiber-measure Jacobian `J_l(y,z)`:

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

Use `lambda` for pulled ray-fiber extinction and reserve `sigma` for world-local
density. That small notation change makes the gauge story clearer.

### 3. Visibility Monoid

Define an optical transfer element:

```text
g = (beta, m)
```

where:

```text
beta = residual transmittance through the element
m    = visible emitted/color contribution of the element against black
```

Composition:

```text
(beta_1, m_1) otimes (beta_2, m_2)
    =
    (beta_1 beta_2, m_1 + beta_1 m_2)
```

Identity:

```text
e = (1, 0)
```

Decode against background:

```text
decode((beta,m), B) = m + beta B
```

This is the central algebra. It is associative, depth-ordered, GPU-prefix
friendly, and gives the discrete VJP directly.

### 4. Optical Transfer Matrix

Represent the monoid as affine matrices:

```text
M(beta,m) =
    [ beta I_C   m ]
    [ 0          1 ]
```

Then:

```text
M(g_1 otimes g_2) = M(g_1) M(g_2)
```

For continuous foam:

```text
A_y(z) =
    [ -lambda_y(z) I_C   eta_y(z) ]
    [ 0                  0        ]

M_y = P exp int A_y(z) dz
```

This is the compact mathematical headline. Use it carefully: it is elegant,
but the implementable object is the discrete event-element scan.

### 5. Event Optical Element

For a certified interval `I_k(y) = [z_k, z_{k+1}]`:

```text
beta_k(y)
    =
    exp(- int_{I_k} lambda(y,s) ds)

m_k(y)
    =
    int_{I_k}
      exp(-(tau(y,z) - tau(y,z_k)))
      eta(y,z) dz
```

The rasterizer is:

```text
G(y) = otimes_k (beta_k(y), m_k(y))
I(y) = decode(G(y), I_bg(y))
```

For constant owner-run material:

```text
g_r(y)
    =
    ( exp(-sigma_owner length_r),
      (1 - exp(-sigma_owner length_r)) c_owner )
```

This is the implementation bridge to owner-run cutwalk / endpoint-record work.

## Theorem Stack To Promote

### T1. Alpha Compositing Is The Atomic Case

A splat with opacity `alpha_i` and color `c_i` is:

```text
g_i = (1 - alpha_i, alpha_i c_i)
```

Depth-sorted alpha compositing is:

```text
G(y) = otimes_i g_i(y)
```

Safe wording:

```text
WorldFoam generalizes splat compositing at the ray-transfer equation level.
```

Unsafe wording:

```text
WorldFoam strictly dominates splatting as a trainable renderer.
```

### T2. Visibility Noncommutativity

For continuous generators:

```text
A_i =
    [ -lambda_i I_C   eta_i ]
    [ 0               0     ]

eta_i = lambda_i c_i
```

The commutator color column is:

```text
[A_1,A_2] ~ lambda_1 lambda_2 (c_1 - c_2)
```

For discrete alpha elements:

```text
Delta m = alpha_i alpha_j (c_i - c_j)
```

This should become a highlighted theorem:

```text
Visibility/order error is controlled by opacity overlap times color contrast.
```

This unifies the World Tubes swap bound and the WorldFoam transfer algebra.

### T3. Same-Representation Replay Equivalence

If per-frame replay and compiled atlas emit the same ordered event elements:

```text
hat g_k(y) = g_k(y)
```

then:

```text
G_compiled(y) = G_replay(y)
I_compiled(y) = I_replay(y)
```

Approximation error is element error plus support/fallback error. This is the
first proof baseline and should precede STAR/GS quality comparisons.

### T4. Monoid Scan VJP

For:

```text
h = a otimes b
beta_h = beta_a beta_b
m_h = m_a + beta_a m_b
```

the direct VJP is:

```text
bar m_a    += bar m_h
bar m_b    += beta_a bar m_h
bar beta_a += beta_b bar beta_h + dot(bar m_h, m_b)
bar beta_b += beta_a bar beta_h
```

For constant segment:

```text
beta = exp(-DeltaTau)
m = (1 - beta) c

bar DeltaTau = -beta bar beta + beta dot(bar m, c)
bar c        = (1 - beta) bar m
```

This is worth implementing before boundary calculus.

### T5. Event Closure, Not Schur Closure

World Tubes:

```text
Gaussian pullback + fiber marginalization -> Schur complement UVT footprint.
```

WorldFoam:

```text
bounded cell pullback -> event intervals + optical transfer elements.
```

Best sentence:

```text
WorldFoam does not have Schur closure. It has event closure.
```

This should be in the introduction or method discussion.

## Promising But Not Mainline Yet

### P1. Interface Flux Adjoint

For boundary movement replacing material `B` by material `A`:

```text
dI/ds =
    T(s) [
      lambda_A (c_A - I_behind)
      - lambda_B (c_B - I_behind)
    ]
```

This is genuinely useful if finite-difference tests pass. It could upgrade
WorldFoam from fixed-topology VJP to fixed-topology moving-boundary VJP.

Do not put it in the main claim until:

```text
face-crossing finite differences pass
sphere endpoint finite differences pass
sign conventions are pinned
near-parallel ray/face denominators are handled
```

### P2. Optical-Depth Coordinate Basis

Parameterize color by optical depth `s = tau(z) - tau(z_k)`.

Then:

```text
m_k = int_0^a e^{-s} c(s) ds
```

Polynomial color basis gives closed-form incomplete-gamma moments. This might
be a good rasterizer variant, but it should not block the first scan/VJP tests.

### P3. Magnus / Commutator Compression

First-order interval collapse:

```text
Lambda = int lambda dz
Q = int lambda c dz
g = ( exp(-Lambda), (1 - exp(-Lambda)) Q/Lambda )
```

Second-order commutator moment:

```text
K = 1/2 int int lambda_1 lambda_2 (c_1 - c_2)
```

Hypothesis:

```text
split intervals where commutator energy is large.
```

This is intellectually attractive and may become a real contribution. But the
first implementation should compare it against simple adaptive splitting.

## Revised Method Section

Use this order:

```text
3.1 Camera-gauged optical matter
3.2 Ray-fiber optical density lambda and color-density eta
3.3 Visibility monoid and optical transfer elements
3.4 Atomic splats as the monoid's atomic-measure case
3.5 WorldFoam event atlas and owner-run rasterization
3.6 Commutator theorem and interval compression criterion
3.7 Monoid-scan VJP
3.8 Fixed-topology boundary flux as optional extension
```

## Revised Experiment Priority

Do these before broad real-scene quality work:

```text
E1. Alpha equivalence:
    sorted splat compositing equals monoid scan of atomic elements.

E2. Commutator prediction:
    two-layer and multi-layer scenes have swap error predicted by
    opacity product times color contrast.

E3. Event-element replay equivalence:
    per-frame live owner-run events and compiled event elements produce the
    same image under fixed certified intervals.

E4. Monoid VJP finite differences:
    beta, m, DeltaTau, c, sigma, and owner-run length gradients match finite
    differences.

E5. Boundary flux finite differences:
    moving face and support-sphere endpoints match finite differences in
    fixed-topology scenes. Keep this after E4.

E6. Compression bakeoff:
    arbitrary depth layers vs commutator-energy splitting vs optical-depth
    basis vs simple adaptive transmittance error.

E7. Frame scaling:
    replay vs compiled event atlas over F = 2,4,8,16,32,64.
```

## Figures To Add

```text
Figure 1: World Tubes vs WorldFoam operator order.
Figure 2: Camera ray fiber with event optical elements.
Figure 3: Visibility monoid scan: beta/m composition.
Figure 4: Alpha splats as atomic transfer elements.
Figure 5: Commutator heatmap: opacity overlap x color contrast.
Figure 6: Owner-run event atlas and prefix/suffix VJP.
Figure 7: Event closure vs Schur closure.
Figure 8: Compression bakeoff and event-density death curve.
```

## Kill Criteria

Do not keep expanding theory if:

```text
monoid VJP fails finite-difference tests
compiled event replay does not match same-representation replay
commutator-energy splitting does not beat simple adaptive splitting
memory grows like per-frame replay
quality remains below the existing weak WorldFoam floor after replay equivalence
```

If these fail, keep WorldFoam as a side-paper theory/prototype lane and let
World Tubes remain the nearer arXiv push.

## Immediate Work Status

Completed in the intake pass:

```text
scientist note records the hard triage of source reformulation

paper draft names optical-transfer algebra, visibility monoid, commutator
theorem, monoid VJP, and event closure

math appendix adds the polished cell-path atlas definition, replay theorem,
VJP, commutator, and validation gates

experiment plan adds alpha equivalence, commutator prediction, event replay
equivalence, monoid VJP, boundary flux, compression bakeoff, and event-density
tests

proof scaffold adds visibility monoid / atomic alpha case and commutator
criterion
```

Historical implementation-facing work, now completed:

```text
Spec: research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md

1. [x] Build the alpha-equivalence fixture.
2. [x] Build the cell-path owner-run fixture and replay equivalence check.
3. [x] Build the monoid/cell-path VJP finite-difference fixture.
4. [x] Build the two-layer commutator prediction fixture.
5. [ ] Keep boundary flux, flux witness scores, and Magnus/commutator
   compression diagnostic-only unless the simpler kinetic compiled route
   fails.
```

The current implementation-facing work is native lowering of the landed
active kinetic multi-chart program and frozen-program VJP, followed by rebuild/
runtime parity, structural recertification, trainer/evaluator integration, and
measured public-scene scaling.
