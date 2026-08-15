# Ray Transfer Lineage And Novelty Audit

Date: 2026-07-26

Status: independent red-team supplement to the normalized ray-holonomy intake;
no new implementation or benchmark result

## Scope

This note audits the proposal preserved in:

```text
research_notes/worldfoam_paper/scientist_notes/
2026-07-26_gauge_invariant_ray_holonomy_intake_and_paper_split.md
```

The original attachment is:

```text
/Users/nicholasbardy/.codex/attachments/
c3f6b522-fd32-4797-941d-8fc2ed5722e2/pasted-text.txt
```

SHA-256:

```text
bbafd893ee7579e8b07934b7df355b3a8fbcec970f92f505ce320afb8cf82a01
```

An exact duplicate exists at:

```text
/Users/nicholasbardy/.codex/attachments/
a41b0a04-7c57-44ba-aa3d-95e748406c02/pasted-text.txt
```

The audit asks four narrower questions:

1. Which equations and system ideas are already WorldFoam?
2. Which parts are genuinely new relative to this repository?
3. Which mathematical claims are correct but framed too broadly?
4. Does the package currently support a new paper claim?

No code was changed and no experiment was launched. Repository artifacts were
inspected as inherited evidence.

## Executive Verdict

The proposal is not presently a new renderer paper.

It combines two separable objects:

```text
A. retained-fiber emission-absorption transfer
B. self-normalized strongly convex polynomial atoms
```

Object A is already the mathematical center of WorldFoam. The existing
WorldFoam appendix already contains:

```text
A_y(z) =
    [ -lambda_y(z) I_C   eta_y(z) ]
    [ 0                  0        ]

M_y = P exp integral A_y(z) dz
```

It also already contains the gauge-invariant optical one-form, visibility
monoid, alpha-equivalence, commutator theorem, product integral, cell-path
atlas, prefix/suffix VJP, same-representation replay theorem, fixed-topology
gradient boundary, and event-complexity scaling claim.

Object B is new relative to the repository's implemented representations. Its
slice-convexity, unique-ridge, one-ray-interval, and exact single-atom optical
depth results are useful. It is nevertheless an unimplemented primitive
hypothesis. The proposal supplies no evidence that its implicit minimizer,
root certificates, overlap integration, atlas subdivisions, or optimizer
behavior are competitive.

The correct project classification is:

```text
World Tubes / STAR UVT:
    primary paper
    early fiber marginalization
    baseline-compatible Gaussian-splat semantics

WorldFoam:
    retained-fiber sibling method
    already owns the optical-transfer algebra
    conditional second paper after quality and native-kernel gates

convex-potential atom:
    optional WorldFoam producer experiment
    separate representation claim only if it wins independently
```

The proposal is architecturally integrated with STAR through the shared
camera-program compiler boundary. It is not a drop-in STAR extension because
it changes operator order and rendering semantics.

## 1. The Exact Operator Fork

Let the camera program define:

```text
pi: E_Gamma -> B
B = Omega x T
Gamma: E_Gamma -> M
F_y = pi^{-1}(y)
```

Both methods begin by pulling world matter onto camera-ray fibers:

```text
Gamma^* W
```

They then apply different operators.

### World Tubes

World Tubes eliminates depth early:

```text
Trace_Gamma[W](y) = pi_* Gamma^* W.
```

For a local Gaussian pullback this produces a compact UVT footprint by a Schur
complement. Visibility is then represented by conditional depth, uncertainty,
order strata, commutation bounds, and fallback.

### WorldFoam

WorldFoam retains depth:

```text
lambda(y,z) dz = Gamma^* dmu
eta(y,z) dz    = Gamma^* dnu
```

and evaluates:

```text
I(y) =
    integral T(y,z) eta(y,z) dz
    + T_back(y) I_bg(y)

T(y,z) =
    exp(-integral_front^z lambda(y,s) ds).
```

Equivalently:

```text
M_y = P exp integral A_y(z) dz.
```

The proposal's "ray-holonomy renderer" is this second operator, not a third
operator.

## 2. Claim-By-Claim Repository Lineage

| Proposal component | Existing repository location | Audit |
| --- | --- | --- |
| Camera program is not a gauge | `gauged_uvt_trace_atlas/00_bundle_foundations/`, `DEPTH_FIBER_CROSS_TRACK_NOTE.md` | Useful wording correction, not a new method. |
| Ray is a fiber over sensor time | Gauged UVT foundations and both paper drafts | Existing shared formalism. |
| Fiber-coordinate Jacobian invariance | `worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md`, section C | Existing theorem and saved STAR value/gradient probes. |
| No order-free `(alpha,c,depth)` summary is universally exact | `worldfoam_paper/proofs/depth_fiber_operator_ordering.md`, Proposition 3 | Existing counterexample with the same formula. |
| Matrix optical generator | `WORLD_FOAM_MATH_APPENDIX.md`, section E | Algebraically identical. |
| Path-ordered exponential | `WORLD_FOAM_MATH_APPENDIX.md`, section E | Algebraically identical. |
| Visibility commutator | `WORLD_FOAM_MATH_APPENDIX.md`, section L | Algebraically identical. |
| Associative transfer elements `(beta,m)` | `WORLD_FOAM_MATH_APPENDIX.md`, section D | Existing visibility monoid. |
| Prefix/suffix or forward/reverse VJP | proof scaffold Proposition 6 and optical-transfer fixture | Existing derivation and finite-difference fixture. |
| Cell/event word compiler | `WORLD_FOAM_MATH_APPENDIX.md`, sections G-H | Existing WorldFoam compiler target. |
| Same-representation replay theorem | proof scaffold Theorem 8 | Existing theorem. |
| Fixed-topology VJP caveat | proof scaffold Theorem 9 | Existing caveat. |
| Event-complexity rather than frame-count scaling | proof scaffold Proposition 10 | Existing conditional systems claim. |
| Self-normalized convex-potential atom | attachment sections 5-8 | New-to-repository primitive proposal. |
| Strong-convexity-safe parameterization | attachment section 5.2 | New-to-repository parameterization proposal. |
| One support interval per straight ray | attachment section 8 | New theorem for the proposed primitive. |
| Polynomial optical-depth integral | attachment section 8 | New theorem for the proposed primitive. |
| Atom-specific discriminant/root compiler | attachment section 10 | A specialization of the existing event-atlas idea; unimplemented. |
| Duhamel transfer derivative | attachment section 11 | Standard propagator derivative; compatible with existing VJP, not a new renderer. |

The proposal's renderer contribution list therefore double-counts the
repository's existing Paper-B theory. Its potentially independent content is
the primitive and the claim that this primitive makes retained-fiber
compilation practical.

## 3. Existing Implementation Evidence

### 3.1 STAR / World Tubes is implemented

The current projective STAR path is not a paper-only sketch.

The principal implementation surface is:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/
torch_gsplat_bridge_star_uvt/projective_trace.py
```

It contains concrete records and routines for:

```text
projective trace fits and windows
support bounds and tile-time bins
conditional depth and visibility sidecars
support and visibility event reports
sensor-time interval partitions
finite-exposure and rolling-shutter lowering
fallback splitting and patching
interval Metal forward
direct interval Metal backward
camera-family evaluation and VJP
```

The trainer integration is in:

```text
src/train/star_uvt_projective_interval_backend.py
src/train/star_uvt_feature_overfit_trainer.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/
research_project/trainer_harness/tile_metal_autograd.py
```

Saved evidence includes:

```text
fiber value gauge max relative error:
    3.5008714326902284e-13

fiber gradient gauge max relative error:
    2.3252318152652155e-12

trained final shared/replay interval-entry ratio:
    0.14836872087001554

trained final shared/replay forward ratio:
    <= 0.26570109456685365

trained final shared/replay backward ratio:
    <= 0.09386445865404805

camera-family Q2 shared payload growth:
    1.0x versus replay 64.0x
```

These are implementation and verifier results. They do not establish broad
dynamic-NVS quality, but they are a substantially stronger systems base than
the new proposal currently has.

### 3.2 WorldFoam transfer is partly implemented

The optical-transfer reference is:

```text
research_experiments/world_foam_lane2/
cell_path_optical_transfer_fixture.py
```

Its tests verify:

```text
visibility-monoid associativity
constant-run alpha equivalence
compiled/replay transfer equivalence
analytic prefix/suffix VJP against finite differences
commutator swap prediction
```

The saved artifact is:

```text
outputs/benchmarks/
2026-07-08_worldfoam_cell_path_optical_transfer_summary.json
```

The larger prototype lane includes moving-ray slab tapes, owner-run and
cutwalk records, fixed-segment VJPs, MPS/Metal smokes, storage scaling, and
matched STAR microgates under:

```text
research_experiments/world_foam_lane2/
```

The strongest honest status remains:

```text
theory plus prototype
native optical-transfer parity incomplete
moving-boundary/topology gradients incomplete
multi-scene heldout RGB quality incomplete
event-density and memory death curves incomplete
```

### 3.3 The proposed atom/compiler is not implemented

No repository implementation was found for:

```text
self-normalized convex-potential atom evaluation
certified minimizer branch r(t)
certified support-root branches
fiber-polynomial trace records for this atom
native GPU polynomial transfer ODE
trust-region incremental recompilation for this atom
```

The proposal must therefore be discussed as a mathematical design, not as a
renderer result.

## 4. Terminology Audit: "Holonomy" Is Misleading Here

For an open interval from `s_0` to `s_1`, the object:

```text
U(s_1,s_0) = P exp integral_s0^s1 A(s) ds
```

is normally a transport operator, propagator, or product integral.

Holonomy normally refers to parallel transport around a closed loop, or to
the group generated by such loop transports. A camera ray is an open fiber,
not a loop. No endpoint identification or closed ray is part of the proposal.

The terminology becomes more problematic because two different notions of
gauge are being mixed.

### 4.1 Ray-coordinate reparameterization

Under an oriented coordinate change:

```text
s' = phi_y(s)
```

the optical one-form satisfies:

```text
A'(s') ds' = A(s) ds.
```

This is ordinary change-of-variables covariance. The resulting physical
transfer is coordinate independent.

### 4.2 Internal feature-basis gauge

If the transported state changes basis by `G(s)`, a genuine connection
transforms as:

```text
A' = G^-1 A G - G^-1 partial_s G.
```

Open-path transport then transforms by endpoint factors:

```text
U'(s_1,s_0) =
    G(s_1)^-1 U(s_1,s_0) G(s_0).
```

It is covariant, not invariant. A closed-loop conjugacy class is the usual
holonomy object. The current proposal does not use a changing RGB/feature
basis. WorldFoam's math appendix already keeps that possibility in a
speculative feature-transfer branch.

### Recommendation

Use:

```text
ordered ray transfer
retained-fiber optical transfer
camera-program-compiled product integral
```

Avoid using "ray holonomy" in the title or headline claim. It may appear as an
optional geometric interpretation with a precise terminology disclaimer.

## 5. A Second Gauge Conflation

The proposal groups:

```text
ordinary depth
inverse depth
projective depth
half-angle orbit parameter
```

as interchangeable fiber coordinates.

The first three can parameterize depth along one ray. A half-angle orbit
parameter normally parameterizes the camera path or sensor-time base. That is
not the same coordinate change.

Reparameterizing ray depth requires a fiber-measure Jacobian.
Reparameterizing camera time affects:

```text
which physical camera pose is queried
the indexing of output samples
the measure used for finite exposure
rolling-shutter timing
```

For an exposure integral, a base-time reparameterization also requires its own
Jacobian. Fiber-coordinate invariance alone does not prove camera-time
reparameterization invariance.

The safe statement is:

> For a fixed physical camera program, optical transfer is invariant to an
> oriented reparameterization of ray depth when the pulled optical measure is
> transformed correctly.

It is not:

> Moving cameras are gauge transformations, so the renderer is invariant to
> camera motion.

Changing the camera program changes the observation.

## 6. The Transfer Algebra Is Standard Rendering Semantics

For:

```text
A(s) =
    [ -lambda(s) I   eta(s) ]
    [ 0              0      ],
```

the product integral is equivalent to:

```text
tau(s) = integral_front^s lambda(r) dr
T(s)   = exp(-tau(s))
I      = integral T(s) eta(s) ds + T_back I_bg.
```

This is standard emission-absorption volume rendering. The matrix form is
useful because it exposes associativity, compression, scans, and adjoints. It
does not create a new physical renderer by itself.

A narrow external prior-art sanity check reinforces this boundary:

- [NeRF](https://arxiv.org/abs/2003.08934) uses differentiable classical
  volume rendering of continuous density and emitted radiance.
- [DIVeR](https://arxiv.org/abs/2111.10427) uses deterministic interval-based
  integration along rays.
- [NeRF Revisited / PL-NeRF](https://arxiv.org/abs/2310.20685) derives exact
  integration under piecewise-linear density.
- [CvxNet](https://arxiv.org/abs/1909.05736) is evidence that learnable convex
  implicit primitives are an established representation direction, although
  it is not the same self-normalized polynomial atom.

This was not a comprehensive literature review. It is enough to rule out
novelty claims based only on:

```text
emission-absorption transfer
path ordering
deterministic segment integration
convex implicit support
```

Any external novelty claim for the combined method still requires a dedicated
prior-art audit.

## 7. What The Convex-Potential Atom Actually Guarantees

Assume:

```text
q: R^3 x I -> R
nabla_x^2 q(x,t) >= lambda I
lambda > 0

mu(t) = min_x q(x,t)
D(x,t) = q(x,t) - mu(t)

sigma(x,t) =
    alpha(t) (1-D(x,t))_+^p.
```

Under smooth coefficients, bounded physical-time domain, and positive
`alpha(t)`, the following claims are sound:

```text
one unique spatial minimizer r(t)
compact convex support slice {D <= 1}
connected support with regular boundary
smooth ridge r(t)
one support interval on every affine spatial ray
polynomial extinction along that ray
algebraically evaluable single-atom optical depth
vanishing endpoint term for optical-depth derivatives
```

These are useful construction theorems.

## 8. What The Atom Does Not Guarantee

### 8.1 "No stored position" is representational rhetoric

The polynomial's linear and higher-order coefficients encode the ridge. The
position is derived rather than explicitly named, but information about it is
still stored.

The cost also moves:

```text
explicit center:
    cheap to read

implicit center:
    solve nabla_x q(r(t),t) = 0
    certify the solution branch
    differentiate or apply envelope identities
```

The scientific question is whether the implicit chart improves capacity,
conditioning, or compilation. Naming the center indirectly is not itself a
contribution.

### 8.2 Self-normalization may destroy low-degree time closure

Even when `q(x,t)` is polynomial, the minimizer:

```text
r(t) = argmin_x q(x,t)
```

and minimum:

```text
mu(t) = q(r(t),t)
```

are generally algebraic or implicit functions of time, not low-degree
polynomials.

The fiber polynomial:

```text
h_y(s) = 1 + mu(t_y) - q(o_y+s d_y,t_y)
```

is polynomial in ray depth `s` at fixed `y`. Its coefficients need not be
low-degree functions of sensor time. Chebyshev fitting them is an
approximation whose rank and chart count must be measured.

This is the largest unproved bridge from atom theorem to camera compiler.

### 8.3 Strong convexity does not give a unique orientation

The Hessian at the ridge is positive definite, but it may have repeated
eigenvalues. At a repeated eigenvalue, eigenvectors inside the repeated
eigenspace are not unique.

Therefore:

```text
scale spectrum:
    derived

orientation:
    derived only where the Hessian spectrum is simple
```

A globally smooth quaternion is not guaranteed. If appearance has an
anisotropic material frame, that frame is additional state.

### 8.4 The parameter count is not the identifiable structured count

The dense polynomial quotient count removes time-only additions to `q`. The
safe factorized parameterization introduces other non-identifiabilities:

```text
F(t)^T F(t) factor gauges
sign and permutation symmetries in even-power terms
redundant repeated rank-one terms
degree growth after polynomial expansion
```

Raw scalar count is not an effective degree-of-freedom or optimizer-complexity
result.

### 8.5 Compact convex support limits topology

One atom cannot represent:

```text
two disconnected components
a torus or shell with a hole
branched support
multiple simultaneous ridges
```

Mixtures can represent these, but then active overlap, compiler records, and
integration cost grow. The correct comparison is against a matched mixture of
Gaussian or cell primitives, not against one weaker primitive.

## 9. "Exact Ray Polynomial Rendering" Is Too Broad

The atom gives a polynomial extinction profile on a straight ray segment.
The complete colored render is not generally polynomial or elementary.

For multiple atoms:

```text
lambda(s) = sum_i sigma_i(s)
eta(s)    = sum_i sigma_i(s) c_i(s)
```

and:

```text
I =
    integral exp(-integral lambda)
             eta(s) ds.
```

Even when every `sigma_i` is polynomial on every active segment, the
transmittance is the exponential of an integrated polynomial. With varying
or differently colored emission, the final integral is generally
non-elementary.

The proposal acknowledges a certified numerical ODE. The paper language must
therefore distinguish:

```text
exact:
    support interval
    single-atom optical depth
    isolated constant-color transfer

numerical with tolerance:
    arbitrary colored overlap
    view-dependent appearance
    finite exposure
    rolling shutter
    VJP of the approximated transfer
```

A more accurate method phrase is:

```text
polynomial-density retained-fiber transfer
```

not:

```text
exact ray-polynomial renderer
```

## 10. Overlap Is Defined Correctly, Not Made Free

Summing extinction and emission before transfer removes the need to assign one
representative total depth to each overlapping volume. This is a semantic
improvement for volumetric matter.

It does not remove:

```text
active-support discovery
tile and frustum incidence
support endpoint sorting or full-interval integration
per-ray active count hbar
quadrature or ODE work q_f and q_r
surface-like jump ordering
near/far clipping
camera-chart and support tangencies
training-time topology refresh
```

It also changes the representation from splat-like atomic opacity events to a
continuous participating medium. Quality changes can come from this semantic
change rather than from compilation. A fair experiment needs both:

```text
same-world replay versus compiled retained transfer
STAR versus retained transfer at matched world capacity
```

The first isolates compilation. The second tests representation semantics.

## 11. Narrow Novelty Ledger

### High-confidence existing or standard components

```text
camera-ray fiber formalism
depth-coordinate Jacobian invariance
emission-absorption equation
affine transfer matrix
path-ordered/product integral
two-layer noncommutation counterexample
prefix/suffix adjoint
event-atlas scaling conditional on event count
```

### New-to-repository mathematical proposal

```text
self-normalized strongly convex spacetime polynomial atom
safe strong-convexity parameterization
one-interval support theorem for that atom
single-atom polynomial optical-depth formula
tangency smoothness order for the compact-support profile
atom-specific minimizer/root discriminant compiler
```

### Not established as field novelty

```text
the atom family relative to convex implicit primitives
the exact combination of atom plus moving-camera compiler
the claimed certificate construction relative to kinetic geometry
the transfer implementation relative to deterministic volume integration
```

### Not established empirically

```text
quality per byte
optimizer stability
GPU integration cost
event-count scaling
root-certificate cost
training-time recompile frequency
public heldout improvement
native forward/backward performance
```

## 12. Paper Classification

The complete proposal should currently be classified as:

```text
an intellectually coherent WorldFoam extension hypothesis
```

It is not yet:

```text
a validated standalone rendering method
a new STAR implementation
a demonstrated replacement for World Tubes
a demonstrated new primitive paper
```

WorldFoam can still become its own Paper B. If it does, the defensible novelty
must be the measured systems result:

> A known moving-camera program is compiled into a reusable retained-fiber
> optical-transfer atlas whose native forward and adjoint preserve a
> high-accuracy volumetric reference while reducing structural replay work
> across time.

The standard transfer equation and the word "holonomy" cannot carry that
paper. The convex-potential atom should remain optional until a matched
capacity/compiler experiment proves it is necessary.

