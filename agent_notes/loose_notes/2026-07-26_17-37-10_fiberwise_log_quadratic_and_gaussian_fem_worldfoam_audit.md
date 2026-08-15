# Fiberwise Log-Quadratic Tubes And Gaussian-FEM WorldFoam

Date: 2026-07-26

Status: critical mathematical and engineering audit; proposals are not
implemented results

Author/role: Codex primary synthesis with independent red-team, repository
overlap, and camera-gauge audits

## Context

The trigger was the attached ChatGPT Pro proposal:

```text
/Users/nicholasbardy/.codex/attachments/
d3e27c47-2005-4080-9515-c4eb7668a86e/pasted-text.txt
```

It proposes a fiberwise log-quadratic spacetime atom

\[
\rho(x,t)
=
\exp\!\left[
-\frac12x^\top P(t)x+q(t)^\top x-r(t)
\right],
\qquad P(t)\succ0,
\tag{1}
\]

with \(P,q,r\) in finite global time bases. It also claims a maximality
theorem under exact Gaussianity on spatial rays and recommends a 24-parameter
degree pattern.

The questions audited here are:

1. Is the mathematics correct?
2. Is the object new relative to DynaWorld's existing notes?
3. Does it advance the sublinear multi-frame rasterization objective?
4. Does a Gaussian finite-element WorldFoam fit the existing camera-ray
   fibers and camera-gauge framework?
5. What is the smallest experiment that distinguishes mathematical elegance
   from an actual renderer improvement?

## Inputs Used

- the attached proposal, read in full;
- `research_notes/spacetime_gaussian_representation/08_native_motion_bundles_and_shared_raster.md`;
- `research_notes/renderer_lane_taxonomy.md`;
- `research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md`;
- `research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md`;
- current WorldFoam Gate1/Gate4 reference and benchmark notes;
- `BASELINES.md`, including the matched 16-frame and full-300-frame rows;
- the primary 4DGS, exact-ray Gaussian, and volumetrically consistent
  Gaussian-rendering papers linked by the proposal.

No new training or renderer benchmark was run. Claims about new quality or
speed are therefore explicitly proposals, not evidence.

## Validation Performed

- Three independent audits checked the proposal algebra, overlap with current
  notes/code, and camera-gauge compatibility.
- The ray integral and inverse-depth Jacobian identity were numerically checked
  for one nonspherical SPD example against a 200,000-panel Simpson reference:

```text
analytic integral                  0.2846654891336991
ordinary-depth absolute error      0.0 at printed precision
inverse-depth-gauge absolute error 0.0 at printed precision
```

- Current performance and memory statements were copied from `BASELINES.md`
  and the accepted paper-run artifacts rather than inferred from an active
  process.
- No quality, convergence, event-density, or new-kernel performance result was
  generated.

## Executive Verdict

The attached answer is mathematically useful but not a renderer breakthrough.

It contributes:

- a clean classification lemma under a strong all-affine-lines axiom;
- a convenient natural-parameter presentation;
- correct field-jet formulas for derived position, velocity, acceleration,
  scale, and simple-spectrum eigenframe motion;
- a concrete 24-parameter swept-Gaussian baseline.

It does not contribute:

- a new object relative to the existing swept-Gaussian field in E079-E080;
- a proof of sublinear geometry, raster, visibility, or backward work;
- a sensor-time support/event compiler;
- exact general visibility for overlapping colored atoms;
- a replacement for the current camera-gauge/projective trace atlas.

The strongest new engineering idea is not the standalone atom. It is a
**log-extinction finite element on a native 4D cell complex**, rendered by
WorldFoam's retained ray-depth transfer algebra. That construction is
compatible with the existing camera rays as fibers and cameras as gauges.
It is not implemented.

## 1. What The Proposed Object Really Is

Completing the square in (1) gives

\[
m(t)=P(t)^{-1}q(t),
\qquad
C(t)=P(t)^{-1},
\tag{2}
\]

\[
w(t)
=
\exp\!\left[
-r(t)+\frac12q(t)^\top P(t)^{-1}q(t)
\right],
\tag{3}
\]

and hence

\[
\rho(x,t)
=
w(t)\exp\!\left[
-\frac12(x-m(t))^\top P(t)(x-m(t))
\right].
\tag{4}
\]

So \(P,q,r\) do not remove a time-varying position function. They encode it
in exponential-family natural coordinates. In the special case \(P=I\),
\(q(t)=m(t)\) literally is the position curve.

This is the same family already written in the repository as

\[
Q(x,t)=x^\top A(t)x-2b(t)^\top x+c(t),
\qquad A(t)\succ0,
\tag{5}
\]

with

\[
p(t)=A(t)^{-1}b(t),
\quad C(t)=A(t)^{-1},
\quad \psi(t)=c(t)-b(t)^\top A(t)^{-1}b(t).
\tag{6}
\]

The coefficient map is only a convention change:

\[
A=P,\qquad b=q,\qquad c=2r
\tag{7}
\]

if (5) is used inside \(\exp(-Q/2)\).

Status: **proved equivalence**.

## 2. The Classification Theorem: Correct, But Narrower Than Advertised

Let \(\ell(x,t)=-\log\rho(x,t)\). Fix \(t\), assume \(\ell\in C^3\), and
assume that for **every affine spatial line**

\[
x(s)=x_0+sd
\tag{8}
\]

the function \(s\mapsto\ell(x_0+sd,t)\) has degree at most two.

Then

\[
D_x^3\ell(x,t)[d,d,d]=0
\qquad\text{for every }d.
\tag{9}
\]

Because \(D_x^3\ell\) is a symmetric trilinear form, polarization implies

\[
D_x^3\ell(x,t)=0.
\tag{10}
\]

Therefore the spatial Hessian is independent of \(x\), and integration gives

\[
\ell(x,t)
=
\frac12x^\top P(t)x-q(t)^\top x+r(t).
\tag{11}
\]

This is a valid elementary maximality theorem.

The proposal switches, however, from "camera rays" to "every affine line."
Those are not equivalent. A finite camera path and finite field of view
observe only a subset of affine lines. Add a smooth bump to \(\ell\) whose
support misses the observed ray set. Every observed ray is unchanged, while
the global field is not quadratic.

Therefore the defensible claim is:

> Fiberwise log-quadratics are maximal among \(C^3\) fields whose negative
> log is quadratic on every affine spatial line.

It is not:

> They classify every field that is Gaussian on the rays of the cameras in
> a dataset.

It also deliberately excludes compactly clipped cell atoms. A cell-truncated
Gaussian is not Gaussian on the whole affine line even though its integral
over the cell interval is analytic.

Status: **theorem proved under a stronger axiom; broad camera-ray
interpretation refuted**.

## 3. Parameter Counts And Derived Motion

For polynomial degrees \((d_P,d_q,d_r)\), the scalar count is

\[
6(d_P+1)+3(d_q+1)+(d_r+1).
\tag{12}
\]

Thus:

| Family | Count, excluding appearance |
| --- | ---: |
| Static natural-parameter Gaussian | \(6+3+1=10\) |
| Full SPD(4) Gaussian in this chart | \(6+6+3=15\) |
| Proposed \((d_P,d_q,d_r)=(1,2,2)\) | \(12+9+3=24\) |

The derivatives of the ridge follow from \(Pm=q\):

\[
\dot m
=
P^{-1}(\dot q-\dot Pm),
\tag{13}
\]

\[
\ddot m
=
P^{-1}(\ddot q-\ddot Pm-2\dot P\dot m).
\tag{14}
\]

These formulas are useful but do not create new degrees of freedom; they
decode motion already stored in the coefficient functions.

A constant \(P\), affine \(q\), and quadratic \(r\), subject to the block
precision being SPD, recover every strict SPD(4) Gaussian. Its conditional
spatial center is affine in time and its conditional covariance is constant.
Variable \(P(t)\) and higher-order \(q(t)\) add curvature, physical
eigenframe rotation, and changing scale.

The eigenprojector/eigenframe formulas are geometric only when eigenvalues
are distinct. At repeated eigenvalues, orientation in the repeated eigenspace
is gauge. If appearance has an anisotropic material frame, that appearance
frame is extra data and cannot automatically be recovered from scalar density.

The proposal's Euclidean spacetime curvature also needs a declared conversion
between seconds and meters. Without a spacetime metric or scale, its norm is
not invariant under changing the unit of time.

Status: **counts and ridge derivatives proved; metric-free spacetime-curvature
claim rejected**.

## 4. Exact Ray Restriction And Its Gauge Boundary

Let \(y=(u,v,\tau)\) denote a sensor-time sample and use a local affine
physical-depth gauge

\[
\Gamma_\ell(y,s)
=
\bigl(o(y)+s\,d(y),\,t_y\bigr).
\tag{15}
\]

At fixed \(y\),

\[
-\log\rho(o+sd,t_y)
=
\frac12a s^2+b s+c,
\tag{16}
\]

where

\[
a=d^\top Pd>0,\qquad
b=d^\top(Po-q),
\tag{17}
\]

\[
c=\frac12o^\top Po-q^\top o+r.
\tag{18}
\]

Thus

\[
s_*=-\frac ba,
\tag{19}
\]

and the full-line integral is

\[
\int_{\mathbb R}\rho(o+sd,t_y)\,ds
=
\sqrt{\frac{2\pi}{a}}
\exp\!\left[-c+\frac{b^2}{2a}\right].
\tag{20}
\]

A finite interval \([s_-,s_+]\) gives the corresponding erf difference.
This is exact for every individual affine pinhole ray, including a moving
camera and rolling shutter.

The camera-gauge invariant statement is instead

\[
\lambda_\ell(y,s)\,ds
=
\rho(\Gamma_\ell(y,s))J_\ell(y,s)\,ds.
\tag{21}
\]

Under \(s=\phi_y(\zeta)\),

\[
\lambda'_\ell(y,\zeta)
=
\lambda_\ell(y,\phi_y(\zeta))
\left|\partial_\zeta\phi_y(\zeta)\right|,
\tag{22}
\]

so

\[
\int\lambda_\ell(y,s)\,ds
=
\int\lambda'_\ell(y,\zeta)\,d\zeta.
\tag{23}
\]

Literal Gaussianity in the depth coordinate is not gauge invariant. In
inverse depth, the exponent contains \(1/\zeta^2\) and \(1/\zeta\); in log
depth it contains \(e^{2\zeta}\) and \(e^\zeta\). The optical one-form and
its integral remain invariant when the Jacobian is retained.

Status: **exact and compatible with the existing camera-gauge formalism**.

## 5. Why This Does Not Establish Sublinear Rasterization

Finite state independent of frame count is not the same as sublinear
rendering work.

Direct evaluation still permits

\[
W_{\mathrm{direct\ geometry}}
=
\Theta(NT)
\tag{24}
\]

for basis evaluation, SPD solves, projection, support, binning, and sorting
over \(N\) atoms and \(T\) times.

For a moving pinhole camera,

\[
a(u,v,t)
=
\widetilde u^\top D(t)^\top P(t)D(t)\widetilde u,
\tag{25}
\]

and the trace contains

\[
a(u,v,t)^{-1/2}
\exp\!\left[
\frac{b(u,v,t)^2}{2a(u,v,t)}-c(u,v,t)
\right].
\tag{26}
\]

This is generally rational-exponential or transcendental over sensor-time,
not one joint UVT quadratic. Arbitrary camera rotation can also take a small
polynomial basis for \(P\) into trigonometric coefficients such as
\(\sin 2\theta(t)\) and \(\cos 2\theta(t)\).

Consequences:

- exact per-ray depth integration survives;
- the strict SPD(4)/affine-camera global quadratic closure does not;
- the current projective/orbit trace atlas, denominator certificates,
  support events, visibility strata, and fallback remain necessary;
- variable \(P(t)\) can be more expensive than strict SPD(4) unless the
  compiler amortizes it.

The required scoped claim is:

\[
W_{\mathrm{structural}}
=
O(E(T)+NB),
\tag{27}
\]

where \(E(T)\) is the number of certified support, chart, face, and order
events. Structural work is sublinear relative to per-frame replay only when

\[
E(T)=o(T\,R),
\tag{28}
\]

where \(R\) is the corresponding per-frame structural work. Full image
materialization still costs at least \(\Omega(PT)\) for \(P\) pixels.

The proposal gives no bound or measurement for \(E(T)\).

Status: **sublinear inference is unsupported**.

## 6. Visibility And Support Overclaims

Analytic density integral for one atom does not solve overlapping colored
volume transport. If differently colored atoms overlap in depth, one alpha
and one total order per atom generally cannot reproduce continuous
interleaving.

A pointwise threshold \(\rho<\varepsilon\) also does not by itself certify an
integrated opacity error. A broad low-density tail can have non-negligible
line mass. A valid culling certificate must bound tail optical depth, total
active atoms, and the induced compositing error.

"Small fixed quadrature" for exposure is an approximation unless an error
bound is supplied. With variable \(P,q,r\) and a moving camera, the time
integrand need not remain Gaussian or even polynomial-exponential of low
degree.

Status: **per-atom interval formulas proved; global visibility, culling-error,
and exposure-exactness claims weakened**.

## 7. Comparison With What Is Already On Disk

| Representation | Expressivity | Exact local ray algebra | Shared-time rendering evidence | Current verdict |
| --- | --- | --- | --- | --- |
| Strict SPD(4) atom | affine ridge, fixed conditional SPD(3) | yes | strongest global quadratic source closure | required baseline |
| Proposed \(P(t),q(t),r(t)\) atom | curved ridge, rotating/changing SPD(3) | yes per time | no new compiler or event bound | expressive producer/baseline |
| Existing E079-E080 swept Gaussian | same family as proposal | yes, already derived | basis/adjoint math already present | proposal mostly repackages it |
| World Tubes / STAR atlas | source-dependent, early depth pushforward | Schur/trace based | implemented projective intervals and VJP | current primary renderer |
| Current full WorldFoam/PowerFoam | bounded cells, retained depth | current full lane mostly constant/cell surface state | Gate4 microgate positive; full lane not compact | theory/prototype, not parity |
| Proposed log-FEM WorldFoam | bounded cells plus local positive field | exact P1/P2 segment optical depth | unimplemented | high-value bounded experiment |

The proposal beats the **currently implemented restricted World-Tube atom**
in per-atom curved/rotating spatial expressivity. It does not beat the existing
mathematical source family or the current compiler on the central
computational objective.

The full-300-frame benchmark exposes a separate implementation problem:

- World Tubes: 14,336 parameters, about 60 KB checkpoint, 3.114 GB sampled
  peak MPS driver allocation, 78.33 s mean train wall, 5.9153 heldout PSNR.
- current WorldFoam lane: 28,569,600 parameters, about 116.7 MB checkpoint,
  15.794 GB sampled peak driver allocation, 361.82 s mean train wall,
  5.6159 heldout PSNR.

The WorldFoam count is exactly

\[
28{,}569{,}600
=
1024\cdot300\cdot93,
\tag{29}
\]

consistent with the current full lane storing 93 trainable scalars per
cell-frame. This is not evidence against the WorldFoam transfer algebra. It
is evidence that the current full-lane parameterization does not yet realize
the intended spacetime sharing. A compact 4D FEM parameterization could
directly attack that gap.

By contrast, the narrow verified Gate4 microgate already shows the value of
compiled cell events: over 2/4/8/16 frames, WorldFoam total time was
3.008/3.014/3.323/4.095 ms, while matched STAR was
5.003/5.943/8.092/9.794 ms. This is a fused-MSE speed microgate, not broad
RGB quality parity.

Status: **measured facts from current baseline catalog; representation-level
extrapolation remains a hypothesis**.

## 8. Does Gaussian-FEM WorldFoam Fit Camera Rays And Gauges?

Yes.

Let a convex spacetime cell be

\[
K
=
\{Z\in\mathbb R^4:A_j^\top Z\le h_j,\ j=1,\ldots,m\}.
\tag{30}
\]

Along a fixed sensor-time ray

\[
Z(y,s)
=
(o(y)+sd(y),t_y),
\tag{31}
\]

each face inequality becomes

\[
\alpha_j(y)+\beta_j(y)s\le0.
\tag{32}
\]

Intersecting its lower and upper bounds gives one exact interval

\[
K\cap F_y=[s_-(y),s_+(y)]
\tag{33}
\]

because \(K\) is convex. This is precisely the current
camera-pullback/ray-fiber construction.

For a cell-local log-quadratic density,

\[
\sigma_K(Z)
=
\mathbf1_{Z\in K}\exp[-\ell_K(Z)],
\qquad
\ell_K(Z)
=
\frac12Z^\top H_KZ+g_K^\top Z+c_K,
\tag{34}
\]

the ray pullback is

\[
\ell_K(Z(y,s))
=
\frac12h(y)s^2+k(y)s+l(y).
\tag{35}
\]

Therefore

\[
\Delta\tau_K(y)
=
\int_{s_-}^{s_+}
J(y,s)e^{-\ell_K(Z(y,s))}\,ds
\tag{36}
\]

is a truncated-Gaussian erf expression in affine metric depth when \(J\) is
constant along the ray. With constant cell color \(c_K\), its exact transfer
element is

\[
\beta_K=e^{-\Delta\tau_K},
\qquad
m_K=(1-\beta_K)c_K.
\tag{37}
\]

The existing associative visibility product is unchanged:

\[
(\beta_1,m_1)\otimes(\beta_2,m_2)
=
(\beta_1\beta_2,m_1+\beta_1m_2).
\tag{38}
\]

Thus the current cell word, prefix/suffix scan, and fixed-topology adjoint can
be reused. Unlike overlapping global Gaussians, disjoint foam cells provide a
real geometric ray order and do not require pretending that overlapping
colored densities have one atomwise total order.

Status: **mathematically compatible; implementation absent**.

## 9. A Cleaner Finite-Element Form Than One \(P,q,r\) Tube Per Cell

The phrase "Gaussian finite element" should be made precise.

The cleanest candidate is an exponentiated finite element for optical
extinction:

\[
\sigma_h(Z)=\exp[-\ell_h(Z)].
\tag{39}
\]

On each 4D cell \(K\), write

\[
\ell_h|_K(Z)
=
\sum_{a\in\mathcal N(K)}
\theta_a N_a(Z).
\tag{40}
\]

Two useful orders are:

### P1 log element

If \(N_a\) are affine barycentric shape functions, then

\[
\ell_h(Z(y,s))=as+b.
\tag{41}
\]

The optical-depth integral is an elementary exponential difference, with a
stable constant-density limit as \(a\to0\).

### P2 log element

If \(N_a\) are quadratic shape functions, then

\[
\ell_h(Z(y,s))=as^2+bs+c.
\tag{42}
\]

The optical-depth integral is erf for \(a>0\), the P1 expression for \(a=0\),
and erfi for \(a<0\). Constraining the ray/spatial Hessian to be PSD avoids the
erfi growth branch, but global SPD is not mathematically required because the
cell is bounded.

This formulation has three advantages over independent \(P,q,r\) records:

1. shared nodal coefficients give \(C^0\) log-density continuity across cell
   faces automatically on a conforming simplicial complex;
2. positivity of extinction is automatic through the exponential;
3. constant-density WorldFoam is included as the constant-\(\ell_h\) case.

There is a geometry choice:

- a 4D simplicial/Delaunay complex gives standard shared-corner FEM;
- power/Voronoi cells give the current owner semantics but favor a
  discontinuous-Galerkin cell polynomial unless generalized barycentric
  coordinates or a subdivision are introduced.

This is a real fork. "Corner-based world cells" most naturally suggests the
4D simplicial version; "owner foam" most naturally suggests cell-local DG
log-polynomials.

Status: **new proposal derived from the current framework**.

## 10. Appearance Is The First Important Limitation

For constant cell color, (37) is exact. For a general corner-interpolated
color \(c(s)\),

\[
m_K
=
\int_{s_-}^{s_+}
T(s)\sigma(s)c(s)\,ds,
\tag{43}
\]

and \(T(s)\) contains the exponential of the cumulative extinction integral.
Even when \(\sigma\) is a Gaussian in \(s\), this emission integral is not
generally elementary.

Initial choices are:

1. constant RGB or constant feature vector per cell;
2. an appearance basis polynomial in cumulative optical depth, for which
   transfer moments can be precomputed;
3. controlled short quadrature inside the cell;
4. refine cells and let adjacent cells represent appearance changes.

The user preference for "let two cells/splats do the piecewise work" favors
option 1 first.

Status: **exact constant-appearance path; varying appearance unresolved**.

## 11. Gradient Boundary

For fixed active faces and topology,

\[
\Delta\tau(\theta)
=
\int_{a(\theta)}^{b(\theta)}\sigma(s,\theta)\,ds
\tag{44}
\]

has derivative

\[
\frac{d\Delta\tau}{d\theta}
=
\int_a^b\partial_\theta\sigma\,ds
+\sigma(b)b_\theta-\sigma(a)a_\theta.
\tag{45}
\]

The first term gives coefficient gradients. The last two terms are boundary
flux from moving active faces. The existing prefix/suffix transfer adjoint can
then propagate these local derivatives.

Face-winner swaps, interval births/deaths, connectivity changes, and
power-diagram adjacency changes are event boundaries. The honest initial
contract remains:

```text
exact fixed-topology VJP
+ endpoint derivatives
+ certified refresh/fallback at topology events
```

The current Gate4 microgates do not yet implement the complete moving-geometry
or topology derivative path.

Status: **fixed-stratum derivative proved; topology-differentiable system
open**.

## 12. Branches And Falsification Tests

### Branch A: Keep The Proposed 24-Parameter Tube As A Source Baseline

Hypothesis:
    Curvature and rotating covariance improve quality per atom enough to
    justify a more complex projective trace compiler.

What would make it false:
    A low-knot moving ordinary 3D Gaussian or two strict SPD(4) atoms match its
    quality at lower bytes and lower compile/event cost.

Cheap test:
    One accelerating, one rotating anisotropic, and one scale-changing
    synthetic atom; match bytes and fit error. Then compile both through the
    same trace-atlas interface.

Decision:
    Keep it as a producer, not a renderer fork.

### Branch B: P1 Log-FEM WorldFoam

Hypothesis:
    Most of the current quality gap can be improved by compact continuous
    extinction while retaining extremely cheap cell transfer.

What would make it false:
    Constant-density cells at matched bytes achieve the same quality, or P1
    gradients become ill-conditioned without any heldout gain.

Cheap test:
    Two adjacent fixed 4D simplices, constant RGB, exact segment integration,
    ordinary-depth/log-depth gauge equivalence, and finite-difference
    coefficient/endpoint gradients.

Decision:
    This is the lowest-risk first Gaussian-FEM experiment.

### Branch C: P2 Log-FEM WorldFoam

Hypothesis:
    A quadratic log field substantially improves quality per cell and reduces
    required cell count.

What would make it false:
    Erf/erfi cost, Hessian constraints, or optimization conditioning erase the
    cell-count gain.

Cheap test:
    Use the same fixed cell words as Branch B; change only P1 to P2
    extinction and compare quality/byte, ray-integral error, forward time, and
    VJP time.

Decision:
    Run only after P1 validates the plumbing.

### Branch D: The Current Foam Geometry, Not Density, Is The Bottleneck

Hypothesis:
    Initialization, support coverage, and moving topology dominate current
    WorldFoam quality; richer density will not help.

What would make it false:
    P1/P2 density improves heldout quality with the exact same geometry and
    cell words.

Cheap test:
    Freeze geometry and appearance, optimize extinction only, and compare
    constant/P1/P2 at matched parameter count.

Decision:
    If false, do not build a new 4D mesh before fixing geometry/init.

### Branch E: The Proposal Is Computationally Worse Than Strict SPD(4)

Hypothesis:
    Variable \(P(t)\) increases atlas subdivisions and event count enough to
    overwhelm its expressivity gain.

What would make it false:
    Matched-quality event count and total forward/backward time are lower
    because fewer atoms are needed.

Cheap test:
    Record atoms, bytes, atlas cells, denominator/support/order events,
    compile time, forward time, backward time, and heldout error—not parameter
    count alone.

## 13. Recommended Minimal Experiment

Do not rewrite the renderer. Add one source/transfer fixture:

1. Use a fixed-topology pair or short chain of 4D convex cells.
2. Reuse Gate4 face endpoints and cell words.
3. Implement constant, P1 log-extinction, and P2 log-extinction modes.
4. Keep one constant RGB vector per cell.
5. Compare analytic segment optical depth with dense quadrature.
6. Verify ordinary-depth versus inverse/log-depth results with the fiber
   Jacobian.
7. Finite-difference nodal coefficients and active-face endpoints.
8. Replay at \(T=4,16,64,256\) using direct per-time evaluation and one
   compiled tape.
9. Report tape bytes, event count, compile time, forward, backward, RGB error,
   and quality per parameter/byte.
10. Only then put the best mode into the matched paper runner.

Suggested fixture gates:

```text
analytic-vs-quadrature optical-depth relative error <= 1e-6
gauge-equivalence relative error                 <= 1e-6
coefficient VJP relative error                   <= 1e-5
endpoint VJP relative error                      <= 1e-5
compiled-vs-direct RGB max error                 <= 1e-6
```

No fixed speed threshold should be declared before measuring the constant-cell
reference on the same kernel. Promotion requires a quality-per-byte gain and
retention of the current event/storage scaling, not mathematical novelty alone.

## 14. Final Decision

The proposal should be retained as:

- a correct all-lines classification/no-go lemma;
- a 24-parameter swept-Gaussian baseline;
- a more polished presentation of the already existing E079-E080 family.

It should not replace:

- strict SPD(4) as the simplest exact source;
- World Tubes/STAR as the current primary compiled renderer;
- the projective camera/event atlas;
- WorldFoam's retained-depth visibility algebra.

The Gaussian-FEM WorldFoam hybrid is worth a bounded experiment because it
targets the full-300-frame lane's actual failure mode: huge per-cell-per-frame
state while preserving the Gate4 event-sharing advantage. The clean initial
object is an exponentiated P1/P2 log-extinction finite element on a native 4D
cell complex, not one unconstrained moving Gaussian record per cell.

## Open Questions

1. Should the cell complex be a 4D simplicial/Delaunay mesh with shared nodal
   FEM, or the current power-cell owner foam with DG log-polynomials?
2. Can constant cell appearance reach useful quality if extinction becomes
   P1/P2, or is an optical-depth appearance basis immediately necessary?
3. What fraction of current WorldFoam quality failure is initialization and
   support rather than density capacity?
4. Does P2 reduce the number of cells enough to pay for erf and Hessian
   constraints?
5. Can the current projective Gate4 compiler certify active faces and event
   counts for a true 4D complex across moving cameras?
6. Can moving-face endpoint gradients be added without destabilizing training
   near topology events?
7. At matched heldout quality, which representation minimizes total asset
   bytes plus atlas bytes rather than model parameters alone?
