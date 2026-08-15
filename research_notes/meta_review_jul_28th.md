# Research memo — literature cutoff: July 26, 2026

## 1. Executive verdict

The full-covariance tube written in the prompt is exactly, bijectively, a full-rank SPD(4) Gaussian under a time-distinguished conditional parameterization.
Therefore STAR already represents real 4D spacetime objects rather than frame states, but that object is not new: native full-rank 4D Gaussian scene representations were published before this work, and rank-deficient spacetime Gaussian surfels appeared by July 2026. ([arXiv][1])
If `spatial precision` in the actual code is arbitrary SPD(3), the world family is not narrower than SPD(4); any narrowing must come from implementation constraints on (C), appearance, support, or—more likely—the camera compiler.
A single atom nevertheless has only a straight conditional centerline, constant conditional spatial Hessian, and temporally gated amplitude, so curved motion and rotating or changing covariance require mixtures or a richer atom.
The next paper should choose **option 1: UVT-STAR**, reframed as an event-stratified camera-program compiler and shared-adjoint architecture with a Gaussian backend, not as a new Gaussian primitive.
The global UVT Gaussian should be replaced by an atlas of local sensor-time charts whose size depends on camera conditioning, projected support, genuine visibility events, and approximation rank—not requested frame count.
The strongest plausible novel mathematics is the rendering-specific combination of finite sensor-time stratification, output-sensitive trace compilation, certified local transfer approximation, and coefficient-space reverse mode including visibility-boundary terms.
Four-dimensional power foam should remain a **parallel backend and stress test**, not replace STAR, because point partitioning does not bound line-stabbing depth and topology gradients remain the dominant risk.
Pentatope transport should be the **exact/certified reference lane**, not the production representation, because its local analysis is excellent but its 4D meshing and training topology are likely prohibitive.
Your fixed-camera timings make the compiler thesis credible, but the paper should be killed unless variable-camera heavy work, interaction memory, and shared-backward cost stay nearly invariant as (T) rises over a fixed interval at matched quality and learned bytes.

---

## 2. The four-layer decomposition: world, compiler, evaluator, adjoint

The clean mathematical object is not “a UVT Gaussian.” It is the composition

[
W_\theta
;\xrightarrow{;\Gamma^*;};
\text{fields or geometry on }B\times D
;\xrightarrow{;\operatorname{Transport}*s;};
\mathcal T*{\theta,\Gamma}:B\to\mathcal S,
]

where (\mathcal S) is an affine transfer semigroup. For scalar attenuation and RGB emission, an element may be represented as

[
\mathcal T(y)=(E(y),A(y)),
\qquad
C(y)=E(y)+A(y)C_{\rm bg},
]

with composition, from front to back,

[
(E_1,A_1)\star(E_2,A_2)
=======================

(E_1+A_1E_2,;A_1A_2).
]

The compiler approximates or exactly represents this section of the transfer semigroup over sensor-time.

| Layer            | Canonical role                                                                                                          | Current STAR                                                                 | Recommended realization                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **A. World**     | Camera-independent spacetime ontology (W_\theta)                                                                        | Linear Gaussian tubes                                                        | Initially the same tubes/full SPD(4); later any backend implementing a trace interface                              |
| **B. Compiler**  | Pull back the world through a continuous camera program and resolve support, transport, visibility, and event structure | One fixed pose/intrinsic approximation projected into a global UVT footprint | Local camera charts; continuous support pushforward; event predicates; adaptive trace or composite-transfer records |
| **C. Evaluator** | Materialize arbitrary samples (y=(u,v,\tau)) cheaply                                                                    | UVT tile lookup and tube evaluation                                          | Atlas lookup plus (r_a) local basis or trace evaluations; (O(Nr)) including unavoidable writes                      |
| **D. Adjoint**   | Reduce all image residuals before differentiating geometry once                                                         | Direct per-tube atomics, apparently without a dense tape                     | Residual-to-atlas coefficient reduction, then one compiler/world VJP plus explicit event-boundary terms             |

A useful formal distinction is:

[
\theta
\longmapsto
a_{\theta,\Gamma}
\longmapsto
{C(y_n)}_{n=1}^N
\longmapsto L,
]

where (a_{\theta,\Gamma}) denotes the finite collection of atlas coefficients, trace descriptors, and event records.

The camera-specific (a_{\theta,\Gamma}) is a **gauge or cache**, not the world. A novel camera path requires recompilation from (W_\theta); editing (W_\theta) remains well-defined.

### Physical time remains distinguished

No SO(4) symmetry is needed. Space and time can first be nondimensionalized,

[
\bar x=x/\ell_0,\qquad \bar t=t/\tau_0,
]

but slicing still uses (\pi_t), shutter time remains part of (B), and camera motion remains an (SE(3))-valued function of physical time. The tube parameterization is arguably more physically legible than a raw 4D Cholesky factor because it exposes the conditional spatial covariance and the time-to-space regression.

### Why one global UVT footprint is the wrong general object

Although (B=\Omega\times I_s) is globally defined for a conventional sensor, the projection of one world primitive can:

* leave and re-enter the field of view;
* pass through the camera plane;
* become highly non-Gaussian under perspective and camera rotation;
* cross panoramic seams;
* undergo multiple visibility and ordering changes.

A 360-degree orbit therefore need not destroy the sensor-time domain, but it generally destroys the compactness and low polynomial degree of a **single footprint model**. The correct answer is an atlas whose chart count grows with angular extent and conditioning, not with temporal sampling density.

---

## 3. SPD(4) versus Gaussian-tube equivalence derivation

Let

[
\tau=t-t_0,\qquad y=x-x_0,
]

and consider

[
g(x,t)
======

\alpha\exp\left[
-\frac12
\left(
(y-v\tau)^T C^{-1}(y-v\tau)
+\frac{\tau^2}{s_t^2}
\right)
\right],
]

where (C\in\operatorname{SPD}(3)) and (s_t^2>0).

Expanding the exponent gives

[
\begin{aligned}
&(y-v\tau)^TC^{-1}(y-v\tau)+s_t^{-2}\tau^2\
&=
y^TC^{-1}y
-2\tau,v^TC^{-1}y
+\tau^2\left(v^TC^{-1}v+s_t^{-2}\right).
\end{aligned}
]

Thus, with (z-\mu=(y,\tau)),

[
Q=
\begin{bmatrix}
C^{-1} & -C^{-1}v[2mm]
-v^TC^{-1} & s_t^{-2}+v^TC^{-1}v
\end{bmatrix}.
]

The Schur complement of (C^{-1}) is

[
Q_{tt}-Q_{tx}Q_{xx}^{-1}Q_{xt}=s_t^{-2}>0,
]

so (Q\in\operatorname{SPD}(4)).

Block inversion gives

[
\Sigma=Q^{-1}
=============

\begin{bmatrix}
C+s_t^2vv^T & s_t^2v[2mm]
s_t^2v^T & s_t^2
\end{bmatrix}.
]

Conversely, let

[
\Sigma=
\begin{bmatrix}
\Sigma_{xx} & \Sigma_{xt}\
\Sigma_{tx} & \Sigma_{tt}
\end{bmatrix}
\in\operatorname{SPD}(4).
]

Then

[
s_t^2=\Sigma_{tt}>0,
]

[
v=\frac{\Sigma_{xt}}{\Sigma_{tt}},
]

and

[
C
=

\Sigma_{xx}
-\Sigma_{xt}\Sigma_{tt}^{-1}\Sigma_{tx}.
]

The last expression is positive definite by the Schur-complement theorem. Substituting these quantities reconstructs (\Sigma) exactly.

Equivalently, directly from precision blocks,

[
C=Q_{xx}^{-1},
]

[
v=-Q_{xx}^{-1}Q_{xt},
]

[
s_t^2
=====

\left(
Q_{tt}-Q_{tx}Q_{xx}^{-1}Q_{xt}
\right)^{-1}.
]

### Proposition 1: exact bijection

There is a bijection

[
\operatorname{SPD}(4)
;\longleftrightarrow;
\operatorname{SPD}(3)\times\mathbb R^3\times\mathbb R_{>0},
]

given by

[
\Sigma\longleftrightarrow(C,v,s_t).
]

Including the four-dimensional mean adds ((x_0,t_0)) on either side. Both parameterizations have fourteen geometric degrees of freedom:

[
4\text{ mean}
+
10\text{ covariance}
====================

3+1+3+6+1.
]

If (\alpha) means peak density, it is unchanged. If it means normalized mass, the covariance determinant contributes the usual normalization factor, but the geometric equivalence still holds.

### Conditional interpretation

The Gaussian conditional at physical time (t) is

[
x\mid t
\sim
\mathcal N!\left(
x_0+v(t-t_0),;C
\right),
]

while the temporal marginal is

[
t\sim \mathcal N(t_0,s_t^2).
]

Therefore:

* (v) is not extra structure; it is the regression coefficient (\Sigma_{xt}\Sigma_{tt}^{-1}).
* Storing (v) is merely a coordinate chart adapted to the distinguished projection (\pi_t).
* A raw SPD(4) factorization has no additional geometric expressivity.
* The tube chart may nevertheless optimize better because velocity and conditional spatial shape are separated.

### Is current STAR narrower?

From the parameter list in the prompt, the answer is conditional:

* If `spatial precision` is an arbitrary SPD(3), then the **conceptual STAR world family is exactly full SPD(4)**.
* If it is diagonal, isotropic, fixed-orientation, clamped, or otherwise constrained, current STAR is a strict subset.
* If geometry is full but color, opacity, temporal support, or appearance are restricted, the **world density family** remains full SPD(4) while the complete radiance family is narrower.
* The fixed-camera projection and one-sequence-pose approximation are compiler restrictions, not restrictions of the world atom.

Native 4D Gaussian scene models, including full spacetime covariance and conditioning at timestamps, are already established; the 2023/2024 native-4D works explicitly model spacetime as a whole, and later work has explored disentangled temporal processing, continuous retiming, and rank-deficient spacetime surfels. ([arXiv][1])

### Strict limitation of one SPD(4) atom

A single full-rank spacetime Gaussian has:

[
\mu_{x\mid t}=x_0+v(t-t_0),
\qquad
\nabla_x^2[-\log g]=C^{-1}.
]

It cannot intrinsically express:

1. curved conditional motion;
2. changing conditional orientation;
3. changing conditional anisotropic scale;
4. multiple simultaneous spatial modes;
5. splitting or merging of one atom;
6. periodic or oscillatory motion;
7. a spatial profile whose normalized Hessian changes with time.

Its temporal amplitude changes, and a fixed density cutoff can make the visible support shrink near the temporal tails, but the intrinsic conditional covariance remains (C).

Polynomial motion and rotation have already appeared in Spacetime Gaussian Feature Splatting, while RetimeGS and FreeTimeGS-style families add continuous temporal trajectories or lifespans. These are richer than a single SPD(4) atom but generally retain timestamp-by-timestamp rasterization. ([CVPR Open Access][2])

---

## 4. Comparative table covering all candidates

| Candidate                         | A. Canonical world                                                                                        | B. Camera compiler                                          | C. Cheap evaluator                                     | D. Shared adjoint                                                                             | Expressivity and exactness                                                                         | Decision                                     |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Structured STAR tube**          | A true global 4D Gaussian if (C) is full SPD(3); linear conditional motion and constant slice Hessian     | Current fixed-camera map to one UVT Gaussian footprint      | UVT tiles and local tube list                          | Direct atomics; no dense interaction tape, but geometry work may still be repeated per sample | Exact world time slice; projection and alpha compositing generally approximate                     | Keep as first backend                        |
| **Native SPD(4) Gaussian**        | Identical to unrestricted tube under reparameterization                                                   | Usually conditioned at every (t), then rendered per frame   | Standard 3DGS-like rasterization                       | Per-frame or timestamp-sliced reverse                                                         | No extra geometry expressivity over full tube; raw factorization may change conditioning           | Baseline and parity test, not a contribution |
| **4D power foam**                 | Convex 4D power cells; physical slices are induced 3D power-cell sections                                 | Beam–face event compiler; neighbor traversal                | Cell trace or compiled transfer                        | Cell-feature reduction; site and weight VJP away from flips                                   | Exact affine boundaries with shared metric; line-stabbing can be large; combinatorial flips remain | Parallel backend                             |
| **4D pentatope/corner complex**   | General conforming piecewise-polynomial spacetime field                                                   | Exact cell traversal stratification, then transfer atlas    | Exact analytic segment transfer or certified local fit | Corner/vertex adjoint plus boundary terms                                                     | Strongest local exactness; arbitrary complex is not equivalent to a power foam                     | Exact/reference lane                         |
| **Bernstein swept-volume atoms**  | Curved, compact-support spline tubes with changing affine cross-section                                   | Certified polynomial root isolation and transfer fitting    | Trace list or composite transfer basis                 | Spline/control-point VJP plus root-event terms                                                | Curved motion and changing shape; transport usually certified numerical                            | Serious independent lane                     |
| **Deformation or frame-state GS** | Continuous canonical deformation can define a genuine spacetime field; independent frame Gaussians do not | Typically instantiate geometry separately at each timestamp | Per-frame rasterization                                | Per-frame reverse                                                                             | Highly expressive, but does not achieve frame-amortized scene work                                 | External baseline                            |

Radiant Foam already establishes differentiable 3D Voronoi-foam rendering, and Power Foam extends the idea to bounded power cells and a unified ray-tracing/rasterization representation; Semantic Foam subsequently reuses the cell partition for semantic decomposition. None of those results by itself establishes a 4D camera-time beam compiler. ([arXiv][3])

Four-dimensional simplex space-time meshes are standard in space-time finite elements, including moving domains and time-varying topology. Radiance Meshes and DiffTetVR establish recent tetrahedral radiance-field and differentiable-volume-rendering backends in 3D. ([arXiv][4])

### Expressivity classification

**Identical under reparameterization**

[
\boxed{\text{unrestricted linear Gaussian tube}=\text{full-rank SPD(4) Gaussian}.}
]

**Strict subsets**

* Diagonal or isotropic STAR tubes are strict subsets of SPD(4).
* A rank-three positive-semidefinite spacetime Gaussian is in the boundary closure of SPD(4), not inside SPD(4); it conditions to a rank-two surfel rather than a volumetric ellipsoid. ([arXiv][5])
* Linear-trajectory atoms are strict subsets of suitable spline-trajectory families when the same cross-section kernel is used.

**Incomparable finite-dimensional families**

At fixed primitive count and fixed degree:

* Gaussian mixtures;
* power-cell fields;
* arbitrary simplicial piecewise-polynomial fields;
* swept spline volumes

are incomparable. With unrestricted refinement, several are dense in broad function classes, but that asymptotic universality says little about matched learned bytes or camera-pushforward complexity.

**Frame-state models**

A table of unrelated ({W_k}_{k=1}^T) is not a spacetime-native world. A canonical field plus continuous deformation (F_t), however, does define a global object

[
W(x,t)=W_0(F_t^{-1}(x),t),
]

even if its renderer inefficiently materializes one frame at a time. Thus “true spacetime ontology” and “frame-amortized renderer” are separate properties.

---

## 5. Strongest independently derived candidate: Bernstein swept-volume atoms

The most serious non-Gaussian, non-foam, non-pentatope candidate is a compact-support **Bernstein swept-volume atom**, abbreviated BSVA.

### 5.1 Definition

Over one physical-time knot span, let

[
c(t)=\sum_{j=0}^{d} B_j^d(\xi(t)),c_j,
]

[
A(t)=\sum_{j=0}^{d} B_j^d(\xi(t)),A_j,
]

where (B_j^d) are Bernstein polynomials and

[
\det A(t)\ge \delta>0.
]

The atom is the image of a unit spatial ball under

[
F(\eta,t)=\big(c(t)+A(t)\eta,;t\big),
\qquad
|\eta|\le 1.
]

Define

[
q(x,t)
======

\left|A(t)^{-1}(x-c(t))\right|^2,
]

and use a compact nonnegative kernel

[
\sigma(x,t)
===========

\beta(t),[1-q(x,t)]_+^k,
\qquad
\beta(t)\ge0.
]

Emission can be represented as

[
j(x,t,\omega)
=============

\sigma(x,t)
\sum_{\ell} b_\ell(x,t),c_\ell(\omega),
]

where the (b_\ell) are low-order Bernstein functions in local coordinates and time.

A multi-span B-spline gives a single globally identified atom with continuity constraints; it is piecewise polynomial, but it is not a chain of independently optimized frame identities.

### 5.2 Semialgebraic support

Because

[
A^{-1}=\frac{\operatorname{adj}A}{\det A},
]

the support condition (q\le1) is equivalent, when (\det A>0), to

[
h(x,t)
======

\left|
\operatorname{adj}(A(t))(x-c(t))
\right|^2
---------

\det(A(t))^2
\le0.
]

Thus each knot-span support is a bounded-degree semialgebraic set.

Under a polynomial or rational camera chart, ray intersections solve

[
h(\Gamma(y,s))=0.
]

Tangencies satisfy

[
h(\Gamma(y,s))=0,
\qquad
\partial_s h(\Gamma(y,s))=0.
]

These roots and multiplicity events can be certified by interval Newton methods, Bernstein subdivision, Bézier clipping, or low-degree Sturm sequences.

### 5.3 Expressivity

A BSVA directly expresses:

* curved center motion through (c(t));
* changing rotation, shear, and anisotropic scale through (A(t));
* compact temporal and spatial support;
* smooth appearance changes;
* long-lived identity without per-frame states.

Position, velocity, and curvature are

[
c(t),\qquad c'(t),\qquad c''(t),
]

while orientation and scale may be derived from the polar decomposition

[
A(t)=R(t)S(t).
]

The center is stored rather than discovered as an implicit invariant, but it is also the unique minimizer of (q(\cdot,t)). A fully implicit ridge would be more philosophically intrinsic and materially harder to optimize.

### 5.4 Transport

At fixed physical time and without finite propagation delay,

[
x(s)=o(t)+sd(t),
]

so (q(x(s),t)) is quadratic in (s). Consequently, ([1-q]^k) is polynomial on each support interval.

However, although the optical depth

[
\int_0^s \sigma(r),dr
]

is polynomial, the emission integral generally contains the exponential of a degree-((2k+1)) polynomial. It is not elementary for general (k).

Therefore:

* homogeneous (k=0) atoms admit exact constant-density segment transport;
* smoother (k\ge1) atoms use certified adaptive quadrature;
* finite propagation time raises the polynomial degree because (t=t_s-s/c), but remains certifiable under bounded-degree charts.

### 5.5 Pushforward and visibility

Each atom has a compact 4D bounding box and a bounded projected support per camera chart. The compiler can:

1. cull atom–chart pairs with a 4D BVH;
2. project conservative sensor-time envelopes;
3. isolate entry, exit, and tangency events;
4. fit the resulting transfer on event-free patches;
5. expose silhouette and occlusion boundaries for the adjoint.

Unlike an SPD(4) atom, one BSVA can bend through sensor-time. That reduces world primitive count only if the added trace curvature does not cause a compensating explosion in atlas rank or chart count.

### 5.6 Learned bytes and GPU realization

For (K) spline controls, geometric storage is approximately

[
K(3+9)
]

scalars for (c_j,A_j), plus opacity and appearance. More compact rotation-scale parameterizations are possible, although log-Cholesky or exponential-map interpolation weakens the low-degree semialgebraic structure.

A practical GPU path is:

* one BVH leaf per atom–knot span;
* interval bounds evaluated with Bernstein convex-hull properties;
* batched low-degree root isolation;
* warp-coherent quadrature on certified intervals;
* the same sensor-time atlas and coefficient-space reverse as STAR.

### 5.7 Verdict on the independent candidate

BSVA is mathematically defensible and genuinely different from the three supplied families. It is **not automatically a paper contribution**: swept ellipsoids, spline motion, compact kernels, and polynomial root isolation are individually established ideas.

It becomes useful only if, at matched learned bytes and image quality:

[
\text{one BSVA}
\quad\text{replaces at least roughly}\quad
3\text{–}4
]

linear SPD(4) atoms without increasing compiled rank or event count by a similar factor. Otherwise it is elegant baggage.

---

## 6. Recommended mathematical construction: an event-stratified sensor-time transfer atlas

I recommend the following construction as the actual STAR object.

### 6.1 Fiberwise transfer

Let

[
p:B\times D\to B,\qquad p(y,s)=y.
]

For each sensor-time point (y), define the pulled-back ray fields

[
\tilde\sigma_\theta(y,s)
========================

\sigma_\theta(\Gamma(y,s)),
]

[
\tilde j_\theta(y,s)
====================

j_\theta(\Gamma(y,s),\omega_y).
]

The exact backend transfer is

[
\mathcal T_{\theta,\Gamma}(y)
=============================

\operatorname{Transport}*s
\left(
\tilde\sigma*\theta(y,\cdot),
\tilde j_\theta(y,\cdot)
\right).
]

This is a nonlinear attenuating pushforward, informally

[
\mathcal T_{\theta,\Gamma}
==========================

p_*^{\rm transport}\Gamma^*W_\theta.
]

### 6.2 Atlas structure

The compiler returns

[
\mathcal A
==========

\left{
(U_a,\chi_a,\mathcal R_a,c_a,\mathcal D_a,\mathcal C_a)
\right}_{a=1}^{A},
]

where:

* (U_a\subset B) is a sensor-time patch;
* (\chi_a:U_a\to[-1,1]^3) is a local coordinate chart;
* (\mathcal R_a) is a trace record or active world dependency set;
* (c_a) are local coefficients;
* (\mathcal D_a) is a compact dependency graph back to world and camera parameters;
* (\mathcal C_a) is an error, support, and topology certificate.

Each patch is one of three types.

#### Type I: certified trace-list patch

The active traces and their semantics are fixed over (U_a). The evaluator computes

[
\mathcal T(y)
=============

T_{i_1}(y)\star\cdots\star T_{i_{d_a}}(y),
]

where each (T_i(y)) is represented by a small basis expansion.

This is appropriate when (d_a) is small.

#### Type II: composite-transfer patch

If line depth or overlap is large, compile the whole transfer:

[
\mathcal T(y)
\approx
\widehat{\mathcal T}_a(y)
=========================

\operatorname{Decode}
\left(
\sum_{m=1}^{r_a} c_{a,m}\psi_{a,m}(y)
\right).
]

A convenient physically constrained encoding is

[
\tau_a(y)
=========

\operatorname{softplus}
\left(
\sum_m c^\tau_{a,m}\psi_{a,m}(y)
\right),
\qquad
A_a(y)=e^{-\tau_a(y)},
]

together with a direct expansion of (E_a(y)).

This makes evaluation (O(r_a)), even when compiling the patch required (O(r_ad_a)) source work.

#### Type III: event-aligned patch

An event surface is represented explicitly as

[
\Sigma_e={y:h_e(y,\theta,\Gamma)=0}.
]

The patch stores:

* two-sided trace or transfer records;
* the event equation and normal;
* parameter derivatives of (h_e);
* an event-aligned integration or boundary-quadrature rule.

A final system should not rely on an expensive per-sample fallback over a fixed positive-measure region. Such a fallback would reintroduce

[
O(N_Fd)
]

scene work as (T) rises. A fallback is acceptable as a development diagnostic; the asymptotic claim requires it to be eliminated, compiled by a secondary method, or negligible at the benchmarked resolutions.

### 6.3 Local basis

The practical choices are:

* anisotropic tensor Chebyshev;
* hierarchical Bernstein;
* sparse-grid polynomials;
* low-rank tensor trains;
* direct analytic Gaussian descriptors in nearly affine charts.

Chebyshev coefficients are attractive because analytic transfer functions converge exponentially until the nearest visibility event or projection singularity. Bernstein coefficients are attractive for range and positivity certification.

The rank (r_a) must count actual coefficients or evaluator operations. It cannot mean “one record containing a thousand candidates.”

### 6.4 Event predicates

Backend-specific event predicates include:

1. support entry or exit;
2. ray–support tangency;
3. a projected silhouette;
4. depth-order equality;
5. disocclusion or frontmost-identity change;
6. cell-face sequence change;
7. root coalescence;
8. traversal through a cell edge or vertex;
9. power-diagram or mesh topology flip;
10. camera projection denominator crossing zero;
11. lens or panoramic chart seam;
12. a certificate or approximation-rank failure.

For smooth physically integrated volume fields, primitive “ordering swaps” need not be genuine events because densities coexist and transport is continuous. They become events when the selected backend approximates volume by sorted alpha layers. The compiler must preserve the semantics of the chosen backend rather than silently mixing the two models.

### 6.5 Structural versus numeric compilation

Training makes full recompilation after every parameter update dangerous. Split compilation into:

[
\text{structural compile}
+
\text{numeric coefficient refresh}.
]

The structural phase determines:

* candidate references;
* event topology;
* patch connectivity;
* trace ordering;
* dependency sets.

Each predicate carries a margin. For a predicate (h_e),

[
|h_e|\ge\gamma_e
]

and a parameter Lipschitz bound

[
|\Delta h_e|
\le L_{e,\theta}|\Delta\theta|
]

give a sufficient trust region

[
|\Delta\theta|
<
\frac{\gamma_e}{L_{e,\theta}}
]

under which its sign and local topology cannot change. Within that region, only coefficients are refreshed. Violations trigger local—not global—recompilation.

---

## 7. Forward algorithm and pseudocode

### 7.1 Compiler interface

A pluggable world backend should expose:

```text
Bounds4D(element, time_interval, ε)
TracePredicates(element, camera_chart)
TraceTransfer(element_or_cell, camera_chart, y)
TraceTransferVJP(...)
TopologyMargin(...)
```

A Gaussian backend returns projected layer descriptors or an exact volumetric trace evaluator. A foam backend returns face intersections and cell segments. A pentatope backend returns affine cell intervals and analytic segment transfers.

### 7.2 Compilation

```text
Compile(Wθ, Γ, ε):

    1. Choose backend semantics and split ε into:
         support/truncation,
         camera approximation,
         root/intersection,
         transport,
         local basis,
         floating-point budgets.

    2. Build or refit the backend's 4D acceleration structure.

    3. Split Γ into local camera charts:
         - bounded projection condition number,
         - bounded pose/intrinsic curvature,
         - no lens or panorama seam crossing,
         - bounded rolling-shutter and exposure-map variation.

    4. For every camera chart:
         refs ← ContinuousPushforwardCull(Wθ, chart)
         create an initial coarse partition of Ω × chart_time.

    5. Process sensor-time patches adaptively:

         active ← certified support references on patch
         predicates ← support, tangency, order, traversal,
                      silhouette, and camera-chart predicates

         if predicates are not sign-stable:
             split patch along the highest-impact predicate
             continue

         trace ← certify active intervals and backend semantics
         d ← source trace depth

         if d is small:
             fit/certify per-trace descriptor functions
             if error ≤ ε_patch:
                 emit Type-I trace-list record
             else:
                 split patch or raise local degree
         else:
             sample exact backend transfer at adaptive nodes
             fit a constrained composite transfer basis
             certify transfer and finite-pixel error
             if error ≤ ε_patch:
                 emit Type-II composite record
             else:
                 split patch or raise local rank

         if an actual event crosses the patch:
             isolate the event surface
             emit Type-III two-sided/event-aligned records

    6. Construct:
         atlas lookup structure,
         compact dependency graph,
         topology margins,
         error certificates,
         incremental invalidation map.

    7. Return Aθ,Γ,ε.
```

### 7.3 Evaluation

```text
Eval(A, y):

    a ← AtlasLookup(A, y)

    if a is Type I:
        evaluate at most r_a local trace descriptors
        compose them in certified order
        return decoded color

    if a is Type II:
        q ← Σ_m c[a,m] ψ[a,m](y)
        return DecodeTransfer(q)

    if a is Type III:
        side ← Sign(h_a(y))
        evaluate the corresponding side record
        apply finite-pixel/event integration when requested
```

For a requested set ({y_n}_{n=1}^N),

[
W_{\rm eval}
============

O\left(
\sum_{n=1}^{N}r_{a(y_n)}
\right)
+
\Omega(N)
]

for output writes.

### 7.4 A minimal variable-camera Gaussian implementation

The first implementation need not begin with full algebraic visibility decomposition. A practical version is:

1. Represent the camera pose by a cubic (SE(3)) spline and intrinsics by low-order splines.
2. Over each camera chart, evaluate conditional 3D Gaussian parameters at Chebyshev time nodes.
3. Project them with exactly the same projection model as the sliced baseline.
4. Fit and certify time-dependent:
   [
   u_i(t),v_i(t),Q_i^{2D}(t),z_i(t),\alpha_i(t).
   ]
5. Compute continuous tile envelopes over the whole chart.
6. Generate order-event roots only among overlapping footprint neighbors.
7. At samples, evaluate these descriptors and the same splat semantics.
8. Reduce reverse residuals to descriptor coefficients before differentiating projection and world geometry.

This already tests the principal claim: variable camera motion without a per-frame projection and geometry-backward loop.

### 7.5 Forward cost decomposition

Let

* (S): canonical world representation size;
* (G): camera-program segment and chart complexity;
* (R): continuous world–chart–patch references;
* (E): resolved support or visibility event objects;
* (A): atlas patches;
* (d_a): source trace depth in patch (a);
* (q_a): backend samples used to fit and certify the patch;
* (r_a): evaluator rank.

Then the intended—not unconditional—cost is

[
W_{\rm heavy}
=============

\widetilde O
\left(
S+G+R+E+\sum_{a=1}^{A}q_ad_a
\right),
]

and, when (q_a=O(r_a\log(1/\varepsilon))),

[
W_{\rm heavy}
=============

\widetilde O
\left(
S+G+R+E+\sum_a r_ad_a
\right).
]

Evaluation is

[
W_{\rm cheap}=O(N\bar r),
]

where (\bar r) is the sample-weighted local rank.

These costs behave differently under different scaling experiments:

* **Increase frame density over a fixed physical interval:** (S,G,R,E,A) should remain invariant; only (N) changes.
* **Extend physical duration:** support references, camera charts, and genuine events may increase.
* **Increase camera angular velocity at fixed duration:** chart count and projection rank may increase.
* **Increase scene motion frequency:** event and rank complexity may legitimately increase.
* **Increase image resolution:** output and evaluator work increase; atlas refinement may also increase if (\varepsilon) is measured in pixels.

### 7.6 FLOPs, bandwidth, sorting, and atomics

The implementation should report them separately.

**Compilation FLOPs**

* continuous bounds;
* root isolation;
* backend trace samples;
* basis fitting;
* coefficient and certificate computation.

**Compilation bandwidth**

* writing continuous reference lists;
* reading world parameters;
* writing atlas coefficients and dependencies.

**Sorting**

* should occur per event patch or trace record;
* must not be repeated independently at each requested frame;
* pair generation should be restricted to projected-overlap neighbors.

**Evaluation bandwidth**

* atlas lookup;
* coefficient reads;
* (r_a) basis evaluations;
* output writes.

**Reverse bandwidth**

* residual reads;
* coefficient-adjoint accumulation;
* deterministic segmented reduction.

A system can be FLOP-sublinear in frame count but still become memory-bandwidth linear because (N) outputs must be read or written. That is acceptable and should be stated plainly.

---

## 8. Shared reverse algorithm and pseudocode

### 8.1 Factorization

Let all atlas coefficients be concatenated into (a(\theta,\Gamma)). For a stable patch,

[
q(y)=\sum_{m=1}^{r_a}a_{a,m}\psi_{a,m}(y),
]

[
C(y)=F_a(q(y)).
]

For residual (\bar C(y)=\partial L/\partial C(y)),

[
\bar q(y)
=========

D F_a(q(y))^T\bar C(y),
]

and

[
\boxed{
\bar a_{a,m}
============

\sum_{y_n\in U_a}
\psi_{a,m}(y_n),\bar q(y_n).
}
]

This is the key reduction. Every frame contributes only to a small atlas coefficient vector.

The world adjoint is then

[
\bar\theta_{\rm interior}
=========================

\left(
\frac{\partial a}{\partial\theta}
\right)^T
\bar a.
]

Projection, root, covariance, and camera-spline differentiation occur once per atlas dependency, not once per pixel-frame interaction.

### 8.2 Reverse pseudocode

```text
Adjoint(A, Wθ, Γ, samples, residuals):

    initialize coefficient adjoints ā = 0
    initialize event adjoints ē = 0

    parallel over blocks of samples:

        local_map ← small block-local hash or sorted accumulator

        for (y, C̄) in block:

            a ← AtlasLookup(A, y)

            if a is Type I or Type II:
                q ← evaluate local basis/trace record
                q̄ ← DecodeTransferVJP(q, C̄)

                for each touched local coefficient m:
                    local_map[CoeffID(a,m)] += ψ[a,m](y) q̄

            if a is Type III:
                accumulate the appropriate side's interior term

        deterministically reduce local_map into ā

    for every event surface intersecting an integrated pixel/exposure:
        integrate the visibility-boundary adjoint
        accumulate into ē

    θ̄, Γ̄ ← CompilerVJP(
                 Wθ, Γ,
                 A.dependencies,
                 ā, ē,
                 checkpoint_or_recompute=True
             )

    return θ̄, Γ̄
```

### 8.3 What must be stored

The persistent reverse state should contain:

* atlas coefficients;
* patch lookup structure;
* compressed world/camera dependency lists;
* basis type and chart transforms;
* event equations;
* support, order, and root-separation margins;
* adaptive fit tree or deterministic regeneration seeds;
* error certificates.

It should not contain:

[
\text{pixel}\times\text{frame}\times\text{primitive}
]

or

[
\text{ray sample}\times\text{world parameter}
]

records.

### 8.4 What can be recomputed

The following are cheap or structurally shared enough to recompute:

* basis values at requested samples;
* local decode Jacobians;
* transfer prefix/suffix products at compiler fit nodes;
* Gaussian projection intermediates per world–chart descriptor;
* root derivatives at certified roots;
* local quadrature values.

For a block of (B_s) requested samples, peak reduction scratch can be

[
O(B_s\bar r)
]

rather than (O(N\bar r)).

### 8.5 Deterministic reduction

Raw floating-point atomics are useful as a speed target but not a correctness reference.

A deterministic mode should use:

1. block-local accumulation;
2. stable sorting or fixed hash ownership by coefficient ID;
3. a fixed reduction tree;
4. one fixed-order global merge.

The deterministic kernel should be bitwise repeatable for identical hardware and launch configuration. Its performance target should be measured against the direct-atomic implementation.

### 8.6 Visibility-boundary derivative

Let an event surface be

[
\Sigma_\theta={y:h(y,\theta)=0}.
]

For an integrated sensor objective,

[
L(\theta)=\int_B f_\theta(y),dy,
]

the derivative is

[
\frac{dL}{d\theta}
==================

\int_{B\setminus\Sigma_\theta}
\partial_\theta f_\theta(y),dy
+
\int_{\Sigma_\theta}
[f_\theta](y),v_n(y),dS,
]

where

[
v_n
===

-\frac{\partial_\theta h}{|\nabla_y h|}.
]

Ordinary pathwise AD follows the currently selected branch at fixed (y) and recovers only the first term. It misses the distributional boundary contribution when silhouettes, hard visibility, or discrete layer ordering move. Edge-sampling and related differentiable-rendering work exists precisely because visibility derivatives contain such singular terms. ([Aaltodoc][6])

The compiler can expose:

* (h);
* (\nabla_yh);
* (\partial_\theta h);
* two-sided transfer or loss values;
* a quadrature parameterization of (\Sigma).

It can then apply event-surface quadrature, edge sampling, or an event-aligned reparameterization.

There is an important qualification:

* For a finite sum of fixed point samples, the discrete loss is piecewise differentiable and pathwise AD is its ordinary derivative away from a sample crossing.
* The boundary term belongs to the continuous finite-pixel, exposure, antialiasing, or sample-expectation objective that those point samples approximate.

For a smooth, fully volumetric renderer, many apparent “occlusion events” produce steep but continuous transfer and ([f]=0). For sorted alpha splats, an order swap can be an artificial renderer discontinuity; eliminating that discontinuity may be better than differentiating it.

---

## 9. Exactness and certification analysis

### 9.1 Legend

* **Exact:** mathematically exact for the stated backend, aside from floating-point implementation.
* **Analytic-delicate:** closed form exists but needs stable special functions or limiting branches.
* **Certified approximate:** numerical error is explicitly bounded.
* **Heuristic:** no valid global bound.
* **Event-nondifferentiable:** smooth within a fixed combinatorial structure, not at a topology change.

### 9.2 Candidate-by-candidate boundary

| Operation                        | STAR tube / SPD(4)                                                                              | 4D power foam                                                | Pentatope complex                                | Bernstein swept volume                                      |
| -------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------ | ----------------------------------------------------------- |
| **Time slice**                   | Exact Gaussian conditioning                                                                     | Exact cell intersection with (t=\tau)                        | Exact polytope slice                             | Exact spline evaluation                                     |
| **Support**                      | Infinite exactly; finite cutoff is certified truncation                                         | Exact convex cell, subject to robust predicates              | Exact simplicial support                         | Exact semialgebraic support per span                        |
| **Ray intersection**             | Gaussian evaluation exact; cutoff ellipsoid roots analytic                                      | Affine/rational boundary roots under local camera model      | Affine face intersections                        | Certified polynomial/rational roots                         |
| **Perspective projection**       | Usually Jacobian/EWA approximation; affine full-line marginal exact                             | Trace rays directly, so no projection approximation required | Trace rays directly                              | Trace rays or fit projected support                         |
| **Depth ordering**               | Center/tile sort is heuristic for overlapping volumes                                           | Exact traversal of disjoint cells                            | Exact traversal of disjoint cells                | Exact support interval order if roots certified             |
| **Transmittance**                | Single primitive can be analytic; overlapping colored Gaussian volume generally not closed form | Exact for constant/P1 cell fields with suitable formulas     | Exact for P1 extinction and polynomial emission  | Usually certified quadrature                                |
| **Standard 3DGS blending**       | Approximate physical volume semantics                                                           | Not applicable                                               | Not applicable                                   | Backend choice                                              |
| **Finite pixel**                 | Numerical or atlas-integrated                                                                   | Numerical or atlas-integrated                                | Numerical or atlas-integrated                    | Numerical or atlas-integrated                               |
| **Exposure / rolling shutter**   | Exact in camera map; integral numerical or atlas-based                                          | Same                                                         | Same                                             | Same                                                        |
| **Interior geometry gradient**   | Exact for the chosen approximate renderer                                                       | Exact away from adjacency flips                              | Exact away from mesh degeneracy/topology changes | Exact or certified implicit-root derivative                 |
| **Visibility-boundary gradient** | Missed by ordinary AD for hard/order events                                                     | Requires face/silhouette boundary term                       | Requires silhouette/cell-boundary term           | Requires support/silhouette term                            |
| **Topology event**               | Split/delete decisions discrete                                                                 | Power-cell adjacency flip nondifferentiable                  | Remeshing/flip nondifferentiable                 | Root coalescence or knot-structure change nondifferentiable |

### 9.3 Gaussian projection exactness

Under an affine ray-bundle map

[
\Gamma(y,s)=L
\begin{bmatrix}
y\s
\end{bmatrix}
+b,
]

pulling back a spacetime Gaussian gives a Gaussian in ((y,s)). If depth is integrated over the entire real line in an optically thin additive model, marginalizing (s) gives an exact Gaussian in (y), with precision obtained by a Schur complement.

For a physical half-ray (s\ge0), the integral generally contains a Gaussian CDF factor and is not a pure Gaussian footprint. Under perspective or nonlinear camera motion, one UVT Gaussian is again only local or approximate.

Thus:

* affine full-line Gaussian marginalization is exact;
* physical ray support, occlusion, and perspective are not made exact merely because the footprint is Gaussian;
* a correct paper must not use “exact 4D projection” for the current general renderer.

Standard 3DGS uses projected alpha-composited billboards and can exhibit depth-order popping; StopThePop addresses per-pixel sorting, while EVER replaces billboard alpha compositing with exact emission-only ellipsoid volume rendering. ([arXiv][7])

3DGUT supports arbitrary nonlinear camera mappings and rolling-shutter effects by approximating Gaussian projection with sigma points; it demonstrates that broad camera support is established, although that is not a continuous camera-program compiler across many requested frames. ([arXiv][8])

### 9.4 Exact local pentatope transport

Within a cell and affine ray segment of length (L), suppose

[
\sigma(s)=a+bs,
\qquad
j(s)=\sum_{n=0}^{q}c_ns^n.
]

Then

[
A=e^{-aL-\frac12bL^2},
]

and

[
E=\sum_{n=0}^{q}c_nJ_n(a,b,L),
]

where

[
J_n(a,b,L)
==========

\int_0^L
s^n
e^{-as-\frac12bs^2},ds.
]

For (b>0),

[
J_0
===

\sqrt{\frac{\pi}{2b}}
e^{a^2/(2b)}
\left[
\operatorname{erf}
\left(
\frac{a+bL}{\sqrt{2b}}
\right)
-------

\operatorname{erf}
\left(
\frac{a}{\sqrt{2b}}
\right)
\right].
]

For (b<0), the corresponding expression uses (\operatorname{erfi}). Stable implementations should use scaled complementary error functions, Dawson-type forms, or series expansions to avoid catastrophic overflow and cancellation.

The recurrences are

[
aJ_0+bJ_1
=========

1-e^{-aL-\frac12bL^2},
]

and, for (n\ge1),

[
bJ_{n+1}+aJ_n
=============

## nJ_{n-1}

L^n e^{-aL-\frac12bL^2}.
]

For (b\approx0), switch to the (b=0) incomplete-gamma form or a local series rather than divide by (b).

This gives exact real-arithmetic segment transport. It does **not** by itself make the following exact:

* determination of all intersected cells;
* floating-point traversal;
* finite-pixel integration;
* exposure integration;
* topology gradients;
* a low-rank sensor-time approximation of the composed transfer.

### 9.5 Power foam exactness

With one shared metric (M),

[
D_i(X)-D_j(X)
]

is affine in (X), so pairwise boundaries are exact 4D hyperplanes. With an affine local camera chart, their ray intersections are rational functions of sensor-time.

The partition property gives point overlap one in cell interiors, but a ray can cross (D) cells. Radiant Foam’s efficient neighbor transition does not make total traversal constant; it makes the work per cell transition small. ([arXiv][3])

Geometry gradients are smooth while the active power-diagram combinatorics remain fixed. At an adjacency flip, the discrete topology changes. Any claim of complete differentiability must either:

* define a generalized derivative;
* smooth or relax the partition;
* use a topology-invariant feature backbone;
* or accept piecewise differentiability with rebuilds.

### 9.6 Finite pixels, exposure, and rolling shutter

Let (R_{p,k}\subset B) be the pixel and exposure support with filter (w_{p,k}). The actual output is

[
I[p,k]
======

\int_{R_{p,k}}
w_{p,k}(y)C(y),dy.
]

Once (C(y)) is represented on an event-free atlas patch by a polynomial basis, this integral may be:

* exact for compatible polynomial filters and domains;
* computed by preintegrated basis moments;
* or certified by adaptive quadrature.

If (R_{p,k}) crosses an event surface, it must be split or integrated with an event-aware rule.

Rolling shutter is not a separate representation problem. It is encoded in

[
t=t(u,v,\tau)
]

inside (\Gamma). Existing Gaussian work already models continuous camera motion, exposure, and rolling shutter, so the novelty must be amortizing the camera program rather than merely supporting those effects. ([arXiv][9])

---

## 10. Complexity theorems and proof sketches

### 10.1 The genuine visibility and trace events

The relevant quantities are different:

* **Point overlap**
  [
  k_{\rm point}
  =============

  \sup_{z\in M}
  #{i:z\in\operatorname{supp}W_i}.
  ]

* **Projected support overlap**
  [
  k_{\rm proj}
  ============

  \sup_{y\in B}
  #{i:\Gamma_y\cap\operatorname{supp}W_i\ne\varnothing}.
  ]

* **Ray depth or line-stabbing number**
  [
  d(y)
  ====

  #{\text{connected world intervals or cells hit by }\Gamma_y}.
  ]

* **Ordering-event count:** number of roots of pairwise depth predicates over the camera program.

* **Event-stratification complexity:** number and arrangement complexity of support, tangency, silhouette, order, traversal, and camera-chart event surfaces in (B).

Bounded point overlap does not bound line-stabbing: (n) disjoint compact cells can be arranged successively along one ray while point overlap remains one.

A foam has point overlap one but can have (d(y)=\Theta(n)).

A compact world representation does not guarantee compact observation complexity. With (n) overlapping layers whose depths are distinct linear functions of time, pairwise depth order can change at (\Theta(n^2)) distinct times. Therefore no universal theorem can remove dependence on genuine event complexity.

Visibility complexes, potentially visible sets, and output-sensitive hidden-surface algorithms already establish the broader idea that visibility can be precomputed or represented in ray/view space and that useful bounds must depend on output complexity. ([ACM Digital Library][10])

### 10.2 Theorem 1: finite sensor-time stratification

**Statement.**
Let (B) and the relevant ray-depth interval be compact. Suppose:

1. the world contains finitely many bounded-degree semialgebraic support, boundary, and cell predicates;
2. the camera map is piecewise semialgebraic of bounded degree;
3. appearance and transport coefficients are analytic inside support cells;
4. no predicate vanishes identically over an open set.

Then there exists a finite semialgebraic stratification

[
B=\bigsqcup_{\alpha}S_\alpha
]

such that, on each connected open stratum:

* the number and multiplicities of ray-boundary roots are constant;
* the identities and ordering of those roots are constant;
* the intersected cells or supports are fixed;
* traversal combinatorics are fixed;
* the resulting transfer is analytic.

**Proof sketch.**

Construct the critical set in (B\times D) from:

[
h_i(y,s)=0,
]

[
h_i(y,s)=\partial_s h_i(y,s)=0,
]

root-coincidence resultants, endpoint crossings, and pairwise order predicates. Project this set to (B). Tarski–Seidenberg preserves semialgebraicity under projection. On the complement, roots are simple and vary continuously; their identities and order cannot change without crossing the projected critical set. Hardt-type semialgebraic triviality yields a finite partition over which the fibers have constant topological type. ([JKMS][11])

**Novelty status.**
Finiteness is a standard consequence of real algebraic geometry. A practical rendering-specific predicate system, constructive GPU atlas, and useful output-sensitive bound would be new.

For Gaussian fields, use an (\varepsilon)-support ellipsoid or extend the argument to a suitable definable analytic setting. The exponential density itself does not create support-topology events.

### 10.3 Theorem 2: conditional output-sensitive heavy-work bound

**Statement.**
Assume fixed ambient dimension and bounded predicate degree, robust root and sign oracles, and an atlas with:

* (R) continuous support references;
* (E) emitted event objects or event cells;
* stable patches (a=1,\ldots,A);
* source trace depths (d_a);
* (q_a) adaptive backend evaluations per patch.

Then an output-sensitive compiler can be organized with work

[
W_{\rm compile}
===============

\widetilde O
\left(
S+G+R+E+\sum_a q_ad_a
\right)
]

and atlas storage

[
M_{\rm atlas}
=============

O
\left(
R+E+\sum_a r_a+\lvert\mathcal D\rvert
\right).
]

Requested evaluations have work

[
W_{\rm eval}
============

O\left(
\sum_{n=1}^{N}r_{a(y_n)}
\right)
+
\Omega(N).
]

**Proof sketch.**

* Build or refit the world accelerator once.
* Emit each continuous support reference once.
* Charge event processing to an emitted event object.
* Charge backend transfer work to each adaptive fit or verification node and its source trace depth.
* Charge evaluation only to local basis or trace entries.

There is no (T) in the compilation term when (T) merely changes the sample density on the same (B) and the same continuous (\Gamma).

**Qualification.**
This is a conditional algorithmic accounting result, not a claim that (R,E,A,r_a), or (d_a) are small. In the worst case they are not.

### 10.4 Theorem 3: shared-adjoint work and memory

**Statement.**
Let the atlas contain (K=\sum_a r_a) coefficients, and suppose every requested sample touches at most (r) coefficients. Then residual reduction can be implemented with:

[
W_{\rm reduce}=O(Nr),
]

and peak additional interaction memory

[
M_{\rm reduce}
==============

O(K+B_sr+Q_\Sigma),
]

where (B_s) is the streaming block size and (Q_\Sigma) is event-boundary quadrature storage.

A single compiler VJP then computes world and camera gradients with work proportional to the retained or recomputed compile dependency graph, rather than (N) times its size.

**Proof sketch.**

Each sample contributes a sparse linear form to coefficient adjoints. Associativity of addition permits block-local accumulation followed by a segmented reduction. After reduction, the chain rule is applied once to the map ((\theta,\Gamma)\mapsto a).

The theorem excludes unavoidable output or residual storage. Those can also be streamed if the loss permits.

### 10.5 Theorem 4: exact local transport lemma

The recurrence in Section 9 follows from

[
\frac{d}{ds}
\left(
s^n e^{-as-\frac12bs^2}
\right)
=======

## n s^{n-1}e^{-as-\frac12bs^2}

\left(
as^n+bs^{n+1}
\right)e^{-as-\frac12bs^2}.
]

Integrating over ([0,L]) yields the stated recurrence. Hence P1 extinction and polynomial premultiplied emission admit exact segment transfer using one special-function base moment and recurrence.

This is mostly standard analysis; the rendering-specific contribution would be integrating it into a 4D event atlas and its reverse mode.

### 10.6 Theorem 5: transfer-error propagation

Let

[
M_i=(E_i,A_i)
]

denote ordered segment transfers, with (0\le A_i\le1), and let (\widehat M_i) satisfy

[
|E_i-\widehat E_i|\le\varepsilon^E_i,
\qquad
|A_i-\widehat A_i|\le\varepsilon^A_i.
]

If every radiance arriving from behind a segment is bounded by (C_{\max}), then recursively,

[
|C_i-\widehat C_i|
\le
\varepsilon^E_i
+
C_{\max}\varepsilon^A_i
+
|C_{i+1}-\widehat C_{i+1}|.
]

Therefore,

[
\boxed{
|C-\widehat C|
\le
\sum_i
\left(
\varepsilon^E_i+C_{\max}\varepsilon^A_i
\right).
}
]

A sharper matrix form follows from telescoping:

[
\left|
\prod_iM_i-\prod_i\widehat M_i
\right|
\le
\sum_i
\left(
\prod_{j>i}|M_j|
\right)
|M_i-\widehat M_i|
\left(
\prod_{j<i}|\widehat M_j|
\right).
]

For optical depth (\tau,\widehat\tau\ge0),

[
|e^{-\tau}-e^{-\widehat\tau}|
\le
|\tau-\widehat\tau|.
]

These bounds allow support, quadrature, and basis errors to be converted into a final image-space certificate.

### 10.7 Analytic approximation and event distance

Suppose transfer on a normalized patch extends analytically to a complex polyellipse with radius (\rho>1). Standard Chebyshev estimates give exponential coefficient decay of the form

[
|\mathcal T-\mathcal T_m|_\infty
\le
C\rho^{-m}.
]

The nearest event surface, projection pole, or root collision limits (\rho). Thus high rank is not merely an implementation nuisance; it is a quantitative signal that the patch should be split along an event or the world primitive should be split.

### 10.8 Atlas-stability lemma

If every support, order, and root predicate in a patch has margin at least (\gamma), and a parameter update satisfies

[
\max_e L_{e,\theta}|\Delta\theta|<\gamma,
]

then the patch’s active set, root multiplicities, and ordering remain unchanged.

This is elementary continuity, but exposing and using the resulting trust radius could be an important practical contribution for training-time incremental compilation.

### 10.9 Camera-pushforward complexity metric

The proposed scalar

[
E_\varepsilon+\sum_l r_{l,\varepsilon}d_l
]

captures two important effects but omits references, certificate work, boundary integration, atlas memory, and fallback.

A better intrinsic object is a **complexity signature**

[
\mathbf K_\varepsilon
=====================

\left(
R,;
E,;
A,;
H,;
K,;
Q_\Sigma,;
F
\right),
]

where

[
H=\sum_a q_ad_a,
\qquad
K=\sum_a r_a,
]

and (F) is the expected cost of any unresolved region.

A hardware-calibrated scalar can then be

[
\begin{aligned}
\kappa_{\varepsilon,\rho}(W,\Gamma)
={}&
c_RR+c_EE+c_AA
+c_H\sum_a q_ad_a\
&+
c_K\sum_a r_a
+c_QQ_\Sigma
+c_F\int_{F}\rho(y)c_{\rm backend}(y),dy.
\end{aligned}
]

Here (\rho(y)) is either a uniform sensor-time measure or an expected query distribution.

#### Can it be bounded?

* It is finite under the bounded-degree compact assumptions.
* Coarse semialgebraic bounds exist.
* Those bounds are too pessimistic for systems work.
* No useful universal small bound exists without separation, curvature, opacity, or event-density assumptions.

#### Can it be measured?

Yes. Every component is emitted or counted by the compiler and should be logged during training.

#### Can it be differentiated?

The exact discrete quantities are not smoothly differentiable. Useful relaxations include:

* soft support occupancy;
* soft projected-overlap counts;
* penalties on small depth-order margins;
* camera projection Jacobian and Hessian norms;
* local singular-value-tail penalties on sampled transfer matrices;
* a smoothed event-density proxy
  [
  \int_B
  \delta_\eta(h(y))
  |\nabla_yh(y)|,dy;
  ]
* soft rank surrogates such as nuclear norms.

#### World split versus camera-chart split

For a problematic patch, estimate

[
\frac{-\Delta\kappa}{\Delta\text{learned bytes}}
]

for a world-primitive split and

[
\frac{-\Delta\kappa}{\Delta\text{compiled bytes}}
]

for a camera-chart split. Choose the intervention with the better predicted reduction subject to the learned-byte budget.

This is more principled than regularizing velocity or imposing a hard support floor without considering the camera pushforward.

---

## 11. Prior-art and novelty boundary

### 11.1 Adversarial audit

| Proposed headline                                                                                                                  | Classification                                                    | Reason                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| “Dynamic scenes are global objects in 4D spacetime”                                                                                | **Established**                                                   | Spacetime ray tracing treated an animation as a static 4D structure in 1988; modern dynamic fields and native 4DGS do the same. ([ACM Digital Library][12])                                                           |
| “A 4D Gaussian encodes linear motion through space-time covariance”                                                                | **Established**                                                   | Native 4D Gaussian works already use full spacetime covariance and timestamp conditioning. ([arXiv][1])                                                                                                               |
| “A Gaussian tube with velocity is more expressive than SPD(4)”                                                                     | **False**                                                         | The families are bijectively identical when (C) is full SPD(3).                                                                                                                                                       |
| “Conditioning an SPD(4) Gaussian yields a moving 3D Gaussian”                                                                      | **Established**                                                   | Direct multivariate-Gaussian algebra and prior 4DGS implementations.                                                                                                                                                  |
| “Continuous-time or nonlinear Gaussian motion”                                                                                     | **Established in several forms**                                  | Spacetime Gaussian Feature Splatting, FreeTimeGS-style methods, RetimeGS, and 2026 dynamic variants cover polynomial, velocity-field, spline, lifespan, or implicit temporal behavior. ([CVPR Open Access][2])        |
| “Low-rank or factorized spacetime fields”                                                                                          | **Established**                                                   | K-Planes, Fourier PlenOctrees, and related dynamic fields factorize space and time. ([CVPR Open Access][13])                                                                                                          |
| “Camera motion, exposure, and rolling shutter in Gaussian rendering”                                                               | **Established**                                                   | Gaussian Splatting on the Move and 3DGUT explicitly handle these camera effects. ([arXiv][9])                                                                                                                         |
| “Exact Gaussian/ellipsoid rendering”                                                                                               | **Established for specific primitive semantics**                  | EVER performs exact emission-only ellipsoid volume rendering rather than billboard alpha compositing. ([arXiv][14])                                                                                                   |
| “Motion-blur-aware Gaussian splatting”                                                                                             | **Established lineage**                                           | EWA splatting and later moving-camera Gaussian methods already address filtering and blur. ([ACM Digital Library][15])                                                                                                |
| “Global or regional visibility represented in ray/view space”                                                                      | **Established**                                                   | Visibility complexes and PVS preprocessing represent visibility across families of rays or view cells. ([ACM Digital Library][10])                                                                                    |
| “Direct ray-to-color/light-field distillation”                                                                                     | **Established**                                                   | R2L distills a volumetric teacher to a one-evaluation neural light field; HyperReel uses ray-conditioned sampling and a compact dynamic volume. ([arXiv][16])                                                         |
| “Differentiable visibility requires boundary terms”                                                                                | **Established**                                                   | Edge-sampling differentiable rendering explicitly samples the singular visibility contribution. ([Aaltodoc][6])                                                                                                       |
| “Differentiable 3D foam”                                                                                                           | **Established**                                                   | Radiant Foam, Power Foam, and Semantic Foam. ([arXiv][3])                                                                                                                                                             |
| “4D simplex space-time worlds”                                                                                                     | **Established outside neural rendering**                          | Space-time FEM work uses pentatopes, including changing topology. ([arXiv][4])                                                                                                                                        |
| “Exact tetrahedral radiance transport and geometry gradients”                                                                      | **Largely established in 3D**                                     | Radiance Meshes and DiffTetVR cover tetrahedral rendering and differentiable geometry. ([arXiv][17])                                                                                                                  |
| “Compile a camera program into a continuous sensor-time trace atlas whose heavy scene work is independent of output frame density” | **Plausibly novel**                                               | Existing work covers spacetime worlds, visibility structures, camera effects, and ray-space distillation separately; I found no primary source combining this exact complexity target and canonical-world separation. |
| “Reduce all frame residuals to atlas coefficients before one world-space geometry VJP”                                             | **Plausibly novel**                                               | Direct backward kernels exist, but the explicit coefficient reduction plus one shared compiler VJP is a narrower and stronger claim.                                                                                  |
| “Certified event-stratified transfer approximation with visibility-boundary adjoint”                                               | **Clearly novel only if it works**                                | The mathematical ingredients are established; the rendering-specific constructive system and measured complexity would be the contribution.                                                                           |
| “A trainable camera-pushforward complexity metric choosing world versus camera splits”                                             | **Plausibly novel**                                               | This is more specific than primitive compactness or occupancy regularization.                                                                                                                                         |
| “A 4D power foam beam-event compiler”                                                                                              | **Clearly novel only if event compression and gradients succeed** | Merely extending sites to four coordinates is not enough.                                                                                                                                                             |

### 11.2 The narrow novelty claim that survives

The defensible paper claim is:

> Given a canonical continuous spacetime world and a continuous camera program over a fixed physical interval, STAR compiles scene-dependent projection, support, visibility, and geometry differentiation into an adaptive sensor-time trace atlas. Increasing requested frame density then adds only bounded-rank atlas evaluation, residual reduction, and output work, while the expensive world and camera pushforward is shared.

This is narrower than “compile a renderer,” because camera-specific baking and light-field distillation are old ideas. It is stronger because it specifies:

* a canonical editable world;
* a continuous camera program, not a finite frame list;
* explicit event and chart complexity;
* a same-world baseline;
* forward and reverse amortization;
* no dense interaction tape;
* error and topology certificates.

### 11.3 What the literature does not permit you to claim

Do not claim novelty for:

* SPD(4) scene atoms;
* velocity extracted from spacetime covariance;
* conditioning or slicing at (t);
* continuous-time Gaussians in general;
* rolling shutter or exposure support;
* 4D world fields;
* ray-space visibility structures;
* differentiable visibility boundary sampling;
* 3D foam;
* pentatope space-time meshes;
* exact isolated Gaussian, ellipsoid, or tetrahedral segment integration.

The July 2026 Grassmannian work is especially important: it introduces rank-three spacetime Gaussian covariances whose time slices are rank-two surfels, showing that even “derive the moving 3D primitive from a more intrinsic spacetime plane” is now occupied prior art. ([arXiv][5])

---

## 12. Primary paper recommendation

### Decision

[
\boxed{
\textbf{Choose 1: UVT-STAR / sensor-time trace compilation.}
}
]

But the paper should define STAR as:

> **a camera-program compiler and shared-adjoint architecture with pluggable canonical-world backends.**

It should not define STAR merely as:

> a particular Gaussian rasterizer in ((u,v,t)).

### Why this has the strongest novelty

The world primitive space is crowded. Native 4D Gaussians, deformation models, polynomial trajectories, continuous retiming, temporal bases, and even rank-deficient spacetime surfels already exist. The open space is the **cross-frame renderer architecture**:

[
\text{continuous world}
+
\text{continuous camera program}
\longrightarrow
\text{shared observation object}
\longrightarrow
\text{many samples and one adjoint}.
]

The forward and reverse complexity target is also sharper than another quality-oriented dynamic-scene representation.

### Why it is implementable

You already have evidence for three difficult components:

1. a spacetime tube world;
2. shared UVT binning and rasterization;
3. a backward that avoids a dense pixel-frame-primitive workspace.

The internal fixed-camera data show:

[
T:8\to32
]

with STAR step time

[
0.024\to0.033\text{ s},
]

a (1.37\times) increase for (4\times) more frames, while the isolated dynamic-GSplat baseline increased (3.39\times). This is not a publication result, but it is exactly the kind of architectural signal that justifies attacking variable-camera compilation.

### The single issue most likely to kill it

The killer is not Gaussian expressivity. It is:

[
\boxed{
\text{variable camera motion causing chart, reference, event, or local-rank explosion.}
}
]

The closely related training-time killer is structural invalidation: if world updates force almost every atlas record to be rebuilt every step, the shared backward will not rescue end-to-end training.

### Parallel lane

Keep **4D World Foam** as a parallel lane with a narrow objective:

> Can one continuous camera-time beam atlas contain materially fewer cell-face events than the sum of independent-ray traversals, while retaining correct site gradients away from topology flips?

It is a valuable exact-visibility stress test for the compiler. It is not yet the primary representation.

### Reference lane

Use the **pentatope/corner complex** for:

* exact local transport tests;
* gradient verification;
* event-surface verification;
* error-certificate validation.

A small synthetic backend is enough. Do not begin by training a production-scale 4D pentatope world.

### Representation lane

Keep Bernstein swept-volume atoms as a later ablation. They should not delay the variable-camera compiler.

### What to abandon

Abandon or demote:

* “SPD(4) is the novel object”;
* “velocity-free native 4D” as a novelty claim;
* one monolithic UVT Gaussian for arbitrary camera programs;
* full 4D foam as the immediate replacement for STAR;
* full production pentatope training as the first milestone;
* primitive-count-only comparisons;
* fixed-camera-only evidence.

---

## 13. Minimal decisive implementation plan

### Stage 0: establish exact semantic parity

Implement one canonical conversion between:

[
(C,v,s_t,x_0,t_0)
\quad\leftrightarrow\quad
(\mu,\Sigma).
]

Tests must show:

* exponent equality at random spacetime points;
* identical conditional 3D means and covariances;
* identical rendered images after conversion;
* identical gradients under a common renderer.

Full SPD(4) versus unrestricted tubes is a **parity test**, not a representation-quality experiment. A quality difference means initialization, regularization, numerical conditioning, or code differs.

### Stage 1: same-world fixed-camera compiler ablation

Use exactly the same learned tubes.

**Baseline A**

1. condition at each requested time;
2. project and bin independently;
3. use ordinary per-frame gsplat;
4. run ordinary reverse.

**STAR B**

1. compile the full fixed camera-time interval once;
2. evaluate the same timestamps;
3. reduce descriptor adjoints;
4. run one world VJP.

Match:

* support cutoff;
* alpha semantics;
* ordering;
* precision;
* appearance;
* loss;
* learned bytes.

This turns the current timing evidence into a scientifically isolated result.

### Stage 2: continuous variable-camera descriptors

Represent:

* pose by a cubic (SE(3)) spline;
* intrinsics and distortion by low-order splines;
* rolling-shutter timing by an explicit sensor map;
* exposure by a continuous shutter interval.

Fit per-tube projected descriptors over camera-time charts. Do not input (T) poses to the compiler; input the same continuous camera program and vary only the evaluation samples.

Required outputs:

* continuous footprint envelope;
* descriptor coefficients;
* approximation error;
* chart condition number;
* candidate reference list independent of (T).

### Stage 3: coefficient-space shared reverse

Add a two-stage reverse:

[
\text{sample residuals}
\to
\text{descriptor coefficient adjoints}
\to
\text{world/camera parameters}.
]

Compare three modes:

1. dense/reference autograd;
2. direct float atomics;
3. deterministic segmented reduction.

This is the central reverse-mode contribution.

### Stage 4: event-aware chart refinement

Initially handle:

* support entry and exit;
* camera-plane or projection-conditioning failures;
* center-depth order swaps among overlapping splats;
* large footprint curvature;
* overflow.

Each failure should trigger:

* time/chart split;
* sensor patch split;
* composite-transfer mode;
* or world-primitive split.

Hard fallback is only a temporary diagnostic.

### Stage 5: exact or cellular proof-of-generality

Implement one small second backend:

* the existing 4D foam toy, or
* a small pentatope transport scene, or
* homogeneous compact swept ellipsoids.

It need not beat Gaussians at reconstruction quality. It must demonstrate that the compiler interface is not secretly equivalent to Gaussian projection.

### Stage 6: structural refresh during training

Attach margins and dependency lists to every atlas record.

Measure per step:

* coefficient-only refresh fraction;
* local structural invalidation fraction;
* global rebuild frequency;
* rebuild work;
* stale-certificate violations.

Without this stage, the architecture may only be useful for inference after training, which is still publishable but is a substantially narrower paper.

---

## 14. Benchmark matrix and kill criteria

### 14.1 Experiment 1: same-world renderer ablation

| Variable       | Baseline A                                             | STAR B                                | Held fixed                |
| -------------- | ------------------------------------------------------ | ------------------------------------- | ------------------------- |
| World          | Same learned spacetime tubes                           | Same tensors, bit-for-bit             | All learned parameters    |
| Time handling  | Slice and project each timestamp                       | Compile interval once                 | Requested timestamps      |
| Camera         | Initially fixed; then same continuous variable program | Same                                  | Pose, intrinsics, shutter |
| Blending/order | Existing gsplat semantics                              | Must reproduce them                   | Support and precision     |
| Backward       | Per-frame geometry work                                | Coefficient reduction plus shared VJP | Loss and residuals        |

This experiment isolates the compiler and backward architecture.

### 14.2 Experiment 2: same-compiler representation ablation

Use one compiler API and compare:

1. current structured STAR parameterization;
2. raw SPD(4) Cholesky parameterization;
3. one genuinely different backend: BSVA or cellular.

Important interpretation:

* Structured versus raw SPD(4) is a conditioning and optimization ablation unless the current (C) is constrained.
* It is not an expressivity comparison.
* The third backend is needed for a real representation comparison.

Report both:

[
\text{learned bytes}
\quad\text{and}\quad
\text{compiled atlas bytes}.
]

### 14.3 Experiment 3: camera-motion stress

Vary one axis at a time:

| Axis                    | Fixed quantity                          | Sweep                                            |
| ----------------------- | --------------------------------------- | ------------------------------------------------ |
| Requested frame density | Physical interval and camera spline     | (T=1,2,4,8,16,32,64,128)                         |
| Physical duration       | Sampling density                        | Short to long duration                           |
| Translation speed       | Orbit angle and duration where possible | Static to fast translation                       |
| Angular velocity        | Translation and duration                | Static to rapid rotation                         |
| Orbit extent            | Nominal angular speed                   | (15^\circ,45^\circ,90^\circ,180^\circ,360^\circ) |
| Rolling-shutter slope   | Camera path                             | Global to severe rolling shutter                 |
| Exposure                | Camera path                             | Instantaneous to long shutter                    |
| Event frequency         | World bytes and image quality           | Few to repeated crossings/disocclusions          |

The critical diagnostic is not only total runtime. Fit

[
t(T)=t_{\rm heavy}+N(T)c_{\rm eval}
]

and report the inferred heavy intercept and per-output slope.

### 14.4 Experiment 4: visibility pathology

Include synthetic scenes with known ground truth:

* two thin opaque sheets crossing in depth;
* many disjoint cells along one ray;
* repeated foreground shutters producing disocclusion;
* grazing tangencies;
* one large primitive passing close to the camera;
* transparent overlapping layers;
* a rotating anisotropic object;
* curved motion that one SPD(4) cannot fit;
* topology split and merge;
* a full 360-degree camera orbit;
* a scene with high point sparsity but high line-stabbing depth.

These should report both image error and event/atlas behavior.

### 14.5 Required report

For every run report:

* structural compile time;
* numeric refresh time;
* forward evaluation time;
* reverse residual-reduction time;
* compiler/world VJP time;
* total training-step time;
* peak output memory;
* peak interaction memory;
* learned bytes;
* compiled atlas bytes;
* continuous candidate references;
* summed sliced candidate references;
* mean, p95, and maximum local candidates;
* mean and p95 evaluator rank;
* stable, event, and unresolved patch fractions;
* chart count;
* support-event count;
* order-event count;
* boundary quadrature count;
* mean and p95 line-stabbing depth;
* local and global recompile frequency;
* deterministic versus atomic parity;
* image quality;
* gradient error;
* speedup versus (T);
* speedup versus angular velocity.

Quality comparison must use matched PSNR, SSIM, LPIPS, learned bytes, and common reference images—not matched primitive count.

### 14.6 Hard kill criteria

#### A. Frame-density invariance

For the same continuous camera program and physical interval:

[
\frac{W_{\rm heavy}(T=32)}
{W_{\rm heavy}(T=8)}
\le 1.10.
]

The following must be exactly or nearly invariant with (T):

* chart count;
* structural atlas records;
* continuous references;
* event count;
* coefficient count.

If any of these scale approximately with (T), frame count has been hidden in the compiler.

#### B. End-to-end speed

At (256^2) and (T=32), moderate variable camera motion:

* forward speedup over same-world slicing: at least (2.0\times);
* reverse speedup: at least (2.0\times);
* total training-step speedup: at least (1.7\times).

Inference compile break-even should occur by approximately (T\le8). Training break-even should occur by (T\le16).

If the same-world compiler cannot beat slicing, representation comparisons are irrelevant.

#### C. Output slope

The marginal cost per extra pixel-frame should be no more than roughly twice a standalone atlas-basis-evaluation plus output-write kernel.

A large world-size-dependent marginal slope indicates hidden per-sample geometry work.

#### D. Reverse memory

Excluding output and residual arrays:

[
\frac{M_{\rm interaction}(T=32)}
{M_{\rm interaction}(T=8)}
\le1.10.
]

There must be no allocation proportional to dense pixel-frame-candidate capacity.

#### E. Deterministic backward

* Bitwise repeatability in deterministic mode.
* Runtime no worse than (1.5\times) the direct-atomic target.
* Relative gradient difference from high-precision reference:
  [
  <10^{-5}
  ]
  for the pure reduction test.
* Cosine similarity:
  [

  > 0.9999.
  > ]

Failure means the current atomic speed is not yet a publishable training result.

#### F. Same-world image parity

For fixed-camera affine/parity tests:

* maximum absolute color error:
  [
  <10^{-4};
  ]
* gradient relative error:
  [
  <10^{-4}
  ]
  away from support and ordering events.

For approximate variable-camera compilation:

* PSNR against sliced renderer:
  [
  \ge 50\text{ dB};
  ]
* LPIPS difference:
  [
  \le 0.001;
  ]
* 99.9th percentile absolute channel error:
  [
  \le 2/255.
  ]

Against ground truth, STAR should lose no more than (0.1) dB PSNR or (0.002) LPIPS relative to the same-world sliced renderer.

#### G. Geometry and visibility gradients

Away from event surfaces:

* median finite-difference relative error:
  [
  <0.5%;
  ]
* p95:
  [
  <2%.
  ]

For finite-pixel tests crossing a known silhouette or order event, the boundary-aware gradient should agree with high-sample finite differences within (5%).

#### H. Candidate and rank compactness

Under moderate camera motion:

* median composite rank:
  [
  r\le8;
  ]
* p95:
  [
  r\le24;
  ]
* p95 direct trace-list size:
  [
  \le64.
  ]

At (T=32),

[
R_{\rm continuous}
\le
0.4
\sum_{k=1}^{32}R_k^{\rm sliced}.
]

The exact ratio is dataset-dependent, but the continuous representation must materially eliminate duplicated references.

#### I. Stable coverage

For moderate motion:

* certified stable or event-aligned coverage:
  [
  \ge98%;
  ]
* expensive unresolved fallback:
  [
  <2%.
  ]

For severe (180^\circ)–(360^\circ) stress:

* compiled coverage:
  [
  \ge90%;
  ]
* unresolved fallback:
  [
  <10%.
  ]

A persistent positive-measure expensive fallback is incompatible with the strongest asymptotic claim.

#### J. Angular motion

The speedup should remain:

* at least (1.5\times) through a (180^\circ) orbit;
* at least (1.2\times) through a (360^\circ) orbit.

Chart count may grow with angular extent but must remain invariant with (T).

If moderate camera rotation collapses almost every chart to per-frame width, the main paper claim fails.

#### K. Training-time validity

After warm-up:

* coefficient-only refresh for most records;
* fewer than (20%) of records structurally invalidated per step on median;
* global structural rebuild no more frequent than approximately once per twenty steps;
* all rebuild work below (25%) of total step time.

If most of the atlas is rebuilt every step, frame-amortized training is not achieved. An inference-only paper remains possible but must be framed that way.

#### L. Independent representation gate

BSVA remains alive only if, at matched quality:

* one BSVA replaces at least three linear SPD(4) atoms on curved-motion tests;
* learned bytes improve by at least (20%), or quality improves materially at equal bytes;
* p95 compiler rank grows by no more than (1.5\times).

Otherwise abandon it.

#### M. Cellular parallel-lane gate

The 4D foam lane continues only if:

[
E_{\rm compiled\ beam}
\le
0.5
\sum_k E_k^{\rm independent\ rays}
]

at (T=32) on a fixed physical interval, with:

* site-geometry finite-difference error below (2%) away from flips;
* at least (1.2\times) end-to-end speedup over the same-cell per-frame renderer;
* no hidden dense beam-by-cell table.

Failure makes 4D foam a dead end for this paper, though not necessarily as a representation generally.

---

## 15. Proposed paper title and abstract

### Proposed title

**STAR: Event-Stratified Sensor-Time Compilation and Shared Adjoints for Frame-Amortized Dynamic Rendering**

### Abstract

Dynamic scene representations increasingly encode a continuous four-dimensional world, yet rendering a denser video commonly repeats projection, visibility processing, sorting, and geometry differentiation at every timestamp. We introduce STAR, a compiler that maps a canonical spacetime world and a continuous camera program to an event-stratified sensor-time trace atlas. The atlas partitions pixel–shutter space only where projected support, ray traversal, camera conditioning, or visibility predicates change. Within each stable patch, STAR stores either a bounded trace list or a certified low-rank approximation of the composite emission–attenuation transfer operator. Requested frames are then materialized using atlas lookup, local basis evaluation, and output writes, while image residuals are first reduced into atlas-coefficient adjoints before one shared world- and camera-space reverse pass. We establish a finite stratification result for bounded-degree world and camera maps, an output-sensitive compilation bound parameterized by continuous support references, genuine events, traversal depth, and local rank, and error bounds for composited transfer approximations. We also expose event surfaces to account for visibility-boundary derivatives that ordinary pathwise differentiation omits. We instantiate STAR with spacetime Gaussian tubes—shown to be exactly a time-distinguished parameterization of full SPD(4) Gaussians—and evaluate against timestamp-sliced rendering using identical learned worlds. The resulting formulation separates canonical scene representation from camera-specific computation and targets frame-density-independent heavy work over a fixed physical interval.

[1]: https://arxiv.org/abs/2310.10642 "https://arxiv.org/abs/2310.10642"
[2]: https://openaccess.thecvf.com/content/CVPR2024/html/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.html "https://openaccess.thecvf.com/content/CVPR2024/html/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.html"
[3]: https://arxiv.org/abs/2502.01157 "https://arxiv.org/abs/2502.01157"
[4]: https://arxiv.org/abs/2210.09831 "https://arxiv.org/abs/2210.09831"
[5]: https://arxiv.org/abs/2607.10489 "https://arxiv.org/abs/2607.10489"
[6]: https://aaltodoc.aalto.fi/items/6cd5119a-d37b-4252-93e6-314bb5113f59 "https://aaltodoc.aalto.fi/items/6cd5119a-d37b-4252-93e6-314bb5113f59"
[7]: https://arxiv.org/abs/2402.00525 "https://arxiv.org/abs/2402.00525"
[8]: https://arxiv.org/html/2412.12507v2 "https://arxiv.org/html/2412.12507v2"
[9]: https://arxiv.org/abs/2403.13327 "https://arxiv.org/abs/2403.13327"
[10]: https://dl.acm.org/doi/10.1145/508357.508362 "https://dl.acm.org/doi/10.1145/508357.508362"
[11]: https://jkms.kms.or.kr/journal/download_pdf.php?doi=10.4134%2FJKMS.2007.44.1.179 "https://jkms.kms.or.kr/journal/download_pdf.php?doi=10.4134%2FJKMS.2007.44.1.179"
[12]: https://dl.acm.org/doi/10.1109/38.504 "https://dl.acm.org/doi/10.1109/38.504"
[13]: https://openaccess.thecvf.com/content/CVPR2023/html/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.html "https://openaccess.thecvf.com/content/CVPR2023/html/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.html"
[14]: https://arxiv.org/abs/2410.01804 "https://arxiv.org/abs/2410.01804"
[15]: https://dl.acm.org/doi/10.1109/TVCG.2002.1021576 "https://dl.acm.org/doi/10.1109/TVCG.2002.1021576"
[16]: https://arxiv.org/abs/2203.17261 "https://arxiv.org/abs/2203.17261"
[17]: https://arxiv.org/abs/2512.04076 "https://arxiv.org/abs/2512.04076"

