# Self-Normalized Convex-Potential Atoms and Ordered Ray Transfer

## Audit of the second external formulation dump

**Status:** one genuinely distinct primitive candidate embedded in a largely
existing WorldFoam transfer/compiler formulation. Mathematics is partly
proved below; novelty and engineering performance remain open.

**Date:** 2026-07-26

**Source intake:** external dump, SHA-256
`bbafd893ee7579e8b07934b7df355b3a8fbcec970f92f505ce320afb8cf82a01`.

## 1. Executive verdict

The dump contains two very different contributions:

1. **Camera bundle + depth gauge + ordered optical transfer.** This is a good,
   more precise explanation of the current WorldFoam core. The transfer
   matrices, path-ordered exponential, alpha commutator, gauge invariance,
   event cells, and prefix/suffix differentiation were already recorded in
   `world_foam_reformulation.md`, `WORLD_FOAM_MATH_APPENDIX.md`,
   `WORLD_FOAM_PAPER_DRAFT.md`, and
   `proofs/depth_fiber_operator_ordering.md`. This is not a new paper.
2. **Self-normalized convex-potential spacetime atom.** This appears genuinely
   distinct from the local representations inspected. It is a compact,
   connected, smooth-slice primitive with position, velocity, curvature,
   orientation, and scale derived from one native field. It is worth
   implementing as a candidate. It is not automatically a WorldFoam, and it
   is not yet a publishable method.

The right action is to preserve the atom as a separate representation branch,
fold the transfer clarifications into WorldFoam, and benchmark the atom only
after cheaper finite-element/polynomial baselines exist.

## 2. Precise camera terminology

Let the observation base be:

\[
B=\{(u,v,t)\}.
\]

The camera program defines a ray bundle:

\[
\Gamma:B\times S\to M,
\qquad
\Gamma(u,v,t,s)
=
\big(o(u,v,t)+s\,d(u,v,t),t\big).
\]

The precise vocabulary is:

- **camera program:** defines the family of measurements/rays;
- **ray fiber:** the set \(\Gamma(\{y\}\times S)\) over one
  \(y=(u,v,t)\);
- **gauge/chart:** a local coordinate \(s\), inverse depth, log depth, or
  projective coordinate along that fiber;
- **compiler:** constructs reusable records over domains in \(B\).

Saying “the camera is a gauge” is a tolerable slogan but mathematically
imprecise. The camera creates the bundle; the gauge chooses coordinates in it.

## 3. What the commutator does and does not rule out

Represent one front-to-back alpha/color element by:

\[
g_i=(\beta_i,m_i)
=
(1-\alpha_i,\alpha_i c_i).
\]

Composition is:

\[
g_i\otimes g_j
=
(\beta_i\beta_j,m_i+\beta_i m_j).
\]

Swapping two elements changes the emitted term by:

\[
\begin{aligned}
\Delta m
&=
\alpha_i c_i+(1-\alpha_i)\alpha_jc_j\\
&\quad-
\alpha_jc_j-(1-\alpha_j)\alpha_ic_i\\
&=
\boxed{\alpha_i\alpha_j(c_i-c_j)}.
\end{aligned}
\]

Therefore two contributors commute only when:

\[
\alpha_i=0,\qquad \alpha_j=0,\qquad\text{or}\qquad c_i=c_j
\]

componentwise, apart from accidental cancellations. This proves that a
**commutative/order-free opacity-color aggregate** cannot reproduce arbitrary
colored visibility.

It does not, by itself, prove that a thin-layer tuple
\((\alpha,c,z)\) fails: exact depths can simply be sorted. Failure of one
representative depth requires an extended-profile counterexample.

Let \(R\) and \(B\) be red and blue, let each full primitive have optical depth
\(\tau>0\), and put \(r=e^{-\tau/2}\). In scene A, split the red primitive into
half-depth layers at \(z=0,2\) and put the full blue layer at \(z=1\). In scene
B, split blue around one full red layer. In both scenes each primitive retains
the same:

- total opacity \(1-e^{-\tau}\);
- constant color;
- mean depth \(\widehat z=1\).

Yet their rendered colors differ by:

\[
\boxed{
(1-r)^2(1-r^2)(R-B)\ne0.
}
\]

Thus total opacity, color, and representative/mean depth do not encode an
extended colored depth profile.

This derivation is correct, important, and already a central WorldFoam theorem.

## 4. Ordered transfer is the existing WorldFoam object

For extinction \(\lambda(s)\) and emitted source
\(\eta(s)=\lambda(s)c(s)\), define:

\[
A(s)
=
\begin{bmatrix}
-\lambda(s)I_C & \eta(s)\\
0 & 0
\end{bmatrix}.
\]

The ray transfer is:

\[
\boxed{
M(s_1,s_0)
=
\mathcal P
\exp\!\left(\int_{s_0}^{s_1}A(s)\,ds\right).
}
\]

Applied to behind/background color:

\[
\begin{bmatrix}I(s_0)\\1\end{bmatrix}
=
M(s_1,s_0)
\begin{bmatrix}I(s_1)\\1\end{bmatrix}.
\]

For piecewise-constant cells, this reduces exactly to the visibility monoid
scan. Its generator commutator is:

\[
[A_1,A_2]
=
\begin{bmatrix}
0 & \lambda_1\lambda_2(c_1-c_2)\\
0 & 0
\end{bmatrix}.
\]

This is the continuous counterpart of the discrete swap formula.

For ordinary scalar extinction plus emission, the full homogeneous matrix is
mainly explanatory. The structured pair:

\[
g=(\beta,m)
\]

and its associative monoid are the exact lower-dimensional solver and are
likely the better GPU implementation. A generic dense \(4\times4\) ODE kernel
would throw away useful structure.

### Terminology: transport, not normally holonomy

For an open interval of one camera ray, \(M(s_1,s_0)\) is ordered parallel
transport or a product integral. **Holonomy** normally refers to transport
around a closed loop. The repository already uses cell-complex holonomy for
closed adjacency loops. Calling every open-ray transfer “ray holonomy” risks
colliding with that distinct diagnostic.

Recommended terms:

- **ordered ray transfer**;
- **vertical parallel transport**;
- **ray-fiber product integral**.

Reserve “holonomy” for an explicitly closed loop.

## 5. Gauge covariance of ray transfer

Let \(s=s(\zeta)\) be an orientation-preserving depth-coordinate change.
Then:

\[
A_\zeta(\zeta)
=
A_s(s(\zeta))\frac{ds}{d\zeta}.
\]

Consequently:

\[
\mathcal P_\zeta
\exp\!\left(\int_{\zeta_0}^{\zeta_1}A_\zeta(\zeta)\,d\zeta\right)
=
\mathcal P_s
\exp\!\left(\int_{s_0}^{s_1}A_s(s)\,ds\right).
\]

This is ordinary coordinate reparameterization invariance of the
matrix-valued one-form \(A(s)\,ds\). It validates the camera-chart language,
but it is not a newly discovered physical law or an independent renderer.

A genuine change of basis in transfer state is a different transformation. If
\(\Psi=G(s)\Psi'\), then the connection coefficients transform as:

\[
A'
=
G^{-1}AG-G^{-1}\frac{dG}{ds}.
\]

Open-path transport is endpoint-covariant:

\[
U'(s_1,s_0)
=
G(s_1)^{-1}U(s_1,s_0)G(s_0),
\]

not literally invariant. The external dump should not conflate a depth
coordinate change with a state-basis gauge transformation.

## 6. The distinct proposed object

Let:

\[
q_\theta:\mathbb R^3\times I\to\mathbb R
\]

be smooth and uniformly strongly convex in space:

\[
\nabla_x^2 q_\theta(x,t)\succeq\lambda I,
\qquad
\lambda>0.
\]

Define the per-time minimum and minimizer:

\[
\mu_\theta(t)=\min_xq_\theta(x,t),
\qquad
r_\theta(t)=\arg\min_xq_\theta(x,t).
\]

The normalized excess potential is:

\[
D_\theta(x,t)=q_\theta(x,t)-\mu_\theta(t)\ge0.
\]

For integer \(p\ge1\), define extinction:

\[
\boxed{
\sigma_\theta(x,t)
=
\alpha_\theta(t)
\big(1-D_\theta(x,t)\big)_+^p,
\qquad
\alpha_\theta(t)\ge0.
}
\]

The name **self-normalized** means the field subtracts its own spatial minimum,
so:

\[
D_\theta(r_\theta(t),t)=0
\]

and the support threshold is always one potential unit above the ridge.

This object is native in spacetime in the useful sense that the learned object
is one scalar field \(q(x,t)\). Position and motion are derived from its
minimizer; they are not separately stored velocity parameters.

It is not a normalized probability distribution, and \(\alpha\) should be
called peak extinction amplitude, not probability mass or opacity without a
ray-length convention.

It is also **foliation-native rather than fully 4D symmetric**: the definition
singles out physical time and minimizes over each spatial slice. That is
physically reasonable for rendering, but it should be stated rather than
marketed as a rotation-symmetric four-dimensional body.

Strong convexity in \(x\) does not make the complete spacetime support convex.
For example:

\[
q(x,t)=10(x-t^2)^2
\]

is uniformly strongly convex in \(x\), and every time slice is an interval,
but its spacetime support is a curved tube around \(x=t^2\) and is not a convex
subset of \((x,t)\). This is desirable motion capacity, but it prevents using
all convex-4D-cell traversal results without modification.

## 7. A structurally safe parameterization

One sufficient family is:

\[
\begin{aligned}
q_\theta(x,t)
&=
\tfrac12x^\top G(t)x+\ell(t)^\top x+c(t)\\
&\quad+
\sum_{r=1}^{R}
w_r(t)\big(a_r(t)^\top x+b_r(t)\big)^{2d_r},
\end{aligned}
\]

with:

\[
G(t)=\lambda_0I+F(t)^\top F(t),
\qquad
\lambda_0>0,
\]

\[
w_r(t)=\epsilon+\operatorname{softplus}(\omega_r(t))
\quad\text{or}\quad
\epsilon+p_r(t)^2.
\]

Even powers with nonnegative weights are convex because
\(z\mapsto z^{2d}\) is convex. The quadratic term gives:

\[
\nabla_x^2q(x,t)\succeq G(t)\succeq\lambda_0I.
\]

Caveats:

- using \(p_r^2\) has a zero derivative at \(p_r=0\);
- high-degree even polynomials can overflow far outside support;
- global polynomials can create large condition numbers;
- evaluating every term before support rejection may be expensive;
- strong convexity forbids a single atom from representing disconnected or
  multiply centered matter.

Two further structural caveats are easy to miss:

1. Self-normalization makes the spatial support nonempty at every time for
   which \(\alpha(t)>0\). Finite lifetime must therefore come from a compact
   amplitude/support law for \(\alpha(t)\), not from \(D\) alone.
2. Polynomial \(q(x,t)\) does not imply polynomial \(r(t)\) or \(\mu(t)\).
   For example:

   \[
   q(x,t)=\tfrac12(1+t^2)x^2+tx
   \]

   is uniformly strongly convex, but:

   \[
   r(t)=-\frac{t}{1+t^2},
   \qquad
   \mu(t)=-\frac{t^2}{2(1+t^2)}.
   \]

   Quartic potentials generally give algebraic minimizer branches. A
   low-degree world polynomial does not automatically become a low-degree
   sensor-time trace; the compiler needs minimizer continuation,
   approximation, or certification.

## 8. Proven spatial-slice properties

Assume \(q(\cdot,t)\) is \(C^2\), coercive, and
\(\lambda\)-strongly convex.

### 8.1 Unique ridge

Strong convexity implies a unique minimizer \(r(t)\) satisfying:

\[
\nabla_xq(r(t),t)=0.
\]

### 8.2 Nonempty bounded support

The support slice is:

\[
S_t=\{x:D(x,t)<1\}.
\]

It contains \(r(t)\), so it is nonempty. Strong convexity gives:

\[
q(x,t)
\ge
q(r(t),t)+\frac{\lambda}{2}\|x-r(t)\|^2.
\]

Thus \(D(x,t)<1\) implies:

\[
\boxed{
\|x-r(t)\|<\sqrt{\frac{2}{\lambda}}.
}
\]

Therefore every slice is bounded by a known ball.

### 8.3 Convexity and connectedness

\(S_t\) is a strict sublevel set of a convex function, hence convex and
connected.

### 8.4 Smooth boundary

On \(D=1\), the spatial gradient cannot vanish. A vanishing gradient would be
the unique minimizer and would have \(D=0\). Therefore:

\[
\nabla_xD\ne0\quad\text{on }D=1.
\]

The implicit function theorem makes the support boundary a smooth
two-dimensional surface whenever \(q\) has the required differentiability.

These claims are real mathematical advantages of the construction.

## 9. Derived position, motion, curvature, orientation, and scale

Let:

\[
H(t)=\nabla_x^2q(r(t),t)\succ0.
\]

Differentiate the ridge equation:

\[
\nabla_xq(r(t),t)=0.
\]

The first derivative is:

\[
H(t)\dot r(t)+q_{xt}(r(t),t)=0,
\]

so:

\[
\boxed{
\dot r(t)=-H(t)^{-1}q_{xt}(r(t),t).
}
\]

This is a derived velocity, not a learned velocity coordinate.

Differentiating again gives:

\[
\boxed{
\ddot r
=
-H^{-1}
\left(
q_{xtt}
+2q_{xxt}\dot r
+q_{xxx}[\dot r,\dot r]
\right),
}
\]

with all derivatives evaluated at \((r(t),t)\).

The minimum obeys the envelope theorem:

\[
\boxed{
\dot\mu(t)=q_t(r(t),t).
}
\]

Its second derivative is:

\[
\boxed{
\ddot\mu(t)
=
q_{tt}
-q_{tx}H^{-1}q_{xt}.
}
\]

Near the ridge:

\[
D(r+\delta x,t)
=
\tfrac12\delta x^\top H(t)\delta x
+O(\|\delta x\|^3).
\]

Therefore:

- eigenvectors of \(H(t)\) give local principal orientation;
- radii are locally proportional to
  \(\sqrt{2/\lambda_i(H(t))}\);
- changes of \(H(t)\) give derived rotation and scale change.

This answers the original conceptual demand cleanly: the primitive can have a
moving center and time-varying orientation/scale while remaining one world
field, without separately storing \(x_0\), velocity, quaternion, and scale
curves.

The Hessian is only a **local** orientation/scale tensor. Nonquadratic level
sets need not share one global ellipsoidal orientation, and eigenvectors remain
gauge-ambiguous at repeated eigenvalues.

## 10. One interval per affine ray

Fix time \(t\) and an affine ray:

\[
x(s)=o+sd,\qquad d\ne0.
\]

Define:

\[
h(s)=1+\mu(t)-q(o+sd,t).
\]

Then:

\[
h''(s)
=
-d^\top\nabla_x^2q(o+sd,t)d
\le
-\lambda\|d\|^2<0.
\]

So \(h\) is strictly concave. Its positive set:

\[
\{s:h(s)>0\}
\]

is therefore empty or one open interval. This is an excellent compiler
property: each atom contributes at most one depth run to each ray.

It does not bound how many atoms hit a ray or how many atoms overlap.

## 11. Exact ray integral: what is true and what is overstated

Along the support interval \([s_-,s_+]\):

\[
\tau
=
\alpha(t)\int_{s_-}^{s_+}h(s)^p\,ds.
\]

If \(q(x,t)\) is polynomial in \(x\), then \(h(s)\) is polynomial in \(s\).
For integer \(p\), \(h(s)^p\) is polynomial and the antiderivative is
elementary once the endpoints are known.

The endpoint claim needs care:

- for quadratic \(q\), the roots are quadratic and cheap;
- for quartic \(q\), roots are algebraic but a stable GPU quartic solver is
  not automatically cheap;
- for degree \(>4\), no general formula by radicals exists;
- certified root isolation can be rigorous but is iterative;
- “exact algebraic optical depth” does not mean constant-cost exact Metal
  evaluation.

There is another hidden dependency: even if the ray polynomial is cheap after
\(\mu(t)\) is known, computing or compiling \(\mu(t)\) may dominate.

Thus the paper-safe statement is:

> Polynomial convex potentials reduce each accepted ray segment to polynomial
> integration with uniquely bracketed support roots. Low degree permits
> closed-form or tightly certified evaluation.

## 12. Endpoint differentiation and tangency regularity

For:

\[
\tau(\theta)
=
\alpha
\int_{s_-(\theta)}^{s_+(\theta)}
h(s,\theta)^p\,ds,
\]

Leibniz gives endpoint terms:

\[
h(s_+)^p\,ds_+-h(s_-)^p\,ds_-.
\]

At regular support roots, \(h(s_\pm)=0\). Therefore for \(p>0\) the endpoint
terms vanish:

\[
\boxed{
\partial_\theta\tau
=
(\partial_\theta\alpha)\int h^pds
+\alpha p\int h^{p-1}\partial_\theta h\,ds.
}
\]

This does not mean root computation is dispensable in forward evaluation or
that every geometry loss has zero boundary term. It is specific to this
zero-at-support density integral.

Near a generic tangency/support birth, the normal form is:

\[
h(s;\delta)
\approx
\delta-\frac{\kappa}{2}(s-s_0)^2,
\qquad
\kappa>0.
\]

Then:

\[
\tau(\delta)
\sim
C_{p,\kappa}\,(\delta_+)^{p+1/2}.
\]

For integer \(p\), this is \(C^p\) but generally not \(C^{p+1}\) at
\(\delta=0\). Increasing \(p\) smooths support birth but also flattens density
near the boundary and increases polynomial degree.

## 13. Parameter derivatives of the self-normalization

For any parameter \(\theta\), the envelope theorem gives:

\[
\partial_\theta\mu_\theta(t)
=
\partial_\theta q_\theta(r_\theta(t),t),
\]

because \(\nabla_xq(r,t)=0\). Therefore:

\[
\boxed{
\partial_\theta D_\theta(x,t)
=
\partial_\theta q_\theta(x,t)
-\partial_\theta q_\theta(r_\theta(t),t).
}
\]

This is attractive: derivatives of the minimizer cancel in the derivative of
the minimum value. However, evaluating \(r(t)\) remains necessary, and
backpropagating objectives that directly use ridge position or Hessian still
requires implicit differentiation through \(H^{-1}\).

## 14. Overlapping atom transfer versus foam ownership

If atoms overlap, total extinction and emission are:

\[
\lambda(x,t)=\sum_i\sigma_i(x,t),
\]

\[
\eta(x,t)=\sum_i\sigma_i(x,t)c_i(x,t).
\]

The transfer generator is:

\[
A(x,t)
=
\begin{bmatrix}
-\lambda I_C & \eta\\
0&0
\end{bmatrix}.
\]

Inside an overlap interval, one cannot generally evaluate each atom as a
separate thick segment in arbitrary order; they occupy the same depths.
Correct evaluation needs:

- integration of the summed generator;
- a subdivision/quadrature scheme;
- or an analytic closure for the combined fields.

WorldFoam owner cells avoid that particular ambiguity because exactly one cell
owns almost every world point. A ray then yields a cell word with disjoint
depth intervals.

Therefore:

\[
\boxed{
\text{compact convex-potential atoms} \ne \text{WorldFoam cells}.
}
\]

They can be:

1. an overlapping primitive renderer;
2. clipped to unique owner cells;
3. used as basis functions inside cells;
4. converted into a partition through a separate competition rule.

Each choice has different transfer and gradient mathematics. The external dump
slides between these cases and needs to choose one before implementation.

## 15. Certified sensor-time compiler

A conservative atom compiler can store, on each accepted sensor-time cell
\(C_\ell\):

\[
\mathcal R_\ell
=
\{
\text{candidate atom ids},
\text{support-root brackets},
\text{coefficient bounds},
\text{depth order/overlap word},
\text{error certificate},
\text{fallback flag}
\}.
\]

For a polynomial atom, interval arithmetic or Bernstein bounds can certify:

- strong convexity on the time/chart interval;
- empty versus nonempty ray support;
- one-root-pair brackets;
- polynomial coefficient ranges;
- absence of order/topology changes;
- transfer approximation error.

The compiler must not relabel support births and overlap swaps as “only
metadata.” They are still events, and their count can dominate:

\[
E=E_{\mathrm{support}}+E_{\mathrm{tangent}}
+E_{\mathrm{overlap}}+E_{\mathrm{chart}}+E_{\mathrm{fallback}}.
\]

The single-atom discriminant equations:

\[
h=0,\qquad \partial_sh=0
\]

detect support birth/death. A presegmented multi-atom compiler additionally
needs endpoint-coincidence and active-word events, or it must perform live
sorting/quadrature inside overlaps.

## 16. Ordered-transfer differentiation

For continuous transport:

\[
\partial_\theta M(s_1,s_0)
=
\int_{s_0}^{s_1}
M(s_1,s)
\partial_\theta A(s)
M(s,s_0)\,ds.
\]

This is the Duhamel/variation-of-constants formula. It is exact under standard
regularity assumptions. For cell words it reduces to the existing
prefix/suffix product VJP.

A generic stability bound needs explicit hypotheses. If, under one
submultiplicative norm:

\[
\|A(s)\|,\|\widehat A(s)\|\le M,
\qquad
\|\widehat A(s)-A(s)\|\le\delta
\]

on an interval of length \(L\), then one standard estimate is:

\[
\|\widehat U(L)-U(L)\|
\le
\delta L e^{ML}.
\]

The exact exponent depends on the convention and which propagators receive the
common \(M\) bound. A VJP error theorem also needs a separate bound on
\(\|\partial_\theta A-\partial_\theta\widehat A\|\); forward generator error
alone is insufficient.

## 17. Complexity and unavoidable failure regime

Let \(N\) atoms, \(P\) pixels, \(T\) times, \(S\) actual
atom-gauge-tile-event trace records, \(B\) persistent incidence payload, and
\(H\) queried ray/atom interactions. Explicit output still costs
\(\Omega(PT)\). A compiled renderer can target memory:

\[
O(NK+S K_{\mathrm{tr}}+B),
\]

and render work at least:

\[
\Omega(PT+H),
\]

plus certified quadrature/root work. Replacing \(S\) by a bare event count is
not valid unless the record multiplicity per split is itself bounded.

There is no unconditional sublinear theorem:

- an oscillating camera can force many chart/event splits;
- many atoms can intersect every ray;
- support births can occur at every sampled time;
- overlapping colors can force fine ordered integration;
- discriminants can approach zero over large sensor-time sets;
- root and minimizer solves can erase projection savings.

The useful claim is empirical and structural:

> on scenes whose continuous supports and transfer records have low event
> complexity, the compiler reuses world-to-ray work across many requested
> frames.

## 18. Prior-art risk

The atom is not safely novel based only on local absence. Close primary work
includes:

- **3D Convex Splatting**, which represents radiance fields with smooth convex
  primitives and a differentiable rasterizer:
  <https://arxiv.org/abs/2411.14974>.
- **Don't Splat Your Gaussians**, which gives closed-form ray transfer for
  compact volumetric kernels and introduces an Epanechnikov alternative:
  <https://arxiv.org/abs/2405.15425>.
- **From ex(p) to poly**, which uses compact polynomial/ReLU splat kernels:
  <https://arxiv.org/abs/2603.18707>.
- **Deformable Beta Splatting**, which uses bounded deformable beta kernels:
  <https://arxiv.org/abs/2501.18630>.
- **Splat the Net**, which uses bounded neural density primitives with exact
  analytic line integrals: <https://arxiv.org/abs/2510.08491>.
- **PhysConvex**, which develops dynamic convex radiance fields:
  <https://arxiv.org/abs/2602.18886>.

The candidate distinction is not merely “convex,” “compact,” “polynomial,” or
“dynamic.” A defensible novelty statement would need to isolate:

- self-normalization by the instantaneous spatial minimum;
- a globally safe strongly-convex spacetime parameterization;
- derived ridge motion/orientation/scale formulas;
- one-interval ray support with tangency smoothness;
- camera-program event compilation and shared analytic adjoint;
- demonstrated temporal compute/memory benefit.

A dedicated literature review must test each of those pieces.

## 19. Claim-by-claim classification

| Claim in the dump | Verdict |
|---|---|
| Camera program, ray fibers, depth gauges | Correct terminology refinement |
| Arbitrary colored overlap cannot use order-free alpha summaries | Correct; already WorldFoam core |
| Path-ordered transfer connection | Correct; already WorldFoam core |
| “Ray holonomy” | Better named open-ray transport/product integral |
| Gauge invariance | Correct reparameterization covariance; already present |
| Unconstrained determinant atom is risky | Reasonable critique |
| Strongly convex self-normalized atom | Distinct and mathematically useful candidate |
| Compact connected smooth slices | Proved under stated regularity/strong convexity |
| Derived position/velocity/curvature/orientation/scale | Correct local differential geometry |
| One support interval per affine ray | Proved |
| Exact polynomial ray integral | Correct after roots; GPU/root-cost claim still open |
| Endpoint terms vanish | Correct for \(p>0\) extinction integral at regular support roots |
| Tangency gives \(C^p\) birth | Correct for the generic quadratic tangency normal form |
| Certified event compiler is sublinear | Candidate architecture, not a theorem without event bounds |
| This is already a complete new paper pair | No; one existing paper core plus one untested primitive candidate |

## 20. Falsification ladder

### Math fixture

1. Random safe potentials; verify Hessian lower bound.
2. Newton solve for \(r(t)\); compare implicit \(\dot r,\ddot r\) with finite
   differences.
3. Intersect random rays; verify empty/one-interval support.
4. Compare analytic polynomial optical depth with high-accuracy quadrature.
5. Sweep tangency offset \(\delta\); fit slope \(p+1/2\) in log-log space.
6. Compare Duhamel/prefix VJP with float64 finite differences.

### Representation counterexamples

1. Two disconnected moving objects: one atom must fail or fill the gap.
2. A thin rotating nonconvex object: measure primitive count required.
3. Dense colored overlap: compare correct summed-generator integration with
   falsely ordered per-atom segments.
4. High polynomial degree: measure root solve, overflow, and conditioning.
5. Oscillatory camera/time coefficients: measure event explosion.

### Matched renderer test

Compare at equal parameter and byte budgets:

- full \(\operatorname{SPD}(4)\) World Tubes;
- P0/P1/P2 WorldFoam cells;
- log-P1/P2 Gaussian FEM cells;
- quadratic convex-potential atoms;
- quartic convex-potential atoms;
- per-frame Dynamic 3DGS.

Do not implement the quartic branch before the quadratic one demonstrates a
specific residual.

## 21. Paper recommendation

Do not call the full dump a new paper. Split it:

### Existing WorldFoam paper

Keep:

- camera program/ray fiber/depth gauge terminology;
- ordered product integral;
- commutator theorem;
- gauge covariance;
- event atlas;
- prefix/suffix and Duhamel backward.

These improve presentation but are extensions/cleanups of material already on
disk.

### New primitive candidate

Incubate:

\[
\text{self-normalized strongly-convex spacetime atom}
\]

as a separately named experiment. It earns its own method paper only if it:

1. survives the novelty audit;
2. has a native Metal forward and analytic backward;
3. beats quadratic/positive-polynomial cell and compact-kernel baselines;
4. shows lower temporal memory or compute at matched output and quality;
5. works on multiple real dynamic scenes and camera paths.

Until then, the honest label is:

> **proved representation proposal with an untested rendering/compiler
> hypothesis.**
