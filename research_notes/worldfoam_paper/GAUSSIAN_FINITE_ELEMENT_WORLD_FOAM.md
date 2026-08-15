# Gaussian Finite-Element WorldFoam

## Native 4D cell fields, exact ray transfer, approximation bounds, and a fair counterbaseline

**Status:** derived method candidate and implementation specification. The core
formulas below are checkable, but the method has not yet cleared native Metal,
training-quality, event-scaling, or publication-novelty gates.

**Date:** 2026-07-26

## 1. Decision in one paragraph

The useful new construction is not “put one Gaussian in every frame.” It is a
single finite-element field on a bounded four-dimensional spacetime complex:

\[
\sigma_h(X)
=
\sigma_\star \exp[-\ell_h(X)],
\qquad
X=(x,t)\in\mathbb R^3\times I.
\]

Each camera ray intersects a short word of spacetime cells. If the local
log-extinction \(\ell_h\) is quadratic in space, its restriction to a ray is a
quadratic polynomial, so optical depth has an exact error-function formula and
an analytic VJP. This plugs directly into the existing WorldFoam
camera-ray/ordered-transfer algebra.

However, the Gaussian/log choice is not yet the preferred winner. Because the
cells already provide compact support, a nonnegative polynomial or Bernstein
finite element has an even cheaper polynomial ray integral and may be more
stable. The first experiment must therefore compare positive direct-density
FEM against log-Gaussian FEM on exactly the same cells, tapes, parameters, and
camera rays.

## 2. Terminology correction

“Gaussian finite element” can mean several different things:

1. radial-basis/Gaussian shape functions;
2. a Gaussian quadrature rule;
3. a finite element whose *physical density* is an exponential quadratic.

Only the third is intended here. A less ambiguous name is:

> **log-quadratic finite-element extinction on a spacetime WorldFoam.**

The world quantity is optical extinction, not a normalized probability
density. With physical ray arclength \(s\),

\[
[\sigma]=\mathrm{length}^{-1},
\qquad
\tau=\int \sigma\,ds
\]

is dimensionless. The exponent \(\ell_h\) is dimensionless. If world time and
world space participate in the same cell metric, a declared conversion scale
\(v_t\) is mandatory:

\[
\widetilde X=(x,v_t(t-t_{\mathrm{ref}})).
\]

Without \(v_t\), a four-dimensional Euclidean distance incorrectly adds
seconds squared to meters squared.

## 3. Native world object

Let

\[
M=\Omega\times I
\]

be the bounded spacetime domain, with \(\Omega\subset\mathbb R^3\) and time
interval \(I\). Let \(\mathcal K_h\) be a conforming four-dimensional cell
complex covering \(M\).

The minimal scene fields are:

\[
\sigma_h:M\to\mathbb R_{\ge 0},
\qquad
c_h:M\times S^2\to[0,1]^C.
\]

Here \(\sigma_h\) is extinction and \(c_h\) is emitted color or a feature
decoded into color. A first exact-transfer implementation should use a
constant color \(c_K\) in each cell. View-dependent appearance is a later,
separable basis problem.

For a local cell \(K\), define:

\[
\ell_K(X)=\sum_{a\in\mathcal N(K)}L_a N_a(X),
\qquad
\sigma_K(X)=\sigma_\star e^{-\ell_K(X)}.
\]

The \(N_a\) are finite-element shape functions and the \(L_a\) are log-density
coefficients. Cell support is explicit:

\[
\sigma_h(X)
=
\sum_{K\in\mathcal K_h}
\mathbf 1_K(X)\,\sigma_K(X)
\]

for a discontinuous Galerkin field, or the usual unique conforming value for a
continuous field.

This is spacetime-native: there is one field on \(M\), not a list of
independent frame states and not a manually stored velocity parameter.
Motion is read from the evolution of level sets, maxima, and material
features through the \(t\) coordinate.

## 4. Which 4D cell complex?

### 4.1 Fixed 4D simplicial complex

A 4-simplex has five vertices. It provides standard barycentric P1/P2 elements,
straight spacetime faces, and robust ray-face intersection. It is the cleanest
mathematical starting point.

Advantages:

- standard finite-element approximation theory;
- P1/P2 basis functions are polynomial;
- fixed topology separates field learning from topology events;
- an affine camera ray intersects affine faces by scalar roots.

Costs:

- a general 4D triangulation can be large;
- sliver simplices damage conditioning;
- direct raster hardware does not natively traverse 4-simplices.

### 4.2 Native 4D power diagram

Let sites \(S_i\in\mathbb R^4\), weights \(w_i\), and spacetime metric
\(G\succ0\) define:

\[
K_i
=
\left\{
X:
(X-S_i)^\top G(X-S_i)-w_i
\le
(X-S_j)^\top G(X-S_j)-w_j,\ \forall j
\right\}.
\]

Pairwise comparisons cancel quadratic terms:

\[
2(S_j-S_i)^\top GX
\le
S_j^\top GS_j-S_i^\top GS_i+w_i-w_j,
\]

so every cell is a convex 4D polytope with affine faces. This matches
WorldFoam ownership semantics well.

Its time slices are structured but restricted. With a block-diagonal metric
and sites \(S_i=(p_i,\tau_i)\), the spatial face normals are fixed by
\(p_j-p_i\), while the effective right-hand side varies affinely with \(t\).
Thus faces translate through time but do not realize arbitrary rotation or
curvature inside one fixed diagram.

### 4.3 Independently moving 3D power diagrams

An apparently more flexible alternative defines a 3D power diagram at each
time from moving sites \(p_i(t)\). Its spacetime owner set need not be a convex
polytope. Even in one spatial dimension, choosing

\[
p_1(t)=t,\qquad p_2(t)=-t
\]

gives an equal-distance boundary

\[
(x-t)^2=(x+t)^2
\quad\Longleftrightarrow\quad
xt=0.
\]

That boundary is the union of two axes in \((x,t)\), not one affine face, and
the associated owner regions are not a standard convex spacetime complex.
This may be expressive, but it forfeits the simplest event compiler.

### 4.4 Recommendation

Use a fixed 4D simplicial complex for the first mathematical/CPU fixture.
Use either its direct representation or a triangulated fixed 4D power complex
for the first native renderer. Do not begin by differentiating topology or
independently rebuilding a 3D foam at every frame.

## 5. Finite-element choices and degrees of freedom

For a 4-simplex:

| Element | Local scalar DOFs | Local form |
|---|---:|---|
| P0 | 1 | constant |
| P1 | 5 | total degree \(\le1\) in \(x,y,z,t\) |
| P2 | 15 | total degree \(\le2\) in \(x,y,z,t\) |

The P2 count follows from:

\[
\dim \mathbb P_2(\mathbb R^4)
=
\binom{4+2}{2}
=15.
\]

These are field coefficients, not a splat center/rotation/scale tuple.

### 5.1 An important expressivity limit of total-degree P2

Write a general total-degree P2 log field as:

\[
\ell(x,t)
=
\tfrac12 x^\top A x
+t\,b^\top x
+d^\top x
+\tfrac12\gamma t^2
+\delta t+\epsilon.
\]

At fixed \(t\), its spatial Hessian is:

\[
\nabla_x^2\ell(x,t)=A,
\]

which is constant in time. If \(A\succ0\), the conditional spatial ridge is:

\[
x_\star(t)=-A^{-1}(bt+d),
\]

which moves affinely. A single total-degree P2 element therefore supports:

- affine within-cell motion;
- fixed spatial covariance/orientation;
- quadratic temporal amplitude.

It does **not** support within-cell rotating or changing spatial covariance.
The aggregate field can still curve or change scale across multiple cells, but
that is mesh approximation, not one-element closure.

### 5.2 Richer tensor-product element

To retain exact quadratic-in-ray integration while allowing time-varying
spatial quadratic coefficients, use:

\[
\ell_K(x,t)
=
\sum_{m=0}^{q}t^m
\left[
\tfrac12x^\top A_mx+b_m^\top x+c_m
\right].
\]

This is \(\mathbb P_2(x)\otimes\mathbb P_q(t)\) with:

\[
10(q+1)
\]

local scalar coefficients because a quadratic in three spatial variables has
\(\binom{3+2}{2}=10\) coefficients.

For each fixed \(t\), restriction to an affine spatial ray remains quadratic
in depth, so the exact optical-depth formula survives. Time-varying
\(A(t)\), \(b(t)\), and \(c(t)\) permit changing local scale/orientation and a
derived nonlinear ridge:

\[
x_\star(t)=-A(t)^{-1}b(t)
\]

whenever \(A(t)\succ0\).

The tradeoff is higher local state and more expensive coefficient evaluation.
Start with \(q=1\) only after P2 total-degree establishes the baseline.

## 6. Continuity across cells

Continuous extinction is physically pleasant but not required for a valid
Beer-Lambert integral. Bounded piecewise density with jumps is integrable, and
its interface crossings are already explicit WorldFoam events.

### 6.1 C0 conforming log field

Sharing finite-element nodal coefficients across adjacent cells makes
\(\ell_h\) continuous, hence \(\sigma_h\) continuous and positive.

For two quadratic polynomials \(\ell_i,\ell_j\) separated by affine face

\[
h(X)=n^\top X-\beta=0,
\]

C0 continuity is equivalent to divisibility:

\[
\ell_i(X)-\ell_j(X)
=
h(X)(a^\top X+b).
\]

### 6.2 C1 condition

For quadratic pieces, equal value and equal normal gradient on the face imply:

\[
\ell_i(X)-\ell_j(X)=c\,h(X)^2.
\]

C1 is not needed for a first renderer. It increases coupling, complicates
local adaptation, and does not eliminate visibility or support events.

### 6.3 Discontinuous Galerkin field

A DG field gives independent cell coefficients. It is simple to optimize and
matches owner-run execution. Add jump regularization only if measured seams or
ill-conditioning require it:

\[
\mathcal L_{\mathrm{jump}}
=
\lambda_J
\sum_{F}
\int_F [\ell_h]^2\,dA.
\]

P0 and DG-P1/P2 are therefore essential baselines, not merely degenerate
versions to omit.

## 7. Camera rays as fibers and camera charts as gauges

For a camera program, let:

\[
\Gamma(u,v,t,s)
=
\big(o(u,v,t)+s\,d(u,v,t),t\big)
\]

map sensor coordinate \((u,v)\), observation time \(t\), and ray parameter
\(s\) into world spacetime.

The ray is the fiber over \((u,v,t)\). A depth gauge is a coordinate choice on
that fiber. If \(s\) is not physical arclength, the pulled-back extinction is:

\[
\widetilde\sigma(u,v,t,s)
=
\sigma_h(\Gamma(u,v,t,s))
\left\|\frac{\partial x}{\partial s}\right\|.
\]

Under a monotone reparameterization \(s=s(\zeta)\):

\[
\widetilde\sigma_\zeta\,d\zeta
=
\sigma_h(\Gamma)\left\|\frac{\partial x}{\partial s}\right\|
\left|\frac{ds}{d\zeta}\right|d\zeta
=
\widetilde\sigma_s\,ds.
\]

Therefore optical depth is gauge invariant:

\[
\tau
=
\int \widetilde\sigma_s\,ds
=
\int \widetilde\sigma_\zeta\,d\zeta.
\]

The camera itself is not “a gauge.” More precisely:

- the camera program defines the observation/ray bundle;
- each ray is a fiber;
- ordinary depth, inverse depth, log depth, or a projective coordinate is a
  gauge/trivialization on that fiber.

This is exactly compatible with the existing camera-as-program,
ray-as-fiber, camera-chart-as-gauge WorldFoam design.

## 8. Exact optical depth for a log-quadratic segment

Fix observation time \(t\), a cell intersection
\([s_-,s_+]\), and an affine world ray \(x(s)=o+sd\). Suppose:

\[
\ell_K(\Gamma(s))
=
a s^2+b s+c.
\]

Then:

\[
\tau_K
=
\sigma_\star
\int_{s_-}^{s_+}
e^{-(a s^2+b s+c)}\,ds.
\]

### 8.1 Positive quadratic coefficient

For \(a>0\):

\[
\boxed{
\tau_K
=
\sigma_\star e^{-c+b^2/(4a)}
\frac{\sqrt\pi}{2\sqrt a}
\left[
\operatorname{erf}
\left(
\sqrt a\left(s+\frac{b}{2a}\right)
\right)
\right]_{s_-}^{s_+}.
}
\]

### 8.2 Linear exponent

For \(a=0\) and \(b\ne0\):

\[
\boxed{
\tau_K
=
\sigma_\star e^{-c}
\frac{e^{-bs_-}-e^{-bs_+}}{b}.
}
\]

For \(a=b=0\):

\[
\boxed{
\tau_K
=
\sigma_\star e^{-c}(s_+-s_-).
}
\]

### 8.3 Negative quadratic coefficient

For \(a=-A<0\):

\[
\boxed{
\tau_K
=
\sigma_\star e^{-c-b^2/(4A)}
\frac{\sqrt\pi}{2\sqrt A}
\left[
\operatorname{erfi}
\left(
\sqrt A\left(s-\frac{b}{2A}\right)
\right)
\right]_{s_-}^{s_+}.
}
\]

The integral remains finite because the cell interval is bounded, but a
negative \(a\) means extinction grows toward the interval ends and the
\(\operatorname{erfi}\) branch can be numerically dangerous. A convex
log-extinction restriction \(a\ge0\) is therefore attractive but is a
representation constraint, not a mathematical necessity.

## 9. Stable numerical evaluation

The direct formulas can lose accuracy when:

- \(|a|(s_+-s_-)^2\ll1\);
- the two erf values are both close to \(1\) or \(-1\);
- \(b(s_+-s_-)\) is small in the linear branch;
- the completed-square prefactor overflows while the final difference is
  moderate;
- the interval is nearly tangent or has nearly zero length.

The shader needs explicit branches:

1. near-constant/linear series using `expm1`;
2. `erfc`/scaled-`erfcx` differences in large positive tails;
3. log-domain prefactor evaluation;
4. a negative-curvature rejection or bounded `erfi` path;
5. a finite interval-length and finite-optical-depth certificate.

A dimensionless switch variable is:

\[
\eta_a=|a|\Delta s^2,\qquad \Delta s=s_+-s_-.
\]

The threshold must be chosen from parity sweeps, not hard-coded from intuition.

## 10. Analytic coefficient VJP by moments

Define:

\[
M_n
=
\int_{s_-}^{s_+}
s^n e^{-(as^2+bs+c)}\,ds.
\]

Then:

\[
\frac{\partial M_0}{\partial a}=-M_2,\qquad
\frac{\partial M_0}{\partial b}=-M_1,\qquad
\frac{\partial M_0}{\partial c}=-M_0.
\]

For \(a\ne0\), integration by parts gives:

\[
\boxed{
M_{n+1}
=
\frac{
nM_{n-1}-bM_n
-\left[s^n e^{-(as^2+bs+c)}\right]_{s_-}^{s_+}
}{2a}.
}
\]

Thus \(M_0,M_1,M_2\) suffice for a quadratic coefficient VJP. If local
coefficients \((a,b,c)\) are affine functions of finite-element parameters
\(\theta\), then:

\[
\nabla_\theta\tau_K
=
\sigma_\star
\left(
-M_2\nabla_\theta a
-M_1\nabla_\theta b
-M_0\nabla_\theta c
\right).
\]

This avoids differentiating a numerically awkward erf difference directly.
The recurrence itself needs a separate near-\(a=0\) branch.

## 11. Moving endpoints and geometry gradients

For a moving cell intersection:

\[
\tau(\theta)
=
\int_{s_-(\theta)}^{s_+(\theta)}
\sigma(s,\theta)\,ds,
\]

Leibniz gives:

\[
\boxed{
d\tau
=
\int_{s_-}^{s_+}\partial_\theta\sigma\,ds
+\sigma(s_+)\,ds_+
-\sigma(s_-)\,ds_-.
}
\]

For one affine face constraint along the ray:

\[
h_j(\Gamma(s),\theta)
=
\alpha_j(\theta)+\beta_j(\theta)s=0,
\]

the candidate root is:

\[
s_j=-\frac{\alpha_j}{\beta_j},
\]

and, away from \(\beta_j=0\),

\[
ds_j
=
-\frac{\beta_j\,d\alpha_j-\alpha_j\,d\beta_j}{\beta_j^2}.
\]

The actual entry/exit endpoint is the active maximum/minimum over face roots.
This derivative is valid only while:

- the active face identity is fixed;
- the cell word is fixed;
- no face is parallel to the ray;
- no zero-length run is born or dies.

Winner swaps, tangencies, and topology changes require an explicit fallback,
subgradient convention, or atlas refresh. A fixed-topology VJP is useful but
must not be reported as a complete moving-boundary gradient.

## 12. Optical-transfer element and VJP

With constant cell color \(c_K\):

\[
\beta_K=e^{-\tau_K},
\qquad
m_K=(1-\beta_K)c_K.
\]

The element acts on background/behind color \(I_+\) as:

\[
I=(1-\beta_K)c_K+\beta_K I_+.
\]

Its differential is:

\[
d\beta_K=-\beta_K\,d\tau_K,
\]

\[
dm_K=\beta_Kc_K\,d\tau_K+(1-\beta_K)\,dc_K.
\]

For a front-to-back cell word, compose:

\[
(\beta_1,m_1)\otimes(\beta_2,m_2)
=
(\beta_1\beta_2,m_1+\beta_1m_2).
\]

This is the existing associative WorldFoam visibility monoid. Prefix/suffix
products give an \(O(R)\) VJP for a ray with \(R\) owner runs and parallel scan
structure across runs.

## 13. Appearance: where exact closure stops

Constant \(c_K\) makes each segment transfer exact from \(\tau_K\). If
\(c(s)\) varies arbitrarily:

\[
m_K
=
\int_{s_-}^{s_+}
\exp\!\left[-\int_{s_-}^{s}\sigma(r)\,dr\right]
\sigma(s)c(s)\,ds,
\]

and the inner cumulative integral of a log-quadratic contains erf. Multiplying
by a generic polynomial color does not normally return an elementary closed
form.

Safe first choices:

1. constant RGB/features per cell;
2. a small number of cells for spatial color variation;
3. a color basis in optical-depth coordinate only after it beats constant
   cells.

An optical-depth polynomial:

\[
c(\tau)=\sum_{n=0}^{p}c_n\tau^n
\]

has incomplete-gamma moments under \(e^{-\tau}\), but it is naturally
ray-oriented rather than obviously a camera-independent world material.

## 14. Approximation guarantees

Assume true extinction can be written:

\[
\sigma=\sigma_\star e^{-\ell}
\]

and the finite-element approximation obeys:

\[
\|\ell-\ell_h\|_{L^\infty(K)}\le\epsilon.
\]

This theorem requires strictly positive target extinction on the approximated
region. Exact vacuum makes \(-\log(\sigma/\sigma_\star)\) singular. Vacuum
should be represented by absent cells, an explicit support mask, or a positive
floor whose modeling error is reported.

Then pointwise:

\[
e^{-\epsilon}
\le
\frac{\sigma_h}{\sigma}
\le
e^\epsilon.
\]

For any ray segment in \(K\):

\[
\boxed{
e^{-\epsilon}\tau
\le
\tau_h
\le
e^\epsilon\tau.
}
\]

Consequently:

\[
|\tau_h-\tau|
\le
(e^\epsilon-1)\max(\tau,\tau_h).
\]

Since \(T(\tau)=e^{-\tau}\) has derivative magnitude at most one on
\(\tau\ge0\):

\[
|e^{-\tau_h}-e^{-\tau}|
\le
|\tau_h-\tau|.
\]

For a fixed cell word, identical colors in \([0,1]^C\), and per-segment
optical-depth perturbations \(\delta\tau_r\), each color channel has the
conservative bound:

\[
\boxed{
|I_h-I|
\le
\sum_r|\delta\tau_r|.
}
\]

The reason is that the derivative with respect to one segment optical depth is
a front-transmittance factor times a difference of two colors in \([0,1]\), so
its channelwise magnitude is at most one.

On a shape-regular mesh, standard interpolation gives, for sufficiently smooth
\(\ell\):

\[
\|\ell-I_h^p\ell\|_{L^\infty(K)}
\le
C_{\mathrm{shape}}h_K^{p+1}
|\ell|_{W^{p+1,\infty}(K)}.
\]

Combining these inequalities converts mesh size and polynomial degree into an
optical-depth and fixed-word image bound. It does not cover wrong cell words,
missing support, or camera-atlas fit error; those require separate terms.

## 15. The essential counterbaseline: positive direct-density FEM

The exponential is not free. On a bounded cell, compact support is already
provided by \(\mathbf1_K\), so a direct nonnegative polynomial density may be
strictly simpler.

On a simplex with barycentric coordinates \(\lambda_i(X)\), define the
Bernstein basis of degree \(p\):

\[
B_\alpha^p(X)
=
\frac{p!}{\alpha_1!\cdots\alpha_5!}
\prod_{i=1}^{5}\lambda_i(X)^{\alpha_i},
\qquad
|\alpha|=p.
\]

It satisfies:

\[
B_\alpha^p(X)\ge0,
\qquad
\sum_{|\alpha|=p}B_\alpha^p(X)=1.
\]

With coefficients \(c_\alpha\ge0\):

\[
\boxed{
\sigma_K(X)=\sum_{|\alpha|=p}c_\alpha B_\alpha^p(X)\ge0.
}
\]

Along an affine ray, \(\lambda_i(\Gamma(s))\) is affine in \(s\), hence
\(\sigma_K(\Gamma(s))\) is a polynomial. Optical depth is an elementary
polynomial antiderivative:

\[
\tau_K
=
\sum_{n=0}^{p}\gamma_n
\frac{s_+^{n+1}-s_-^{n+1}}{n+1}.
\]

Advantages over log-FEM:

- no `exp`, `erf`, `erfi`, or tail cancellation;
- coefficient gradients are linear;
- nonnegativity is structural;
- the same P1/P2 cell and event tapes can be reused;
- exact integral cost is lower.

Possible disadvantages:

- multiplicative density ranges may require more degree/cells;
- zero density is easy, but very large dynamic range may optimize poorly;
- coefficient positivity can bias approximation;
- direct polynomial interpolation gives additive rather than relative error.

This counterbaseline can invalidate the “Gaussian” part while preserving the
finite-element WorldFoam contribution. It must be implemented first-class.

The clean Bernstein positivity guarantee above is simplex-specific. On a
general power polytope, one needs a triangulation, a polytope-specific positive
basis, or a positivity certificate such as SOS. Log-FEM retains the practical
advantage that positivity is automatic on any cell.

## 16. Relationship to self-normalized convex-potential atoms

A compact convex-potential atom:

\[
\sigma_\theta(x,t)
=
\alpha(t)\big(1-[q_\theta(x,t)-\min_yq_\theta(y,t)]\big)_+^p
\]

has a connected compact spatial slice and one ray interval under strong
convexity. It is a useful overlapping primitive family.

It is not automatically a finite-element foam:

- its support may overlap other atoms;
- it does not define unique cell ownership;
- summing atoms changes the local transfer generator;
- its per-atom minimizer and root solver add work that a bounded cell does not
  need.

One could place such a field *inside* a cell, but then the cell already gives
support and the self-normalization needs to justify its cost through quality or
compression. Treat the convex atom as a separate later shader/representation,
not as the definition of Gaussian FEM WorldFoam.

## 17. Prior-art boundary

The following nearby work makes the novelty burden substantial:

- **Tetra-NeRF** uses Delaunay tetrahedra as an adaptive neural radiance-field
  representation: <https://arxiv.org/abs/2304.09987>.
- **Radiance Meshes** uses constant-density Delaunay tetrahedral cells and
  exact fast volume rendering by rasterization/ray tracing:
  <https://arxiv.org/abs/2512.04076>.
- **DiffTetVR** develops differentiable tetrahedral volume rendering and
  geometry derivatives: <https://arxiv.org/abs/2601.00114>.
- **Don't Splat Your Gaussians** derives closed-form ray transfer for compact
  volumetric kernels, including an Epanechnikov kernel:
  <https://arxiv.org/abs/2405.15425>.
- **From ex(p) to poly** replaces exponential splat kernels with compact
  polynomial/ReLU kernels for rendering speed:
  <https://arxiv.org/abs/2603.18707>.

Therefore none of these alone is a safe novelty claim:

- tetrahedral radiance fields;
- constant-density cell volume rendering;
- differentiable tetrahedral traversal;
- closed-form compact polynomial ray kernels;
- replacing exponentials with compact polynomials.

The potentially distinctive combination is:

> a camera-program-compiled **4D spacetime** cell complex with an exact
> finite-element optical-transfer basis, event-atlas temporal reuse, and a
> shared analytic backward across a long camera/time program.

That combination still requires a formal literature search and experiments;
it is not novel merely because the components are combined in a note.

## 18. Complexity statement

Let:

- \(C\) be the number of world cells;
- \(P\) the number of pixels per frame;
- \(T\) the number of requested frames;
- \(R_y\) the number of owner runs on ray \(y\);
- \(B\) the compiled atlas records;
- \(E\) the support/topology event count.

Explicit image output remains:

\[
\Omega(PT).
\]

For a compiled fixed-topology atlas, a useful decomposition is:

\[
W
=
O(C_{\mathrm{compile}}+B+E)
+O\!\left(\sum_{y\in\mathrm{outputs}}R_y\right)
+O(PT).
\]

The finite-element basis changes the constant cost of each run; it does not
remove ray transfer. Temporal improvement exists only if atlas construction
and reuse replace larger per-frame cell reconstruction/traversal work.

The paper must report:

\[
\frac{B}{CT},\qquad
\frac{E}{CT},\qquad
\frac{\sum_yR_y}{PT},\qquad
\text{fallback fraction},\qquad
\text{tape bytes per output pixel}.
\]

## 19. Minimal implementation ladder

### Stage 0: CPU/Torch analytic fixture

Implement on one fixed 4D simplex/tiny complex:

- P0 constant extinction;
- positive P1/P2 Bernstein extinction;
- log-P1 and log-P2 extinction;
- exact forward optical depth;
- coefficient VJP;
- endpoint VJP under fixed active faces;
- finite differences in float64;
- gauge reparameterization parity.

No Metal or training claim is needed to falsify the formulas.

### Stage 1: fixed-tape Metal transfer microkernel

Feed identical precomputed cell words/endpoints to shader variants. This
isolates material-basis cost from compiler cost:

1. P0;
2. positive P1;
3. positive P2;
4. log-P1;
5. log-P2.

Measure forward, backward, memory, and numerical parity.

### Stage 2: native spacetime cell compiler

Add:

- 4D face/event compilation through the camera program;
- active-face/endpoints tape;
- event/fallback certificates;
- fixed-topology geometry VJP;
- refresh behavior at winner swaps/tangencies.

### Stage 3: matched training and paper breadth

Run the same datasets, cameras, frame selections, pixel budget, parameter/byte
budgets, seeds, and metrics as World Tubes, Dynamic 3DGS, and current
WorldFoam.

## 20. Acceptance and kill criteria

Promote log-quadratic FEM only if it demonstrates at least one matched
advantage over positive polynomial FEM:

- better heldout quality per parameter/byte;
- fewer cells/runs at matched quality;
- better conditioning or optimization stability;
- lower event density;
- an expressive dynamic behavior that P2 positive density cannot match.

Kill or demote the Gaussian/log branch if:

- erf/exp evaluation dominates transfer cost;
- negative-curvature cells repeatedly overflow;
- positive P1/P2 matches quality with lower memory/time;
- appearance closure requires expensive per-ray quadrature;
- event/tape scaling remains effectively per-frame.

Preserve the spacetime FEM/WorldFoam idea if only the Gaussian exponent loses.

## 21. Current recommendation

The most disciplined next object is:

\[
\boxed{
\text{fixed 4D cell complex}
+
\text{constant color}
+
\{\text{positive P1/P2},\ \text{log-P1/P2}\}
+
\text{exact segment transfer}
+
\text{compiled camera-ray event tape}.
}
\]

This is compatible with camera rays as fibers and depth coordinates as gauges.
It directly attacks the current WorldFoam memory problem by replacing
per-frame cell state with a small native spacetime field. It is substantial
enough to become a WorldFoam representation section and ablation now. It
becomes a separate method paper only after the native compiler, analytic
backward, multi-scene quality, and measured temporal-reuse gates clear.
