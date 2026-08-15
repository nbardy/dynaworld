# Foundations: 3DGS, Full Spacetime Gaussians, and Rotation

## Claim-status legend

- **Definition:** establishes notation or a modeling convention.
- **Proved:** follows from the derivation in this note.
- **Known theorem:** standard linear algebra or Lie-group fact.
- **Implementation fact:** verified against code or repository documentation.
- **Recommendation:** design judgment requiring experiments.
- **Open:** unresolved or convention-dependent.

## 1. Ordinary 3D Gaussian splatting

### 1.1 The geometric object

**Definition.** A peak-normalized 3D Gaussian kernel is

\[
g(x)=\alpha
\exp\!\left[-\frac12(x-\mu)^	op\Sigma^{-1}(x-\mu)\right],
\quad
\mu\in\mathbb R^3,
\quad
\Sigma\in\operatorname{SPD}(3).
\]

A symmetric \(3\times3\) matrix has

\[
3+2+1=\frac{3(3+1)}2=6
\]

independent entries. Positive definiteness restricts the allowed values but
does not reduce the manifold dimension. Ordinary full-anisotropy 3DGS geometry
therefore has 3 mean DOF and 6 covariance DOF, for 9 total.

“Width, height, and depth” name only the three principal standard deviations.
The other three covariance DOF orient those principal axes.

### 1.2 Why implementations use log-scales and a quaternion

Every SPD(3) covariance has the spectral form

\[
\Sigma=R(q)
\operatorname{diag}(s_x^2,s_y^2,s_z^2)
R(q)^\top,
\qquad R(q)\in SO(3),\quad s_j>0.
\]

Implementations commonly store unconstrained log-scales \(\ell_j\) and set

\[
s_j=e^{\ell_j},
\qquad
s_j^2=e^{2\ell_j}.
\]

Thus “log-scale” normally means the logarithm of a principal standard
deviation, not the logarithm of a covariance eigenvalue. Exponentiation makes
the widths positive without a constrained optimizer.

A quaternion stores four real components, but a rotation quaternion satisfies
\(\lVert q\rVert=1\), leaving three continuous DOF. The equality
\(R(q)=R(-q)\) is a discrete double-cover redundancy, not another continuous
constraint. If scales repeat, some orientation parameters are additionally
unidentifiable because rotating within an equal-eigenvalue subspace leaves
\(\Sigma\) unchanged.

The repository renderer constructs this covariance in
[`src/train/renderers/common.py`](../../src/train/renderers/common.py), and the
official 3DGS implementation likewise exponentiates scales, normalizes
quaternions, and sigmoid-activates opacity.

### 1.3 Why a Gaussian also has opacity

**Definition/implementation convention.** The covariance controls spatial
falloff; a separate learned scalar controls the peak:

\[
a_i(p)=\sigma(o_i)
\exp\!\left[-\frac12(p-\bar p_i)^\top Q_i(p-\bar p_i)\right].
\]

So opacity is position-dependent at a pixel, exactly as the question suggests,
but it is the product of two different things:

1. peak opacity \(\sigma(o_i)\);
2. covariance-controlled Gaussian falloff.

Standard splatting does not force every Gaussian to have equal integrated
mass. Making a Gaussian wider while holding its peak fixed increases its
integral. This peak-versus-mass convention becomes important when pushing a 4D
world Gaussian through ray depth.

With simple RGB, the parameter count is:

| Item | Stored floats | Effective DOF |
|---|---:|---:|
| Center | 3 | 3 |
| Log-scales | 3 | 3 |
| Raw normalized quaternion | 4 | 3 |
| Opacity logit | 1 | 1 |
| RGB | 3 | 3 |
| Total | 14 | 13 |

The original 3DGS uses spherical-harmonic appearance rather than only RGB, so
appearance cost depends on the chosen SH degree.

## 2. The full 4D spacetime Gaussian

Let \(y=(x,t)\in\mathbb R^3\times\mathbb R\). Define

\[
\rho(y)=\alpha
\exp\!\left[-\frac12(y-\mu_4)^\top
\Sigma_4^{-1}(y-\mu_4)\right],
\qquad \Sigma_4\succ0.
\]

### 2.1 Degree count

**Known theorem.** Symmetric \(n\times n\) matrices have dimension
\(n(n+1)/2\). Hence SPD(4) has 10 DOF. The mean has 4, so full spacetime
geometry has 14.

This is not merely the “3D six parameters expanded to eight.” Eight is the
axis-aligned count:

\[
\underbrace{4}_{\text{center}}
+
\underbrace{4}_{\text{axis widths}}
=8.
\]

An axis-aligned 4D covariance has no space-time cross terms and therefore no
moving conditional center. A moving Gaussian needs the three additional
\(xt,yt,zt\) covariances, and a generally oriented spatial cross-section needs
the remaining three spatial off-diagonal covariances.

### 2.2 Full block form

Partition the covariance as

\[
\Sigma_4=
\begin{bmatrix}
A&b\\
b^\top&c
\end{bmatrix},
\]

where \(A=A^\top\in\mathbb R^{3\times3}\), \(b\in\mathbb R^3\), and
\(c>0\). The ten covariance DOF visibly split as

\[
\underbrace{6}_{A}
+
\underbrace{3}_{b}
+
\underbrace{1}_{c}
=10.
\]

The marginal spatial covariance is \(A\). It is not the spatial covariance of
a time slice; that conditional covariance is the Schur complement derived
next.

## 3. Exact tube-equivalence theorem

### Theorem

For every \(\Sigma_4\in\operatorname{SPD}(4)\), define

\[
v=b/c,
\qquad
C=A-bb^\top/c.
\]

Then \(C\in\operatorname{SPD}(3)\), and for
\(\tau=t-t_0\), \(\eta=x-x_0\),

\[
\begin{bmatrix}\eta\\\tau\end{bmatrix}^{\!\top}
\Sigma_4^{-1}
\begin{bmatrix}\eta\\\tau\end{bmatrix}
=
(\eta-v\tau)^\top C^{-1}(\eta-v\tau)
+\frac{\tau^2}{c}.
\]

Conversely, every \(C\succ0\), \(c>0\), and \(v\in\mathbb R^3\) defines one
SPD(4) covariance through

\[
\Sigma_4=
\begin{bmatrix}
C+cvv^\top&cv\\
cv^\top&c
\end{bmatrix}.
\]

### Proof

Because \(\Sigma_4\succ0\), the Schur complement of \(c\) is
\(C=A-bb^\top/c\succ0\). Substituting \(b=cv\) and \(A=C+cvv^\top\) gives
the displayed covariance. Direct block inversion yields

\[
\Sigma_4^{-1}=
\begin{bmatrix}
C^{-1}&-C^{-1}v\\
-v^\top C^{-1}&c^{-1}+v^\top C^{-1}v
\end{bmatrix}.
\]

Multiplying out its quadratic form gives

\[
\eta^\top C^{-1}\eta
-2\tau\eta^\top C^{-1}v
+\tau^2v^\top C^{-1}v
+\tau^2/c,
\]

whose first three terms are
\((\eta-v\tau)^\top C^{-1}(\eta-v\tau)\).

For the converse, for every nonzero \((a,r)\),

\[
\begin{bmatrix}a\\r\end{bmatrix}^{\!\top}
\Sigma_4
\begin{bmatrix}a\\r\end{bmatrix}
=a^\top Ca+c(v^\top a+r)^2>0.
\]

Thus the reconstructed matrix is SPD. Both transformations are inverse, so
the parameterizations are bijective. \(\square\)

### Corollary: the repository's precision normal form is full when \(C\) is full

Let \(Q=C^{-1}\) and \(\lambda_t=c^{-1}\). Then

\[
\Lambda_4=\Sigma_4^{-1}=
\begin{bmatrix}
Q&-Qv\\
-v^\top Q&v^\top Qv+\lambda_t
\end{bmatrix}.
\]

This is exactly the Phase 2 World Tube formula. If \(Q\) is a full SPD(3)
matrix, it represents every SPD(4) covariance. In the active scaffold,
`precision_xy[2]` is not a full SPD(3), which is why that code is restricted.

### Interpretation

The three values in \(v\) are not an extra motion mechanism glued onto the
Gaussian. They are the three space-time cross-covariance DOF expressed in
physical units:

\[
v=\frac{\Sigma_{xt}}{\Sigma_{tt}}.
\]

At time \(t\), the Gaussian's center is

\[
m(t)=x_0+v(t-t_0),
\]

its conditional covariance is exactly \(C\), and its peak multiplier is

\[
w(t)=\exp\!\left[-\frac{(t-t_0)^2}{2c}\right].
\]

This gives a precise meaning to “the spacetime center”: \((x_0,t_0)\) is the
point of maximum field value, \(x_0\) is the spatial center at \(t_0\), and
\(\sqrt c\) is temporal standard deviation.

## 4. Rigidity theorem: what one Gaussian cannot do

### Theorem

Every nondegenerate joint Gaussian on \((x,t)\), when sliced at fixed time,
has an affine conditional mean and a time-independent conditional covariance.

### Proof

Partition its precision as

\[
\Lambda_4=
\begin{bmatrix}Q&r\\r^\top&s\end{bmatrix}.
\]

At fixed \(t\), the spatial part of the negative log kernel is

\[
\frac12x^\top Qx+x^\top r(t-t_0)+\text{terms independent of }x.
\]

The quadratic coefficient \(Q\) is independent of time, so the slice
covariance \(Q^{-1}\) is constant. Completing the square makes the center an
affine function of \(t\). \(\square\)

### Consequences

One exact 4D Gaussian cannot intrinsically:

- follow a curved trajectory;
- rotate its 3D conditional ellipsoid over time;
- change its three conditional principal widths over time;
- have a multimodal or arbitrary temporal presence curve.

Perspective and camera motion can still make its *screen-space* covariance
change with time because the projection Jacobian changes along the trajectory.
That apparent change is not a changing world covariance.

Conversely, the conditional family

\[
\rho(x,t)=a(t)
\exp\!\left[-\frac12(x-m(t))^\top C(t)^{-1}(x-m(t))\right]
\]

is one joint Gaussian only when \(m(t)\) is affine, \(C(t)\) is constant, and
\(\log a(t)\) is quadratic with the compatible coefficients. Otherwise it is
a generalized Gaussian tube or field. Calling it that preserves the
distinction between an exact Gaussian compiler and a richer learned family.

## 5. Where rotation went in four dimensions

Every SPD(4) covariance has

\[
\Sigma_4=R_4\operatorname{diag}(s_1^2,s_2^2,s_3^2,s_4^2)R_4^\top,
\qquad R_4\in SO(4).
\]

**Known theorem.** \(\dim SO(n)=n(n-1)/2\), so \(SO(4)\) has six DOF. Four
principal widths plus six orientation DOF give all ten covariance DOF.

Three of those orientation directions can be understood as spatial
orientation; the three directions mixing time with space become spacetime
tilt, hence linear conditional velocity. This physical split is exact in the
block form even though a generic spectral eigenbasis does not canonically label
its rotations as “spatial” and “temporal.”

### Pair of quaternions, not an octonion

Identify \(\mathbb R^4\) with the quaternions. An orientation-preserving 4D
rotation can be represented as

\[
p\mapsto q_Lp q_R^{-1},
\]

where \(q_L,q_R\) are unit quaternions. This realizes

\[
\operatorname{Spin}(4)\cong SU(2)\times SU(2)
\]

as a double cover of \(SO(4)\). The pair stores eight numbers with two unit
constraints, so it has six continuous DOF. Simultaneously flipping both signs
does not change the rotation.

A unit octonion lies on a seven-dimensional sphere and octonion multiplication
is nonassociative; it is not a minimal parameterization of \(SO(4)\). It is the
wrong object here.

The native 4D Gaussian model of Yang et al. uses four scales plus left/right
quaternions, confirming this practical parameterization. But pair-quaternion
storage is not required: a ten-parameter Cholesky factor of \(\Sigma_4\), or
the exact block form above, already contains the same covariance.

### 4D orientation is not time-varying 3D rotation

An \(SO(4)\) factor describes the fixed orientation of one ellipsoid in
spacetime. It does not provide a physical quaternion function \(q(t)\). A
rotating rigid anisotropic splat needs \(C(t)=R(t)C_0R(t)^\top\), which violates
the rigidity theorem unless \(R(t)\) leaves \(C_0\) invariant.

## 6. Units and identifiability

Space and time do not share physical units. A raw eigenvector that rotates
meters into seconds depends on the arbitrary time normalization. If
\(\tilde t=t/\tau_\star\), the covariance must transform as

\[
\tilde\Sigma=D\Sigma D^\top,
\qquad
D=\operatorname{diag}(1,1,1,1/\tau_\star),
\]

and the numerical tilt changes accordingly.

This is a strong reason to:

- define the public object invariantly as \((\mu_4,\Sigma_4)\);
- record the spatial and temporal unit normalization in the asset metadata;
- optimize in \((C,v,c)\), whose units are space\(^2\), space/time, and
  time\(^2\);
- avoid interpreting arbitrary \(SO(4)\) angles without a declared metric on
  spacetime coordinates.

Opacity/temporal-width tradeoffs, repeated covariance eigenvalues, mixture
component permutations, and camera-depth ambiguity remain statistical
identifiability issues even when the parameterization itself is algebraically
unique.

## 7. Recommended storage chart

**Recommendation.** Preserve the finite-lifetime semantic ABI

\[
(\mu_4,\Sigma_4,\alpha,\text{appearance}),
\]

but optimize geometry as:

| Field | DOF | Constraint-safe storage |
|---|---:|---|
| \(x_0\) | 3 | unconstrained |
| \(t_0\) | 1 | bounded/normalized clip time if desired |
| \(C\in\operatorname{SPD}(3)\) | 6 | lower Cholesky; softplus/exp diagonal |
| \(v\) | 3 | unconstrained or physically bounded |
| \(c>0\) | 1 | log temporal standard deviation |
| Geometry total | 14 | exact full SPD(4) |

This is globally SPD-safe, has no quaternion normalization or spectral
permutation gauge, exposes physical units, and maps directly to the existing
tube exponent. A pair-quaternion spectral form should be an optional
serialization/debug view, not a second simultaneous set of trainable geometry
parameters.

For practical long-lived content, use the same precision chart but allow a
typed persistent mode at \(\lambda_t=0\) on a bounded time interval. This is a
positive-semidefinite boundary of the strict SPD(4) family, not another spatial
geometry model; see
[02_slicing_projection_and_opacity.md](02_slicing_projection_and_opacity.md).
