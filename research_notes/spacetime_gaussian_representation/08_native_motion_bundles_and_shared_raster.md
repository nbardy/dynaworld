# Native Motion, Gaussian Worldtubes, and Shared Rasterization

**Date:** 2026-07-23

**Status:** mathematical backtrack and architecture decision; exact claims are
proved under their stated assumptions, engineering proposals remain unbuilt

**Question:** Can position and rotation move through time while the primitive
remains one native four-dimensional volume, and can the existing shared
forward/backward raster machinery survive if the source is not one strict
SPD(4) Gaussian?

## Executive answer

Yes, with three important distinctions.

1. A full strict SPD(4) Gaussian already has a moving spatial position. Its
   space-time covariance terms tilt the 4D ellipsoid, and its fixed-time slice
   center moves linearly. Writing that tilt as a velocity is only a coordinate
   choice.
2. One strict SPD(4) Gaussian cannot have a curved centerline or a spatial
   covariance that physically rotates or changes scale. Its normalized
   fixed-time covariance is constant. A fixed 4D rotation is not a
   time-varying 3D rotation.
3. A curved, rotating, or changing-scale Gaussian cross-section can still
   define one native density on \(\mathbb R^3\times I\). The clean larger
   object is a transported or swept Gaussian worldtube. Describing its
   worldline as \(p(t)\) does not make it a per-frame bank; it is merely a
   coordinate description of one spacetime field.

The renderer result is equally important:

> The ray-depth Schur/completion-of-squares calculation does not require the
> source to be globally Gaussian in spacetime. It only requires the source
> exponent to be quadratic along each ray-depth fiber.

Therefore an ordinary spatial Gaussian with arbitrary shared
\(p(t),C(t),a(t)\) retains exact conditional ray-depth mean, variance, and
integrated trace at every sensor-time point. What is lost is the ability to
encode the entire trace as one global UVT quadratic. It must instead be
compiled into piecewise polynomial/rational trace cells with error and event
certificates.

The most honest matched experiment is therefore:

- **A:** one full strict SPD(4) atom, with movement derived from its cross
  covariance;
- **B:** one low-knot moving/rotating ordinary 3D Gaussian compiled through the
  same trace atlas and shared adjoint;
- **C:** an adaptive piecewise/mixture chain of SPD(4) atoms;
- promote a general swept Gaussian worldtube only if B materially wins and C
  needs too many pieces.

## What the current source actually does

The premise that all position motion was removed is too broad.

- [`WorldTubeBatch`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py)
  stores `x0` and a three-vector `velocity`.
- `project_world_tubes_ortho` and `project_world_tubes_pinhole` put projected
  motion into the \(ut\) and \(vt\) entries of the joint UVT precision.
- Depth motion is put into `depth_beta[:, 2]`.
- [`FeatureScreenTimeTubeModel`](../../src/train/star_uvt_feature_tube_model.py)
  likewise trains a UV velocity and constructs the UVT cross terms.

In these records, `ma` stores the joint UVT center at \(t_0\); it is not the
center at every time. The conditional center at another \(t\) is obtained from
the joint quadratic. What is genuinely restricted is the world spatial shape:
the current `WorldTubeBatch` has only two fronto-parallel spatial precisions,
not a full world SPD(3) covariance, and several trainer paths detach depth
metadata.

## Claim-status legend

- **Exact definition:** chosen mathematical object or notation.
- **Proved:** follows by the displayed algebra or a named elementary theorem.
- **Computational evidence:** numerically checked, not a proof.
- **Proposal:** implementation or representation choice requiring experiments.
- **Open:** unresolved semantic or empirical question.

## Assumptions and notation

- \(I\subset\mathbb R\) is a bounded physical-time interval.
- \(x\in\mathbb R^3\), \(t\in I\), and \(z=(x,t)\in\mathbb R^4\).
- A strict SPD(4) Gaussian is defined on ambient \(\mathbb R^4\) and merely
  queried on \(I\), unless truncation and renormalization are stated
  explicitly.
- A raw field value carries temporal activity. A normalized conditional
  distribution does not.
- All covariance matrices called SPD are strictly positive definite.
- Camera rays are affine in the selected ordinary depth coordinate inside a
  local chart. Nonlinear depth gauges keep their Jacobian inside the integral.
- Visibility, clipping, and alpha/extinction semantics are separate from
  Gaussian geometry and must remain explicitly typed.

## One hundred numbered equations

### I. What “native in spacetime” means

A native world domain is the total space

\[
\mathcal M=\mathbb R^3\times I.
\tag{E001}
\]

Physical time is the projection

\[
\pi_T:\mathcal M\rightarrow I,\qquad \pi_T(x,t)=t.
\tag{E002}
\]

A native primitive is one field or measure on that total space:

\[
\rho:\mathcal M\rightarrow\mathbb R_{\ge0},\qquad
(x,t)\mapsto \rho(x,t).
\tag{E003}
\]

Its frame at time \(t\) is the pullback along the inclusion

\[
i_t:\mathbb R^3\rightarrow\mathcal M,\qquad
i_t(x)=(x,t),\qquad \rho_t=i_t^*\rho.
\tag{E004}
\]

Thus writing \(\rho_t\) does not replace a 4D object by a frame bank. It merely
restricts one field to a physical-time fiber. Define the negative log field

\[
q(x,t)=-2\log \rho(x,t).
\tag{E005}
\]

When the spatial ridge is unique, its position can be *derived* from the 4D
field rather than stored as a privileged function:

\[
p(t)=\underset{x}{\operatorname{argmin}}\;q(x,t).
\tag{E006}
\]

Likewise, the local spatial precision and inverse-Hessian shape proxy can be
derived from the ridge Hessian:

\[
Q_{\mathrm{loc}}(t)=\frac12\nabla_x^2q(p(t),t),\qquad
C_{\mathrm{Lap}}(t)=Q_{\mathrm{loc}}(t)^{-1}.
\tag{E007}
\]

For a Gaussian spatial slice, \(C_{\mathrm{Lap}}\) is its exact covariance.
For a general field it is only a local Laplace-shape descriptor, not the
global statistical covariance.

An independent per-frame parameter bank has state scaling

\[
\Theta_{\mathrm{frame}}
=\{\theta_{i,f}:1\le i\le N,\;1\le f\le T\},
\qquad |\Theta_{\mathrm{frame}}|=\Theta(NTD).
\tag{E008}
\]

A finite shared temporal representation instead has

\[
\theta_i(t)=\sum_{j=0}^{B-1}\Theta_{ij}\phi_j(t),
\qquad |\Theta_{\mathrm{basis}}|=\Theta(NBD).
\tag{E009}
\]

Both can define values at every time. The representation distinction is
whether the asset stores \(T\) independent states or one finite shared object:

\[
\theta_{\mathrm{world}}
\xrightarrow{\;\mathcal C_{\Gamma,D}\;}
\theta_{\mathrm{trace},D}
\xrightarrow{\;\mathcal R_D\;}
I_D.
\tag{E010}
\]

### II. A strict SPD(4) Gaussian already moves position

Let the world field be

\[
\rho(z)=\alpha
\exp\!\left[-\frac12(z-\mu_4)^\top\Sigma_4^{-1}(z-\mu_4)\right],
\qquad \Sigma_4\succ0.
\tag{E011}
\]

Partition its center and covariance:

\[
\mu_4=
\begin{bmatrix}x_0\\t_0\end{bmatrix},
\qquad
\Sigma_4=
\begin{bmatrix}A&b\\b^\top&c\end{bmatrix}.
\tag{E012}
\]

The geometric degree count is

\[
\dim \mu_4+\dim\operatorname{SPD}(4)
=4+\frac{4(4+1)}2=14.
\tag{E013}
\]

Define the conditional coordinates

\[
v=\frac bc,\qquad
C=A-\frac{bb^\top}{c}.
\tag{E014}
\]

The Schur criterion gives

\[
\Sigma_4\succ0
\iff c>0\ \text{and}\ C\succ0.
\tag{E015}
\]

Conversely, every \(C\succ0,c>0,v\in\mathbb R^3\) reconstructs

\[
\Sigma_4=
\begin{bmatrix}
C+cvv^\top&cv\\
cv^\top&c
\end{bmatrix}.
\tag{E016}
\]

This covariance has the exact block factorization

\[
\Sigma_4=
\begin{bmatrix}I&v\\0&1\end{bmatrix}
\begin{bmatrix}C&0\\0&c\end{bmatrix}
\begin{bmatrix}I&0\\v^\top&1\end{bmatrix}.
\tag{E017}
\]

Its determinant is therefore

\[
\det \Sigma_4=c\,\det C.
\tag{E018}
\]

Block inversion yields

\[
\Sigma_4^{-1}=
\begin{bmatrix}
C^{-1}&-C^{-1}v\\
-v^\top C^{-1}&c^{-1}+v^\top C^{-1}v
\end{bmatrix}.
\tag{E019}
\]

For \(\eta=x-x_0\) and \(\tau=t-t_0\), completing the square gives

\[
\begin{bmatrix}\eta\\\tau\end{bmatrix}^{\!\top}
\Sigma_4^{-1}
\begin{bmatrix}\eta\\\tau\end{bmatrix}
=
(\eta-v\tau)^\top C^{-1}(\eta-v\tau)+\frac{\tau^2}{c}.
\tag{E020}
\]

The raw time slice is consequently

\[
\rho_t(x)=
\alpha e^{-\tau^2/(2c)}
\exp\!\left[
-\frac12(x-x_0-v\tau)^\top C^{-1}(x-x_0-v\tau)
\right].
\tag{E021}
\]

Its spatial center moves:

\[
m(t)=x_0+v(t-t_0).
\tag{E022}
\]

Its normalized spatial covariance and center derivatives satisfy

\[
\operatorname{Cov}(X\mid T=t)=C,\qquad
\dot m(t)=v,\qquad
\ddot m(t)=0,\qquad
\dot C(t)=0.
\tag{E023}
\]

Its raw peak activity is

\[
\alpha(t)=\alpha
\exp\!\left[-\frac{(t-t_0)^2}{2c}\right].
\tag{E024}
\]

The central worldline is the 4D curve

\[
\gamma(t)=\bigl(x_0+v(t-t_0),\,t\bigr).
\tag{E025}
\]

Its tangent is

\[
\dot\gamma(t)=
\begin{bmatrix}v\\1\end{bmatrix}.
\tag{E026}
\]

Thus velocity is exactly the covariance tilt in physical units:

\[
\Sigma_{xt}=cv,\qquad
v=\frac{\Sigma_{xt}}{\Sigma_{tt}}.
\tag{E027}
\]

If the precision is partitioned as
\(\Lambda_4=\bigl[\begin{smallmatrix}Q&r\\r^\top&s\end{smallmatrix}\bigr]\),
the same motion is

\[
C=Q^{-1},\qquad
v=-Q^{-1}r,\qquad
c=(s-r^\top Q^{-1}r)^{-1}.
\tag{E028}
\]

Only a block-diagonal world covariance forces fixed position:

\[
\Sigma_{xt}=0
\iff v=0
\iff m(t)=x_0.
\tag{E029}
\]

For the ambient, untruncated Gaussian of E011, the time-marginalized spatial
covariance is

\[
\operatorname{Cov}_{\mathbb R^4}(X)=A=C+cvv^\top.
\tag{E030}
\]

The additional \(cvv^\top\) is trajectory smear over the temporal lifetime,
not the instantaneous spatial size. If the field is instead truncated to
\(T\in I\) and renormalized, \(c\) in this decomposition is replaced by
\(\operatorname{Var}(T\mid T\in I)\).

### III. The existing shared UVT quadratic also moves position

Let \(y=(u,v)\in\mathbb R^2\) and write a joint UVT exponent as

\[
q_{\mathrm{uvt}}(y,t)=
\begin{bmatrix}y-y_0\\\tau\end{bmatrix}^{\!\top}
H
\begin{bmatrix}y-y_0\\\tau\end{bmatrix}.
\tag{E031}
\]

Partition its fixed precision:

\[
H=
\begin{bmatrix}P&r\\r^\top&s\end{bmatrix},
\qquad H\succ0\quad(\text{hence }P\succ0).
\tag{E032}
\]

Completing the spatial square produces

\[
q_{\mathrm{uvt}}
=
\left(y-y_0+P^{-1}r\,\tau\right)^\top
P
\left(y-y_0+P^{-1}r\,\tau\right)
+
\left(s-r^\top P^{-1}r\right)\tau^2.
\tag{E033}
\]

Hence its conditional screen center is

\[
m_{uv}(t)=y_0-P^{-1}r\,(t-t_0).
\tag{E034}
\]

Its conditional screen covariance is fixed:

\[
C_{uv}(t)=P^{-1}.
\tag{E035}
\]

Its effective temporal precision is the Schur complement

\[
\lambda_t=s-r^\top P^{-1}r>0.
\tag{E036}
\]

Equivalently, for a desired screen velocity \(v_{uv}\),

\[
H=
\begin{bmatrix}
P&-Pv_{uv}\\
-v_{uv}^\top P&v_{uv}^\top Pv_{uv}+\lambda_t
\end{bmatrix}.
\tag{E037}
\]

The \(ut\) and \(vt\) cross entries are therefore

\[
r=-Pv_{uv}.
\tag{E038}
\]

Position gradients remain ordinary continuous gradients inside this
quadratic:

\[
\frac{\partial m_{uv}(t)}{\partial y_0}=I_2,\qquad
\frac{\partial m_{uv}(t)}{\partial v_{uv}}=(t-t_0)I_2.
\tag{E039}
\]

The current affine UVT tube constructors implement exactly the map

\[
(y_0,t_0,P,v_{uv},\lambda_t)
\longmapsto
\left(
ma=(y_0,t_0),\
q_{\mathrm{uvt}}=(P,-Pv_{uv},v_{uv}^\top Pv_{uv}+\lambda_t)
\right).
\tag{E040}
\]

A fixed `ma` is therefore not a fixed center across time. The motion lives in
the shared precision cross terms.

### IV. Why one global quadratic cannot rotate or curve

Consider the most general quadratic spacetime exponent

\[
\Phi(x,t)=
x^\top Kx+2t\,r^\top x+s t^2
+2d^\top x+2et+f.
\tag{E041}
\]

Its spatial Hessian is

\[
\nabla_x^2\Phi(x,t)=2K.
\tag{E042}
\]

Its spatial ridge is

\[
x_*(t)=-K^{-1}(rt+d).
\tag{E043}
\]

Therefore

\[
\ddot x_*(t)=0.
\tag{E044}
\]

The covariance of its normalized spatial cross-section obeys

\[
\frac{d}{dt}K^{-1}=0.
\tag{E045}
\]

A physically rotating anisotropic covariance with fixed principal widths would
instead have

\[
C(t)=R(t)D^2R(t)^\top,\qquad R(t)\in SO(3),\qquad \dot D=0.
\tag{E046}
\]

Define its angular-velocity algebra element

\[
\Omega(t)=\dot R(t)R(t)^\top,\qquad \Omega^\top=-\Omega.
\tag{E047}
\]

Then its covariance derivative is

\[
\dot C(t)=\Omega(t)C(t)-C(t)\Omega(t).
\tag{E048}
\]

Visible physical rotation requires

\[
[\Omega(t),C(t)]\ne0.
\tag{E049}
\]

Rotation is unobservable only inside an isotropic or repeated-eigenvalue
subspace:

\[
[\Omega,C]=0
\quad\Longrightarrow\quad
\dot C=0.
\tag{E050}
\]

A fixed Mahalanobis level \(R^2\) of the SPD(4) atom intersects time \(t\) in

\[
(x-m(t))^\top C^{-1}(x-m(t))
=R^2-\frac{(t-t_0)^2}{c}.
\tag{E051}
\]

This contour shrinks near the temporal ends, but after normalization

\[
\operatorname{Cov}(X\mid T=t)=C
\tag{E052}
\]

still does not rotate or rescale. A spectral view of the same object is

\[
\Sigma_4=
R_4\operatorname{diag}(s_1^2,s_2^2,s_3^2,s_4^2)R_4^\top,
\qquad R_4\in SO(4).
\tag{E053}
\]

Its dimensional accounting is

\[
\dim SO(4)=6,\qquad
4\ \text{widths}+6\ \text{orientation DOF}=10.
\tag{E054}
\]

Those are the fixed orientation DOF of one spacetime ellipsoid. A general
conditional Gaussian field is one joint SPD(4) Gaussian **if and only if**

\[
C(t)\equiv C,\qquad
m(t)=x_0+v(t-t_0),\qquad
a(t)=\alpha e^{-(t-t_0)^2/(2c)}.
\tag{E055}
\]

Necessity follows from the constant spatial Hessian and affine ridge above;
sufficiency follows by substituting into equation E020.

### V. A swept Gaussian is still one native 4D volume

The direct larger object is

\[
\rho(x,t)=
a(t)
\exp\!\left[
-\frac12(x-m(t))^\top Q(t)(x-m(t))
\right],
\qquad Q(t)\succ0.
\tag{E056}
\]

Its radius-\(r\) spacetime worldtube is the single subset

\[
\Omega_r=
\left\{(x,t)\in\mathcal M:
(x-m(t))^\top Q(t)(x-m(t))\le r^2
\right\}.
\tag{E057}
\]

Its center is the worldline

\[
\gamma:I\rightarrow\mathcal M,\qquad
\gamma(t)=(m(t),t).
\tag{E058}
\]

If a smooth 4D worldline \(W\) is transverse to physical time,

\[
d(\pi_T|_W)\ne0
\quad\Longrightarrow_{\mathrm{IFT}}\quad
W\cap U=\{(m(t),t):t\in J\}.
\tag{E059}
\]

Thus a native worldline and a local function \(m(t)\) contain the same
geometric information. The relevant Gaussian-state bundle is

\[
\mathcal E=
I\times
\left(
\mathbb R^3\times\operatorname{SPD}(3)\times\mathbb R_+
\right)
\longrightarrow I.
\tag{E060}
\]

A moving Gaussian is a section

\[
s(t)=\bigl(t,m(t),C(t),a(t)\bigr).
\tag{E061}
\]

Because an interval is contractible, every ordinary finite-dimensional bundle
over it is trivializable:

\[
\Gamma(I,\mathcal E)
\cong
\{m(t),C(t),a(t)\}.
\tag{E062}
\]

Bundle language clarifies geometry but does not create free compression. A
transported Gaussian can be defined by a moving affine chart, with
\(R(t)\in SO(3)\) and \(A(t)\in GL(3)\):

\[
\Phi:I\times\mathbb R^3\rightarrow\mathcal M,\qquad
\Phi(t,\xi)=\bigl(m(t)+R(t)A(t)\xi,\ t\bigr).
\tag{E063}
\]

Its Jacobian determinant is

\[
\left|\det D\Phi(t,\xi)\right|=|\det A(t)|.
\tag{E064}
\]

Under a mass-density convention, pushing forward a canonical Gaussian fiber
gives

\[
\rho(x,t)=
\frac{a(t)}{(2\pi)^{3/2}|\det A(t)|}
\exp\!\left[
-\frac12
\left\|A(t)^{-1}R(t)^\top(x-m(t))\right\|^2
\right].
\tag{E065}
\]

Here \(a(t)\) is total spatial mass. The constant can instead be absorbed into
the definition of \(a\). Under a peak-amplitude convention, both the
normalizer and \(|\det A|^{-1}\) are omitted. This is one reason the amplitude
convention must be typed before implementation.

Its spatial covariance is

\[
C(t)=R(t)A(t)A(t)^\top R(t)^\top.
\tag{E066}
\]

The transport connection exposes translation and rotation rates:

\[
u(t)=R(t)^\top\dot m(t),\qquad
\Omega_b(t)=R(t)^\top\dot R(t)\in\mathfrak{so}(3).
\tag{E067}
\]

In the comoving coordinate \(\xi=A^{-1}R^\top(x-m)\), scalar composition and
density pullback are distinct:

\[
\begin{aligned}
\rho(\Phi(t,\xi))
&=
\frac{a(t)}{(2\pi)^{3/2}|\det A(t)|}e^{-\|\xi\|^2/2},\\
\Phi^*\!\left(\rho\,dx\,dt\right)
&=
\frac{a(t)}{(2\pi)^{3/2}}e^{-\|\xi\|^2/2}\,d\xi\,dt.
\end{aligned}
\tag{E068}
\]

Only the Gaussian *shape* is stationary; its amplitude may still evolve. The
Jacobian cancels in the density pullback. An object-frame gauge change
\(S(t)\in SO(3)\), accompanied by the inverse change of fiber
coordinates/shape so that \(RA\) is unchanged, transforms the rotational
connection as

\[
R'=RS,\qquad A'=S^\top A,\qquad
\Omega_b'=S^\top\Omega_b S+S^\top\dot S.
\tag{E069}
\]

This is not a ray-depth gauge, and it does not include the separate
scale/strain connection \(A^{-1}\dot A\).

A monotone time reparameterization \(\theta=\varphi(t)\) transforms apparent
velocity by

\[
\frac{d\widetilde m}{d\theta}
=
\frac{\dot m(t)}{\dot\varphi(t)}.
\tag{E070}
\]

Neither gauge removes physical relative motion. A comoving chart simply moves
that motion into the camera and connection.

### VI. Larger native objects and coordinate alternatives

If \(g_t(x)=R(t)^\top(x-m(t))\), a camera in the comoving object chart becomes

\[
\mathcal C'_t=\mathcal C_t\circ g_t^{-1}.
\tag{E071}
\]

Thus straightening the object does not make projection or occlusion
time-independent. A coordinate-invariant thickened worldline can instead be
defined as a normal-bundle pushforward. Let \(s\) be arclength,
\(N_{\gamma(s)}\) the Euclidean normal space, and \(\kappa_s\) a normalized
density on that fiber:

\[
\nu=F_\#\!\left(w(s)\kappa_s(n)\,ds\,dn\right),\qquad
F(s,n)=\gamma(s)+n,\quad
n\in N_{\gamma(s)},\quad \|\dot\gamma(s)\|=1.
\tag{E072}
\]

Where this pushforward is absolutely continuous, its field is
\(\rho=d\nu/dz\); self-overlaps add measure. A closely related affine Gaussian
sweep uses physical-time fibers rather than Euclidean normal fibers and gives
a strict SPD(4) Gaussian. Let

\[
S\sim\mathcal N(0,c),\qquad
\epsilon\sim\mathcal N(0,C),\qquad
Z=
\begin{bmatrix}x_0\\t_0\end{bmatrix}
+
\begin{bmatrix}v\\1\end{bmatrix}S
+
\begin{bmatrix}\epsilon\\0\end{bmatrix}.
\tag{E073}
\]

Then

\[
\operatorname{Cov}(Z)=
\begin{bmatrix}C&0\\0&0\end{bmatrix}
+
c
\begin{bmatrix}v\\1\end{bmatrix}
\begin{bmatrix}v^\top&1\end{bmatrix}
=
\begin{bmatrix}C+cvv^\top&cv\\cv^\top&c\end{bmatrix}.
\tag{E074}
\]

A finite mixture is also one native spacetime field:

\[
\rho(z)=
\sum_{k=1}^K\pi_k\,
\mathcal N_4(z;\mu_k,\Sigma_k),
\qquad \pi_k>0.
\tag{E075}
\]

Its component responsibility at time \(t\) is

\[
\omega_k(t)=
\frac{\pi_k\,\mathcal N_1(t;t_k,c_k)}
{\sum_j\pi_j\,\mathcal N_1(t;t_j,c_j)}.
\tag{E076}
\]

Its effective conditional center can curve:

\[
\bar m(t)=
\sum_k\omega_k(t)
\left[x_k+v_k(t-t_k)\right].
\tag{E077}
\]

Its effective covariance can rotate or change scale:

\[
\bar C(t)=
\sum_k\omega_k(t)
\left[
C_k+
(m_k(t)-\bar m(t))(m_k(t)-\bar m(t))^\top
\right].
\tag{E078}
\]

This moment covariance does not make the mixture itself unimodal. A coherent
single generalized tube can instead be defined implicitly by

\[
q(x,t)=x^\top A(t)x-2b(t)^\top x+c(t),
\qquad A(t)\succ0.
\tag{E079}
\]

Its position, covariance, and residual temporal activity are derived:

\[
p(t)=A(t)^{-1}b(t),\qquad
C(t)=A(t)^{-1},\qquad
\psi(t)=c(t)-b(t)^\top A(t)^{-1}b(t).
\tag{E080}
\]

This is a native 4D scalar field without making position a privileged public
parameter, although some equivalent time-dependent coefficients remain
mathematically unavoidable.

### VII. Ray-depth Schur elimination does not require a joint 4D Gaussian

Let the camera ray bundle use \(y=(u,v,t)\) and affine ordinary ray depth
\(z\), with a nonzero ray direction:

\[
X_\Gamma(y,z)=o(y)+z\,d(y),\qquad d(y)\ne0.
\tag{E081}
\]

At each physical time, allow an arbitrary moving and rotating spatial Gaussian:

\[
\rho(x,t)=
a(t)\exp\!\left[
-\frac12(x-p(t))^\top Q(t)(x-p(t))
\right].
\tag{E082}
\]

Substitution into the ray gives a scalar quadratic

\[
q_\Gamma(y,z)=h(y)z^2+2b(y)z+c(y).
\tag{E083}
\]

The three coefficients are

\[
\begin{aligned}
r(y)&=o(y)-p(t),\\
h(y)&=d(y)^\top Q(t)d(y)>0,\\
b(y)&=d(y)^\top Q(t)r(y),\\
c(y)&=r(y)^\top Q(t)r(y).
\end{aligned}
\tag{E084}
\]

Completing the scalar square yields

\[
q_\Gamma(y,z)=
h(y)\left(z+\frac{b(y)}{h(y)}\right)^2
+
\left(c(y)-\frac{b(y)^2}{h(y)}\right).
\tag{E085}
\]

Therefore the conditional ray-depth mean and variance are

\[
\widehat z(y)=-\frac{b(y)}{h(y)},\qquad
\sigma_z^2(y)=\frac1{h(y)}.
\tag{E086}
\]

The depth-eliminated exponent is the scalar Schur complement

\[
q_\perp(y)=c(y)-\frac{b(y)^2}{h(y)}\ge0.
\tag{E087}
\]

For a depth-independent fiber Jacobian \(J(y)\), the unbounded trace is

\[
\bar\rho(y)=
J(y)a(t)\sqrt{\frac{2\pi}{h(y)}}
\exp\!\left[-\frac12q_\perp(y)\right].
\tag{E088}
\]

Near/far clipping is also analytic:

\[
\bar\rho_{[z_n,z_f]}(y)=
\bar\rho(y)
\left[
\Phi\!\left(\sqrt h(z_f-\widehat z)\right)
-
\Phi\!\left(\sqrt h(z_n-\widehat z)\right)
\right].
\tag{E089}
\]

For an optical line integral, \(J(y)=\|d(y)\|\), or \(J=1\) for a unit ray.
A coarea or volume-pushforward Jacobian may depend on \(z\); then E088 must
retain that Jacobian inside the integral. Under a nonlinear depth coordinate
\(\zeta=\varphi_y(z)\), the pushed-forward law acquires
\(|dz/d\zeta|\) and is generally not Gaussian in \(\zeta\), although the
physical integral is invariant.

The exact differentials needed for a retained/smooth trace VJP are

\[
\begin{aligned}
dq_\perp
&=dc-\frac{2b}{h}\,db+\frac{b^2}{h^2}\,dh,\\
d\widehat z
&=-\frac1h\,db+\frac{b}{h^2}\,dh,\\
d\sigma_z^2
&=-\frac1{h^2}\,dh.
\end{aligned}
\tag{E090}
\]

Equations E081–E090 never assume that \(p(t)\), \(Q(t)\), or \(a(t)\) arise
from one global SPD(4) Gaussian. The time-conditioning Schur complement is
optional; the ray-depth Schur complement remains exact.

### VIII. Shared forward/backward computation without one global SPD(4)

For fixed spatial precision and a degree-\(d\) trajectory, write

\[
p(t)=\sum_{j=0}^d p_j\tau^j.
\tag{E091}
\]

Its spatial exponent expands into shared coefficients:

\[
\begin{aligned}
(x-p(t))^\top Q(x-p(t))
={}&x^\top Qx
-2\sum_{j=0}^d\tau^j x^\top Qp_j\\
&+\sum_{i=0}^d\sum_{j=0}^d
\tau^{i+j}p_i^\top Qp_j.
\end{aligned}
\tag{E092}
\]

Thus the temporal polynomial degree is at most \(2d\). More generally, collect
compiled sensor-time fields in a local basis:

\[
\chi_i(y)\approx
\widehat\chi_i(y)=
\sum_{\ell=0}^{B-1}\Theta_{i\ell}\phi_\ell(y),
\qquad
\|\chi_i-\widehat\chi_i\|_{\infty,D}\le\varepsilon_D.
\tag{E093}
\]

A degree-\(d\) Taylor cell can certify its trajectory error by

\[
\left\|
p(t)-\sum_{j=0}^d\frac{p^{(j)}(t_c)}{j!}(t-t_c)^j
\right\|
\le
\frac{\sup_{\xi\in D}\|p^{(d+1)}(\xi)\|}
{(d+1)!}|t-t_c|^{d+1}.
\tag{E094}
\]

At sampled queries, basis evaluation is a matrix operation

\[
X=\Phi\Theta.
\tag{E095}
\]

Its exact reverse-mode adjoint is

\[
\overline\Theta=\Phi^\top\overline X.
\tag{E096}
\]

For a trajectory \(p(t_f)=B(t_f)\theta\), the world-parameter gradient is

\[
\nabla_\theta L=
\sum_f B(t_f)^\top\nabla_{p(t_f)}L.
\tag{E097}
\]

Within a fixed \(B\)-dimensional linear basis—or a stable bounded-precision
model with the same effective dimension—the state counts and exact
independent-sample requirement are

\[
M_{\mathrm{frame}}=\Theta(NTD),\qquad
M_{\mathrm{basis}}=\Theta(NBD),\qquad
\text{arbitrary \(T\) independent samples require }B\ge T.
\tag{E098}
\]

Compilation can reduce repeated geometry work. Let \(E\) be the average
number of certified atlas cells per primitive, so \(NE\) is the total cell
count. A scoped leading-order cost model—not a universal complexity theorem—is

\[
\begin{aligned}
W_{\mathrm{frame\ geometry}}
&\sim NT\,c_{\mathrm{proj/bin/sort}},\\
W_{\mathrm{compile/update}}
&\sim NB\,c_{\mathrm{coeff}}+NE\,c_{\mathrm{cert}}+W_{\mathrm{rebuild}},\\
W_{\mathrm{trace\ eval}}
&\sim PT\bar A\,c_{\mathrm{eval}}(B),\\
W_{\mathrm{shade,direct}}
&\sim PT\bar A\,c_{\mathrm{shade}},\\
W_{\mathrm{output}}
&=\Omega(PT)\quad\text{if all \(T\) full images are materialized}.
\end{aligned}
\tag{E099}
\]

Here rebuild, sorting, culling, and camera-cell costs are implementation
dependent. The representation/compiler relationship is a branching partial
order, not a single containment chain. The subset statements admit the
singleton SPD(4) case; compilation is partial and may split, fall back, or
fail:

\[
\boxed{
\begin{aligned}
\mathrm{SPD4}
&\subset \mathrm{Mixture/Piecewise\ SPD4},\\
\mathrm{SPD4}
&\subset \mathrm{SweptGaussian},\\
\mathcal S_{\mathrm{regular,certifiable}}
&\subset
\bigl(
\mathrm{Mixture/Piecewise\ SPD4}
\cup
\mathrm{SweptGaussian}\bigr),\\
\mathcal C_{\Gamma,D,\varepsilon}:
\mathcal S_{\mathrm{regular,certifiable}}
&\rightharpoonup \mathrm{FiberTraceAtlas}_{\varepsilon}.
\end{aligned}
}
\tag{E100}
\]

Membership in \(\mathcal S_{\mathrm{regular,certifiable}}\) includes the
required smoothness, bounded support/event complexity, camera-denominator
separation, chart validity, and requested error tolerance. Low event count by
itself is not enough.

The reusable contribution is therefore broader than a “4D Gaussian
rasterizer.” It is a compiler and shared adjoint for low-event-complexity
sensor-time traces. Strict SPD(4) is its simplest exact source language.

## Proofs and derivation summaries

### Proposition 1 — Native-position theorem

**Claim.** A full strict SPD(4) Gaussian has a linearly moving conditional
spatial center.

**Proof.** Equations E014–E020 are an exact block decomposition of every
strict SPD(4) covariance. At fixed \(t\), the only \(x\)-dependent term is
\((x-x_0-v\tau)^\top C^{-1}(x-x_0-v\tau)\), whose unique minimizer is equation
E022. Conversely, equation E016 constructs an SPD(4) covariance for every
spatial SPD(3) cross-section, temporal variance, and linear motion. Therefore
the movement is intrinsic spacetime tilt, not an external trajectory glued to
the Gaussian. \(\square\)

### Proposition 2 — No-rotation/no-curvature theorem

**Claim.** One nondegenerate joint Gaussian cannot have a curved spatial ridge
or changing normalized spatial covariance.

**Proof.** Every joint Gaussian has a quadratic negative log field of the form
E041. Its spatial Hessian E042 is independent of time, so its covariance is
constant. Solving the first-order condition gives E043, which is affine.
Equations E044–E045 follow. A rotating anisotropic covariance has nonzero
commutator E048–E049, contradicting E045. \(\square\)

### Proposition 3 — A swept tube is genuinely native 4D

**Claim.** Equation E056 defines one spacetime volume even though it is
described by functions over physical time.

**Proof.** Equation E056 assigns one nonnegative scalar to every point of
\(\mathcal M\), so it is definitionally a field on the total 4D space.
Equation E057 gives its global support level set. Conversely, equation E059
shows that any smooth worldline transverse to physical time is locally the
graph of some \(m(t)\). Avoiding the notation \(m(t)\) can change storage or
coordinates but cannot remove the underlying degrees of freedom. \(\square\)

### Proposition 4 — Gauges relocate motion; they do not erase it

**Claim.** A comoving gauge can make one cross-section stationary but cannot
remove physical relative motion or rendering complexity.

**Proof.** Equation E068 makes the canonical fiber shape stationary, while
retaining any amplitude evolution. But the same transformation changes the
camera according to E071, while connection terms transform by E069. Relative
positions between independently moving objects and camera rays are invariant.
The motion has moved from the object coordinates into the connection/camera
rather than disappeared. \(\square\)

### Proposition 5 — Ray-depth Schur theorem without global SPD(4)

**Claim.** Any spatial Gaussian at each time has exact Gaussian conditional
depth along an affine ray, even when its center and covariance evolve
arbitrarily through time.

**Proof.** At fixed \(y=(u,v,t)\), equation E081 is affine in scalar depth.
Substitution into the spatial quadratic E082 yields E083–E084. Completing one
scalar square yields E085 and immediately gives E086–E089. No step assumes
that the coefficients are globally quadratic in \(t\). \(\square\)

### Proposition 6 — Compact-basis shared-adjoint sufficiency theorem

**Claim.** If temporal traces admit a compact basis and a fixed certified event
partition, coefficient evaluation and its transpose can share that basis and
partition; a global SPD(4) source is not required.

**Proof.** Any trace family represented by E093 evaluates through E095.
Reverse-mode differentiation of a linear basis evaluation is exactly its
transpose E096, and the chain rule gives E097. State compression follows when
\(B\ll T\), while E098 scopes the independent-frame obstruction to this basis
or bounded-precision setting. The certifier must additionally bound support,
camera-denominator, and visibility-order events. This does not join the entire
render and backward pass: compiler VJPs, bin/sort/composite backward kernels,
and event-boundary handling remain separate operations. \(\square\)

## The two Schur complements must be kept separate

| Operation | Formula | Requires joint SPD(4)? | What it gives |
|---|---|---:|---|
| Time conditioning | \(C=A-bb^\top/c,\ v=b/c\) | Yes, for this exact source interpretation | Linear moving center and fixed spatial covariance |
| Ray-depth elimination | \(q_\perp=c-b^2/h\) | No | Exact trace amplitude, depth mean, and depth variance for any spatial Gaussian fiber |
| Ordinary 3DGS projection | \(C_{uv}\approx J_\pi RCR^\top J_\pi^\top\) | No | Conventional projected conic; may also be compiled over time |

The first Schur complement explains the capacity of a strict 4D Gaussian. The
second is the renderer/compiler algebra worth preserving even if the world
representation is generalized.

## Representation branches

### Branch A — Full strict SPD(4), correctly implemented

**Status:** required exact baseline.

**Why it may be enough:** It already moves position and has only 14 geometric
DOF. In an affine/orthographic camera chart it gives exact global quadratic
trace lowering. Under a projective camera, the ray-depth formulas remain exact
pointwise, but the UVT trace fields generally require certified atlas cells.

**What would falsify it:** An accelerating or physically rotating synthetic
sequence where the matched model cannot fit at acceptable bytes despite correct
initialization, full spatial SPD(3), and nonzero \(xt\) gradients.

**Cheap test:** Fit one full atom to:

- constant-velocity translation;
- constant angular rotation of an anisotropic ellipsoid;
- constant acceleration.

It should fit the first and fail the latter two for the predicted mathematical
reason.

### Branch B — Low-knot dynamic ordinary 3DGS through the shared atlas

**Status:** highest-value matched control.

Use shared bases for:

- position \(m(t)\);
- rotation in \(\mathfrak{so}(3)\) or unit-quaternion splines;
- log scales;
- activity/opacity and optionally appearance.

Project or ray-integrate them into trace fields, fit/certify local trace cells,
and use the existing tile/order/event atlas plus basis-transpose backward.

**What it tests:** Whether the paper contribution is primarily the compact
world representation, the temporal trace compiler, or both.

**What would falsify it:** Trace residuals or visibility events force
\(B+E=\Theta(T)\), eliminating storage and geometry-work sharing.

### Branch C — Adaptive piecewise/mixture SPD(4)

**Status:** recommended first expressivity extension.

Each piece retains exact quadratic compilation. Split where:

- center curvature exceeds a projected-error threshold;
- covariance/rotation residual exceeds tolerance;
- camera denominator or support fit fails;
- visibility event density becomes too high.

For a twice differentiable center with
\(\sup_t\|m''(t)\|\le M\), a linear segment of width \(h\) has interpolation
error at most \(Mh^2/8\). That gives a direct subdivision rule.

**Risk:** A density mixture is linear, but ordinary alpha-splat splitting is
not render-neutral. Amplitude/extinction semantics must be frozen first.

### Branch D — Swept Gaussian worldtube

**Status:** mathematically clean larger object; promote only if B beats A and C
requires too many pieces.

Recommended semantic form:

```text
SweptGaussianWorldTube:
    identity / appearance
    bounded time domain and activity convention
    low-knot worldline
    transported SPD(3) cross-section
    optional rigid rotation and log-scale evolution
```

Recommended internal coordinates:

- cubic or low-knot \(m(t)\);
- \(R(t)\in SO(3)\) via Lie-algebra increments;
- SPD-safe scale/shape through log-Cholesky or
  \(C(t)=C_0^{1/2}\exp S(t)C_0^{1/2}\), with \(S(t)=S(t)^\top\);
- local jets and error bounds for compilation.

This is not a per-frame bank. It is one shared finite-dimensional spacetime
field.

### Branch E — General implicit 4D field

**Status:** expressive but not the first engineering choice.

Equation E079 is the most useful restricted implicit family because it remains
quadratic in space and therefore preserves exact ray-depth integration. A
general neural or high-degree implicit \(q(x,t)\) loses simple support bounds,
conditioning, and analytic depth elimination. It should not replace a cheaper
Gaussian-section model without evidence.

### Branch F — Flow/shared deformation field

**Status:** useful if many primitives share coherent motion.

A shared flow can transport a canonical field:

```text
dx/dt = u(x,t)
rho(x,t) = rho_0(Phi_t^{-1}(x))
```

This may amortize motion parameters across objects but introduces deformation
entanglement and a harder compiler. It is a different hypothesis from one
primitive having richer motion.

## Engineering consequence

### What should be implemented first

1. **Restore the exact full SPD(4) source lane.**
   Store a safe \(4\times4\) Cholesky or the equivalent
   \((x_0,t_0,C,v,c)\) chart. If philosophical clarity matters, store the
   Cholesky and derive \(v\); do not expose velocity as the canonical ABI.
2. **Verify moving position explicitly.**
   Test nonzero world \(xt\) covariance, UVT \(ut/vt\) terms, screen movement,
   depth movement, and gradients to those parameters.
3. **Restore full world spatial covariance.**
   Replace the current two fronto-parallel precisions with a full SPD(3)
   cross-section before adding curvature.
4. **Complete `WorldObject -> FiberTrace`.**
   Carry trace amplitude, conditional depth mean/slopes/variance, gauge, and
   fit/visibility certificates.
5. **Build matched Branch B.**
   Add a low-knot ordinary 3DGS source that targets the same `FiberTraceAtlas`.
   It may use conventional projected conics first and exact ray-depth Schur as
   a second semantic mode.
6. **Generalize the compiler, not the compositor.**
   Add local trace jets/polynomial coefficients and residual bounds. Reuse
   STAR binning, event strata, interval Metal, exposure, rolling shutter, and
   fallback policies.
7. **Keep hard-order gradients honest.**
   Only the discrete permutation derivative is zero within a stable
   hard-sorting stratum; image gradients can still flow through projection,
   footprint, clipping, attenuation, and amplitude. At a generic swap the
   hard-order map is nonsmooth, although commuting contributions can make a
   swap neutral. Pure order-key gradients require retained/soft ordering or an
   auxiliary depth objective.
8. **Compare A/B/C before D.**
   Use equal active primitives, equal learned bytes, equal trace metadata,
   fallback/event counts, and equal wall time.

### Suggested isolated code boundary

```text
research_experiments/spacetime_world_objects/
    world_object_protocol.py
    affine_spd4.py
    basis_dynamic_3dgs.py
    piecewise_spd4.py
    swept_gaussian.py
    ray_depth_trace.py
    trace_jet_compiler.py
    synthetic_motion_suite.py
```

The production renderer should receive one versioned trace contract rather
than know which world representation produced it.

## Falsification suite

### Algebra and numerical tests

1. Random SPD(4) covariance/precision block roundtrip.
2. Slice minimizer equals the predicted moving center.
3. UVT cross-term velocity recovery.
4. Full SPD(4) finite differences reach world \(xt\) terms.
5. Direct curved/rotating spatial-Gaussian ray quadrature matches E088–E089.
6. Joint SPD(4) lowering and arbitrary-fiber lowering agree in their overlap
   case.

### Capacity tests

1. Constant-velocity, fixed-shape scene: A should be exact.
2. Constant acceleration: A must fail; B/C/D should improve predictably.
3. Rotating anisotropic ellipsoid: A must fail while isotropic rotation should
   be observationally invisible.
4. Changing scale: only B/C/D should fit.
5. Long-lived object: compare strict temporal Gaussian with a typed bounded
   persistent tube.

### Systems tests

1. Compare per-frame bin/sort versus one trace atlas with identical rendered
   states.
2. Report coefficient count \(B\), trace-cell count \(E\), fallback fraction,
   and event roots—not only model parameters.
3. Measure forward geometry work, shading work, and backward accumulation
   separately.
4. Increase curvature until \(B+E\) approaches \(T\); that is the empirical
   sharing failure boundary.
5. Keep the current full-scale MPS guard closed until CPU algebra and a tiny
   externally monitored native microprofile pass.

## Computational checks performed for this note

CPU-only NumPy checks were run; no MPS work was launched.

- 500 random well-conditioned SPD(4) matrices:
  - covariance/precision center-slope agreement maximum absolute error
    \(1.776\times10^{-15}\);
  - completed-square maximum relative error
    \(2.627\times10^{-15}\).
- 500 random UVT quadratic constructions:
  - recovered cross-term velocity maximum absolute error
    \(7.494\times10^{-15}\).
- 200 samples of an arbitrarily curved, rotating spatial Gaussian:
  - analytic per-time ray integral versus dense numerical quadrature maximum
    relative error \(1.138\times10^{-14}\).

These checks corroborate the derivations but do not replace the proofs.

## Decision

The project did not make a fundamental mistake by choosing a native 4D
Gaussian. The mistaken inference was that a fixed 4D covariance meant a fixed
spatial position. Its cross terms already encode linear movement.

The real limitation is sharper:

> One global quadratic spacetime volume has an affine ridge and constant
> spatial Hessian.

That is excellent for exact compilation and temporal sharing but weak for
acceleration, physical rotation, and scale evolution.

We should not discard the 4D Gaussian or the Schur machinery. We should:

1. finish the full moving SPD(4) lane;
2. preserve the ray-depth Schur transform;
3. make the renderer accept low-event-complexity trace fields from more than
   one world source;
4. run the low-knot moving/rotating 3DGS matched control;
5. choose piecewise SPD(4) or a swept Gaussian worldtube based on measured
   curvature, rotation, trace-cell, and fallback costs.

The concise conceptual replacement is:

```text
not: “a 4D Gaussian renderer”

but: “a shared compiler and adjoint for certifiable spacetime trace fields,
      with strict SPD(4) as the simplest exact source language.”
```
