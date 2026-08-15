# Slicing, Projection, Opacity, and Visibility

## 1. Four operations that must not be conflated

Let \(X\in\mathbb R^3\), \(T\in\mathbb R\), and let
\(\rho(x,t)\) be a peak-normalized spacetime kernel.

| Operation | Formula | Semantic result |
|---|---|---|
| Raw time slice | \(\rho_t(x)=\rho(x,t)\) | Instantaneous spatial field, including temporal activity |
| Normalized conditional | \(p(x\mid t)=p(x,t)/p_T(t)\) | Spatial distribution with activity normalized away |
| Time marginal | \(\rho_X(x)=\int\rho(x,t)\,dt\) | Static trajectory-integrated occupancy / motion smear |
| Shutter integral | \(I_{[a,b]}=\int_a^b R[\rho_t,\mathcal C_t]w(t)\,dt\) | Exposure and motion blur after camera/render semantics |

For an instantaneous frame, the raw slice is the appropriate semantic object.
The normalized conditional throws away whether the primitive is active at that
time. The time marginal throws away frame identity. A shutter integral is not
generally equal to rendering the time-marginalized density because visibility,
transmittance, and alpha compositing are nonlinear.

## 2. Exact slice of a full spacetime Gaussian

For the block coordinates from [01_foundations.md](01_foundations.md),

\[
\rho(x,t)=
\alpha
e^{-\frac12\lambda_t(t-t_0)^2}
e^{-\frac12(x-x_0-v(t-t_0))^\top
Q(x-x_0-v(t-t_0))},
\]

where \(Q=C^{-1}\) and \(\lambda_t=1/c\).

The three slice components are therefore:

\[
\text{center: }m_x(t)=x_0+v(t-t_0),
\]

\[
\text{covariance: }C(t)=Q^{-1}=C,
\]

\[
\text{peak: }\alpha(t)=\alpha
e^{-\frac12\lambda_t(t-t_0)^2}.
\]

The separate \(\alpha\) is not redundant with covariance. It sets the maximum
of the peak-normalized kernel. Temporal covariance controls the relative peak
at other times; spatial covariance controls falloff away from the moving
center.

### Persistent-tube boundary case

A strict SPD(4) Gaussian requires \(\lambda_t>0\), so every primitive fades in
both temporal directions. An exactly persistent constant-opacity line over an
unbounded time domain requires \(\lambda_t=0\), making

\[
\Lambda_4=
\begin{bmatrix}Q&-Qv\\-v^\top Q&v^\top Qv\end{bmatrix}
\]

positive semidefinite with null direction \((v,1)\), not positive definite.
It is a Gaussian cross-section swept along a worldline: a cylinder/tube rather
than a normalizable joint Gaussian on all of \(\mathbb R^4\).

This is not a defect on a bounded clip. Three honest choices are:

1. keep strict SPD(4) and use a temporal scale much larger than the clip;
2. include the closure \(\lambda_t\ge0\), with an explicit finite support
   interval for persistent tubes;
3. factor activity \(a(t)\) from geometric support and let a persistent class
   use \(a(t)=1\).

**Recommendation:** treat \(\lambda_t=0\) as a typed persistent/static boundary
case rather than forcing an ill-conditioned huge covariance. The strict
SPD(4) object remains the localized atom and all block formulas continue by a
well-defined precision-space limit.

## 3. Position is a mean; covariance is support

There is not a separate “covariance of the position” and another covariance of
the volume in this deterministic primitive. The parameters have different
roles:

- \(m_x(t)\) is the center/mean of the spatial slice;
- \(C(t)\) is the spatial spread and orientation around that center;
- uncertainty over the learned parameter \(m_x\) would be a Bayesian posterior
  covariance, which is a different object and is not part of ordinary 3DGS;
- a distribution of multiple primitive centers is a mixture, again a different
  object.

A generalized model may make both \(m_x(t)\) and \(C(t)\) time-dependent. That
does not mean the position “has covariance”; it means the center and support
are two time-indexed functions.

## 4. From world XYZT to screen UVT

UVT means \((u,v,t)\): two image coordinates plus time. It is not a relabeling
of \((x,y,z,t)\). A UVT Gaussian has already selected a camera/chart and has
eliminated or summarized ray depth.

### 4.1 Camera-ray pullback

Let a local camera ray be

\[
\gamma_y(s)=b+Ay+sd,
\qquad y=(u,v,t),
\]

where \(s\) is ray depth, \(A\) maps sensor-time differentials into world
spacetime, and \(d\) is the lifted ray direction. For a world precision
\(\Lambda\), pull the exponent into local \((y,s)\) coordinates. Its precision
is

\[
H=J^\top\Lambda J,
\qquad J=[A\;d].
\]

Partition it into sensor-time and depth blocks:

\[
H=
\begin{bmatrix}
H_{yy}&H_{ys}\\
H_{sy}&H_{ss}
\end{bmatrix}.
\]

### 4.2 Ray-depth pushforward

Integrating the unbounded scalar depth fiber produces UVT precision

\[
S=H_{yy}-H_{ys}H_{ss}^{-1}H_{sy}.
\]

This is the Schur complement already derived in the repository. The same
completion of squares produces a conditional depth mean that is affine in
\(y\), and a depth variance \(H_{ss}^{-1}\). The current STAR ABI stores a UVT
mean/precision plus `depth0` and `depth_beta`; the fuller semantic contract
should also store or certify depth variance and chart validity.

For a locally affine map, Gaussian pullback and depth integration are exact.
For a perspective camera, the ordinary 3DGS renderer already uses a local
Jacobian approximation:

\[
\Sigma_{uv}\approx
J_\pi R_{cw} C R_{cw}^\top J_\pi^\top.
\]

The spacetime analogue can use local affine charts, segmented camera-time
charts, projective/rational coefficients, or exact per-ray evaluation. The
selected approximation must carry an error/support certificate or a fallback
path.

### 4.3 When slice and projection commute

For an affine map \(Y=MX+d\) that preserves the time coordinate,

\[
M(X\mid T=t)=MX\mid T=t,
\qquad
\operatorname{Cov}(MX\mid T=t)=MCM^\top.
\]

Thus slice-then-project and joint affine-pushforward-then-slice agree. This is a
useful exact test.

They need not commute under:

- nonlinear perspective without local linearization;
- time-varying chart boundaries;
- occlusion/order changes;
- nonlinear alpha/transmittance compositing;
- shutter integration after a moving camera transform.

The distinction explains why “take a slice” is the right frame semantics but
not the whole renderer architecture.

## 5. Opacity normalization choices

There are at least three defensible, inequivalent amplitude conventions.

### 5.1 Peak-preserving splat convention

Store \(\alpha\) as a learned peak opacity. Projection changes the footprint
but not the base peak by a determinant-normalization law. This is closest to
ordinary 3DGS and makes parameter transfer easiest.

**Risk:** widening a kernel increases integrated opacity/mass, and depth
marginalization is not a literal conserved-density operation.

### 5.2 Mass-preserving Gaussian convention

Treat the world Gaussian as a normalized density times mass \(m_i\). Integrating
depth introduces the exact Gaussian determinant factor; time slices have
normalization that depends on conditional determinants.

**Risk:** the resulting peak semantics differ from ordinary 3DGS, and direct
alpha compositing of integrated density still needs a physically justified
extinction conversion.

### 5.3 Optical-depth convention

Treat the field as extinction \(\sigma\), integrate optical depth along a ray,
and use Beer-Lambert transmittance. This is closest to WorldFoam/volume
semantics.

**Risk:** it is a different renderer and cannot be used to claim drop-in
equivalence with the standard sorted-alpha baseline.

**Open decision:** the representation should record whether amplitude means
peak alpha, Gaussian mass, or extinction. The compiler cannot silently switch
between them. For the first World Tubes comparison, peak-preserving semantics
are the cleanest baseline match; mass-preserving ray integration should be a
separate ablation.

## 6. Visibility is not in covariance

A covariance describes local support, not which primitive is in front. A full
depth extent introduces several complications:

- two primitives' depth distributions can overlap;
- their conditional depth order can change across \((u,v,t)\);
- a single depth-at-center sort can be wrong over a footprint;
- depth integration before nonlinear compositing can destroy the evidence
  needed for correct occlusion.

The UVT compiler therefore needs one or more of:

- conditional depth mean coefficients;
- depth variance or a conservative interval;
- an order-stability certificate over the chart;
- interval subdivision at order crossings;
- a fallback to per-sample or per-frame depth evaluation.

The canonical world covariance and the compiled visibility sidecar are
separate layers. Adding full \(z\)-covariance does not by itself solve sorting.

## 7. A monocular identifiability counterexample

Under an orthographic source camera looking along \(z\), the two covariances

\[
C_1=\operatorname{diag}(s_x^2,s_y^2,\varepsilon^2),
\qquad
C_2=\operatorname{diag}(s_x^2,s_y^2,M^2)
\]

have the same projected \((x,y)\) footprint for any positive
\(\varepsilon,M\). They can fit the source view identically while producing
radically different heldout views and depth overlap.

Thus restoring full spatial SPD(3) introduces a real depth-thickness cheating
gauge. It requires heldout-camera pressure, multiview data, a scale/depth prior,
or explicit rate/shape regularization. This is not an argument for deleting
the missing DOF; it is a required identifiability test.

## 8. Finite exposure and rolling shutter

For global exposure weighting \(w(s)\), a pixel is

\[
I(u,v)=\int
R\!\left[\rho_{t+s},\mathcal C_{t+s}\right](u,v)w(s)\,ds.
\]

Rolling shutter replaces the nominal time with a row- or pixel-dependent
function \(t=t(u,v)+s\). A compiled UVT trace is attractive because it can be
evaluated at that sensor-time directly, without rebuilding world splats for
each sub-sample. But exact exposure still integrates the rendered/composited
signal, not merely the Gaussian support in isolation.

## 9. Required operator-order tests

1. Raw slice retains the analytic temporal peak envelope.
2. Conditional normalization removes it, demonstrating why condition alone is
   the wrong frame field.
3. Time marginal gives \(A=C+cvv^\top\), demonstrating trajectory smear.
4. Affine slice/project commutation holds numerically.
5. Numerical ray-depth integration matches the Schur complement and amplitude
   factor.
6. Perspective projection shows screen covariance can change while world
   \(C\) stays fixed.
7. Two depth-overlapping splats expose where center-depth sorting fails.
8. Shutter quadrature after compositing differs from compositing a
   time-marginal footprint in an occlusion-changing scene.
9. Time-unit rescaling transforms all covariance/precision blocks
   consistently.
10. The \(\lambda_t\to0\) persistent-tube limit remains numerically stable in
    precision coordinates.

