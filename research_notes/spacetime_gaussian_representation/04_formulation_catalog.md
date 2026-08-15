# Formulation Catalog

This catalog separates two questions that are easy to mix up:

1. **How do we parameterize the same mathematical object?**
2. **Which mathematical object/class do we want?**

Changing Cholesky to pair-quaternions does not increase expressivity. Changing
one joint Gaussian into a spline-indexed covariance family does.

## A. Exact coordinate systems for the same full SPD(4) atom

All rows below represent the same 14-DOF geometry
\((\mu_4,\Sigma_4\in\operatorname{SPD}(4))\).

| Coordinates | Stored / effective geometry | Advantage | Main issue |
|---|---:|---|---|
| Direct symmetric \(\Sigma_4\) entries | 14 / 14 | semantically transparent | unconstrained updates can leave SPD |
| Covariance Cholesky \(LL^\top\) | 14 / 14 | globally SPD-safe; no gauge | coordinate-order dependent |
| Precision Cholesky \(LL^\top\) | 14 / 14 | efficient exponent/slice evaluation | projection often wants covariance solves |
| Symmetric log matrix \(\Sigma=e^S\) | 14 / 14 | global Sym(4)→SPD(4) map | matrix exponential cost/conditioning |
| Four log-scales + six plane angles | 14 / 14 | explicit principal axes | angle charts and spectral degeneracy |
| Four log-scales + quaternion pair | 16 / 14 | practical \(SO(4)\) rotor | two norm constraints, sign and spectral gauges |
| Conditional covariance block \((C,v,c)\) | 14 / 14 | exact physical space/time meaning | privileges declared time axis |
| Conditional precision block \((Q,v,\lambda_t)\) | 14 / 14 | exact tube exponent; stable as persistence limit | \(\lambda_t=0\) reaches PSD boundary |
| LDLᵀ factor | 14 / 14 | positive diagonal separated from correlations | ordering dependent |
| Free square root \(BB^\top\) | 20 / 14 | simple construction | six-DOF right-orthogonal gauge |

**Selection:** use conditional precision or a Cholesky internally. Expose
\((\mu_4,\Sigma_4)\) semantically. Use the quaternion pair only as an optional
spectral view or to reproduce a native 4DGS baseline.

## B. Twenty model families

“Exact compiler” means that an affine camera-ray pullback and depth Gaussian
pushforward stay inside one Gaussian UVT record. Perspective/visibility still
need chart and order handling in every row.

| # | Model family | Geometry cost per identity | Curved center | World \(C(t)\) changes | Camera-portable | One-record Gaussian compiler | Principal use/failure |
|---:|---|---:|:---:|:---:|:---:|:---:|---|
| 1 | Static shared 3DGS | 9 | no | no | yes | per-frame 3D projection | static baseline; cannot model motion |
| 2 | Independent per-frame 3DGS | \(9T\) | arbitrary lookup | arbitrary lookup | yes per frame | no temporal reuse | generous quality upper bound; \(O(TN)\) storage |
| 3 | Persistent tracked dynamic 3DGS | \(9T\), usually regularized | arbitrary lookup | arbitrary lookup | yes | no temporal reuse | identity/regularity stronger; still frame-scaled |
| 4 | Polynomial/STG-style 3DGS | basis-dependent | yes | rotation can; often scale fixed | yes | generally no | compact smooth dynamics; global polynomial failure/overshoot |
| 5 | Anchor splat + low-rank temporal bases | \(9+K\times\) coefficients | yes | if parameterized | yes | generally no | shared temporal compression; basis bias |
| 6 | Grouped \(SE(3)\) motion + residual | group + local state | yes | rigid rotation; residual optional | yes | piecewise/approximate | good for objects; segmentation/group errors |
| 7 | Canonical 3DGS + neural deformation field | 9 + shared network | yes | yes | yes | no closed single record | high capacity; compute and entanglement |
| 8 | Material flow \(\Phi_t\) with transported covariance | 9 + flow | yes | \(D\Phi_t C D\Phi_t^\top\) | yes | local approximation | coherent geometry; Jacobian/training cost |
| 9 | Axis-aligned 4D Gaussian | 8 | no | no | yes | yes | cheapest 4D ellipsoid; no space-time tilt/motion |
| 10 | Full native SPD(4) Gaussian | 14 | affine only | no | yes | yes | clean finite-lifetime atom; fixed slice shape |
| 11 | Persistent semidefinite Gaussian tube | 12 + activity/support | affine only | no | yes | Gaussian limit | exact persistent motion; not normalizable over unbounded time |
| 12 | Current restricted World Tube | 10 | affine only | no | yes | yes in current scaffold | no \(z\)-width, full spatial anisotropy, or orientation |
| 13 | Full screen-space UVT Gaussian + depth model | 9 + depth sidecar | affine in chart | screen support fixed | no | already compiled | elegant STAR state; not canonical multiview world geometry |
| 14 | Piecewise gated affine UVT segments | \(K(9+\text{sidecar})\) | piecewise | piecewise | no | compiled pieces | moving-camera locality; duplication/events |
| 15 | Projective rational/homogeneous trace | coefficient-dependent | screen-projective | chart-dependent | no | compiled projective record | exacter camera traces; not curved world motion |
| 16 | Mixture/chain of \(K\) full SPD(4) atoms | \(14K\), shareable appearance | piecewise/aggregate | aggregate changes | yes | yes per component | recommended first curvature extension; split/opacity ambiguity |
| 17 | B-spline center + fixed spatial SPD(3) | \(3K+6+\) activity | yes | no | yes | segmented local records | efficient curved motion; not one joint Gaussian |
| 18 | Lie-spline center, rotation, and log-scale | basis-dependent | yes | yes | yes | segmented/approximate | full desired dynamics; more gauges and optimization risk |
| 19 | Gauge-charted spacetime Gaussian \((U,\chi,\eta,\Lambda)\) | 14 + chart | chart-induced | chart-induced | yes if chart is world-side | local exact | elegant nonlinear local geometry; chart learning/certification |
| 20 | General world field → gauged trace atlas \(\pi_*\Gamma^*\rho\) | representation-dependent | arbitrary | arbitrary | world asset yes | record-dependent | broad architecture; compiler certificates become central |

Opacity and appearance are excluded from the geometry counts. Add one scalar
for simple peak opacity and three for RGB. A time-varying appearance basis adds
its own coefficients and is independent of the geometric choice.

## C. Detailed notes on the most relevant candidates

### External peer parameter accounting

Let \(A\) be per-Gaussian appearance floats. Simple RGB has \(A=3\); the
original degree-3 3DGS SH basis has \(A=48\). Raw coordinate counts are:

| Peer | Raw floats / Gaussian | What changes with time |
|---|---:|---|
| Standard 3DGS | \(11+A\) | nothing |
| Independent per-frame bank | \(T(11+A)\) | every stored parameter |
| Current repo RGB bank | \(14T\) | mean, scale, quaternion, opacity, RGB |
| Luiten Dynamic 3D Gaussians | \(7T+8\) in its simple accounting | per-time position + quaternion; other properties persistent |
| Native full 4D Gaussian, pair quaternions | \(17+A\) | one joint XYZT Gaussian |
| STG lite / feature form | 29 / 35 | cubic center, linear raw-quaternion path, temporal opacity; fixed spatial scales |
| Deformation-field 4D-GS | \(G(11+A)+P_\phi\) total | shared network deforms canonical 3DGS |

The native full-4D RGB model therefore stores 20 raw floats but has 18
effective DOF because its two unit quaternions store six effective rotation
DOF in eight coordinates. A direct SPD(4) Cholesky model stores the same 18
effective RGB parameters in 18 raw floats.

STG is **not** the pair-quaternion native 4D Gaussian. In its cited default
form it uses cubic spatial position (12 raw coefficients), a linear raw
quaternion trajectory (8), fixed spatial scales and temporal
opacity/center/width fields, plus either RGB or learned features. It is a
generalized conditional trajectory model.

At \(T=300,G=1024\), FP32, simple appearance, representative storage is:

| Peer | Raw storage |
|---|---:|
| Current restricted World Tube | 56 KiB |
| Native full 4D pair-quaternion RGB | 80 KiB |
| STG lite / full | 116 / 140 KiB, plus small shared decoder where used |
| Luiten Dynamic 3D Gaussians | about 8.23 MiB |
| Current independent per-frame RGB bank | about 16.41 MiB |

Thus “300× smaller” is accurate for the current 14-float World Tube versus the
current 14-float-per-frame bank. It is not a general comparison against compact
dynamic representations.

### #9: diagonal 4D Gaussian

\[
\Sigma_4=\operatorname{diag}(s_x^2,s_y^2,s_z^2,s_t^2).
\]

It is the literal “four coordinates plus four widths” eight-parameter geometry,
but its zero space-time cross-covariance gives zero conditional velocity. It is
useful only as an ablation showing that 4D axes alone do not create motion.

### #10: full native SPD(4) Gaussian

This is the clean finite-lifetime atom:

\[
\mu_4\in\mathbb R^4,
\qquad
\Sigma_4\in\operatorname{SPD}(4).
\]

It has linear conditional motion, fixed conditional spatial shape, and a
Gaussian temporal gate. Yang et al.'s native 4DGS belongs here and uses a pair
of quaternions to parameterize \(SO(4)\).

### #11: persistent tube closure

Set the conditional precision to

\[
\Lambda_4=
\begin{bmatrix}Q&-Qv\\-v^\top Q&v^\top Qv+\lambda_t\end{bmatrix},
\qquad \lambda_t\ge0.
\]

For \(\lambda_t>0\), this is exactly #10. For \(\lambda_t=0\), it is a
rank-three cylinder with a Gaussian spatial cross-section and constant temporal
activity. On a bounded timeline with an explicit interval, this is often a
better primitive for static backgrounds and persistent moving surfaces than an
ill-conditioned enormous temporal variance.

### #13: direct UVT Gaussian

\[
m_{uvt}\in\mathbb R^3,
\qquad
Q_{uvt}\in\operatorname{SPD}(3).
\]

This has exactly the clean center-plus-covariance form in sensor time. The
feature STAR model already instantiates it. Its limitation is categorical, not
algebraic: it belongs to a camera chart and has integrated/summarized ray depth,
so it cannot be the camera-independent world asset.

### #16: mixture or piecewise chain

For components \(k=1,\ldots,K\),

\[
\rho(x,t)=\sum_k w_k\rho_k(x,t)
\]

under density semantics. Conditional component responsibilities change with
time, so the aggregate conditional mean can curve and aggregate covariance can
change even though each component has affine center and fixed covariance.

Under standard sorted-alpha splatting, however, primitive contributions do not
add as a linear density mixture. Splitting one alpha splat into two is not
render-neutral. A split rule and opacity convention must be tested rather than
assuming mixture identities from probability theory.

### #18: fully dynamic conditional Gaussian tube

\[
\rho(x,t)=a(t)
\exp\!\left[-\frac12(x-m(t))^\top C(t)^{-1}(x-m(t))\right],
\]

with, for example,

\[
C(t)=R(q(t))
\operatorname{diag}(e^{2\ell_1(t)},e^{2\ell_2(t)},e^{2\ell_3(t)})
R(q(t))^\top.
\]

This directly answers “why not unfix position, rotation, and scale?” We can.
But it is a different class from one 4D Gaussian. It needs an SPD-valued curve,
a rotation interpolation convention, more temporal bases, and segmented or
approximate compilation. It should earn that complexity against #16 at equal
bytes and equal visibility error.

### #19 versus #20

A gauge-charted spacetime Gaussian changes the world-side coordinates in which
the local field is Gaussian. The general trace-atlas architecture is broader:
the world primitive need not be Gaussian at all, as long as the compiler emits
bounded local sensor-time records with support and visibility certificates.
The former is a representation; the latter is a renderer/compiler framework.

## D. Recommended ladder

### G0 — algebra reference

Strict full SPD(4), implemented in CPU double precision. Purpose: prove all
equivalences and reproduce native 4D Gaussian behavior.

### T0 — practical minimal World Tube

Full \(Q\in\operatorname{SPD}(3)\), affine center tilt \(v\), and typed
activity: persistent or localized Gaussian. Purpose: restore all spatial
covariance DOF while supporting long-lived scene content.

### M1 — adaptive mixture/piecewise chain

Split only where center curvature, covariance residual, projection denominator,
or visibility events demand it. Share appearance/identity when appropriate.

### D1 — generalized dynamic covariance

Low-knot \(m(t)\), \(R(t)\), and log-scale curves. Promote only if M1 requires
materially more bytes or compiler events at the same heldout quality.

### Compiler — orthogonal layer

For every level, compile the world object to a gauged UVT atlas with conditional
depth and validity/order sidecars. Do not train UVT records as though they were
portable world coordinates unless the experiment is explicitly a single-chart
screen-space baseline.

## E. Promotion criteria among formulations

Compare candidates on at least these axes:

1. stored trainable scalars/bytes over the full clip;
2. active primitives and primitive-pixel/tile pairs per rendered frame;
3. compiler work and atlas rebuild/split frequency;
4. peak memory, including gradients and optimizer state;
5. heldout-camera quality, not source-view loss alone;
6. trajectory, covariance, and temporal-activity residuals on analytic scenes;
7. visibility/order correctness and fallback rate;
8. invariance to time/spatial unit rescaling;
9. ability to represent static/persistent content without conditioning collapse;
10. complexity of backward differentiation across chart/support/order events.

No single scalar “number of splats” makes these formulations fair. The baseline
suite in [05_decision_and_experiments.md](05_decision_and_experiments.md) reports
both storage and active rendering work.
