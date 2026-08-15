# The Shader Boundary and the Missing Depth-Fiber Contract

**Status:** repository audit complete; Gaussian identities proved under the
stated affine/ray-linear assumptions; architecture recommendation proposed;
end-to-end implementation and empirical validation remain open.

**Date:** 2026-07-23

## 1. Executive conclusion

The earlier shorthand, “we did not implement the mathematics in the shaders,”
is too broad.

The repository has a substantial **compiled trace renderer**. The active Metal
backend evaluates full symmetric UVT quadratics, bins in tile-time cells,
evaluates projective trace records, uses affine pixel-dependent depth for
ordering, composites front-to-back, and implements extensive backward kernels.
What did not become an active end-to-end system is the front half:

\[
\boxed{
(\mu_{XYZT},\Sigma_4)
\longrightarrow
\text{gauged camera-ray pullback}
\longrightarrow
\text{UVT marginal + conditional depth law}
\longrightarrow
\text{certified trace atlas}
\longrightarrow
\text{Metal rasterizer}
}
\]

In particular, the active trainers do not learn a canonical full
world-spacetime Gaussian and differentiate through that complete chain. They
mostly learn camera-specific UVT records or restricted world tubes. The main
browser modes still reproject dynamic 3D splats at sampled times; the
standalone STAR browser implements only a bounded affine UVT subset.

The clean architectural statement is:

> Learn labeled Gaussian atoms in world spacetime. For each camera program,
> apply a gauged camera-ray transform and compile an event-certified
> sensor-time atlas carrying both the UVT marginal and the conditional
> ray-depth law.

UVT is therefore a good **compiled camera-query representation**, not the
canonical scene object.

## 2. Terminology corrections

### 2.1 “Position is the mean” is not special to 4D

Every ordinary 3D Gaussian splat already has a mean

\[
\mu_3=(x,y,z),
\]

which is its spatial center. Its covariance describes extent and orientation
around that center. It is not a second “covariance of the learned position”
unless the entire model is being given a separate Bayesian interpretation.

A joint spacetime Gaussian instead has

\[
\mu_4=(x_0,y_0,z_0,t_0).
\]

That is its center in spacetime. Space-time cross covariance makes the
**conditional spatial mean at a fixed time** move. If

\[
\Sigma_4=
\begin{bmatrix}A&b\\b^\top&c\end{bmatrix},
\]

then

\[
\mathbb E[X\mid T=t]
=x_0+\frac{b}{c}(t-t_0).
\]

Thus “position is the mean” holds in both 3D and 4D. The genuinely 4D fact is
that one spacetime covariance couples the spatial and temporal coordinates.

### 2.2 A strict 4D Gaussian is an ellipsoid, not a cylinder

For \(\Sigma_4\in\operatorname{SPD}(4)\), a level set

\[
(z-\mu_4)^\top\Sigma_4^{-1}(z-\mu_4)=r^2
\]

is a bounded four-dimensional ellipsoid. This is the cleanest minimal atom for
a finite event in spacetime.

A “Gaussian cylinder along a worldline” refers only to the singular precision
limit in which the primitive has zero temporal precision and therefore
constant activity along an unbounded direction. Its covariance is not SPD; it
is a semidefinite Gaussian measure or an improper density. That boundary is
useful for exactly persistent background, but it should not define the first
canonical dynamic primitive.

Preferred wording:

> Start with finite 4D Gaussian ellipsoids in world spacetime. Treat static or
> exactly persistent content as a separate typed 3D primitive, a bounded
> persistent tube, or a deliberate semidefinite limit.

### 2.3 “Camera compiler” is engineering shorthand

Let

- \(M=\mathbb R^3\times\mathbb R\) be world spacetime;
- \(B=\Omega\times\mathcal T\) be sensor-time coordinates
  \(y=(u,v,t)\);
- \(E_\Gamma\) be the camera-ray bundle over \(B\);
- \(\pi:E_\Gamma\to B\) forget ray depth;
- \(\Gamma:E_\Gamma\to M\) map a sensor-time-depth query to world
  spacetime.

The mathematical operator is

\[
\mathcal T_\Gamma=\pi_*\Gamma^*,
\]

a **ray-bundle pullback followed by fiber pushforward**. Good short names are
“gauged camera-ray transform” or “camera-ray trace transform.”

The implementation may reasonably be called a compiler because it specializes
this transform to a known camera program, partitions it into valid local gauge
domains, caches polynomial/rational coefficients, and emits renderer packets.
The more precise systems phrase is **event-certified sensor-time trace-atlas
lowering**. “Camera compiler” alone hides the ray-depth measure, gauge, and
validity conditions.

## 3. What is implemented and what is missing

| Layer | Intended object | Repository state |
|---|---|---|
| Canonical scene | Labeled world atoms \((\mu_4,\Sigma_4,\alpha,a)\) | Reference/scaffold only; not the active trainer state |
| Ray transform | \(\Gamma^*\), fiber measure, \(\pi_*\), Schur lowering | Derived in notes and NumPy/reference experiments; not wired end-to-end |
| Trace-atlas construction | Projective/gauge domains, support and visibility certificates | Substantial Python/Torch machinery, but generally starts from UVT/projective records rather than full world SPD(4) atoms |
| STAR Metal packet | `ma`, full `q_uvt`, `depth0`, `depth_beta`, opacity, appearance | Implemented |
| STAR Metal rasterization | UVT evaluation, tile-time binning, order/fallback, projective interval replay, compositing | Implemented |
| STAR backward | UVT/projective geometry, opacity, appearance | Substantial, but depth-order metadata is not an end-to-end trainable world-state path |
| Active feature model | Learned screen-space UVT tubes | Implemented; emits zero depth slopes in the ordinary path and detaches depth metadata |
| Restricted world model | Position, velocity, time, two fronto-parallel spatial precisions | Implemented; not a full spatial SPD(3), hence not a full SPD(4) Gaussian |
| Main browser | Dynamic anisotropic 3DGS sampled/reprojected per time | Implemented; not the STAR atlas or canonical 4D Gaussian path |
| Standalone browser STAR | Small affine UVT prototype | Implemented subset; no full atlas/certifier or SPD-safe world-to-trace chain |

Relevant implementation evidence:

- [`star_uvt_v0/README.md`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/README.md)
  defines Gate 0 as consuming already-projected UVT tensors.
- [`star_uvt_kernels.metal`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal)
  contains the full UVT quadratic, affine per-pixel depth evaluation,
  ordering, interval replay, compositing, and backward kernels.
- [`projective_trace.py`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py)
  constructs projective traces and support/visibility sidecars.
- [`spacetime_v0/docs/handoff.md`](../../third_party/fast-mac-gsplat/variants/spacetime_v0/docs/handoff.md)
  preserves the intended world `float4` mean, full 4D precision, and
  ray-depth integration, but that variant never became the active product
  path.

The correct verdict is therefore:

> The renderer back half is real. The missing part is the canonical
> \(XYZT+\operatorname{SPD}(4)\) producer, exact/certified world-to-ray
> lowering, the complete conditional depth law, and gradients back to world
> and camera parameters.

## 4. Why UVT appeared to replace XYZT

The original STAR Gate 0 deliberately began after projection. This isolated
the systems question: can one screen-time trace be binned and replayed across
many frames more cheaply than projecting, binning, and sorting independent 3D
splats at every frame?

That choice made sense for a rasterizer gate. It then leaked into the model
boundary: some active trainers optimize the compiled UVT packet directly. This
is convenient for a single known camera program but it is not a camera-portable
world representation.

The layers should be kept distinct:

1. **World object:** camera-independent \(XYZT\) atom or field.
2. **Sensor object:** camera-conditioned ray trace over \(UVT\) with depth.
3. **Raster packet:** coefficients chosen for tile support, order tests, and
   shader evaluation.

The world object should not be “pure UVT.” The sensor object legitimately is.

## 5. Depth should be a conditional fiber law, not a sidecar

Let \(\nu_i\) be one labeled world atom. Pull it onto the ray bundle. The
clean object is the disintegration

\[
\Gamma^*\nu_i(dy,dz)
=
\underbrace{(\pi_*\Gamma^*\nu_i)(dy)}_{\text{UVT marginal }m_i(dy)}
\underbrace{K_i(y,dz)}_{\text{conditional ray-depth law}}.
\]

This is a **pushforward-disintegration pair**. The UVT trace says how much of
the primitive reaches each sensor-time coordinate. The conditional kernel
\(K_i\) says where that contribution lies along the corresponding ray.

Under a change of depth gauge, \(K_i\) is pushed into the new coordinate and
the fiber Jacobian changes. The physical trace remains invariant. This is why
the Jacobian and gauge identity are semantic requirements, not optional
metadata.

Calling depth a “sidecar” was pragmatically useful but mathematically
misleading. It is half of the lossless factorization of the pulled-back
primitive.

## 6. Exact affine Gaussian factorization

### Claim

In an affine camera gauge, a joint Gaussian over \((u,v,t,z)\) is exactly
equivalent to a UVT Gaussian plus an affine conditional Gaussian in ray depth.

### Assumptions

- \(y=(u,v,t)\in\mathbb R^3\) and scalar depth \(z\in\mathbb R\);
- the world-to-gauge map is affine over the local domain;
- the fiber measure is \(J(y)\,dz\), with \(J\) constant in \(z\) inside the
  domain;
- the pulled-back precision is positive definite.

### Derivation

Write the pulled-back peak-normalized field as

\[
f(y,z)=A\exp\!\left[-\frac12
\begin{pmatrix}\delta y\\\delta z\end{pmatrix}^{\!\top}
\begin{pmatrix}P&r\\r^\top&h\end{pmatrix}
\begin{pmatrix}\delta y\\\delta z\end{pmatrix}
\right],
\]

where \(P\in\mathbb S^3\), \(r\in\mathbb R^3\), and \(h>0\). Complete the
square:

\[
\begin{aligned}
\delta y^\top P\delta y+2\delta z\,r^\top\delta y+h\delta z^2
&=\delta y^\top S\delta y
+h(\delta z-\beta\delta y)^2,\\
S&=P-\frac{rr^\top}{h},\\
\beta&=-\frac{r^\top}{h},\\
s_z^2&=\frac1h.
\end{aligned}
\]

Therefore

\[
Z\mid Y=y
\sim
\mathcal N\!\left(m_z+\beta(y-m_y),s_z^2\right),
\]

and the depth-integrated trace is

\[
\bar f(y)
=J(y)A\sqrt{2\pi s_z^2}
\exp\!\left[-\frac12(y-m_y)^\top S(y-m_y)\right].
\]

Conversely, writing \(C=S^{-1}\), the joint covariance reconstructed from the
factorized object is

\[
\operatorname{Cov}(Y,Z)=
\begin{bmatrix}
C&C\beta^\top\\
\beta C&s_z^2+\beta C\beta^\top
\end{bmatrix}.
\]

This proves a bijection between the two parameterizations.

### Parameter count

| Geometry field | Scalars |
|---|---:|
| UVT mean | 3 |
| UVT symmetric SPD precision | 6 |
| Conditional depth intercept | 1 |
| Conditional depth slopes in \(u,v,t\) | 3 |
| Conditional depth variance | 1 |
| Geometry total | 14 |
| Compiled amplitude | 1 |
| RGB | 3 |
| Total | 18 |

This is exactly the same count as a world spacetime mean (4), SPD(4)
covariance (10), amplitude (1), and RGB (3).

The current STAR packet has 17 scalars:

```text
ma[3] + q_uvt[6] + depth0[1] + depth_beta[3]
+ opacity[1] + color[3]
```

It omits exactly one scalar required for this lossless affine Gaussian
geometry: **conditional depth variance**. It preserves a conditional mean
plane, not the full conditional fiber law.

An additional semantic issue remains: after fiber integration the compiled
amplitude includes \(J A\sqrt{2\pi s_z^2}\). It is not automatically identical
to a world-space peak-alpha parameter. Before implementation, amplitude must be
typed as peak alpha, Gaussian mass, or optical thickness and transformed
accordingly.

## 7. Exact depth along nonlinear perspective rays

A global affine \(UVTZ\) Gaussian is not required to obtain exact depth along
each perspective ray.

Let one ray be linear in its chosen depth coordinate:

\[
X_\Gamma(y,z)=a(y)+z\,d(y)\in\mathbb R^4,
\]

and let the world atom have mean \(\mu\), precision \(\Lambda\succ0\), and
peak amplitude \(A\). Define

\[
r(y)=a(y)-\mu,
\quad
h(y)=d(y)^\top\Lambda d(y),
\quad
b(y)=d(y)^\top\Lambda r(y),
\quad
c(y)=r(y)^\top\Lambda r(y).
\]

Completing the one-dimensional square gives

\[
\hat z(y)=-\frac{b(y)}{h(y)},
\qquad
s_z^2(y)=\frac1{h(y)},
\qquad
q_\perp(y)=c(y)-\frac{b(y)^2}{h(y)}\ge0.
\]

For an unbounded fiber with depth-independent Jacobian inside the ray,

\[
\bar\rho(y)
=J(y)A\sqrt{\frac{2\pi}{h(y)}}
\exp[-q_\perp(y)/2].
\]

For a clipped interval \([z_n,z_f]\), multiply by

\[
\Phi\!\left(\sqrt h\,(z_f-\hat z)\right)
-
\Phi\!\left(\sqrt h\,(z_n-\hat z)\right).
\]

Thus perspective does not destroy Gaussianity **along a fixed ray**. It makes
the trace amplitude, mean depth, and variance nonlinear functions of UVT. The
atlas should approximate or encode the functions

\[
q_\perp(y),\qquad \hat z(y),\qquad h(y)
\]

over event-certified domains, with residual and denominator certificates.
This is cleaner than pretending that one large perspective region is globally
affine.

For a global-shutter camera, the depth direction has zero time component. A
rolling-shutter or finite-light-travel model may give the 4D direction a time
component; the same algebra applies if the camera map and fiber measure are
defined consistently.

## 8. Visibility is where early depth collapse becomes lossy

The marginal-plus-conditional factorization above is lossless for one
primitive. Loss occurs when the renderer discards the conditional law or
performs nonlinear visibility after collapsing depth.

For two alpha layers,

\[
I_{12}-I_{21}=\alpha_1\alpha_2(c_1-c_2).
\]

Identical UVT marginals can therefore render differently when their depth
order swaps.

This yields a practical hierarchy:

1. Retain UVT trace, conditional depth mean, and conditional variance.
2. Build effective depth intervals, for example
   \([\hat z-rs_z,\hat z+rs_z]\), with a declared tail tolerance.
3. If intervals are disjoint and order is stable over the cell, use fast
   front-to-back alpha replay.
4. If supports overlap but a commutator/error bound is below tolerance, allow
   the approximation explicitly.
5. Otherwise split the gauge domain or retain the depth fiber and perform a
   small one-dimensional optical-transfer evaluation, as in the WorldFoam
   branch.

The two approaches are not contradictory:

- **World Tubes** pushes depth early and carries sufficient conditional/order
  information for a certified fast path.
- **WorldFoam** retains the depth fiber longer and resolves transmittance before
  collapsing it.

A hybrid can use the same canonical world atom and choose the renderer path per
cell.

## 9. Candidate compiled objects

| Construction | Information retained | Verdict |
|---|---|---|
| Pure UVT marginal | Sensor-time footprint only | Too lossy for visibility |
| Current STAR packet | UVT marginal + affine conditional mean depth | Strong fast-path scaffold; missing depth variance and full producer |
| Joint local UVTZ Gaussian | Full local joint Gaussian | Lossless, but awkward for UVT support/binning |
| **Gaussian FiberTrace** | UVT marginal + conditional Gaussian depth | Recommended exact affine packet |
| Projective FiberTrace | Functions \(q_\perp(y),\hat z(y),h(y)\) + certificates | Recommended perspective extension |
| Retained fiber field | Extinction/emission over \((y,z)\) | Strongest visibility semantics; more expensive |

Recommended affine packet:

```text
FiberTrace:
    ma[3]
    q_uvt_cholesky_or_safe_chart[6]
    depth0[1]
    depth_beta[3]
    log_depth_variance[1]
    trace_amplitude[1]
    appearance[A]
    gauge_domain_id
    support / fit / denominator certificates
    order-or-fallback certificate
```

The first six lines contain the exact 14 geometric degrees of freedom of one
local full Gaussian. The remaining certificate fields are not extra scene
capacity; they record where the compiled approximation is valid.

## 10. Recommended implementation order

1. Implement a CPU-double canonical strict SPD(4) atom with an SPD-safe
   Cholesky parameterization.
2. Implement the affine gauged ray transform and emit a complete
   `FiberTrace`, including conditional depth variance and transformed
   amplitude.
3. Prove in tests that joint UVTZ and marginal-plus-conditional forms round
   trip to numerical precision.
4. Match the existing 17-float STAR path by deliberately dropping/fixing the
   new variance scalar; require exact parity on the restricted fixture.
5. Add compiler VJPs back to \(\mu_4\), \(\Sigma_4\), camera parameters, and
   amplitude. Treat visibility events as piecewise-smooth boundaries, not as
   ordinary smooth sorting.
6. Extend the Metal packet and order certificate to consume depth variance.
7. Compile exact raywise perspective functions into certified local domains;
   compare polynomial/rational approximation against direct ray integration.
8. Add retained-fiber fallback only for ambiguous cells.
9. Integrate this path into the trainer and browser under a name that does not
   overstate the current dynamic-3DGS implementation.

Start with strict finite SPD(4) atoms. Do not add spline trajectories,
time-varying covariance, neural deformation, and retained volumetric transfer
in the same first experiment.

## 11. Falsification and proof obligations

### Algebra gates

- Random SPD(4) world atom \(\leftrightarrow\) conditional block chart round
  trip.
- Joint UVTZ \(\leftrightarrow\) UVT marginal plus conditional depth round
  trip.
- Covariance pushforward and precision Schur complement agree.
- Analytic unbounded and clipped ray integrals agree with high-precision
  numerical quadrature.
- Depth-gauge changes agree only when the correct Jacobian is included.

An immediate 200-case CPU-double spot check passed before this note was
closed: covariance reconstruction max absolute error \(6.44\times10^{-15}\),
precision-Schur agreement \(2.22\times10^{-15}\), conditional-slope agreement
\(1.55\times10^{-15}\), unbounded analytic ray integral max relative error
\(2.05\times10^{-14}\), and clipped-CDF integral max relative error
\(5.56\times10^{-10}\) against a 20,001-sample trapezoidal reference. This is
computational evidence for the formulas, not a substitute for the stated
algebraic proofs or production gradient tests.

### Rendering gates

- Existing STAR parity when depth variance is disabled/frozen.
- Two crossing-depth atoms trigger the certified split/fallback and reproduce
  brute-force ray compositing.
- Stable disjoint depth intervals reproduce ordinary sorted-alpha replay.
- Camera motion and rolling-shutter synthetic fixtures remain consistent across
  gauge-domain subdivision.

### Gradient gates

- Finite-difference checks for mean, all ten SPD(4) coordinates, amplitude,
  camera pose/intrinsics, conditional variance, and compiled coefficients.
- Explicit tests on both sides of a visibility event; no claim of a classical
  derivative exactly at a discrete order change.

### Representation comparison

At matched bytes and matched renderer semantics, compare:

- independent per-frame 3DGS;
- current restricted world tube;
- strict full SPD(4) atoms;
- mixtures/piecewise SPD(4) atoms only after the single-atom residual is
  measured.

The full SPD(4) proposal is weakened if it cannot beat the restricted tube on
controlled linear full-anisotropy scenes, or if the exact compiler cannot
match numerical ray integration before training.

## 12. STAR naming

No authoritative expansion of “STAR” was found in the repository or its
available history. The project uses phrases such as “STAR-GS projected UVT
tube renderer” and “4D screen-time tube renderer,” but never freezes an
acronym expansion. Treat **STAR** as an internal backend name. Any phrase such
as “Sensor-Time Atlas Rasterizer” would be a new backronym and should be
labeled as such rather than presented as recovered history.

## 13. Superseding clarification

Earlier notes used “conditional depth sidecar” and emphasized the
semidefinite persistent-tube boundary. This note sharpens both points:

- conditional depth is properly the disintegration kernel paired with the UVT
  pushforward, not an optional afterthought;
- the canonical minimal experiment should be the strict finite SPD(4)
  spacetime Gaussian; persistence is a separate typed case rather than a
  reason to weaken the core atom.

These corrections do not invalidate the block-Gaussian derivations. They
clarify the object hierarchy and the exact implementation boundary.
