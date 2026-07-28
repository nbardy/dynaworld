# Depth-Fiber and Shader-Boundary Audit

**Date:** 2026-07-23 02:23:17 +0900

## Context

The user challenged three parts of the preceding spacetime-Gaussian audit:

1. whether the intended mathematics ever reached the shaders;
2. whether a true XYZT Gaussian had accidentally been replaced by UVT;
3. whether the repository contained a more principled treatment of ray depth
   than a camera “compiler” and a depth sidecar.

This session audited the active Metal, Torch, trainer, browser, and research
layers and wrote the durable result in
[`06_shader_boundary_and_depth_fiber.md`](../../research_notes/spacetime_gaussian_representation/06_shader_boundary_and_depth_fiber.md).

## Inputs inspected

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/README.md`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py`
- `third_party/fast-mac-gsplat/variants/spacetime_v0/`
- active STAR feature model, interval backend, and trainer files
- main and standalone browser trainers
- `research_notes/gauged_uvt_trace_atlas/`
- `research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md`
- May and July depth/gauge loose notes
- the preceding spacetime representation audit

Three parallel read-only audits covered shader boundaries, representation
archaeology, and alternative depth constructions. No implementation code was
changed.

## Observed facts

### Metal is more complete than the old Gate-0 README implies

Current Metal code implements full packed UVT precision evaluation,
tile-time binning, affine per-pixel depth evaluation, projective interval
replay, ordering/fallback, compositing, and substantial backward kernels.
Later May depth notes and current source supersede older notes that described
the Metal path as scalar-depth only.

### The canonical world front end is not active

The `spacetime_v0` scaffold preserves `mu[4]`, full 4D precision/covariance,
and ray-depth integration in reference form, but it never became the active
Metal/WGSL training chain. Active STAR training generally begins from UVT or
projective records. The restricted world-tube producer lacks full world-z
extent and full spatial covariance.

### Current depth is compiled, fixed metadata

The Metal interval path consumes an affine pixel-dependent depth plane, but
the ordinary feature model emits zero depth slopes, the trainer detaches depth
metadata, the interval backward does not optimize the depth plane as world
geometry, and no conditional depth variance is carried.

### UVT was a deliberate rasterizer boundary

Gate 0 accepted already-projected records to isolate the screen-time binning
and rendering problem. Training those records directly later blurred the
distinction between a camera-independent world object and its camera-specific
compiled trace.

### STAR has no recorded expansion

No canonical acronym expansion was found. Any expansion would be a newly
invented backronym.

## Proved identity: exact Gaussian fiber factorization

In local coordinates \((y,z)=((u,v,t),\text{depth})\), partition the pulled-back
precision as

\[
H=\begin{bmatrix}P&r\\r^\top&h\end{bmatrix}.
\]

Completing the square gives

\[
S=P-rr^\top/h,
\qquad
\mathbb E[Z\mid Y=y]=m_z-r^\top(y-m_y)/h,
\qquad
\operatorname{Var}(Z\mid Y)=1/h.
\]

The depth integral is a UVT Gaussian with precision \(S\) and amplitude
multiplier \(\sqrt{2\pi/h}\), up to the fiber Jacobian. Conversely these
fields reconstruct the joint covariance exactly. Therefore a joint UVTZ
Gaussian is losslessly equivalent to a UVT Gaussian plus affine conditional
Gaussian depth.

Parameter accounting exposes the current restriction:

```text
current STAR geometry:
    ma 3 + q_uvt 6 + depth0 1 + depth_beta 3 = 13

lossless affine Gaussian geometry:
    current 13 + conditional depth variance 1 = 14
```

With amplitude and RGB, these totals are 17 and 18. The canonical world
Gaussian also has 18 scalars with simple appearance.

## Proved identity: exact Gaussian section along each perspective ray

For a ray \(X(y,z)=a(y)+z d(y)\) and world precision \(\Lambda\), define

\[
h=d^\top\Lambda d,
\quad b=d^\top\Lambda(a-\mu),
\quad c=(a-\mu)^\top\Lambda(a-\mu).
\]

Then the conditional ray depth is exactly Gaussian with

\[
\hat z=-b/h,
\qquad s_z^2=1/h,
\]

and the unbounded ray integral is proportional to

\[
\sqrt{2\pi/h}\exp[-(c-b^2/h)/2].
\]

A finite near/far interval adds the corresponding Gaussian CDF difference.
Perspective therefore makes these parameters nonlinear over UVT; it does not
make the one-dimensional ray section non-Gaussian.

### Numerical spot check

A deterministic 200-case CPU-double probe used random SPD(4) covariance and
precision matrices plus random 4D rays. Results:

```text
covariance factorization round trip, max abs   6.439e-15
precision Schur agreement, max abs             2.220e-15
conditional depth slope agreement, max abs     1.554e-15
unbounded analytic ray integral, max relative  2.051e-14
clipped-CDF ray integral, max relative          5.562e-10
```

The integral reference used 20,001-point trapezoidal quadrature. These numbers
are computational corroboration, not a production implementation test.

## Current model

**Current belief:** the clean first system should use strict full SPD(4)
Gaussian atoms in world XYZT, then compile each atom into a trace-plus-
conditional-fiber representation over certified camera-ray gauge domains.

**Confidence:** high for the affine Gaussian equivalence and implementation
boundary; medium for the proposed hybrid visibility policy until benchmarked.

**Could be wrong if:** the amplitude/opacity semantics chosen by the renderer
cannot be made consistent with the fiber pushforward, projective certificate
cost overwhelms the saved per-frame work, or visibility ambiguity forces
retained-fiber evaluation on most pixels.

## Backtracks and terminology changes

- “We did not implement the math in shaders” is **invalidated as a broad
  claim**. The compiled renderer math is substantial. The missing boundary is
  the world producer and differentiable lowering.
- “Depth sidecar” is **weakened terminology**. The exact object is a UVT
  marginal paired with its conditional fiber law.
- “Camera compiler” remains acceptable **only as systems shorthand**. The
  mathematical operator is the gauged camera-ray pullback/fiber-pushforward.
- Making a semidefinite persistent cylinder the practical center of the model
  is **de-emphasized**. Start with finite SPD(4) ellipsoids; type persistence
  separately.

## Branches

### Branch A: Gaussian FiberTrace fast path

Store UVT marginal plus conditional Gaussian mean and variance. Certify stable
depth intervals and use sorted alpha compositing.

Cheap falsifier: compare with numerical ray rendering on two crossing,
overlapping anisotropic atoms.

### Branch B: exact projective functions

Compile \(q_\perp(y)\), \(\hat z(y)\), and \(h(y)\) over small gauge domains
instead of forcing a globally affine UVTZ Gaussian.

Cheap falsifier: measure atlas size and residual as field of view, primitive
extent, and camera motion increase.

### Branch C: retained-fiber fallback

When depth support overlaps or order is unstable, evaluate a small 1D
conditional-Gaussian mixture or WorldFoam-style optical transfer.

Cheap falsifier: count the ambiguous-cell fraction and compare quality/cost
against simple subdivision.

### Branch D: direct UVT remains sufficient

For one fixed camera path, direct UVT optimization may remain the best
engineering baseline despite lacking a reusable world object.

Cheap falsifier: compare held-out cameras and camera-path edits at matched
bytes and time. This branch should not be dismissed merely because its object
is less elegant.

## Falsification plan

1. Random SPD(4) round trips through both conditional block charts.
2. Analytic ray integral versus high-precision quadrature, including clipping.
3. Gauge-change test with and without the required Jacobian.
4. Exact parity with the current STAR ABI when depth variance is disabled.
5. Crossing-depth visibility fixture and ambiguous-cell fallback.
6. Finite-difference compiler VJP for all world covariance coordinates and
   camera parameters.
7. Matched-byte comparison against per-frame 3DGS, direct UVT, and restricted
   world tubes.

## Decision implications

- Do not rewrite the Metal renderer first; preserve it as the compiled back
  end.
- Build the missing world-to-`FiberTrace` producer in a reference lane.
- Add the one missing conditional-depth variance scalar and type the compiled
  amplitude.
- Keep XYZT as canonical, UVT as camera-conditioned, and the atlas as a cache.
- Make the browser labels honest until the canonical path is genuinely wired.

## Open questions

- Which amplitude semantics should be frozen first: peak alpha, normalized
  Gaussian mass, or optical thickness?
- What tail probability defines an effective conditional depth interval?
- Is subdivision or retained-fiber quadrature cheaper for the observed
  visibility-event distribution?
- Can the projective trace coefficients and certificates be differentiated
  stably enough for joint camera optimization?
- How much of the current UVT Metal backward can be reused after introducing
  the canonical world compiler VJP?
