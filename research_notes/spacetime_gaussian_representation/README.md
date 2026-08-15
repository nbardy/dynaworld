# Spacetime Gaussian Representation Audit

**Status:** representation decision recommended; implementation and benchmark
gates remain open.

**Audit date:** 2026-07-23

## Decision in one sentence

Define the canonical **finite-lifetime atom** as

\[
\mathcal G_i=(\mu_{4,i},\Sigma_{4,i},\alpha_i,a_i),
\qquad
\mu_{4,i}\in\mathbb R^3\times\mathbb R,
\quad
\Sigma_{4,i}\in\operatorname{SPD}(4),
\]

optimize the same object in the exact conditional block coordinates
\((x_0,t_0,C,v,\sigma_t)\), and compile it through the camera ray bundle into
gauged sensor-time records before rasterization. Start the canonical
implementation with strict finite SPD(4) ellipsoids. Treat exactly persistent
background as a separate typed static primitive, a bounded persistent tube, or
a deliberate \(\lambda_t=0\) semidefinite limit rather than weakening the core
dynamic atom.

This is not a compromise between a “true 4D Gaussian” and a “velocity tube.”
With a full spatial covariance, the two are exactly the same Gaussian written
in different coordinates. The current paper implementation is restricted
because it replaced the full spatial \(C\in\operatorname{SPD}(3)\) with only
two fronto-parallel precisions, not because it exposes velocity.

## The object, without implementation shorthand

Let \(z=(x,y,z,t)\). One primitive is the peak-normalized field

\[
\rho_i(z)=
\alpha_i
\exp\!\left[-\frac12(z-\mu_{4,i})^\top
\Sigma_{4,i}^{-1}(z-\mu_{4,i})\right].
\]

Its geometric degrees of freedom are:

| Parameter | Effective degrees of freedom |
|---|---:|
| Spacetime center \(\mu_4=(x_0,y_0,z_0,t_0)\) | 4 |
| Symmetric positive-definite \(4\times4\) covariance | 10 |
| Geometry total | 14 |
| Peak opacity/amplitude | 1 |
| Simple RGB appearance | 3 |
| Simple primitive total | 18 |

The covariance already contains all four principal widths and the full
six-degree-of-freedom orientation of a 4D ellipsoid. There is no missing
rotation field. If a spectral representation is wanted, \(SO(4)\) can be
represented by a **pair of unit quaternions**, not an octonion. In code, a
Cholesky or conditional block representation is less redundant and makes the
space/time units clearer.

## What one such Gaussian does over time

Write

\[
\Sigma_4=
\begin{bmatrix}A&b\\b^\top&c\end{bmatrix},
\qquad
v=b/c,
\qquad
C=A-bb^\top/c.
\]

At a time \(t=t_0+\tau\), its unnormalized spatial slice is

\[
\rho_t(x)=
\alpha e^{-\tau^2/(2c)}
\exp\!\left[-\frac12
(x-x_0-v\tau)^\top C^{-1}(x-x_0-v\tau)\right].
\]

Therefore one full 4D Gaussian gives exactly:

- an affine spatial center \(x(t)=x_0+v(t-t_0)\);
- a constant conditional 3D covariance \(C\), including a fixed spatial
  orientation and three principal widths;
- a Gaussian temporal presence envelope centered at \(t_0\);
- an independent peak amplitude \(\alpha\).

It does **not** give a spatial covariance that rotates or changes scale over
time. A time-varying \(C(t)\), curved center \(x(t)\), or non-Gaussian temporal
envelope defines a richer conditional Gaussian tube, not one joint Gaussian.
That distinction is mathematical, not naming preference.

## Why “take a slice” is correct but incomplete

An instantaneous frame is a raw time slice of the world field, preserving its
temporal amplitude. Normalizing to \(p(x\mid t)\) would erase birth/death.
Marginalizing time would produce a static motion-blurred occupancy rather than
a frame.

The renderer should not necessarily materialize that 3D slice as a fresh
per-frame splat bank. The repository's more mature operator ordering is

\[
\boxed{
\text{world primitive}
\xrightarrow{\Gamma^*}
\text{camera-ray pullback}
\xrightarrow{\pi_*}
\text{ray-depth pushforward}
\longrightarrow
\text{gauged UVT trace}
\longrightarrow
\text{time slice / shutter integral / composite}.
}
\]

For an affine local camera chart and a Gaussian primitive, the depth
pushforward is again a Gaussian and is computed by a Schur complement. Under
perspective cameras this is local, segmented, rational/projective, or guarded
by a fallback certificate. Visibility/order cannot in general be marginalized
away with no side information.

## What was recovered from the repository

The elegant formulation was preserved, not deleted:

- [`spacetime_v0/docs/handoff.md`](../../third_party/fast-mac-gsplat/variants/spacetime_v0/docs/handoff.md)
  defines one world-spacetime `float4` mean and full `float4x4` precision, then
  integrates it along the ray-depth fiber into a sensor-time Gaussian.
- [`phase_2_world_tube_projection.md`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/phases/phase_2_world_tube_projection.md)
  records the exact block precision normal form, while explicitly calling the
  implemented fronto-parallel projection a weaker scaffold that does not cover
  full anisotropic 3D covariance.
- [`star_uvt_notes.md`](../../research_experiments/star_uvt_notes.md)
  later names a gauge-charted primitive with \(\eta_i\in\mathbb R^4\) and
  \(\Lambda_i\in\mathbb S_{++}^4\), with the constant-velocity implementation
  treated as a special chart.
- [`02_gaussian_fiber_pushforward/README.md`](../gauged_uvt_trace_atlas/02_gaussian_fiber_pushforward/README.md)
  derives the full world-to-UVT Schur complement and identifies missing depth,
  gauge, projective, and validity metadata.

The active `WorldTubeBatch`, by contrast, stores only `x0`, `velocity`, `t0`,
`precision_xy[2]`, `lambda_t`, opacity, and RGB. It has no world \(z\) extent,
no full \(3\times3\) spatial covariance, and no world spatial orientation.
That implementation narrowing is the source of the current conceptual
confusion.

The audit recursively searched the repository's Markdown corpus for the
representation vocabulary and fully inspected the relevant design, experiment,
handoff, and code-contract hits. It did not linearly reread unrelated product,
dataset, or infrastructure notes. Git history exposed no relevant deleted
Markdown formulation; some older third-party history is unavailable before the
submodule consolidation, so exact authorship dates for those imported files
remain uncertain.

## Recommendation hierarchy

1. **G0 reference atom:** full world \((\mu_4,\Sigma_4)\), opacity, and
   appearance for finite-lifetime content.
2. **T0 practical primitive:** the same full spatial Gaussian/tilt form with
   typed localized or persistent activity; persistent is the
   \(\lambda_t=0\) precision-space boundary on a bounded interval.
3. **Optimization chart:** full spatial SPD(3) Cholesky + spacetime tilt
   \(v\) + positive temporal scale. This is lossless relative to SPD(4).
4. **Compilation contract:** camera-specific UVT precision plus conditional
   depth statistics, gauge identity, support/fit certificate, and ordering or
   fallback metadata.
5. **First capacity extension:** a short mixture or piecewise chain of full 4D
   Gaussians. It preserves Gaussian compiler algebra while approximating curved
   paths and changing aggregate orientation/scale.
6. **Only after a measured residual:** introduce a generalized tube with
   spline \(x(t)\) and SPD-valued \(C(t)\). This is a distinct primitive and
   should be named as such.

Do not add spline velocity, time-varying rotation, time-varying scale, and
neural deformation simultaneously. First restore the six spatial covariance
degrees of freedom that the current world scaffold removed, then falsify the
single-Gaussian hypothesis on controlled curved/rotating scenes.

## Document map

- [01_foundations.md](01_foundations.md) — standard 3DGS parameters, full 4D
  covariance, exact tube equivalence, rotation, units, and rigidity proofs.
- [02_slicing_projection_and_opacity.md](02_slicing_projection_and_opacity.md)
  — slice/condition/marginal distinctions, depth-fiber pushforward, exposure,
  opacity conventions, and visibility.
- [03_repository_archaeology.md](03_repository_archaeology.md) — chronological
  recovery of the repository's candidate representations and exact narrowing.
- [04_formulation_catalog.md](04_formulation_catalog.md) — twenty concrete
  formulations, their capacity, parameter cost, strengths, and failure modes.
- [05_decision_and_experiments.md](05_decision_and_experiments.md) — staged
  implementation, fair baselines, falsifiers, kill criteria, and unresolved
  choices.
- [06_shader_boundary_and_depth_fiber.md](06_shader_boundary_and_depth_fiber.md)
  — exact implemented/missing boundary, UVT-plus-conditional-depth
  factorization, perspective ray integral, visibility, and the recommended
  `FiberTrace` compiler contract.
- [07_concern_status_and_build_plan.md](07_concern_status_and_build_plan.md)
  — exhaustive thread concern ledger, on-disk version inventory, exact scope
  of the moving-camera/gauge/occlusion path, and the ordered mathematics,
  implementation, testing, baseline, and memory-safety plan.
- [08_native_motion_bundles_and_shared_raster.md](08_native_motion_bundles_and_shared_raster.md)
  — one hundred numbered equations proving how SPD(4) already moves position,
  why it cannot rotate or curve, how a swept Gaussian remains a native 4D
  volume, why ray-depth Schur elimination survives arbitrary temporal motion,
  and how the shared raster/adjoint can support several world source languages.
- [09_screen_time_compiler_followup_audit.md](09_screen_time_compiler_followup_audit.md)
  — audits the external screen-time proposal, separates shared parameters from
  shared structural work, states the output lower bound and conditional
  trace/event complexity, restores the affine hypothesis for exact Gaussian
  closure, and classifies the result as a World Tubes paper refinement.

## Primary external anchors

- Kerbl et al., [3D Gaussian Splatting for Real-Time Radiance Field
  Rendering](https://arxiv.org/abs/2308.04079) and the
  [official implementation](https://github.com/graphdeco-inria/gaussian-splatting).
- Yang et al., [Real-time Photorealistic Dynamic Scene Representation and
  Rendering with 4D Gaussian Splatting](https://arxiv.org/abs/2310.10642), a
  native full 4D Gaussian model using four scales and a pair of quaternions for
  4D rotation.
- Li et al., [Spacetime Gaussian Feature Splatting for Real-Time Dynamic View
  Synthesis](https://arxiv.org/abs/2312.16812), a different conditional model
  with temporal Gaussian opacity and polynomial position/rotation.
- Luiten et al., [Dynamic 3D Gaussians](https://arxiv.org/abs/2308.09713), a
  persistent 3DGS model whose Gaussians move and rotate while retaining other
  properties.
- Wu et al., [4D Gaussian Splatting for Real-Time Dynamic Scene
  Rendering](https://arxiv.org/abs/2310.08528), which uses 3D Gaussians plus a
  4D neural deformation encoding rather than one native SPD(4) Gaussian per
  primitive.
