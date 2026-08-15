# SPD(4) build and status plan

**Time:** 2026-07-23 02:42:32 +0900

**Objective:** consolidate every concern raised in the current thread into an
independently checkable map of mathematics completed, engineering completed,
missing mathematics, missing engineering, on-disk implementations, and the
shortest safe build sequence.

**Why:** the discussion was conflating four different objects: the canonical
world primitive, its camera-conditioned trace, hard visibility/order, and a
particular restricted/paper/browser implementation.

## Inputs inspected

- the seven-note `research_notes/spacetime_gaussian_representation/` audit;
- gauged UVT trace-atlas foundations, projective camera, visibility, and Metal
  acceptance notes;
- current STAR UVT projective trace/compiler/trainer/Metal source;
- current moving-camera, visibility, gauge, exposure, rolling-shutter, and
  mixed-fallback tests;
- `spacetime_v0`, `star_uvt_v0`, `star_uvt_prt_v0`, `star_prt_v0`, dynamic
  3DGS, WorldFoam, paper-runner, and browser lineages;
- memory-incident, baseline, and progressive paper-run evidence;
- three independent focused audits: concern matrix, camera/gauge/occlusion
  source audit, and on-disk version/build inventory.

No external literature search was needed for this repository-status synthesis.
No MPS workload was launched.

## Checks run

Two CPU-only current-source invocations completed:

1. bundle gauge invariance, gauge-gradient invariance, projective visibility,
   and visibility-stress suites: **46 passed**;
2. orbit windows, exposure/rolling quadrature/backward, and mixed-fallback
   backward, excluding Metal/MPS cases: **44 passed, 3 deselected**.

Total: **90 passed, 3 deselected**.

## Principal findings

1. The moving-camera memory was correct but needed a scope qualification.
   `star_uvt_v0` has per-pixel/time affine depth, event-root splits, UV
   subdivision, certified order strata, stale-atlas refresh, and mixed fallback.
   It is correct relative to its compiled trace/candidate contract inside
   bounded certified domains. It is not the main browser path and is not the
   full paper-runner camera path.
2. Gauge invariance concerns a monotone coordinate change on one physical ray.
   A moving viewpoint changes the ray bundle and physical visibility; the
   compiler/atlas must recompute that order.
3. The canonical strict world object remains
   `(mu_XYZT, SPD4, typed amplitude, appearance)`. The exact conditional block
   chart `(x0, t0, SPD3 C, v, temporal variance)` loses no covariance DOF.
4. In an affine camera chart, a joint UVTZ Gaussian is exactly a UVT marginal
   plus affine conditional Gaussian depth. The compact current affine STAR
   contract is missing conditional depth variance, the one scalar needed for a
   lossless depth law.
5. Perspective preserves an exact one-dimensional Gaussian along each affine
   depth ray; the trace amplitude, conditional mean, and variance become
   nonlinear UVT functions suitable for certified local compilation.
6. The substantial STAR raster/compiler back half exists. The missing front
   half is full world SPD(4) production, exact/typed trace amplitude, conditional
   depth variance, world/camera VJP, trainer integration, broad acceptance, and
   browser integration.
7. The memory incident supports temporal state amortization but not a quality
   victory or a causal attribution of all driver memory. It used four sampled
   frame/view items per step, not all 300 rendered at once; all 300 per-frame
   parameter banks and eager process state still remained resident.

## Output

The durable synthesis is:

- [`research_notes/spacetime_gaussian_representation/07_concern_status_and_build_plan.md`](../../research_notes/spacetime_gaussian_representation/07_concern_status_and_build_plan.md)

It contains the 27-item concern ledger, exact on-disk version table, math tasks
M0–M8, code tasks C0–C7, acceptance ladder, baseline matrix, and ordered build
sequence.

## Claim status

- **Proved/exact under stated assumptions:** SPD(4) conditional block
  equivalence; fixed-time slice; affine UVTZ marginal/conditional factorization;
  perspective one-dimensional ray Gaussian; monotone gauge/Jacobian invariance.
- **Implemented/tested:** compiled UVT/projective rendering; per-pixel/time
  conditional-mean depth order; event splits; UV subdivision; fallback;
  exposure/rolling-shutter and fixed-atlas VJP fixtures.
- **Conditional:** arbitrary camera programs only through local compiled charts,
  residual/support/order certificates, and fallback.
- **Prototype:** canonical full world `spacetime_v0`, retained-depth WorldFoam,
  browser STAR, and several moving-camera forks.
- **Missing:** active full-SPD(4) world-to-FiberTrace compiler, complete depth
  variance/amplitude ABI, world/camera VJP, retained conditional-fiber fallback,
  end-to-end trainer/browser, safe MPS profile, and broad fair baselines.

## Precise next action

Freeze amplitude/units/support/persistence semantics, then implement a CPU-only
strict block-Cholesky `WorldAtom` and exact affine `FiberTrace` with conditional
depth variance and inverse reconstruction. Do not begin with splines, a new
representation name, browser integration, or a full-scale MPS run.
