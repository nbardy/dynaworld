# Native motion and shared raster mathematics

**Time:** 2026-07-23 13:38:03 +0900

**Role:** coordinator, derivation author, source auditor, and numerical
falsifier

**Objective:** backtrack on the claim that the native SPD(4) representation
removed position motion; determine whether moving/rotating position can remain
one native spacetime volume; determine whether ray-depth Schur elimination and
shared forward/backward raster work require a globally 4D-Gaussian source.

**Why attempted:** the prior representation audit correctly recovered full
SPD(4), but the discussion still conflated a fixed joint covariance with a
fixed spatial center, treated \(p=F(t)\) as necessarily frame-local, and did
not cleanly separate the time-conditioning Schur complement from the
ray-depth Schur complement.

## Inputs used

- `research_notes/spacetime_gaussian_representation/01_foundations.md`
- `02_slicing_projection_and_opacity.md`
- `04_formulation_catalog.md`
- `06_shader_boundary_and_depth_fiber.md`
- `07_concern_status_and_build_plan.md`
- current `WorldTubeBatch`, pinhole/orthographic projection, feature-tube, and
  STAR UVT source paths;
- project meta-philosophy, framing, training-contract, and code-organization
  constraints;
- three independent derivation packets:
  - strict SPD(4) movement/rigidity;
  - arbitrary trajectory shared-raster and complexity analysis;
  - bundle/gauge/worldsheet alternatives.

No external literature novelty search was performed. Consequently, the note
claims exact project mathematics and synthesis, not publication-level novelty.
No MPS work was launched.

## Backtracks and corrections

### Prior belief: position motion may have been removed

**Status:** invalidated as a general mathematical and source claim.

**Evidence:** A strict SPD(4) covariance with nonzero \(\Sigma_{xt}\) has

\[
m(t)=x_0+\frac{\Sigma_{xt}}{\Sigma_{tt}}(t-t_0).
\]

Current `WorldTubeBatch` also stores `velocity` and constructs the corresponding
UVT cross terms. A fixed joint `ma`/precision packet therefore does not imply a
fixed conditional center.

**Surviving concern:** the active world producer still omits full spatial
SPD(3) geometry, and one strict SPD(4) atom cannot curve, physically rotate,
or change normalized spatial covariance.

### Prior belief: Schur sharing may require a full 4D Gaussian

**Status:** split into two different claims.

1. The time-conditioning Schur complement is specific to the strict SPD(4)
   source interpretation.
2. The ray-depth completion of squares needs only a spatial Gaussian along an
   affine ray. It remains exact for arbitrary shared \(p(t),Q(t),a(t)\).

This second result means an ordinary low-knot moving/rotating 3DGS source can
reuse the trace-atlas/compiler architecture.

### Prior belief: bundle or time gauge may avoid a trajectory function

**Status:** weakened.

Over a time interval, the Gaussian-state bundle is trivializable. A smooth
worldline transverse to time is locally the graph of \(m(t)\). A comoving
gauge can straighten it, but then the camera and connection become
time-dependent. The bundle language is useful for typed transport and
gauge-correct derivatives, not free representation capacity.

## Derivations and results

The durable note contains exactly 100 numbered equations and six explicit
proof summaries. Principal results:

- **Proved:** every strict SPD(4) Gaussian is exactly a Gaussian-thickened
  affine worldline with a constant spatial cross-section and Gaussian
  lifetime.
- **Proved:** its fixed-time center moves linearly; its normalized covariance
  cannot rotate or change scale.
- **Proved:** a swept Gaussian with low-knot \(m(t),C(t),a(t)\) is one native
  4D scalar field, not a frame bank.
- **Proved:** a ray bundle is the renderer domain, not the motion object.
- **Proved:** exact ray-depth mean, variance, and trace survive arbitrary
  time-varying spatial Gaussian motion in an affine depth coordinate with a
  depth-independent line-measure Jacobian.
- **Proved:** when traces admit a compact basis and a fixed certified event
  partition, coefficient evaluation and its transpose can reuse them without
  an SPD(4) source. This is a sufficient shared-adjoint result, not a theorem
  that joins the whole renderer and backward pass.
- **Proposal:** treat the renderer as a compiler/adjoint for certifiable
  spacetime trace fields, with strict SPD(4) as its simplest exact source.

## Independent red-team corrections

The final mathematical audit preserved the core result and required these
scope corrections:

- inverse ridge Hessian is an exact covariance only for Gaussian slices;
- the \(C+cvv^\top\) marginal formula is for the ambient untruncated Gaussian;
- transported scalar fields and pulled-back density measures have different
  Jacobian factors;
- an object-frame gauge is not a ray-depth gauge and does not remove strain;
- nonlinear depth gauges generally destroy Gaussian depth coordinates even
  though the physical integral remains invariant;
- the basis-size obstruction is scoped to fixed linear or stable
  bounded-precision representations;
- the compute expressions are a cost model, not asymptotic lower bounds;
- certified trace compilation is partial/failable and needs regularity,
  support, denominator, event, and tolerance hypotheses;
- hard sorting kills the discrete permutation derivative inside a stratum,
  not all image gradients involving depth-dependent geometry.

## Numerical checks

CPU-only NumPy checks:

- 500 random well-conditioned SPD(4) matrices:
  - covariance/precision center-slope maximum absolute error:
    \(1.776\times10^{-15}\);
  - completed-square maximum relative error:
    \(2.627\times10^{-15}\).
- 500 random UVT quadratic constructions:
  - recovered velocity maximum absolute error:
    \(7.494\times10^{-15}\).
- 200 curved, rotating spatial Gaussian/ray samples:
  - analytic ray integral versus dense quadrature maximum relative error:
    \(1.138\times10^{-14}\).

These are computational evidence supporting, but not replacing, the proofs.

## Branches preserved

1. full strict SPD(4), correctly implemented;
2. low-knot dynamic ordinary 3DGS feeding the same trace atlas;
3. adaptive mixture/piecewise SPD(4);
4. coherent swept Gaussian worldtube;
5. quadratic-in-space implicit spacetime field;
6. shared flow/deformation field.

The matched A/B comparison is the highest-value next experiment. Branch 3 is
the cheapest expressivity extension. Branch 4 should be promoted only if
curvature/rotation wins materially and a piecewise chain requires too many
components or events.

## Output

- [`research_notes/spacetime_gaussian_representation/08_native_motion_bundles_and_shared_raster.md`](../../research_notes/spacetime_gaussian_representation/08_native_motion_bundles_and_shared_raster.md)

## Precise next actions

1. Add a CPU-only full SPD(4) `WorldObject` reference and assert nonzero
   cross-covariance moves its conditional center.
2. Add the matched low-knot moving/rotating 3DGS source behind the same
   `WorldObject -> FiberTraceAtlas` boundary.
3. Compare constant velocity, acceleration, rotation, scale change, and
   persistence on analytic scenes.
4. Report learned bytes, trace coefficients, event cells, fallback fraction,
   geometry work, shading work, and gradient parity.
5. Keep MPS fail-closed until the CPU gates and a tiny externally monitored
   native microprofile pass.
