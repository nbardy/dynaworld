# Ray Transfer Falsification And Publication Strategy

Date: 2026-07-26

Status: pre-registration and publication decision for the retained-fiber
ray-transfer proposal; thresholds are proposed gates, not measured results

Companion audits:

```text
research_notes/worldfoam_paper/scientist_notes/
2026-07-26_gauge_invariant_ray_holonomy_intake_and_paper_split.md

research_notes/worldfoam_paper/scientist_notes/
2026-07-26_ray_transfer_lineage_and_novelty_audit.md
```

## Executive Decision

Do not turn the current proposal into the current STAR / World Tubes paper.

Do not open a new implementation lane merely because the formulation is
elegant.

Use this publication sequence:

```text
1. Finish World Tubes / STAR Paper A.
2. Run one bounded retained-transfer falsification ladder.
3. Promote WorldFoam to Paper B only if correctness, native systems, and
   public quality gates all clear.
4. Promote the convex-potential atom as a separate representation
   contribution only if it wins independently of the transfer renderer.
```

This sequence is not conservative for its own sake. It follows the evidence:

```text
STAR:
    implemented compiler, Metal forward/VJP, saved scaling artifacts,
    one public matched split, incomplete breadth

WorldFoam:
    complete core algebra, reference fixture, owner-run/Metal prototypes,
    incomplete native parity and public quality

convex-potential proposal:
    theorems and design only
```

The immediate scientific bottleneck remains public breadth and protocol
closure for Paper A, not missing renderer mathematics.

## 1. Three Possible Paper Outcomes

### Outcome A: No separate paper

This is the correct outcome if:

```text
STAR certificates and live fallback already control overlap error
retained transfer adds substantial per-ray cost
convex-potential atoms do not beat simpler producers
real scene event density destroys amortization
```

Preserve the result as:

```text
a World Tubes limitation theorem
a WorldFoam theory/prototype note
a negative systems result
```

This is scientifically useful and avoids publishing a renamed standard volume
renderer.

### Outcome B: WorldFoam is a separate Paper B

This is justified only if the retained-depth compiler demonstrates all three:

```text
semantic necessity:
    a measured overlap/fallback regime where early marginalization loses

systems value:
    native compiled forward/backward beats same-world replay over camera time

representation viability:
    competitive heldout quality and memory on public scenes
```

The convex-potential atom may be one producer ablation. It does not need to be
the paper's main representation.

### Outcome C: Convex-potential atoms become a separate representation paper

This requires a result that survives changing the renderer:

```text
better quality per byte than strict SPD(4), swept Gaussian, and cell/FEM
better or comparable optimization stability
lower matched-quality atom count
no worse camera-compiler event complexity
```

If the atom wins only when paired with a new retained-transfer renderer, the
causal contribution is ambiguous and should remain inside Paper B.

## 2. Claims Allowed Now

The following are defensible today when scoped precisely.

### Shared framework

> A camera program defines the observation bundle; a gauge selects local
> coordinates within that bundle.

> Ray-depth reparameterizations preserve optical transfer when the pulled
> extinction and emission measures include the correct Jacobian.

### Operator boundary

> Visibility generally does not commute with depth marginalization.

> Two colored alpha events fail to commute by
> `alpha_i alpha_j (c_i-c_j)`.

> World Tubes handles this boundary with conditional depth, order strata,
> commutation bounds, and fallback; WorldFoam retains depth and evaluates
> optical transfer.

### Convex-potential construction

Under declared smoothness, bounded-time, positivity, and uniform
strong-convexity assumptions:

> The self-normalized potential has one unique spatial ridge, compact convex
> slices, a regular support boundary, at most one support interval on a
> straight ray, and an algebraically evaluable single-atom optical depth.

### Current evidence

> STAR has implemented projective/orbit interval compilation, Metal forward
> and direct VJP, gauge value/gradient checks, visibility/fallback paths,
> finite exposure, rolling shutter, and measured world-side reuse.

> WorldFoam has a tested optical-transfer reference, fixed-word VJP checks, and
> owner-run/Metal prototype evidence, but not a completed native
> multi-scene rendering result.

## 3. Claims That Must Not Be Made Yet

### Novelty overclaims

Do not claim:

```text
the first gauge-invariant renderer
a new emission-absorption equation
a new path-ordered visibility operator
a new noncommutativity theorem relative to this repository
a new direct-adjoint principle
```

The transfer algebra is standard, and the repository already contains the
same WorldFoam equations and fixtures.

### Terminology overclaims

Do not claim:

```text
moving cameras are gauge transformations
open-ray transfer is holonomy without qualification
half-angle orbit time is merely a ray-depth coordinate
```

Use:

```text
camera-program-compiled retained-fiber optical transfer
```

as the default paper language.

### Exactness overclaims

Do not claim:

```text
exact arbitrary colored overlap in closed form
exact finite exposure for general camera/appearance programs
exact rolling shutter for general programs
endpoint-free gradients through all topology events
globally smooth derived orientation
```

The exact boundary is narrower:

```text
single-atom support and optical depth:
    exact or algebraically certified

arbitrary colored overlap:
    numerical transfer with a declared tolerance

fixed topology:
    direct VJP can be exact for the implemented approximation

topology changes:
    refresh, fallback, smoothing, or boundary terms are required

orientation:
    unique only on simple-spectrum Hessian strata
```

### Complexity overclaims

Do not claim:

```text
sublinear total rendering
frame-independent pixel work
constant event complexity
no visibility events
no per-ray sorting or active-set work
cheap incremental recompilation
```

The paper-safe statement is:

> Structural world-side work is factored through a compiled object whose size
> follows measured chart, support, discriminant, incidence, and integration
> complexity rather than being forced to follow primitive-frame count.

Output remains at least `Omega(P T)`.

### Representation overclaims

Do not claim:

```text
the atom stores no position information
strong convexity gives a unique quaternion
one convex atom represents arbitrary dynamic topology
the structured parameter count equals identifiable degrees of freedom
```

## 4. Falsification Gate 0: Semantic And Naming Correctness

Goal:
    remove framing errors before any kernel work.

Required derivations:

```text
1. State camera-program, base-chart, and ray-depth coordinate changes
   separately.

2. Derive depth, log-depth, and inverse-depth transfer with their physical
   distance Jacobians.

3. Derive an exposure-time reparameterization separately, including the
   shutter measure.

4. Show the affine matrix product integral is exactly equivalent to scalar
   emission-absorption rendering.

5. State open-path basis covariance:
   U' = G(s1)^-1 U G(s0).
```

Numerical gates:

```text
depth/log/inverse value relative error:
    <= 1e-8

depth/log/inverse VJP relative error:
    <= 1e-6

missing-Jacobian negative control:
    >= 1e-2 relative error

matrix-product versus scalar transfer:
    <= 1e-10 in float64 reference
```

Kill or rename if:

```text
the claimed gauge invariance depends on changing the physical camera
the internal-basis and coordinate gauges remain conflated
the method requires "holonomy" to sound novel
```

## 5. Falsification Gate 1: Primitive Capacity

Goal:
    determine whether the convex-potential atom is more than a clean
    parameterization.

Baselines:

```text
strict SPD(4) Gaussian
existing low-knot swept Gaussian P(t),q(t),r(t)
two- or four-component SPD(4) mixture
capped determinant atom
P1 direct-extinction cell
P1 and P2 log-extinction cells
```

Fixtures:

```text
accelerating anisotropic blob
rotating covariance
changing scale
thin curved support
two-component or branched shape as an explicit failure control
```

Hold fixed:

```text
stored bytes
appearance model
training samples
optimizer budget
renderer and integration tolerance
```

Report:

```text
fit error on training rays
heldout ray error
support Hausdorff or occupancy error
iterations to threshold
gradient and Hessian condition diagnostics
atom count
stored bytes
minimizer solve time
root solve time
trace-fit rank
chart and event count
```

Proposed promotion criterion:

```text
The convex atom must be Pareto-optimal on at least two nontrivial fixtures and
deliver either:

    >= 20% lower heldout error at matched bytes

or:

    >= 20% fewer bytes at matched heldout error

without:

    > 1.5x optimizer steps
    > 1.25x compiler/event cost at matched quality.
```

Kill the atom as a main contribution if:

```text
a small Gaussian mixture matches it
P1/P2 cells match it
implicit minimizer/root costs erase its state advantage
the simple-spectrum orientation domain is too fragile
the support topology is too restrictive
```

The transfer compiler may continue with another primitive if this gate fails.

## 6. Falsification Gate 2: Decisive Overlap Correctness

Goal:
    prove a practical semantic need for retained depth.

Build a deterministic reference scene containing:

```text
two or three differently colored volumetric supports
substantial optical-depth overlap
a moving camera whose rays move through the overlap
one control with equal colors, where ordering should not matter
one disjoint-support control
one thin surface-like control
```

Reference:

```text
high-accuracy dense emission-absorption integration
double precision
finite-difference camera and world gradients
declared integration tolerance
```

Comparators:

```text
STAR representative-depth/order path
STAR stratified/fallback path
per-frame retained-fiber replay
compiled retained-fiber transfer
```

Report by pixel and event stratum:

```text
RGB max, mean, p95, p99 error
VJP max and p95 relative error
commutator energy
order-strata count
fallback fraction
active atoms per pixel
support endpoints per ray
integration evaluations per ray
```

Proposed promotion criterion:

```text
On the predeclared overlap case, compiled retained transfer must:

    reduce STAR non-fallback RGB and VJP error by >= 5x
    match retained replay to <= 1e-4 RGB max error
    match retained replay VJP to <= 1e-3 relative error

while controls remain within the same tolerances.
```

Kill the "practical visibility fix" claim if:

```text
STAR fallback already matches the reference cheaply
commutator energy does not predict error
the gain exists only after changing world capacity or appearance
retained transfer fails the disjoint/equal-color controls
```

## 7. Falsification Gate 3: Native Systems Path

Goal:
    distinguish a renderer from a Python reference.

Requirements:

```text
native GPU forward
native GPU direct VJP
no CPU root solver in the timed steady-state path
no hidden N x T transformed-primitive array
no hidden per-frame total-order table
same-world retained replay baseline
separate compile and materialization timing
```

Frame counts:

```text
T = 4, 8, 16, 32, 64, 128
```

Scene axes:

```text
active atoms per pixel
support endpoint count
polynomial degree
integration tolerance
camera orbit speed
near-camera support
opacity depth
color contrast
```

Report:

```text
atlas bytes
root/minimizer certificate bytes
active incidence bytes
compile time
recompile time
forward time
backward time
peak device memory
integration steps mean/p95/max
event count and chart count
fallback fraction
break-even frame count
```

Proposed promotion criteria:

```text
compiled versus retained replay at T=128:
    persistent structural payload ratio <= 0.20
    structural compile-work ratio <= 0.25

steady-state:
    break-even by T <= 32
    >= 2x total forward+backward speedup over retained replay at T >= 64

low-commutator control:
    <= 1.5x STAR total forward+backward time

accuracy:
    all ratios reported at matched RGB/VJP tolerance
```

Kill the systems-paper claim if:

```text
integration work grows like dense per-atom replay
hbar or endpoint count dominates
root certification remains on CPU
the compiled payload hides sample-linear state
break-even occurs after the intended sequence length
```

## 8. Falsification Gate 4: Training And Recompilation

Goal:
    determine whether the compiled object survives optimization.

Gradient checks:

```text
potential coefficients
opacity coefficients
constant and varying appearance
camera pose
ray-depth gauge parameters
support endpoints
minimizer normalization
```

Required controls:

```text
fixed topology finite differences
regular endpoint motion
generic tangency
endpoint-order crossing
chart denominator approach
support birth/death
```

Numerical gates:

```text
fixed-topology coefficient VJP:
    <= 1e-5 relative error

regular endpoint/camera VJP:
    <= 1e-4 relative error

compiled versus live training loss:
    <= 1e-5 relative difference on fixed records

NaN/Inf:
    zero accepted steps
```

Instrumentation:

```text
dirty-record fraction per step
certificate margin quantiles
records split per step
records falling back per step
compile/recompile time fraction
optimizer step size and trust-region rejection
```

Proposed promotion criteria:

```text
median dirty-record fraction <= 10%
p95 dirty-record fraction <= 30%
median compile/recompile time <= 20% of step time
```

If median dirty fraction exceeds `50%` or recompilation dominates, demote the
method to:

```text
frozen-world inference/export compiler
```

rather than claiming a shared training renderer.

## 9. Falsification Gate 5: Public Evidence

Goal:
    establish that the method matters outside a constructed counterexample.

Minimum breadth:

```text
>= 3 public scenes
>= 3 camera train/heldout splits per scene where available
>= 3 seeds
one controlled synthetic overlap set
one low-overlap control set
one moving/revolving camera stress set
```

Required methods:

```text
World Tubes / STAR
STAR with fallback
per-frame retained-transfer replay
compiled WorldFoam transfer
dynamic 3DGS contextual baseline
convex-potential producer only if Gate 1 passed
```

Required metrics:

```text
PSNR
SSIM
LPIPS
L1
heldout camera metrics
train-view metrics
compile / forward / backward time
peak memory
checkpoint and atlas bytes
event density
fallback rate
recompile rate
```

Paper-B promotion requires:

```text
1. Same-world compiled/replay quality and gradient parity.

2. A statistically stable heldout improvement over STAR on a predeclared
   overlap/fallback subset.

3. No material degradation on low-overlap controls.

4. A systems win over retained replay at the intended camera-path length.

5. At least one native GPU platform with complete forward and VJP.
```

Do not require or claim broad SOTA unless the actual table supports it. A
strong compiler paper can be based on same-representation equivalence and
systems scaling, with external methods used as context.

## 10. Falsification Gate 6: External Novelty

Goal:
    make the final paper claim defensible outside the repository.

Before submission, perform a dedicated primary-source review covering:

```text
classical direct volume rendering
product integrals and transfer matrices
differentiable participating-media rendering
deterministic and exact segment integration
dynamic radiance fields
compact-support radial and polynomial primitives
learned convex implicit primitives
kinetic data structures and certified root tracking
camera-path or light-field precomputation
4D Gaussian and spacetime-cell rendering
```

Required output:

```text
one table of closest methods
one sentence per claimed difference
one implementation-level distinction
one experiment that isolates each claimed difference
```

Kill or narrow the paper if:

```text
the closest prior method already compiles equivalent transfer over camera time
the only difference is geometric vocabulary
the only new component is an uncompetitive primitive
the contribution cannot be stated without combining several standard facts
```

## 11. Hard Kill Criteria

Stop the standalone Paper-B push if any of the following persists after one
targeted repair:

```text
K1. No measured STAR overlap/fallback failure on public data.

K2. Retained transfer does not materially reduce RGB/VJP error on the
    predeclared crossing scene.

K3. Same-world compiled transfer cannot match retained replay.

K4. Native integration cost erases camera-time amortization.

K5. Active overlap, support endpoints, or chart count scales pathologically.

K6. Training dirties most records on most steps.

K7. Convex-potential atoms lose to a small Gaussian mixture or P1/P2 cell
    field at matched bytes.

K8. Gains disappear when camera, appearance, and world capacity are matched.

K9. Results remain synthetic-only.

K10. The novelty statement reduces to standard emission-absorption plus the
     word "holonomy."
```

Failure of the atom gate does not kill WorldFoam. Failure of retained-transfer
systems or quality gates does.

## 12. Publication Strategy For Paper A

Keep the current public method:

```text
World Tubes in Gauged Camera Space
implemented by projective STAR UVT
```

Add only:

```text
1. A precise camera-program versus gauge distinction.

2. The two-layer noncommutation counterexample as the boundary of early
   marginalization.

3. A diagram showing the early-pushforward and retained-fiber operator fork.

4. A limitation paragraph explaining that WorldFoam is a different
   representation with different quality and systems requirements.
```

Do not add:

```text
the convex-potential primitive
an unimplemented transfer ODE
root-certification machinery
new "ray holonomy" branding
speculative Paper-B performance claims
```

Reason:

```text
Paper A already has a coherent causal claim and implementation.
Adding Paper B changes the representation, renderer, baselines, and proof
burden while public breadth for Paper A is still incomplete.
```

## 13. Publication Strategy For Paper B

Preferred working title:

```text
WorldFoam: Camera-Program Compilation of Retained-Fiber Optical Transfer
```

Alternative:

```text
Compiled Ray-Fiber Transfer for Dynamic Camera Programs
```

Avoid leading with:

```text
Gauge-Invariant Ray Holonomy
```

because:

```text
depth-coordinate invariance is necessary but not the main systems novelty
open-ray transport is not standard holonomy terminology
the phrase invites a "standard volume rendering in geometric language"
review
```

Recommended contribution order:

```text
1. Retained-fiber camera-program intermediate representation.

2. Certified event/interval compiler and same-world replay theorem.

3. Native forward and direct adjoint with measured structural reuse.

4. Decisive noncommuting-overlap correctness and fallback comparison.

5. Public heldout quality, memory, and break-even results.

6. Optional convex-potential or FEM producer ablation.
```

The theorem section may use:

```text
product integral
parallel transport
visibility monoid
commutator
```

as explanatory mathematics. The paper title and abstract should lead with the
implemented compiler and measured result.

## 14. Publication Strategy For The Atom

Keep the atom out of the Paper-B title unless Gate 1 shows an independent
Pareto win.

If it passes, a separate representation paper could claim:

```text
self-normalized strongly convex spacetime potentials
certified compact connected support
intrinsic ridge and local shape
one-interval straight-ray support
quality/byte and compiler-complexity gains
```

That paper must compare against:

```text
strict SPD(4)
swept Gaussians
mixtures
compact radial bases
convex implicit primitives
P1/P2 finite-element fields
```

The paper must not treat "derived center" as removing positional information.
Its contribution must be measured capacity, conditioning, or certified
renderability.

## 15. Recommended Next Action

Do not launch a broad implementation.

When Paper A's current public matrix is no longer the bottleneck, run exactly
two small gates in this order:

```text
1. Decisive overlap reference:
   prove that retained transfer closes a real STAR error under matched world
   state and measure the added per-ray cost.

2. Primitive capacity:
   compare convex-potential, swept Gaussian, mixture, and P1/P2 cell fields at
   matched bytes through the same retained-transfer reference.
```

Decision:

```text
overlap gate fails:
    stop Paper B systems work

overlap gate passes, atom gate fails:
    continue WorldFoam with simpler cells or Gaussian density

both pass:
    build the smallest native retained-transfer kernel and run Gate 3
```

This ordering isolates the two hypotheses and prevents a large combined
implementation from hiding which idea, if either, is useful.

