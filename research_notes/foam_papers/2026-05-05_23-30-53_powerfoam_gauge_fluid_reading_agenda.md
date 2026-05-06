# PowerFoam / Gauge / Fluid Reading Agenda

Date: 2026-05-05 23:30:53 +0700

Scope: paper-reading agenda only. Do not treat this note as implementation
evidence. It assumes the current repo-local boundary: the Metal PowerFoam path
is a partial trainable bounded-cell raster/raytrace/backward core, not full
official PowerFoam; RadFoam/Radiant Foam is not ported locally; dynamic feature
foam can repaint unless held-out camera/time tests force transported motion.

Goal: turn paper mechanisms into falsifiable local tests before they become
architecture work. The reading posture should be skeptical: every attractive
mechanism must answer what state it adds, what invariant it protects, what
gradient/replay data it needs, and what result would make it not worth moving
into the Metal path.

## One-Page Reading Card

For each paper or codebase, extract this card before writing any new code:

```text
Mechanism name:
    ...
Paper object:
    cells / tetrahedra / splats / rays / charts / particles / grid / mesh
State variables:
    trainable tensors, discrete topology, per-ray cache, per-frame state
Observable:
    RGB, alpha, depth, normals, flow, density, material coordinate, energy
Invariant or contract:
    visibility correctness, gauge covariance, conservation, topology, transport
Discontinuities:
    sort, topology refresh, ray-cell crossing, occlusion, resampling, remeshing
Gradient route:
    analytic backward, replay backward, finite difference, surrogate, stopgrad
Hardware shape:
    bounded loops, fixed buffers, global sort, dynamic allocation, graph update
Local falsifier:
    smallest unit/synthetic/heldout test that could kill the idea
Metal disqualifier:
    one specific systems demand that would make the path nonlocal or too slow
```

If a paper cannot fill the card, do not elevate it beyond "background reading."

## PowerFoam Questions

Read PowerFoam for mechanisms that explain quality, not just primitive names.
The highest-value questions:

1. What is the minimal paper primitive that cannot be deleted without losing
   held-out quality: bounded power cells, Cech/AABB adjacency, quaternion frame,
   height/detail sites, spherical-Voronoi color, raytracing, densification, or
   loss stack?
2. Which parts are geometry, and which are appearance shortcuts? In particular,
   does a height/detail/SV mechanism improve held-out geometry, or only source
   RGB?
3. Is the selected adjacency a correctness condition or a speed heuristic? A
   true Cech/AABB superset can add false faces safely; a missed radical face can
   enlarge a cell and break held-out views.
4. What replay data is necessary for backward: active cell id, interval start
   and end, endpoint-winning neighbor, local texel coordinate, SV basis choice,
   depth quantile, normal-distance diagnostic?
5. Does the paper depend on nonlocal resampling/grow/prune statistics for
   quality? If yes, what is the smallest local equivalent: contribution EMA,
   point error EMA, visibility count, face witness, low-support pruning?
6. Are official losses optional regularizers or required supports? Separate RGB
   reconstruction, SSIM, normal, sparse/contribution, interpenetration, depth,
   and external normal supervision.
7. Which claims are about full official scale and which claims are about the
   primitive math? Do not let a local 4K synthetic kernel proof become a paper
   acceptance claim.

Local conversion tests:

```text
Primitive parity:
    Torch direct vs Metal on fixed tiny scenes; finite-difference gradients for
    center, radius, density, quaternion/frame, texel site, height, SV color.

Adjacency correctness:
    compare dense true-overlap/Cech, cech_aabb, and approximate/KNN graphs on
    the same cells; report missed-face count, false-face count, render delta,
    heldout residual by missed/unwitnessed face.

Height/SV value:
    train matched configs with only one primitive feature toggled; selector is
    held-out camera/time PSNR/SSIM/L1 plus motion/normal diagnostics, not source
    view fit.

Replay completeness:
    require a debug dump that can reconstruct the exact forward contribution
    path used by backward for a handful of pixels.
```

PowerFoam disqualifiers for the Metal path:

- a mechanism requires unbounded ray traversal or dynamic per-ray allocation in
  the hot kernel.
- backward requires differentiating through topology selection before a
  stopgrad/piecewise-constant diagnostic has shown held-out value.
- quality depends primarily on CUDA/Warp-specific infrastructure with no fixed
  buffer or MPS custom-op equivalent.
- the evidence is source-view PSNR without held-out camera/time separation.
- the paper-scale claim only holds with external depth/normal supervision that
  we do not have for the target Dynaworld setting.

## RadFoam / Radiant Foam Questions

Read RadFoam as a different geometry contract, not as "PowerFoam but faster."
The agenda is to determine whether its triangulation/ray-walk ideas can inform
local tests without becoming a full port.

Key questions:

1. What is the topological object: Delaunay triangulation, regular
   triangulation, AABB tree, tetrahedra, cell adjacency, or a hybrid?
2. Is the topology updated during optimization, and if so at what cadence? What
   state is invalidated by an update?
3. Does ray traversal have bounded work per pixel, or can adversarial geometry
   create long walks and branch divergence?
4. What gradients exist through geometry and density, and what is treated as
   piecewise constant?
5. Does the method rely on global CPU/GPU rebuilds that are acceptable offline
   but hostile to a local Mac training loop?
6. What is the smallest transferable idea: traversal ordering, occupancy
   pruning, tetrahedral mass conservation, graph refresh statistics, or density
   regularization?

Local conversion tests:

```text
Traversal proxy:
    compare current cech_aabb / raytrace selected graph against a regular-
    triangulation-like neighbor oracle on synthetic cells; measure active steps,
    render delta, missed support, and 4K memory.

Topology refresh audit:
    freeze topology for K steps vs refresh every step vs refresh by motion
    threshold; report heldout metrics and graph churn.

RadFoam-negative control:
    if a borrowed traversal improves speed but changes rendered alpha/RGB on a
    fixed scene beyond tolerance, treat it as approximate, not correctness.
```

RadFoam disqualifiers:

- requires a full dynamic Delaunay/regular-triangulation rebuild inside the
  Metal hot path.
- requires pointer-heavy graph traversal or recursion with no fixed-size replay
  representation.
- has no way to produce deterministic backward parity on small fixtures.
- achieves quality mainly through a scene/topology contract unlike our bounded
  cell primitive, making the port a separate renderer rather than an upgrade.

## Differentiable Rendering Questions

Read differentiable rendering papers for visibility-gradient discipline. The
useful mechanisms are usually about where gradients are honest, where they are
biased but useful, and where topology/sorting is held fixed.

Questions:

1. What discontinuity is being handled: occlusion order, triangle edge, cell
   boundary, density threshold, sample count, topology update, sort key?
2. Is the gradient exact for the rendered estimator, a smoothed surrogate, or a
   straight-through estimator?
3. What replay state makes backward deterministic?
4. Does the method prove local finite-difference agreement on tiny scenes, or
   only train end-to-end?
5. How does it avoid exploding gradients near grazing rays and near-zero alpha?
6. Can it report the contribution owner for diagnostics, not just aggregate
   color?

Local conversion tests:

```text
Visibility finite difference:
    choose one active pixel and one boundary pixel; perturb center/radius/frame/
    density/height; require gradient sign and magnitude to match finite
    difference within a declared tolerance outside known discontinuities.

Replay determinism:
    run forward/backward twice with the same tensors; active ids, intervals,
    alpha, color, and gradients must match bitwise or within tight tolerance.

Boundary stress:
    sweep a cell boundary across a pixel; plot RGB/alpha and gradient. A useful
    estimator has predictable smoothing or a declared stopgrad region.
```

Disqualifiers:

- the backward is only justified by "training works" without fixture parity.
- the gradient needs global sorted lists whose memory grows with all cells per
  pixel at 4K.
- the estimator hides visibility ownership, blocking witness/heldout failure
  diagnostics.
- it removes discontinuities by blurring support so much that held-out geometry
  becomes a source-view appearance fit.

## Gauge-Field Questions

Read gauge-field and geometric-learning papers for invariants that prevent
representational cheating. The goal is not to import terminology; it is to find
which internal choices should be unobservable and which failures should become
diagnostics.

Questions:

1. What is the gauge group: permutation, local SO2 tangent rotation, SO3 frame,
   chart coordinate change, scale, depth shift, or topology relabeling?
2. Which variables transform covariantly, which are invariant, and which are
   physical observables?
3. Is there a connection/transport rule between adjacent charts? What state is
   transported across a face or over time?
4. What is the holonomy or loop residual, and does it predict held-out failure?
5. Is the proposed invariant actually benign? Source-camera-only depth shifts
   are not benign if they alter held-out rays.
6. Can the mechanism be measured without changing training?

Local conversion tests:

```text
Frame-rotation invariance:
    rotate tangent frames and counter-rotate vector payloads; scalar renders
    should be invariant, tangent-vector payloads should transform covariantly.

Face transport witness:
    use replay endpoint data to estimate which radical faces are actually
    witnessed by train rays; correlate low face witness with heldout residual.

Holonomy diagnostic:
    for ray paths through cell sequences, compose chart transports around small
    loops; high residual should predict bad heldout more than source loss.

Wrong-gauge negative control:
    intentionally use the wrong neighbor/frame/time transport. If it passes the
    same metric, the metric is not a gauge test.
```

Gauge disqualifiers:

- the "gauge" can improve source loss while changing held-out observables.
- the invariant cannot be computed from renderer replay, trained state, or a
  cheap diagnostic pass.
- enforcing the invariant requires dense all-pairs chart comparisons.
- it adds a loss whose optimum is a degenerate collapse of opacity, scale, or
  support.

## Fluid / Wave Simulation Questions

Read fluid papers for compact transport and conservation mechanisms, not for a
full water solver. The relevant split is material identity vs repainting.

Questions:

1. Is the method Lagrangian, Eulerian, ALE, particle-grid, level set, mesh, or
   shallow-water? What state moves, and what state is re-sampled?
2. What is conserved: mass, volume, area, momentum, vorticity, phase, energy,
   material color, opacity, or only visual plausibility?
3. Are sources/sinks explicit? Foam birth/death is allowed, but it should pay a
   source budget rather than silently repainting.
4. What stability condition exists: CFL, timestep, viscosity, pressure solve,
   remeshing cadence, smoothing radius?
5. Can the mechanism work as a diagnostic or auxiliary residual without making
   the renderer a simulator?
6. Does the method require dense grids at target resolution, or can it attach to
   cells/texels/charts already present in PowerFoam?

Local conversion tests:

```text
Advected checker:
    material texture should move under transport; repaint-only control may fit
    train RGB but should fail heldout phase/flow.

Sinusoidal height wave:
    tests moving geometry, normals, and silhouette. Flat repaint should fail
    amplitude, phase, normal, and alpha IoU.

Area-preserving stretch:
    checks whether deformation changes opacity/mass incorrectly. Report
    det-J error, mass drift, and pullback texture error.

Occlusion handoff:
    crossing surfaces with known ownership should expose whether visibility is
    transported or smeared.

Foam birth/death budget:
    allow explicit source terms, but require alpha/mass trajectory and source
    localization metrics.
```

Fluid disqualifiers:

- requires a full pressure solve or dense 3D grid every training step before a
  smaller material-coordinate diagnostic has shown value.
- stability depends on tiny timesteps incompatible with video-frame training.
- conservation is enforced on RGB instead of density/material state, confusing
  lighting/view dependence with physics.
- the mechanism cannot distinguish advected material from per-frame color
  rewrite under held-out camera/time tests.
- 4K memory scales with image pixels times simulator state instead of visible
  cells or bounded replay records.

## Mechanism-To-Local-Test Pipeline

Use this funnel before implementation:

```text
0. Reading card:
       fill the card above; name the disqualifier up front.

1. No-grad diagnostic:
       compute on existing trained outputs or synthetic states. No trainer
       change. Accept only if the metric separates heldout failure from source
       fit or catches a known negative control.

2. Torch reference:
       implement the smallest scalar/vector formula in readable Python/Torch.
       Add finite-difference checks if gradients are claimed.

3. Synthetic scene:
       one analytic target, one repaint/shortcut negative control, one heldout
       camera/time split. Define thresholds before looking at results.

4. Trainer toggle:
       config-controlled auxiliary loss or primitive feature. Compare matched
       configs; only one mechanism changes.

5. Metal fixture:
       fixed-size forward/backward fixture with replay determinism and finite-
       difference parity on active parameters.

6. 4K verifier:
       saved artifact with forward/backward time and memory. A mechanism that
       passes 128px but explodes at 4K stays research-only.

7. Baseline row:
       only after heldout metrics and verifier results exist. Append to
       BASELINES.md; do not overwrite old rows.
```

## Skeptical Claim Taxonomy

Treat claims as different evidence classes:

```text
Useful background:
    clarifies math, but no local test yet.

Diagnostic candidate:
    can be measured on current outputs without changing training.

Local primitive candidate:
    has Torch parity path and a finite-difference gradient surface.

Metal candidate:
    has bounded replay state, fixed buffers, deterministic backward, and 4K
    memory shape.

Baseline candidate:
    improves held-out camera/time metrics against matched configs and negative
    controls.

Disqualified for current lane:
    requires nonlocal systems, unbounded traversal, dense simulator state, or
    evidence that does not touch heldout behavior.
```

The default prior should be that a beautiful paper mechanism is a diagnostic
candidate, not an implementation task. It graduates only by surviving the local
falsifiers.

## Highest-Value Reading Outputs

Ask side agents to produce these instead of broad summaries:

1. A table of PowerFoam primitive components ranked by likely held-out impact,
   with one local falsifier per component.
2. A RadFoam traversal/topology card that says exactly what is transferable to
   bounded PowerFoam cells and what would require a separate renderer.
3. A differentiable-visibility gradient card: discontinuity, estimator, replay
   state, finite-difference fixture, 4K memory implication.
4. A gauge-transport card: group, covariant variables, invariant observables,
   face/time transport rule, wrong-gauge negative control.
5. A fluid-transport card: material state, conservation law, source budget,
   stability limit, smallest non-simulator diagnostic.
6. A disqualifier log: every mechanism that looks attractive but fails the
   Metal/hardware/heldout contract, with the exact reason.

## Immediate Reading Questions To Hand Off

- In PowerFoam, which exact primitive feature first explains held-out quality:
  height, SV color, Cech/AABB correctness, raytrace traversal, or grow/prune?
- In RadFoam, is the triangulation useful as a diagnostic oracle for our current
  graph, or is it inseparable from a different renderer?
- In differentiable rendering papers, what is the cleanest replay record for
  visibility gradients through bounded cells?
- In gauge-field papers, can face witness or holonomy be computed from existing
  replay data before any new loss is added?
- In fluid/wave papers, what material-coordinate residual best separates real
  motion from repaint on the existing synthetic wave/checker benchmarks?
- For every promising mechanism, what is the first result that would make us
  stop: source-only improvement, failed finite difference, unbounded traversal,
  4K memory blowup, or no heldout correlation?
