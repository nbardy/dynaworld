# DynaWorld Renderer And Paper Lane Taxonomy

Date: 2026-07-19

Status: canonical naming and priority map for the camera-compiled renderer
work. This document separates mathematical framework, paper method,
implementation family, benchmark infrastructure, and exploratory probes.

## 1. Correction: Gauges Are Not Closed

The camera-gauge and ray-fiber mathematics remains part of the main result.
It was not rejected, superseded, or reduced to historical context.

The retained invariant is:

```text
B = Omega x T
pi: E_Gamma -> B
Gamma: E_Gamma -> M
Trace_Gamma[w] = pi_* Gamma^* w
```

The retained machinery includes camera-ray fibers, fiber-measure Jacobians,
ordinary-depth/log-depth/inverse-depth/projective gauges, projective/orbit
gauge domains, continuous denominator certificates, support and near-plane
events, visibility/order strata, finite exposure, rolling shutter, and
compiled interval forward/adjoint paths.

What is closed is the open-ended creation of additional gauge theories,
fiber-bundle names, and chart systems without a measured failure in the
current compiler. A new gauge is admissible only when a replayable camera path
falsifies an existing denominator, support, visibility, accuracy, or scaling
certificate.

## 2. Canonical Naming Hierarchy

These names are not peers. They occupy different levels.

| Level | Canonical name | Meaning | Public-paper role |
| --- | --- | --- | --- |
| Project | DynaWorld | The larger video-to-world-token and dynamic rendering project. | Umbrella only. |
| Shared mathematics | Gauged camera-ray bundle / Gauged UVT Trace Atlas | Coordinate-invariant pullback through a camera program and pushforward or transfer along ray fibers. | Shared theory used by both renderer papers. |
| Paper method A | World Tubes in Gauged Camera Space | Early ray-fiber marginalization of dynamic Gaussian primitives into reusable sensor-time traces plus conditional depth and visibility certificates. | Primary renderer paper. |
| Implementation A | STAR UVT | Sparse/tiled UVT renderer and trainer family used to implement and test World Tubes. | Code/backend name, not a competing paper. |
| Implementation A2 | Projective STAR UVT / PRT | Rational/projective trace and interval-atlas extension of STAR UVT for moving and revolving cameras. | Internal implementation label; paper-facing name remains World Tubes. |
| Paper method B | WorldFoam in Gauged Camera Space | Retains the ray-depth fiber as an opacity/transmittance axis and compiles cell-path optical-transfer events. | Distinct second paper. |
| Implementation B | Gate4 / owner-run / cutwalk / fused-slab WorldFoam | Metal and reference prototypes for compiled cell/event traversal and VJP. | Prototype implementation family for WorldFoam. |
| Ancestor/baseline B | PowerFoam | Existing bounded-cell/foam trainer and local Metal reproduction lineage. | Baseline and engineering ancestor, not a synonym for WorldFoam. |
| Baseline | Dynamic 3DGS / fast-mac | Conventional per-frame Gaussian representation and Metal rasterizer. | Same-budget and per-frame baseline. |
| Probe | Softmax-GS | Alternative overlap-aware compositing experiment. | Ablation/probe only unless repeat/scale clears. |
| Benchmark infrastructure | Paper protocol and paper runner | Dataset, budget, sampling, cost, media, and verifier contracts shared across representations. | Required evidence machinery, not a representation. |
| Demo | Browser WebGPU trainer | Simplified interactive source-view dynamic-splat/tube prototype. | Demo/prototyping surface, not native parity or paper evidence. |

Paper-facing vocabulary should therefore be:

```text
World Tubes in Gauged Camera Space
implemented by the projective STAR UVT interval renderer
```

and:

```text
WorldFoam in Gauged Camera Space
implemented by the owner-run/cutwalk optical-transfer prototype
```

Do not present "Gauged UVT", "STAR UVT", "PRT", and "World Tubes" as four
independent model proposals.

## 3. The Mathematical Fork

Both papers start from the same camera-ray bundle. They differ in operator
order.

### 3.1 World Tubes: Early Fiber Pushforward

For a world primitive `w_i`, World Tubes forms a sensor-time trace:

```text
T_i(y) = pi_* Gamma^* w_i,
y = (u, v, tau).
```

For a local Gaussian pullback, depth marginalization gives the Schur
complement:

```text
S_i = H_yy - H_yz H_zz^{-1} H_zy.
```

The compiled trace stores or evaluates opacity, color, conditional depth,
depth variance, support, and order/fallback certificates. This preserves
baseline Gaussian-splat semantics and makes per-frame replay the clean
same-representation comparator.

### 3.2 WorldFoam: Retained Fiber Optical Transfer

WorldFoam does not immediately integrate out depth. It keeps:

```text
lambda(y, z) >= 0
eta(y, z) = lambda(y, z) c(y, z)
```

and renders with transmittance:

```text
I(y) = integral exp(-integral_{z' < z} lambda(y,z') dz') eta(y,z) dz.
```

The compiled object is a sequence of cell/owner-run optical-transfer elements
over the ray fiber. Composition is associative, which permits prefix/suffix
evaluation and VJP. The retained depth axis is the reason WorldFoam can model
visibility more directly.

The key theorem boundary is:

```text
visibility generally does not commute with depth marginalization.
```

This is not evidence that World Tubes is wrong. It explains why World Tubes
needs order strata and fallback, and why WorldFoam is a meaningful second
operator ordering rather than a renamed implementation.

## 4. Lane Inventory

### Lane A: Gauged Camera-Ray Bundle Theory

Status: retained and substantially validated; not an open-ended expansion lane.

Mathematically complete enough to use: gauge-invariant values and gradients
with the fiber Jacobian, projective/orbit gauges for revolving cameras,
continuous denominator certificates, event-certified domains,
visibility/order strata, and finite-exposure/rolling-shutter sampling.

Remaining work is exposition and consolidation, not another foundational
formalism.

### Lane B: World Tubes Paper And Projective STAR UVT

Status: primary paper and renderer lane.

Paper target:

```text
Compile a known or low-dimensional camera program into reusable sensor-time
traces so projection, support, binning, visibility metadata, and backward
replay scale with trace/event complexity instead of frame count.
```

Implemented: affine and projective traces, interval forward/direct VJP,
revolving-camera orbit windows, camera-family gauge tests, visibility
stress/fallback, finite exposure, rolling shutter, a decisive demo, a paper
runner, and one matched three-seed Neural3D `coffee_martini` split.

Not finished: multi-triplet/multi-scene public breadth, full-sequence paper
protocol runs, matched runtime/cost tables, final figures and LaTeX manuscript,
and optional portability evidence beyond Metal/MPS.

### Lane C: WorldFoam Paper And Optical-Transfer Prototype

Status: real second-paper theory/prototype; parked as a scheduling priority,
not invalidated.

Paper target:

```text
Compile bounded world matter into a reusable ray-fiber optical-transfer event
atlas whose visibility scan, replay, and VJP are associative and temporally
shared.
```

Worked out and tested: visibility monoid, constant-run alpha equivalence,
same-representation replay fixture, analytic prefix/suffix VJP with finite
differences, commutator/swap probe, cell-path fixture, owner-run/Metal bridge,
many Gate4/cutwalk/frame-scaling microgates, and one matched coffee_martini row.

Not finished: native Metal optical-transfer parity rather than a bridge,
competitive heldout RGB quality across multiple scenes, event-density/memory
death curves, the fixed-topology versus moving-boundary gradient boundary, and
a compression bakeoff before promoting Magnus or boundary-flux ideas.

The current evidence supports a theory/prototype paper. A full rendering-paper
claim requires the quality and native-parity gates.

### Lane D: STAR UVT Source-View And Feature-Tube Engineering

Status: implementation support lane.

The RGB `direct_atomic + index_add` path is the practical high-tube route.
Feature/F32 tubes and sparse support diagnostics are useful for renderer and
appearance experiments, but their visual coverage/composition problems should
not be confused with the camera-gauge theorem. Direct-serial at 512px remains
a kernel probe until trainer parity/repeat validates it.

### Lane E: Dynamic 3DGS / fast-mac Baseline

Status: required baseline, not a new paper lane.

Keep it healthy for same-budget quality comparisons, per-frame replay, RGB/F32
raster timing, and heldout-camera controls. Do not spend research cycles
rearchitecting it unless a paper comparison is blocked.

### Lane F: Softmax-GS Probe

Status: bounded probe.

The tiny heldout K=16 result was positive, but repeat/scale was mixed. Do not
port it into STAR UVT or WorldFoam until one larger repeated heldout gate is
positive on PSNR, SSIM, and train-view quality together.

### Lane G: Paper Training Protocol And Runner

Status: active engineering priority.

This lane is currently on disk and in flight. It is adding typed dataset
contracts, full 300-frame coffee_martini manifests, progressive and fixed
matched-pixel protocols, coverage-exact spacetime sampling, active primitive
schedules, target/rasterized pixel accounting, and selected-time World Tubes
rendering instead of accidental full-sequence replay.

The current integration surface is
`research_experiments/paper_runner_suite/run_unified_paper_ablation.py`, with
focused coverage in `tests/test_unified_paper_ablation.py` and checked-in
protocols under `src/train_configs/paper_protocols/`. Treat these files as
active in-flight work until their runtime smokes and cost-contract tests pass.

This is the right immediate work because it turns the existing mathematics
and kernels into credible paper evidence.

### Lane H: Mixed Same-View + Heldout World-Token Training

Status: active broader DynaWorld model lane, separate from the renderer paper.

Its target is a world-token representation that survives novel cameras. It
uses renderer infrastructure but should not inflate a World Tubes or WorldFoam
claim. Finish its benchmark contract independently.

### Lane I: Gaussian 300-Clip Scale Training

Status: blocked at 512px promotion.

Do not resume broad scaling until the promotion NaN is isolated with a saved
pre-promotion checkpoint and camera/FOV diagnostics. This lane is not evidence
for or against World Tubes.

### Lane J: V-JEPA/F32 Multicam Heldout

Status: active benchmark-contract lane.

Keep only the source/camera-disjoint manifest, leakage, pose-error, fisheye,
and heldout metric work. It is an appearance/world-token benchmark, not the
camera-path compiler paper.

### Lane K: Browser WebGPU Trainer

Status: useful demo, parked for research claims.

Keep it runnable and visually inspectable. The active train17/holdout1 demo now
has a full-frame opacity-aware tiled raster, exact windowed DSSIM, trainable
temporal mixture, and bounded fixed-capacity split/recycle. Its fast fork uses
staged projected gradients, one 3D VJP per splat, compact hot/cold projection
packets, and checkpoint-block replay; the older direct tiled path remains a
live control. It is still a systems demo rather than a paper renderer.
`Harmonic trajectory splats` is not native World Tubes or native 4DGS parity.
The standalone affine STAR and bounded DynamicGs shaders are microbenchmarks,
not interchangeable SPA backends. Further promotion requires a matched
quality/performance ablation and the native backend contracts described in
`browser_4dgs_baseline.md`; browser work does not create a Python trainer or
paper lane.

### Lane L: PowerFoam Post-Audit

Status: parked unless selected.

PowerFoam remains useful as implementation lineage, bounded-cell baseline, and
parity surface. Do not merge the entire reproduction backlog into WorldFoam.
Only import components that satisfy the WorldFoam optical-transfer contract.

## 5. Consolidate, Drop, And Finish

### Consolidate now

1. Use **Gauged camera-ray bundle** for shared mathematics.
2. Use **World Tubes** for the primary method and paper.
3. Use **STAR UVT** only for the implementation/backend.
4. Use **projective interval atlas** instead of introducing PRT as another
   public method name.
5. Use **WorldFoam** for the retained-depth method and second paper.
6. Use **Gate4/owner-run/cutwalk** as WorldFoam implementation labels.
7. Keep **PowerFoam** explicitly marked as ancestor/baseline.
8. Put all representations through one paper protocol and paper runner.

### Drop or stop by default

```text
new umbrella gauge/fiber theories without a falsifier
new public-facing aliases for existing operators
duplicate shader forks without a new contract or gate
repeated support/alpha knob sweeps after dense metrics stay flat
Softmax-GS ports before repeat/scale promotion
browser optimizer variants without matched quality improvement
Gaussian 300-clip scaling before the 512px NaN gate
WorldFoam Magnus/boundary-flux expansion before simpler compression tests
```

Drop means stop scheduling work, not delete the evidence.

### Finish 1: Freeze the shared paper protocol

Acceptance:

```text
typed protocol parses checked-in configs
dataset contract overrides old smoke manifests
progressive and fixed protocols match declared pixel budgets
World Tubes samples only requested sensor times
WorldFoam and dynamic 3DGS report the same target/rasterized cost schema
one-step runtime smokes pass for all three representations
```

### Finish 2: Run World Tubes breadth

Run more coffee_martini camera triplets, then multiple Neural3D scenes with
heldout-camera metrics, repeats, media, fallback fraction, compile/render/
backward cost, and amortization point.

Decision gate:

```text
If same-representation quality parity and sublinear world-side scaling survive,
finish Paper A.
If they fail, localize the failure to support, order strata, fallback, or
native overhead before changing the theory.
```

### Finish 3: Write and package World Tubes Paper A

Required output: one canonical command, public synthetic correctness suite,
public multi-scene table, frame-scaling/break-even chart,
visibility/fallback figure, finite-exposure or rolling-shutter demonstration,
compiled-adjoint gradient table, limitations, and LaTeX/arXiv source.

### Finish 4: Re-evaluate WorldFoam with the same breadth table

Do not start with another shader fork. First ask whether heldout quality and
event density justify the retained-depth method. If yes, implement native
optical-transfer Metal parity and finish Paper B. If no, publish or preserve
it as a theory/prototype result and stop the systems expansion.

### Finish 5: Resume broader DynaWorld model work

After the renderer paper contract is stable, return to mixed same-view plus
heldout world-token training and the V-JEPA/F32 benchmark contract. Keep those
claims separate from renderer/compiler claims.

## 6. Current Confidence

High confidence:

```text
the gauge/ray-fiber framework is real and retained
the projective/orbit implementation handles revolving cameras within certified domains
World Tubes and WorldFoam are distinct operator orders
the one-split paper runner and local fixtures are reproducible
```

Medium confidence:

```text
World Tubes will retain its one-split quality lead across scenes
sublinear metadata/backward gains will survive full 300-frame protocols
WorldFoam event compression will remain practical at real scene complexity
```

Low confidence until measured:

```text
WorldFoam can match World Tubes heldout RGB quality
native optical-transfer Metal will preserve the prototype speed advantage
either renderer is a broad SOTA dynamic-NVS replacement
```

The immediate scientific bottleneck is no longer missing mathematics. It is
matched public breadth under a frozen training and cost protocol.

## 7. 2026-07-26 Open-Ray Transfer Clarification

The moving-camera ordered-transfer proposal sharpens Paper B without changing
the lane priority above. The original intake used “ray holonomy,” but an open
camera ray ordinarily has a transfer operator or parallel transport.
“Holonomy” is reserved here for closed loops, including the older cell-graph
loop diagnostic.

Terminology:

```text
camera program:
    defines the measurement bundle and changes the observation

gauge:
    chooses local coordinates/trivialization within that bundle

open-ray transfer:
    path-ordered optical transfer accumulated along one ray fiber
```

The proposal's retained-fiber connection:

```text
H_C(y) = P exp integral_{F_y} Gamma_C^* A
```

is the continuous form of WorldFoam's existing optical-transfer monoid. Its
commutator theorem is the exact limitation behind World Tubes visibility
certificates:

```text
[T_i,T_j] color = alpha_i alpha_j (c_i - c_j).
```

Therefore:

```text
World Tubes / STAR UVT:
    remains Paper A and the current finish-first lane

WorldFoam / gauge-covariant open-ray transfer:
    remains Paper B and receives the retained-fiber, convex-potential,
    discriminant-compiler, and transfer-ODE research branches
```

Paper A may use the terminology correction and noncommutation theorem to state
its approximation boundary. It should not absorb the unimplemented
self-normalized convex-potential atom or path-ordered transfer renderer.

The complete intake, equations, claims, failure regimes, and falsification
ladder are preserved at the historical filename below. Its terminology is
superseded by this clarification:

```text
research_notes/worldfoam_paper/scientist_notes/
2026-07-26_gauge_invariant_ray_holonomy_intake_and_paper_split.md
```
