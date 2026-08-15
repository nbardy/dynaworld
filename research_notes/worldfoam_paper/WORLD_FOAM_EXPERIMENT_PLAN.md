# WorldFoam Paper: Gauge-Invariant Ordered Ray Transfer — Ablations, Charts, Baselines, and Acceptance Gates

Draft date: 2026-08-03

This is the execution plan for the second paper lane:

```text
WorldFoam in Gauged Camera Space:
Gauge-Invariant Ordered Ray Transfer for Moving Cameras
```

It is intentionally stricter than the theory draft. It says what must be run
before we can claim anything beyond "promising theory and prototype."

Current measured-row ledger (2026-08-15): G6 native training-memory `0/21`;
G4 public heldout quality `0/36`.  Source/unit/runtime gates are preflight and
do not count toward either matrix.  The G6 analytic state bound is
frame-invariant but allocator/RSS evidence is absent.  The G4-v1 full-pixel
schedule remains the exact correctness reference but currently requires about
`113--115` million cold `(view,pixel)` compiles per seed; it must remain
fail-closed as the correctness reference.  The separately versioned G4-v2 is
now source-implemented: it freezes identical selected rays, targets, optimizer
steps, RGB-MSE, and full heldout evaluation for all four routes.  Each row uses
`1,228,800` target pixels; WorldFoam directly rasterizes that count while the
Gaussian controls full-rasterize `235,929,600`, so target budget is matched and
raster work is explicitly not claimed equal.  WorldFoam's full-temporal
heldout path is spatial-major (`196,608` cold tracks, `1,536` host calls,
`15,360` bounded native bundles per camera) and uses `843.75 MiB` of temporary
dual-spool disk rather than a resident dense video.  A bounded real-native
pilot and all 36 measured rows remain absent.

## 0. Claim Boundary

Current strongest claim:

```text
WorldFoam's ordered ray-transfer formulation is gauge invariant when the fiber
measure is included. The bounded CPU reference currently verifies ordinary-
depth affine rescaling; general log/nonlinear chart parity remains open. A new
P0 Metal path is source-complete but unbuilt, while historical tiny paired
microgates show a temporal-reuse signal rather than matched systems evidence.
The CPU direct-kinetic path now includes active owner-chart compilation, exact
multi-chart dispatch, continuous primal/referenced-material-action
certification, and a frozen-program sparse geometry/material VJP. Prepared
native tokens own no global/chart-local time clone and receive only live `K`
times. Native kinetic lowering, continuous geometry-Jacobian approximation
certification, structural recertification, real public quality, and full
official parity are not yet proven.
```

Do not claim yet:

```text
SOTA dynamic novel-view synthesis
replacement for STAR UVT / dynamic 3DGS
quality parity with mature Gaussian baselines
official CUDA/Warp parity
sublinear total work in output pixels
```

The target claim after experiments:

```text
For known camera programs, WorldFoam compiles bounded world cells into a lifted
ray-fiber transmittance atlas. This reuses cell/intersection/prefix work across
time and avoids discrete primitive sorting, while matching per-frame foam
quality and improving robustness in controlled visibility-stress scenes.
```

## 0.5 Optical-Transfer Operator Tests

These tests come directly from the depth-fiber/operator-ordering proof scaffold
and the optical-transfer reformulation plan:

```text
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
research_notes/worldfoam_paper/WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md
research_notes/gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
```

They should run before broad public-quality sweeps, because they test whether
the math is actually implemented.

The first code-level implementation spec is:

```text
research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md
```

| Test | Pass condition | Failure meaning |
| --- | --- | --- |
| Gauge Jacobian | ordinary-depth and log-depth pullbacks match with Jacobian and diverge without it | gauge math is decorative or measure is wrong |
| Alpha equivalence | sorted splat compositing equals monoid scan of atomic `(1-alpha, alpha c)` events | optical-transfer algebra is not wired to baseline alpha semantics |
| Non-commutation slab | two matched-UVT translucent scenes differ by the predicted order term | pure UVT marginal is being overclaimed |
| Commutator prediction | measured swap/order error is predicted by opacity overlap times color contrast | commutator theorem is only decorative |
| Cell-path replay equivalence | compiled atlas emits the same independently validated cell/event word and image as per-frame WorldFoam replay | compiler changes semantics instead of amortizing structure |
| Cell-path VJP | beta/m/DeltaTau/sigma/color/run-length finite differences match direct monoid-scan VJP | backward tape/recompute contract is wrong |
| Boundary flux VJP | moving face and sphere endpoints match finite differences under fixed topology | boundary-gradient theory is not ready for paper claims |
| Flux witness diagnostic | interface-flux witness score predicts heldout-free residuals, source leave-one-camera-out error, or topology churn | topology math is decorative |
| Compression bakeoff | commutator-energy splitting beats or matches simple adaptive splitting at equal error/memory | Magnus/commutator compression is not worth mainline status |
| Event-density death curve | speedup/memory/fallback degrade predictably as event complexity rises | sublinear claim lacks a measured failure boundary |

Minimal synthetic fixtures:

```text
constant sphere
two translucent crossing slabs
bounded power-cell pair with analytic face crossing
constant-density owner-run cell word
atomic splat stack lowered to optical transfer elements
fast orbit with ordinary/log-depth gauge variants
fixed-topology coefficient perturbation for VJP finite differences
face-crossing, center/radius, and support-sphere perturbation for boundary flux
heldout-free residual / source leave-one-camera-out splits for witness scores
```

## 0.6 Pass-4 Fixed-Tape Material Gate

Before another renderer fork, compare all segment laws behind one shared
interface:

```text
identical owner/event word
identical segment endpoints and physical lengths
identical camera gauge and Jacobian
identical front-to-back (beta,m) scan
identical output samples and loss
```

| ID | Extinction | Appearance | Required reason |
| --- | --- | --- | --- |
| M0 | P0 constant | constant RGB | Existing material reference |
| M1 | P0 constant | affine RGB | Appearance-only counterbaseline |
| M2 | positive Bernstein P1 | constant RGB | Cheap density-capacity step |
| M3 | positive Bernstein P2 | constant RGB | Mandatory polynomial counterbaseline |
| M4 | log-P1 | constant RGB | Exponential-linear control |
| M5 | convex log-P2 | constant RGB | Gaussian-like `erf` control |

M3 and M5 use the same three scalar segment-control slots. M3 must not be
replaced by nonnegative P2 Lagrange samples, because those do not guarantee
positive density between nodes. M5 initially rejects/falls back on negative
quadratic curvature rather than silently relying on an unstable `erfi` path.

The implementation ladder for this gate is:

1. [x] **Float64 reference:** analytic optical depth and coefficient/length VJP
   against independent quadrature, autograd, and central differences.
2. [x] **Fixed-tape Metal microkernel:** tiny forward/VJP parity for M0--M5 using
   precomputed records. This forks the material evaluator, not the renderer.
3. [x] **Material-value test:** fit controlled synthetic density on shared
   partial chords, evaluate on disjoint held-out chords, and compare
   M2/M3/M4/M5 against both M0 and M1.
4. [ ] **Native-4D P0 geometry compiler:** advance constant material first so
   geometry, events, gauge measure, memory, and adjoints are tested without
   conflating material selection.
5. [x] **CPU adaptive M3/M5 selection gate:** select a per-cell basis on
   disjoint validation chords and evaluate it on disjoint heldout chords at
   matched 24-byte payloads plus one basis-tag bit. This clears the synthetic
   basis-selection ablation only; it does not promote a rich material into the
   native-4D path or replace the P0 systems oracle.

As of 2026-08-03, the CPU portion of step 4 includes the ordinary-depth fiber
Jacobian, exact constant-state chunked replay, active owner-chart compilation,
exact multi-chart dispatch, sparse boundary/ray/site/velocity/weight/material
VJPs, track/time-blocked scratch, and continuous certification for the primal
and referenced-material actions. A suffixed source-only Metal bridge exists
but is not built or runtime verified. Step 4 remains unchecked because the
direct-kinetic multi-chart program/VJP is only source-lowered and has not passed
rebuilt-native parity, while bounded-cell
sphere/vacuum events and continuous geometry-Jacobian approximation
certification are open. A narrow event-free directional trust-region
certificate exists, but there is no general or trainer-integrated geometry
recertification/update path,
production trainer/evaluator, or public evidence. See
`../../TODO/worldfoam_memory_light_native4d.md`.

The checked boxes are supported only by the bounded fixed-segment artifact:

```text
artifacts/foundation_gates/worldfoam_material_m0_m5_cpu_metal_20260727.json
artifacts/foundation_gates/worldfoam_material_m0_m5_cpu_12record_20260727.json
artifacts/foundation_gates/worldfoam_material_value_fit_cpu_20260727.json
outputs/benchmarks/2026-08-15_worldfoam_adaptive_material_basis_cpu/summary.json
```

The first establishes reference/shader forward and explicit-VJP parity on the
shared tape. The second expands CPU evidence to twelve records, all-mode
central differences, and the tiny-tau series branch. The third is a
three-seed, float64 CPU capacity gate with an
independent target oracle, disjoint train/held-out partial chords, and matched
24-byte M3/M5 material payloads. On held-out chords, M3 reaches
`5.26e-17` loss on the positive-P2 target while M5 reaches `8.80e-5`; on the
convex-log-P2 target, M5 reaches `6.19e-15` while M3 reaches `1.33e-3`.
Therefore step 3 finds **no universal material winner**: each basis wins its
own generating family. It clears the controlled capacity comparison, but not
end-to-end replay, native-4D, trained image quality, or systems gates. Rich-
material native integration remains blocked pending a separate real-data and
native promotion gate; P0 geometry integration is not blocked on a universal
material winner.

The fourth artifact clears the CPU synthetic portion of step 5. Its strict
verifier recomputes `72` candidate rows and `36` selection rows across seeds
`17/29/43` and twelve target cells with disjoint train, selection, and heldout
chords. Pure-family basis-selection accuracy and heldout-oracle agreement are
both `1.0`; adaptive mean heldout loss is `0.313405` times the best fixed-basis
mean and exactly matches the heldout oracle (`1.0` ratio). This supports an
explicit per-cell M3/M5 basis tag as a controlled Paper-B ablation. It is not
real-scene evidence and makes no native integration, public-quality,
renderer-speed, or memory claim. Rich-material native promotion therefore
still requires real heldout material/image evidence after the P0 systems path
is sound.

Acceptance thresholds for steps 1--2:

```text
float64 analytic vs independent quadrature max error <= 1e-10
reference VJP relative error <= 1e-5
tiny Metal forward max absolute error <= 1e-5
tiny Metal VJP normalized error <= 1e-4
zero NaN/Inf on accepted inputs
all series/erf/scaled-tail/reject branches counted
M0 regression reported on the identical tape
```

Every JSON artifact must record material mode, coefficient convention, gauge
and length convention, dtype/device, branch counts, tolerances, seed, source
revision, tape dimensions, and whether timings were synchronized.

Claim boundary:

```text
Passing the fixed-tape gate proves local material parity only.
It is not trained-quality evidence, a systems-speed result, or proof of
sublinear parameter/tape scaling across frames.
```

In particular, shader-only work cannot fix a model whose parameters or cell
tape are duplicated per frame. That scaling claim requires the native-4D
compiler in step 4 and explicit reporting of parameter bytes, owner/event
records, and fallback growth versus `T`.

The local workstation remains under the post-incident MPS guard. CPU reference
tests and tightly bounded mechanical Metal parity are allowed; publication
training, the full 300-frame matrix, and broad MPS sweeps require a separately
approved execution host/run.

## 1. Baselines

### 1.1 Same-representation baselines

These isolate the compiler.

| ID | Baseline | Purpose |
| --- | --- | --- |
| F0 | Dense raymarch / analytic reference | Ground truth for synthetic density scenes. |
| F1 | Sequential per-frame WorldFoam replay | Main same-representation baseline: one exact frame forward/reverse, accumulate global bars, release frame scratch, then continue. Peak may be `F`-invariant; world work is `O(F)`. |
| F2 | Per-frame replay with cached camera constants | Separates trivial camera caching. |
| F3 | Cached cell active sets, live ray/cell intersections | Tests active-set caching alone. |
| F4 | Compiled cell intersections, live transmittance prefix | Tests geometry compilation. |
| F5 | Compiled intersections + depth-layer prefix | Proposed core foam atlas. |
| F6 | Full compiled atlas + constant-state recomputed VJP | Training-speed target. |
| F7 | Full atlas with fallback/reference cells | Robustness target. |

Required measurements:

```text
quality: RGB PSNR, SSIM, LPIPS, alpha error, transmittance error
speed: compile ms, forward ms, backward ms, optimizer step ms
memory: atlas bytes, prefix bytes, peak GPU memory
structure: active cells, intersection records, depth layers, fallback fraction
gradients: max/mean gradient error vs reference, optimizer loss decrease
work split: world/intersection/prefix/reverse interactions vs selected-pixel/ray/sample interactions
```

The required memory/work ablation must not turn F1 into a dense retained-tape
straw man.  Run F1 sequentially at the same `S`, selected pixels, frames,
material, geometry, camera, optimizer, and native kernels as the compiled
route.  Measure its peak memory and its repeated world-side work at
`F=8/64/300`.  Compare that against staged compiled replay at `F=8` for parity
and fused union-local replay at `F=8/64/300` for the primary scaling curve.
An all-frame retained active tape may appear as a separately named optional
ablation only when its exact allocated tensors are implemented and measured;
an all-site dense tape is a stress control, not F1.

### 1.2 Gaussian and tube baselines

These position the paper against the existing renderer ecosystem.

| Baseline | Why include |
| --- | --- |
| Dynamic 3DGS / per-frame gsplat | Familiar dynamic primitive baseline. |
| STAR UVT direct atomic | Strong local route in this repo; speed/quality anchor. |
| World Tubes compatibility mode | Same camera-compiler family but GS semantics. |
| Sort-free GS-style weighted sum | Sort-removal comparator. |
| Gaussian Blending-style alpha/transmittance distribution | Blending-semantic comparator. |
| 4DGS / STG / Deformable 3DGS | External dynamic-rendering context. |

Policy:

```text
Same-representation WorldFoam replay is the main proof baseline.
STAR/World Tubes show whether foam is worth switching semantics.
External methods contextualize but should not carry the proof.
```

## 2. Datasets and Scenes

### 2.1 Synthetic exact suite

Mandatory. This is where WorldFoam can be cleanly falsified.

Scenes:

```text
S1 constant-density sphere
S2 two crossing translucent slabs
S3 crossing Gaussian density sheets
S4 thin foreground occluder
S5 dense semi-transparent cloud
S6 moving cell complex
S7 near-camera large cell
S8 fast object + fast orbit camera
```

Camera programs:

```text
C1 static
C2 linear dolly
C3 orbit
C4 fast orbit
C5 orbit + finite exposure
C6 rolling shutter
C7 revolving camera with near-plane crossings
```

Ground truth:

```text
analytic optical-depth integral where available
dense raymarch otherwise
high-sample shutter integration for exposure
```

Required charts:

```text
RGB/alpha/transmittance error vs depth-layer count
compile/eval time vs frame count
fallback fraction vs camera speed
order-flip/flicker metric for sorted baselines
```

### 2.2 DeepView / calibrated multicam

Purpose:

```text
prove the representation is not only synthetic.
```

Minimum gates:

```text
train2/test1 split
real 16f and real 32f loaded frames
matched primitive/cell budget against dynamic gsplat and STAR UVT
heldout camera metrics
```

Do not promote if:

```text
heldout PSNR remains below 13
heldout SSIM remains below 0.15
WorldFoam loses badly to same-budget dynamic gsplat or STAR UVT
```

### 2.3 Neural 3D Video / Technicolor-style dynamic scenes

Purpose:

```text
public dynamic-camera/video comparison once DeepView gates are no longer weak.
```

Required:

```text
scene subset with public camera calibration
same train/eval split for all methods
frame-count scaling sweep
training step time including per-step transfer rebuild
inference-only topology/transfer amortization point reported separately
```

### 2.4 D-NeRF synthetic dynamic scenes

Purpose:

```text
controlled dynamic-object stress with public evaluation.
```

Use after synthetic exact suite, before broad real-video claims.

## 3. Ablations

### A1. Frame-count scaling

Sweep:

```text
F = 2, 4, 8, 16, 32, 64
```

For each method:

```text
total step ms
forward ms
backward ms
cell/ray intersection ms
prefix/transmittance ms
loss/eval ms
atlas bytes
quality delta vs reference
```

Primary plot:

```text
x-axis: frame count
y-axis: normalized step / forward / backward time
```

Pass:

```text
compiled WorldFoam grows slower than per-frame replay in intersection/prefix/
backward-replay components. Training timing includes the mandatory `J`-node
transfer rebuild after every world update; only topology/camera structure may
be reused while its validity token holds. Report frozen-world inference
amortization separately.

Sequential replay and compiled replay both remain under the declared absolute
accelerator/RSS peak limits.  No pass condition requires the sequential route
to have an `O(F)` memory peak: its expected failure is repeated world-side
work and bandwidth, not inability to stream frames.
```

### A2. Depth-layer / prefix representation

Compare:

```text
uniform depth layers
adaptive depth layers
cell-event layers
Gaussian depth basis
hybrid event + Gaussian basis
```

Metrics:

```text
transmittance error
alpha error
memory
prefix scan time
backward gradient error
```

### A3. Tape versus recompute

Compare:

```text
no per-run reverse tape; recompute with constant prefix state
scalar prefix tape
scalar contribution/weight tape
per-layer compact tape
large dense tape
```

Reject:

```text
per-channel tapes that grow linearly in feature dimension and frame count.
```

Expected sweet spot:

```text
exact constant-state two-pass P0 replay; no per-run suffix/reverse tape.
```

### A4. Cell support graph

Compare:

```text
Cech/AABB
regular triangulation teacher/verifier
witnessed Cech subset
uncertainty-weighted witnessed graph
```

Metrics:

```text
traversal count
false-edge fraction
edge churn
heldout residual correlation
train leave-one-camera-out residual
quality at matched cell budget
```

Important falsifier:

```text
If witnessed topology does not predict heldout-free validation signals, do not
make it a major paper contribution.
```

### A5. Gauge connection / holonomy

Log first:

```text
weighted face holonomy mean/p90
holonomy vs heldout residual
holonomy vs edge churn
holonomy vs source leave-one-camera-out residual
```

Only train with a connection loss if logging predicts failure.

Reject as regularizer if:

```text
holonomy improves but heldout quality drops or only source-view quality improves.
```

### A6. Visibility stress

Scenes:

```text
crossing translucent slabs
thin foreground occluder
semi-transparent cloud
near-camera support crossing
```

Compare:

```text
sorted alpha splats
World Tubes visibility gauge atlas
WorldFoam transmittance atlas
sort-free weighted sum
Gaussian blending-style comparator
```

Metrics:

```text
PSNR/SSIM/LPIPS
flicker
order-flip count
gradient variance near crossing
transmittance error
fallback fraction
```

This is the most important qualitative argument for WorldFoam.

### A7. Training behavior

Run:

```text
source-view overfit
train2/test1 heldout
real32 loader smoke
short public dynamic scene
```

Measure:

```text
loss decrease
nonzero gradients
parameter update
quality curves
gradient norms per cell field
cell topology churn
alpha/transmittance saturation
```

## 4. Existing Internal Evidence to Preserve

Use these as engineering anchors, not final paper rows unless rerun cleanly.

### 4.1 Gate4 native-cutwalk 2/4/8/16f

```text
WorldFoam total:
    3.008 / 3.014 / 3.323 / 4.095 ms

WorldFoam backward:
    2.739 / 2.517 / 2.561 / 3.796 ms

Scale over 2 -> 16f:
    1.361x total / 1.386x backward

STAR total:
    5.003 / 5.943 / 8.092 / 9.794 ms

STAR backward:
    2.629 / 3.411 / 5.083 / 6.768 ms
```

Interpretation:

```text
good local speed gate
not quality parity
not public benchmark
```

### 4.2 Repeated-fixture 32f

```text
WorldFoam total:
    2.829 / 3.248 / 4.414 / 4.643 / 6.371 ms

WorldFoam backward:
    2.557 / 2.965 / 4.054 / 4.254 / 6.001 ms

Scale over requested 2 -> 32f:
    2.252x total / 2.347x backward
```

Interpretation:

```text
useful shader scaling smoke
32f repeats 16 loaded frames
not a real 32f video benchmark
```

### 4.3 Render96/site48 gate

```text
WorldFoam total:
    3.760 / 4.125 / 4.619 ms for 2/4/8f

WorldFoam backward:
    3.480 / 3.847 / 4.331 ms

STAR total:
    5.773 / 7.583 / 9.692 ms

STAR backward:
    3.614 / 5.161 / 6.719 ms
```

Interpretation:

```text
larger fused-MSE speed/scale evidence
not RGB system parity
```

### 4.4 Real32 loader smoke

```text
loaded_frame_count = 32
no frame repeats
loss decreases
gradient nonzero
parameter update exists
```

Interpretation:

```text
trainability/data-path gate only
not a quality or timing row under clean benchmark conditions
```

### 4.5 Quality gap

```text
best WorldFoam train PSNR:   about 12.248
best WorldFoam heldout PSNR: about 12.857
gap to solid same-source:    about 9.112 dB
gap to STAR UVT source:      about 17.575 dB
```

Interpretation:

```text
this is the main reason WorldFoam is a separate second paper, not a hidden
section inside World Tubes.
```

## 5. Required Figures

### Figure 1: Representation split

Show:

```text
World Tubes:
    primitive -> UVT footprint + certified order

WorldFoam:
    cell complex -> lifted sigma(u,v,t,z) + transmittance prefix
```

### Figure 2: Ray-fiber foam atlas

Show:

```text
world cells
camera ray bundle
pullback to (u,v,t,z)
depth-layer/prefix scan
frame slice
```

### Figure 3: Frame-count scaling

Plot:

```text
F vs total/forward/backward/intersection time
compiled WorldFoam vs per-frame replay vs STAR/World Tubes
```

### Figure 4: Translucent crossing

Plot:

```text
sorted splat artifact/flicker
World Tubes fallback/order atlas behavior
WorldFoam smooth transmittance
```

### Figure 5: Quality-speed frontier

Plot:

```text
heldout PSNR/SSIM vs step time
WorldFoam current rows
STAR UVT rows
dynamic gsplat rows
```

This figure may currently be unfavorable. That is useful. It tells us whether
the paper is ready.

### Figure 6: Topology diagnostics

Plot:

```text
witnessed edge score vs residual
holonomy vs residual
edge churn over training
```

Only include if the signal is real.

## 6. Acceptance Gates

### Gate G0: Synthetic exactness

Pass if:

```text
compiled atlas matches dense raymarch/analytic transmittance within tolerance
on S1-S5, with bounded fallback.
```

### Gate G1: Same-representation speed

Pass if:

```text
compiled WorldFoam beats per-frame WorldFoam replay for F >= 8 or F >= 16 at
equal quality when the training comparison includes every per-step transfer
rebuild. Frozen-world inference may report a separate amortized threshold.
```

### Gate G2: Backward correctness

Pass if:

```text
gradient errors vs reference are bounded,
optimizer steps decrease loss,
no gradient field is silently zero.
```

### Gate G3: Visibility stress advantage

Pass if:

```text
WorldFoam reduces crossing/translucency artifacts or flicker versus sorted
baselines at comparable cost.
```

### Gate G4: Public quality floor

Pass if:

```text
DeepView-like heldout PSNR >= 13
DeepView-like heldout SSIM >= 0.15
quality is competitive with same-budget replay and not catastrophically behind
STAR/dynamic-gsplat baselines.
```

### Gate G5: Official parity

Pass if:

```text
CUDA/Warp or official fixture parity is validated,
or the paper clearly scopes itself to the Metal prototype and avoids official
PowerFoam reproduction claims.
```

### Gate G6: Memory and temporal-sharing contract

Pass on fixed-duration unique-frame sweeps only if:

```text
world parameters are invariant in F
reverse interaction bytes F=256 / F=16 <= 1.10
per-step transfer/scratch peak scales with spatial block B_p and temporal block K
loss and world gradients are invariant across B_p/K schedules
production training retains no full O(PF) target/ray tensor or Python O(PR) word graph
event/topology records track physical complexity rather than sampling density
```

Report logical tensor payload separately from measured allocator, driver,
process-resident, host-I/O, optimizer, and media-retention peaks.

## 7. Work Queue

Immediate:

1. [x] Complete the float64 M0--M5 fixed-segment reference, including
   independent quadrature, explicit/autograd/finite-difference VJPs, density
   bounds, and numerical branch codes.
2. [x] Complete one parameterized Metal fixed-tape segment-material
   microkernel and fail-loud wrapper; the owner-run renderer remains unchanged.
3. [x] Run the bounded CPU/Metal forward and explicit-VJP parity fixture and
   save its schema-rich artifact. This is numerical validation infrastructure,
   not a paper-quality or speed result.

Existing support fixture (not counted here as a newly cleared paper gate):
keep
`research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py`
and its tests as the constant-density scan/replay baseline; the M0--M5
evaluator lowers to the same `(beta,m)` contract.

4. [x] Run the matched material-value fitting gate on controlled partial-chord
   observations, including disjoint held-out chords and an independent target
   oracle. The three-seed result finds no universal M3/M5 winner, so do not
   promote either law from its own-family exactness.
5. [x] Run adaptive per-cell M3/M5 basis selection at matched 24-byte material
   payloads plus one basis-tag bit. The verified CPU synthetic gate achieves
   `1.0` pure-family selection accuracy and oracle agreement and reduces mean
   heldout loss to `0.313405` of the best fixed basis. Keep P0 in the native
   systems path until a separate real-data/native promotion gate passes.
6. [x] Build crossing translucent slab/Gaussian-sheet fixtures beyond the
   constant-density and fixed-segment numerical gates. The accepted float64
   CPU ray-section suite at
   `outputs/benchmarks/2026-08-15_worldfoam_synthetic_visibility_cpu/summary.json`
   expands this to all `S1--S8 x C1--C7` contexts, depth-layer convergence,
   adaptive fallback, sorted and depth-marginal comparators, temporal error,
   and an independent gauge-Jacobian gate. It closes representation-level
   G0/G3 only; it is not native speed/memory, full kinetic compilation, or
   public-quality evidence.
7. [ ] Select and register one existing per-frame WorldFoam replay as the F1
   baseline so compiler speed is isolated from representation differences.
8. [ ] Convert existing Gate4 results into a reproducible clean rerun script with
   saved JSON/Markdown summary, without treating those historical shader rows
   as native finite-element evidence.

Near term:

1. Do not promote a universal material law from the synthetic gates: M3 and M5
   have identical six-scalar payloads and complementary exact-family wins.
   CPU adaptive validation selection is now green; test real heldout material
   or image observations before any native rich-material promotion.
2. Native-lower the landed direct-kinetic multi-chart program and frozen-
   program VJP, then rebuild and measure parameter bytes, allocator peak,
   bandwidth, and event/atlas growth versus frame count. Promote a richer
   material law separately after its own gate.
3. Only after native same-representation parity, run clean real
   `2/4/8/16/32f` loaded-frame sweeps without repeated frames.
4. Build a public DeepView/Neural3D subset config and matched STAR/GS
   comparators on an approved host.
5. Log Cech/AABB/witness/holonomy diagnostics on existing foam checkpoints.
6. Decide whether WorldFoam is a theory+prototype paper now or waits for the
   quality gate.

Do not do yet:

```text
long hyperparameter sweeps before synthetic exactness and same-representation
replay baselines are clean.
```

## 8. Paper Decision Rule

Submit as a theory/prototype paper if:

```text
G0 + G1 + G2 + G3 pass,
and the paper is explicit that real-scene quality is preliminary.
```

Submit as a full rendering paper only if:

```text
G0 + G1 + G2 + G3 + G4 + G5 pass.
```

Otherwise:

```text
keep WorldFoam as an internal second-paper lane while World Tubes carries the
nearer-term arXiv push.
```
