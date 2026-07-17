# WorldFoam Paper: Ablations, Charts, Baselines, and Acceptance Gates

Draft date: 2026-07-05

This is the execution plan for the second paper lane:

```text
WorldFoam in Gauged Camera Space:
Ray-Fiber Transmittance Fields for Dynamic Rendering
```

It is intentionally stricter than the theory draft. It says what must be run
before we can claim anything beyond "promising theory and prototype."

## 0. Claim Boundary

Current strongest claim:

```text
WorldFoam is a camera-gauged lifted opacity/transmittance formulation with a
Metal prototype. Focused local microgates show favorable speed/frame-count
scaling against matched STAR UVT comparators, but real public quality and full
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
| Cell-path replay equivalence | compiled atlas emits the same certified cell/event word and image as per-frame WorldFoam replay | compiler changes semantics instead of amortizing structure |
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

## 1. Baselines

### 1.1 Same-representation baselines

These isolate the compiler.

| ID | Baseline | Purpose |
| --- | --- | --- |
| F0 | Dense raymarch / analytic reference | Ground truth for synthetic density scenes. |
| F1 | Per-frame WorldFoam replay | Main same-representation baseline. |
| F2 | Per-frame replay with cached camera constants | Separates trivial camera caching. |
| F3 | Cached cell active sets, live ray/cell intersections | Tests active-set caching alone. |
| F4 | Compiled cell intersections, live transmittance prefix | Tests geometry compilation. |
| F5 | Compiled intersections + depth-layer prefix | Proposed core foam atlas. |
| F6 | Full compiled atlas + backward prefix/suffix reuse | Training-speed target. |
| F7 | Full atlas with fallback/reference cells | Robustness target. |

Required measurements:

```text
quality: RGB PSNR, SSIM, LPIPS, alpha error, transmittance error
speed: compile ms, forward ms, backward ms, optimizer step ms
memory: atlas bytes, prefix bytes, peak GPU memory
structure: active cells, intersection records, depth layers, fallback fraction
gradients: max/mean gradient error vs reference, optimizer loss decrease
```

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
compile amortization point
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
backward-replay components, and total step improves once compile is amortized.
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
no tape, recompute prefix/suffix
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
compact scalar prefix/weight tape or fused recompute.
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
compiled WorldFoam beats per-frame WorldFoam replay after amortization for
F >= 8 or F >= 16, at equal quality.
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

## 7. Work Queue

Immediate:

1. Implement
   `research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py`
   and
   `research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py`
   from `experiment_designs/cell_path_optical_transfer_fixture.md`.
2. Build a synthetic transmittance correctness suite with analytic/dense
   references.
3. Add crossing translucent slab/Gaussian-sheet visibility stress.
4. Convert current Gate4 results into a reproducible clean rerun script with
   saved JSON/Markdown summary.
5. Add per-frame WorldFoam replay baseline if missing, so compiler speed is
   isolated from representation differences.
6. Add gradient correctness test for lifted prefix/suffix transmittance beyond
   the constant-density cell-path fixture.

Near term:

1. Run clean real `2/4/8/16/32f` loaded-frame sweeps without repeated frames.
2. Build a public DeepView/Neural3D subset config and matched STAR/GS
   comparators.
3. Log Cech/AABB/witness/holonomy diagnostics on existing foam checkpoints.
4. Decide whether WorldFoam is a theory+prototype paper now or waits for the
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
