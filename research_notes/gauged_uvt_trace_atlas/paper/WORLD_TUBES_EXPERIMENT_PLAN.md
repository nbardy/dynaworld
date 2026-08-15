# World Tubes Paper: Ablations, Charts, Baselines, and Datasets

Draft date: 2026-07-28

This is the execution plan for turning the current Gauged UVT / projective
interval evidence stack into a publishable arXiv paper. It is deliberately
organized around falsifiable experiments rather than a narrative wish list.

## 0. Paper Claim Boundary

Primary claim:

```text
Known or low-dimensional camera programs let dynamic Gaussian primitives be
compiled into reusable sensor-time world tubes. This makes the dominant
world-side bottlenecks in the tested fixed-topology training-step/world-VJP
regime scale with trace/event complexity rather than frame count.
```

Sharper wording for the paper:

```text
We do not claim an information-theoretic sublinear bound in output samples.
We claim and observe sublinear scaling of the dominant projection/support/
binning/visibility/backward-replay bottlenecks under the tested fixed-topology
compiler/evaluator regime. End-to-end training growth under structural
invalidation and recompilation remains unclaimed.
```

Do not claim:

```text
universal replacement for all 4DGS methods
state-of-the-art dynamic novel-view synthesis quality
arbitrary single-frame novel-view speedup
complete solution for fallback-heavy visibility chaos
```

The strongest publishable comparison is against **per-frame replay of the same
representation**, then contextual comparison against external dynamic Gaussian
baselines.

The frozen-world executor is implemented in
`research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py`.
It trains one World Tubes world, hashes its final checkpoint, and evaluates
per-frame replay versus one compiled interval atlas with identical heldout
targets and world-parameter VJPs. The full-frame public result remains pending
on an approved execution host.

## 0.5 Pass-4 Strict-SPD(4) Source Gate

The production STAR Metal back half already evaluates anisotropic
`spatial_precision_uv` and pixel-varying affine `depth_affine_uv`. The new
work is therefore a source/compiler gate, not a renderer fork:

```text
strict mean_xyzt + covariance_xyzt in Sym++(4)
  -> affine camera-ray gauge
  -> UVT mean and precision
  -> affine conditional-depth plane
  -> positive conditional-depth variance
  -> confidence-band order certificate
  -> existing STAR interval-atlas fields
```

Required CPU/reference tests:

1. Random well-conditioned SPD(4) Cholesky/reconstruction error.
2. Affine-gauge pushforward against direct joint Gaussian evaluation.
3. Covariance-block conditional formulas against the equivalent
   precision-block Schur complement.
4. Motion-from-cross-covariance slices against direct Gaussian conditioning.
5. Peak-preserving versus fiber-integrated amplitude conventions.
6. Exact geometry and peak-preserving adapter parity with the existing UVT
   atlas on its representable subset, plus an explicit test that the
   fiber-optical-depth mapping is only a thin-opacity approximation to physical
   Beer--Lambert alpha.
7. Thick-depth overlap fixtures where separated means but overlapping
   confidence bands reject hard ordering.
8. Dense retained-fiber reference counterexamples for differently colored
   overlap.

Session acceptance:

```text
float64 algebra/reconstruction max error <= 1e-10
gradient relative error <= 1e-5
geometry/peak-preserving adapter parity at the declared dtype tolerance
zero silent projection from full SPD(4) to the restricted legacy tuple
```

The reference/compiler gate and controlled tilt/depth-width capacity test now
pass. The float32 source is wired as an opt-in `full_spd4` producer beside the
default `legacy_tube` producer. The synthetic three-camera design has rank six
over symmetric spatial covariance, begins with losses matched within 0.2%,
and ends at `1.16e-13` MSE for full SPD(4) versus `2.07e-4` for the restricted
source. This is a controlled capacity result, not a public-scene quality
claim.

Pass 4 has also moved beyond the source-only gate:

1. The native Metal tile compiler carries conditional-depth variance into a
   confidence-band order certificate. It accepts the fast hard-order route
   only for separated bands and routes ambiguous, invalid, or overflowing
   cells to retained depth.
2. Physical Beer--Lambert alpha and its forward/VJP are implemented for the
   selected static native-SPD(4) route.
3. `retained_fiber_metal` and `hybrid_retained_fiber` provide differentiable
   retained-depth optical transfer on that static route. The retained branch
   consumes conditional depth variance; the ordinary peak-splat fast ABI still
   does not.
4. `dynamic_first_order` and `projective_first_order` compile a moving camera
   through the tested homogeneous one-chart affine gauge. These are
   first-order camera-program charts, not an exact nonlinear/projective
   camera-path result; unsupported segmented programs remain fail-loud.

The bounded Coffee Martini comparison now supplies short-run computational
evidence. All rows use `cam04/cam09 -> cam06`, seed 17, 16 frames, 40 optimizer
steps, four targets per step, and the same `direct_atomic+index_add` training
route.

| Representation / transfer | Atoms | Trainable scalars | Heldout PSNR (dB) | Train wall (s) | Sampled peak driver bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy tube / peak splat | 256 | 3,584 | 5.9865 | 4.9020 | 63,356,928 |
| full SPD(4) / peak splat | 199 | 3,582 | 7.0054 | 4.7512 | 46,596,096 |
| full SPD(4) / Beer, fiber integrated | 199 | 3,582 | 7.1333 | 4.6758 | 46,596,096 |

Evidence roots:

```text
artifacts/spd4_bounded_16f_40step/legacy_256/
artifacts/spd4_bounded_16f_40step/full_spd4_199_param_matched_optimized/
artifacts/spd4_bounded_16f_40step/full_spd4_199_beer_fiber_optimized/
artifacts/spd4_retained_hybrid_smoke/
```

These one-seed, short-protocol rows refute the earlier interpretation that a
roughly tenfold slowdown was inherent to native SPD(4) in this runner. They do
not establish public-scene convergence superiority, a general speedup, or a
representation-only memory advantage: the matched-SPD(4) rows use fewer atoms
to match trainable-scalar count.

The retained/hybrid smoke establishes a narrower result and exposes the next
failure mode. At 16 atoms, `hybrid_retained_fiber` sends `10/64` tiles to
retained depth and matches the full-retained result at recorded metric
precision. At 199 atoms, the conservative certificate sends `64/64` tiles to
fallback. Thus the mixed route is wired consistently on the small smoke, but
dense-scene selectivity remains unresolved; the 199-atom result is a negative
control, not a hybrid speed result.

Still missing for a production-complete retained-fiber extension:

```text
certified nonlinear/projective camera charts with retained-fiber fallback
adaptive, error-controlled retained-depth quadrature for forward and VJP
a selective dense-scene certificate that preserves the declared error bound
```

These retained-fiber items do not block the central projective interval-atlas
paper claim. The seven-row Coffee Martini progressive/control subset is the
minimum matched-protocol public context table. The remaining 14 rows in the
full public matrix are breadth targets, not blockers for the narrow compiler
paper. Every publication run requires a separately approved, adequate
execution host. The bounded local rows may enter an engineering table only
with their one-seed/short-protocol label intact.

### 0.6 World Tubes + Ordered Ray Transfer ablation

The retained-depth extension should be described as **ordered ray transfer**,
not as open-ray holonomy. It is inspired by connection and parallel-transport
mathematics, while holonomy is reserved here for closed-loop transport. This
is an opt-in World Tubes ablation, not a paper rename.

| ID | Representation | Alpha / amplitude | Backend |
| --- | --- | --- | --- |
| WT-OT0 | `legacy_tube` | `peak_splat` | `metal_tile` |
| WT-OT1 | `full_spd4` | `beer_lambert / fiber_integrated` | `metal_tile` |
| WT-OT2 | `full_spd4` | `beer_lambert / fiber_integrated` | `retained_fiber_metal` |
| WT-OT3 | `full_spd4` | `beer_lambert / fiber_integrated` | `hybrid_retained_fiber` |

WT-OT2 is the all-retained oracle. WT-OT3 must report fallback fraction,
reason bits, active atoms, minimum separation, fixed-quadrature identity,
forward/backward time, memory, heldout quality, and image/VJP parity against
WT-OT2. Run the checked-in 16-frame/40-step seed-17 protocol first. Do not add
these rows to the frozen public matrix or spend multi-seed compute until the
199-atom `64/64` negative control becomes selective. The full execution and
falsification contract is `TODO/world_tubes_ordered_transfer_ablation.md`.

## 1. Baselines

### 1.1 Same-representation baselines

These isolate the contribution, but they are not all separate submission
blockers.

| ID | Baseline | Purpose | Closeout status |
| --- | --- | --- | --- |
| B0 | Per-frame STAR UVT / Gaussian-tube replay | Main causal baseline. | Implemented; public frozen sweep pending. |
| B1 | Per-frame replay with cached camera constants | Separates camera caching from compilation. | Optional attribution row. |
| B2 | Cached active set, live depth/order | Tests whether support caching alone explains speed. | Optional attribution row. |
| B3 | Affine UVT trace atlas | Tests projective/gauge domains vs simple affine UVT. | Bounded synthetic artifact complete. |
| B4 | Projective trace atlas, no interval compression | Tests interval compression. | Optional attribution row. |
| B5 | Marginalized conditional depth only | Tests whether mean-depth visibility is enough. | Crossing negative control complete. |
| B6 | Projective interval atlas + visibility gauge atlas | Proposed method. | Implemented; public frozen sweep pending. |
| B7 | Full atlas with fallback enabled | Robustness under visibility stress. | Bounded stress complete; public fallback evidence pending. |
| B8 | WorldFoam transmittance teaser | Non-baseline-compatible retained-depth context. | Optional and parked. |

Required outputs:

```text
quality: PSNR, SSIM, LPIPS, alpha error
speed: compile, forward, backward, total step
memory: payload bytes, tile entries, interval entries, peak GPU memory
metadata: trace count, active-set size, fallback fraction, order-strata count
visibility: overlap graph edges, certified-order pairs, commutable pairs
```

### 1.2 External dynamic Gaussian baselines

Use these as contextual baselines, not the main proof.

| Baseline | Why include |
| --- | --- |
| 3DGS | Static ancestor and renderer-speed reference. |
| 4D-GS | Core dynamic Gaussian baseline. |
| Deformable 3DGS | Canonical Gaussian + deformation baseline. |
| Dynamic 3D Gaussians | Persistent moving Gaussian baseline. |
| Spacetime Gaussian Feature Splatting | Closest spacetime-Gaussian representation. |
| 4DGS-1K / compressed 4DGS variants | Redundancy/compression context. |
| Gaussian Splatting on the Move | Finite exposure / rolling-shutter context. |
| 3DGUT | Nonlinear camera / rolling-shutter context. |

External comparison policy:

```text
If code is practical to run, train/evaluate on public subsets.
If code is not practical, cite reported numbers and compare only methodology.
Never mix hardware FPS claims without hardware normalization.
Always report our same-representation baseline first.
```

## 2. Datasets

### 2.1 Synthetic controlled suite

This is the most important dataset for the paper's theorem-like claims.

Scenes:

```text
S1 isolated moving Gaussian
S2 two crossing translucent Gaussians
S3 near-camera expanding Gaussian
S4 thin foreground occluder
S5 dense semi-transparent cloud
S6 rigid object approximated by many Gaussians
S7 fast dynamic object plus moving camera
```

Camera programs:

```text
C1 static camera, increasing frame samples
C2 linear dolly
C3 orbit
C4 fast orbit
C5 orbit + rolling shutter
C6 finite exposure with K shutter samples
C7 local camera family Q1
C8 local camera family Q2
```

Ground truth:

```text
dense ray/fiber integration
high-sample temporal integration for exposure
per-sample live depth sort
```

Why:

```text
Only synthetic data gives exact trace error, depth-order error, fallback
fraction, and controlled event complexity.
```

### 2.2 Neural 3D Video

Use for real multiview dynamic scenes. The repository format includes multiple
camera videos and `poses_bounds.npy`, which is convenient for camera-path
sampling and heldout views.

Experiments:

```text
train same primitive family on selected scenes
render heldout camera paths with F = 4,8,16,32,64
compare per-frame replay vs trace atlas
report quality tether and scaling
```

### 2.3 D-NeRF

Use D-NeRF as a controlled posed-frame negative/control. Official matched-time
train and test poses are discontinuous under the current adapter, so each frame
forms a separate gauge chart. Report correctness, chart and fallback counts,
and the absence of cross-frame reuse. Do not aggregate this row with
synchronized multicamera scaling or present it as bounded-chart sublinear
scaling.

### 2.4 HyperNeRF / DyCheck

Use as stress tests for topology, monocular capture, and disocclusion. These
should not be the first acceptance dataset because the method is not primarily
a monocular hallucination model.

Experiments:

```text
fallback fraction under topology/disocclusion
quality degradation near event boundaries
compile amortization when camera path is smooth
```

### 2.5 Technicolor / Google Immersive style scenes

Use if practical for high-resolution multiview dynamic comparisons and to
align with Spacetime Gaussian Feature Splatting's evaluation ecosystem.

Experiments:

```text
high-resolution frame-count scaling
memory pressure from tile-time bins
quality tether at 1K or lower crop first, full-res later
```

### 2.6 Internal broad10 real-video set

Use in engineering appendix only unless we can release exact source list,
splits, preprocessing, and scripts.

Current internal evidence:

```text
10 broad quality sources
10 broad media sources
20 compiled trainer case payloads
4 frame-count points
compiled-adjoint replacement accepted
```

## 3. Ablations

### A1. Frame count scaling

Sweep:

```text
F = 4, 8, 16, 32, 64, 128
```

Plot:

```text
x-axis: frame count
y-axis: payload bytes / per-frame replay payload bytes
y-axis: forward ms / replay forward ms
y-axis: backward ms / replay backward ms
y-axis: interval entries / replay bin entries
```

Pass condition:

```text
world-side ratios decrease or stay sublinear as F grows;
quality delta stays below tolerance.
```

Current internal anchor:

```text
orbit payload-growth ratio 0.125
trained interval-growth ratio 0.148
max trained final backward ratio 0.094
```

### A2. Trace representation

Compare:

```text
per-frame splat
affine UVT Gaussian trace
projective/rational trace
projective trace with event-certified gauge domains
```

Metrics:

```text
trace fit residual
support coverage
tile entries
image error
fallback fraction
```

Expected result:

```text
affine UVT is fine for small motion;
projective/gauged traces dominate on orbit/revolving paths.
```

### A3. Gauge invariance and Jacobian

Compare the same trace in:

```text
ordinary depth
log depth
inverse depth
projective orbit coordinate
```

Controls:

```text
with measure Jacobian
without measure Jacobian
orientation reversal without order boundary
```

Current internal anchor:

```text
value max relative error with Jacobian: 3.50e-13
value error without Jacobian: >= 0.600
gradient relative error with Jacobian: 2.33e-12
gradient error without Jacobian: >= 0.592
```

### A4. Visibility handling

Compare:

```text
compiled total order
compiled partial order + commutable pairs
visibility gauge atlas over pairwise depth-difference signs
live per-pixel sort
depth-bin compositing
mixed fallback
reference fallback
optional world-foam transmittance
```

Metrics:

```text
order-change surfaces per tile
local support-overlap graph edge count
certified order-pair fraction
ambiguous pair count
commutation-bound accepted count
fallback fraction
image error near event boundaries
speed lost to fallback
```

Pass condition:

```text
ordinary scenes stay below 10-20% fallback;
stress scenes fail gracefully with measured fallback cost.
visibility gauge atlas matches the selected baseline order up to certified
commutation error and trace approximation error.
```

Important theorem target:

```text
For every support-overlapping pair in cell C_l, either sign(Delta_ij) is
certified, the pair is certified commutable below epsilon, or the region is
fallback. Under that condition compiled compositing matches the baseline
sorted renderer up to epsilon plus trace approximation error.
```

### A5. Interval compression

Compare:

```text
fixed temporal slabs
interval active_start/active_stop
interval + tile-time cell grouping
interval + active-set strata
```

Metrics:

```text
tile entries
metadata bytes
pack time
forward/backward time
```

Expected:

```text
interval compression matters most as frame count grows.
```

### A6. Backward pass / compiled adjoint

Compare:

```text
per-frame autograd replay
materialized batch replay
interval direct VJP
interval direct VJP + shared camera-family chain rule
```

Metrics:

```text
gradient relative error
backward time
gradient payload bytes
peak memory
stale refreshes / support rebins
```

Current internal anchor:

```text
20 broad10 compiled trainer payloads
all gradient flags present
all projective interval main path
compiled trainer replacement gap 0
max trained final backward ratio 0.094
```

### A7. Camera-family sharing

Sweep:

```text
Q1 grid sizes: 3,5,7,9
Q2 grid sizes: 3x3,5x5,7x7
```

Compare:

```text
replay one atlas per q
shared Q atlas
shared Q atlas + native family eval
```

Metrics:

```text
payload growth
chart/domain count
fit residual px
gradient error
forward/backward launches
```

Current internal anchor:

```text
Q2 replay payload growth 64x
Q2 shared payload growth 1x
final payload ratio 0.0625
final chart ratio 0.015625
max UV fit residual 0.111 px
```

### A8. Exposure and rolling shutter

Sweep:

```text
shutter samples K = 1,4,8,16,32
rolling readout fraction = 0, 0.25, 0.5, 1.0 frame
camera angular velocity
object velocity
```

Compare:

```text
baseline per-sample render
compiled atlas sampled at K
compiled atlas adaptive quadrature
high-sample reference
```

Metrics:

```text
quality vs high-sample reference
unique time samples reused
payload growth
forward/backward time
```

### A9. Memory bandwidth and peak memory

Measure:

```text
atlas metadata bytes
tile/bin bytes
primitive parameter bytes
gradient payload bytes
peak MPS/CUDA memory
read/write traffic proxy if profiler available
```

Plots:

```text
memory vs frame count
memory vs camera-family grid
memory vs fallback fraction
```

### A10. Compile amortization point

Question:

```text
How many frames/shutter samples are needed before compile cost pays back?
```

Metric:

```text
F_break_even = compile_time / (per_frame_replay_ms - atlas_eval_ms)
```

Report per scene/camera path.

## 4. Charts To Generate

### Main paper charts

1. **Frame scaling line chart**
   - x: `F`
   - y: normalized payload, interval entries, forward, backward
   - series: per-frame replay, affine UVT, projective atlas

2. **Memory scaling chart**
   - x: `F`
   - y: metadata/tile/bin bytes
   - series: per-frame bins, fixed slabs, interval atlas

3. **Quality tether scatter**
   - x: baseline PSNR or loss
   - y: compiled PSNR or loss
   - diagonal reference

4. **Gradient parity bar chart**
   - coeffs, opacity, temporal opacity, spatial precision, color
   - y: relative error

5. **Camera-family sharing chart**
   - x: number of q samples
   - y: payload growth
   - series: replay per q, shared Q, shared Q2

6. **Fallback stress curve**
   - x: visibility complexity / crossing count / opacity density
   - y: fallback fraction and image error

7. **Break-even plot**
   - x: scene/camera path
   - y: required frame count to amortize compile

### Appendix charts

```text
support over/under coverage
active set size histogram
tile occupancy p50/p95/max
order-strata count histogram
stale refresh count
support rebin count
contact-sheet pixel delta
timing variance fresh-process vs warm-state
```

## 5. Acceptance Targets

Strong paper target:

```text
>= 3x reduction in projection/support/binning metadata work for F>=32
>= 1.3x end-to-end speedup on finite exposure or dense video inference
<= 0.1-0.3 dB PSNR loss against same-representation per-frame replay
gradient relative error <= 1e-5 for differentiable trace parameters
fallback fraction <= 10-20% on ordinary public scenes
break-even <= 16-32 frames for smooth camera paths
```

Minimum arXiv target:

```text
clear sublinear metadata/backward scaling on synthetic + internal/public subset
same-representation quality parity
honest limitation section for fallback-heavy scenes
one public dataset table
one runnable demo command
```

## 6. Immediate Work Queue

Submission-critical, in order:

1. Run one lane-isolated frozen identical-world job on an approved host with
   explicit frame counts `0,4,8,16,32,64,128`. The implementation trains and
   saves once, samples every `F` across the same full physical interval, and
   rejects checkpoint, world-state, target-grid, or evaluator drift. First
   verify non-unit selected-time full-atlas versus chunk-slice forward/VJP
   parity. The runner preserves the original single-shot route timings as
   correctness diagnostics and separately collects alternating paired,
   synchronized timings with one warmup and five reported repeats by default.
   Only the repeated timing summaries are eligible for speed claims. The live
   report must also retain a topology-inclusive serialized compiled-atlas
   artifact and route-scoped synchronized allocator baselines/peaks for replay
   and compilation. Logical tensor volume is a work proxy only; it is not a
   retained-storage or peak-memory result, and the interleaved parity replay
   must not contaminate the compiled-route peak.
2. Run the implemented bounded variable-camera closure/death gate while
   holding the world, physical interval, and requested sample count fixed.
   It compares the compiled atlas against an exact rational, per-sample
   live-depth-order oracle and reports chart/event/trace counts, fallback,
   image parity, and world-VJP parity. The static public sweep alone cannot
   support the moving-camera claim.
3. Complete the seven-row Coffee Martini schema-v2 submission subset:
   progressive seeds `17/29/43`, pixel-matched fixed seeds `17/29/43`, and
   global-shuffle seed `17`.
4. Feed the verified JSON into
   `generate_world_tubes_paper_artifacts.py`. Its default command must reject
   incomplete evidence; only the verified complete bundle may feed the final
   tables and figures. Package the one-command paper demo.
5. Finish citations, venue LaTeX, reproducibility metadata, and rendered-PDF
   verification.

Breadth target after the minimum paper cut:

- six alternate-triplet rows;
- six additional-Neural3D rows;
- one controlled D-NeRF row;
- one separately labelled deterministic timing audit.

Deferred extensions -- not submission blockers:

- generic Type-II composite-transfer and Type-III event-boundary records;
- event-boundary gradient estimators and structural trust-region refresh;
- full `360/720` multi-chart orbit transitions;
- nonlinear/projective retained-fiber fallback;
- adaptive retained-depth quadrature;
- dense-scene retained-fiber certificate calibration;
- multi-seed WT-OT0--3 convergence;
- WorldFoam material-basis selection and native-4D integration;
- external SOTA reproduction and CUDA portability.

## 7. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Reviewers expect SOTA 4DGS quality | Rejection for wrong claim | Lead with renderer/compiler and same-representation baselines. |
| Public dataset setup consumes time | Delays paper | Start with 1-2 scenes and synthetic suite. |
| Fallback-heavy scenes erase speedup | Weakens generality | Treat as limitation; quantify fallback fraction. |
| Visibility gauge pair graph becomes dense | Kills sublinear claim | Consider only local support-overlap pairs; report edge growth. |
| WorldFoam distracts from tube paper | Dilutes thesis | Keep foam as optional teaser and second-paper lane. |
| Current route is RGB/MPS-specific | Portability concern | State as implementation; add CUDA/CPU reference if possible. |
| Internal broad10 not reproducible | Evidence skepticism | Use internal only as engineering appendix; public data for main table. |
| Timing variance on MPS | Noisy speed claims | Use fresh-process median protocol and report variance. |
| Visual quality lane still active | Confused claims | Separate renderer equivalence from model-quality advancement. |

## 8. Suggested Paper Result Tables

### Table A: Existing internal evidence to port into paper

| Claim | Existing value | Artifact |
| --- | --- | --- |
| Gauge-invariant trace value | `3.50e-13` max rel error | `2026-05-25_star_uvt_projective_bundle_gauge_invariance` |
| Missing Jacobian breaks value | `>=0.600` rel error | same |
| Gauge-invariant gradients | `2.33e-12` max rel error | `2026-05-25_star_uvt_projective_bundle_gauge_gradient` |
| Missing Jacobian breaks gradients | `>=0.592` rel error | same |
| Orbit payload-growth ratio | `0.125` | `projective_shared_work_goal_audit` |
| Trained interval-growth ratio | `0.148` | same |
| Final backward ratio | `0.094` | same / final completion audit |
| Fresh-process no-first timing ratio | `0.565` | timing protocol acceptance |
| Projective total timing ratio | `0.836` | timing protocol acceptance |
| Broad trainer case payloads | `20` | compiled-adjoint replacement |
| Broad distinct trainer sources | `10` | compiled-adjoint replacement |
| Completion promotion | `is_goal_complete=true` | goal completion promotion audit |

### Table B: Needed public-facing results

| Result | Needed before paper? | Notes |
| --- | --- | --- |
| Synthetic exact correctness | Yes | This is the clean math proof. |
| Public Neural3D frozen replay-versus-compiled scaling | Yes | One learned checkpoint and identical targets; include the full-frame result and frame-count sweep. |
| D-NeRF posed-frame fallback control | Yes, separately labelled | Official pose/time discontinuities require one frame per chart; report correctness and fallback behavior, not bounded-chart sublinear scaling. |
| External 4DGS/STG comparison | Nice but not required for first arXiv | Contextual; avoid overclaim. |
| Rolling shutter / exposure comparison | Strong if ready | Could be synthetic plus one real/video path. |
| CUDA portability | Nice | Current MPS/Metal is acceptable for arXiv prototype if honest. |
