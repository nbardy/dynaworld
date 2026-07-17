# World Tubes Paper: Ablations, Charts, Baselines, and Datasets

Draft date: 2026-07-04

This is the execution plan for turning the current Gauged UVT / projective
interval evidence stack into a publishable arXiv paper. It is deliberately
organized around falsifiable experiments rather than a narrative wish list.

## 0. Paper Claim Boundary

Primary claim:

```text
Known or low-dimensional camera programs let dynamic Gaussian primitives be
compiled into reusable sensor-time world tubes. This makes the dominant
world-side training bottlenecks scale with trace/event complexity rather than
frame count.
```

Sharper wording for the paper:

```text
We do not claim an information-theoretic sublinear bound in output samples.
We claim and observe sublinear scaling of the dominant projection/support/
binning/visibility/backward-replay bottlenecks; in the tested training regime
this yields sublinear end-to-end training-time growth.
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

## 1. Baselines

### 1.1 Same-representation baselines

These are mandatory because they isolate the contribution.

| ID | Baseline | Purpose |
| --- | --- | --- |
| B0 | Per-frame STAR UVT / Gaussian-tube replay | Main baseline: project/bin/sort/render each frame. |
| B1 | Per-frame replay with cached camera constants | Separates camera math caching from trace atlas. |
| B2 | Cached active set, live depth/order | Tests whether support caching alone explains speed. |
| B3 | Affine UVT trace atlas | Tests projective/gauge domains vs simple affine UVT. |
| B4 | Projective trace atlas, no interval compression | Tests interval compression contribution. |
| B5 | Projective interval atlas + marginalized conditional depth only | Tests whether mean-depth visibility is enough. |
| B6 | Projective interval atlas + visibility gauge atlas | Proposed baseline-compatible method. |
| B7 | Full atlas with fallback enabled | Robustness version under visibility stress. |
| B8 | World-foam transmittance teaser | Optional non-baseline-compatible alpha/transmittance mode. |

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

Use for controlled synthetic dynamic object sequences. Good for comparisons
where geometry/motion is known-ish and public scripts exist.

Experiments:

```text
source-view or novel-view path replay
orbit camera programs
visibility stress on synthetic non-rigid motion
```

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

1. Build `paper_demo/` command:

```text
compile atlas -> render frame stack -> run backward -> emit JSON + contact sheet
```

2. Generate synthetic trace suite.

3. Add the visibility gauge atlas synthetic test:

```text
pairwise Delta_ij sign certificates
interval depth predicates
commutation-bound acceptance
fallback mask
baseline sorted-render equivalence
```

4. Add one world-foam teaser scene:

```text
crossing translucent slabs or crossing Gaussian sheets
compare baseline center sorting, visibility-gauge split/fallback, and
foam transmittance integral
```

5. Export current internal artifacts into a paper table:

```text
artifact path
claim
metric
value
figure/table target
```

6. Run same-representation public subset:

```text
Neural 3D Video, 1-2 scenes, low resolution first
D-NeRF, 2 synthetic scenes
```

7. Add baseline wrappers:

```text
per-frame replay
affine UVT atlas
projective interval atlas
projective interval atlas + visibility gauge atlas
fallback-enabled atlas
```

8. Generate charts from JSON summaries.

9. Only then decide whether to spend time on external SOTA training runs.

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
| Public dataset same-representation frame scaling | Yes | At least N3DV + D-NeRF subset. |
| External 4DGS/STG comparison | Nice but not required for first arXiv | Contextual; avoid overclaim. |
| Rolling shutter / exposure comparison | Strong if ready | Could be synthetic plus one real/video path. |
| CUDA portability | Nice | Current MPS/Metal is acceptable for arXiv prototype if honest. |
