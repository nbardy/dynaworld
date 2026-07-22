---
title: "World Tubes in Gauged Camera Space: Sublinear Frame Scaling for Dynamic Gaussian Splatting"
author: Anonymous
date: 2026-07-22
---

Status: arXiv-style working manuscript with generated LaTeX. The certified
correctness and same-representation scaling tables are populated, and the
three-seed progressive Coffee Martini row is accepted. The remaining public
matrix is paused after the local-memory incident. This is not a final
submission until the pixel-matched/sampler controls and public-scene breadth
in `WORLD_TUBES_EXPERIMENT_PLAN.md` are complete.

The frozen public workload now has one 21-row manifest: seven primary Coffee
Martini/control rows, six alternate-triplet rows, six additional-Neural3D rows,
one controlled D-NeRF row, and one separately labelled deterministic audit.
Only the three primary progressive rows are currently accepted.

## Abstract

Dynamic Gaussian splatting methods render a time-varying scene by evaluating
or deforming primitives at a target timestamp, projecting the resulting
Gaussians, binning them into tiles, resolving visibility, and compositing the
visible contributors. This pipeline is efficient for isolated frames, but it
repeats the same world-side work when a renderer must produce many temporally
nearby samples from a known camera program: video playback, finite exposure,
rolling shutter, dense temporal supervision, or repeated training/evaluation
views.

We introduce **World Tubes in Gauged Camera Space**, a camera-path compiler for
dynamic Gaussian primitives. The core object is a sensor-time trace atlas over
`(u,v,t)`: each spacetime primitive is pulled back through the camera-ray
bundle and pushed forward along the ray fiber to produce a reusable
viewport-time footprint, conditional depth model, support certificate, and
adjoint structure. Locally, a spacetime Gaussian pulled through a camera gauge
and integrated over ray depth yields a UVT footprint by a Schur-complement
fiber marginalization. Globally, camera gauges and event-certified domains make
this construction invariant to depth coordinates on bounded camera-path chart
segments, including the tested orbit segments, finite exposure, and rolling
shutter. The renderer evaluates frames as slices of the
compiled atlas, while training accumulates gradients through a compiled
interval VJP.

On our current STAR UVT / projective interval implementation, the compiled
atlas shows sublinear world-side growth across frame count: the rerun over
`F={4,8,16,32,64,128}` keeps fixed payload growth at `1.0x` while per-frame
replay grows `32.0x` (final payload ratio `0.03125`), with final fixed/replay
CPU compile, forward, and backward ratios of `0.0477`, `0.181`, and `0.392`;
trained high-motion traces keep final
shared/per-frame interval-entry ratio below `0.149`, final trace-count ratio at
`0.1`, final forward ratio below `0.266`, and final backward ratio below
`0.094`. A broad real-video audit covers 10 source-distinct cases, 20
projective-interval trainer payloads, gradient-preserving compiled-adjoint
replacement, and fresh-process median timing with no-first/projective-total
ratios of `0.565`/`0.836`. These results suggest that camera-path compilation
is a practical complement to dynamic Gaussian representations: it does not
claim an information-theoretic sublinear bound in the number of output pixels,
but in the tested training regimes the dominant bottlenecks scale with
trace/event complexity rather than frame count. The remaining per-pixel
shading term is real, but projection, support, binning, visibility metadata,
and backward replay are amortized across time.

## 1. Introduction

3D Gaussian Splatting (3DGS) made real-time neural rendering practical by
replacing expensive volumetric ray marching with visibility-aware anisotropic
splatting. Dynamic extensions such as 4D Gaussian Splatting, Deformable 3D
Gaussians, Dynamic 3D Gaussians, and Spacetime Gaussian Feature Splatting
extend the representation through deformation fields, persistent motion, or
spacetime primitives. These methods improve dynamic scene modeling, but they
usually retain a per-target-view rendering loop: evaluate the primitive state
at the requested timestamp, project to screen, build tile bins, estimate or
sort depth order, shade, composite, and backpropagate through that target
render.

This is the wrong computational unit for several common workloads. A video
renderer needs many temporally adjacent frames. A finite-exposure renderer
needs many shutter samples per output image. A rolling-shutter camera couples
image row to capture time. Training often revisits the same camera path or a
small family of nearby paths many times. In these settings, the output image
samples still scale as `O(F H W)`, but the world-side work should not have to
scale linearly with frame count `F`. The paper's claim is therefore not that
pixels disappear. It is that the expensive training bottlenecks that dominate
dynamic Gaussian pipelines--projection, support, tile membership, ordering
metadata, and backward replay--can scale with trace/event complexity instead
of frame count. In our tested regimes this produces sublinear measured
training-time growth for the compiled route.

We propose to compile the dynamic world through the camera program itself. A
world primitive is not projected independently into each frame; it becomes a
**world tube** observed as a **sensor-time trace**:

```text
world spacetime primitive -> alpha_i(u,v,t), c_i(u,v,t), z_i(u,v,t).
```

The compiled object is a **trace atlas** over the sensor-time base:

```text
B = Omega x T,      y = (u, v, tau).
```

Each atlas domain stores active primitive traces, support bounds, local depth
models, visibility/order certificates, fallback metadata, and differentiable
state. Frames are slices of this atlas, and finite-exposure or rolling-shutter
images are integrals or row-coupled samples through the same object.

This construction is not merely a renamed STAR UVT renderer. STAR UVT supplies
the spacetime Gaussian representation and the sparse Metal execution lineage;
the gauged camera-ray formulation supplies the new compiler semantics. In
particular, it pulls primitives through a moving camera program, marginalizes
the ray-depth fiber without discarding conditional depth, and partitions
support-overlap regions at certified depth-order events. A single raw interval
can therefore fail under a large-motion order crossing while the corresponding
visibility-stratified gauge domains remain replay-equivalent. This distinction
is part of the method and must not be removed by implementation cleanup.

Our contributions are:

1. **A camera-ray bundle formulation for dynamic Gaussian rendering.** We
   define a trace as `pi_* Gamma^* world_primitive`, i.e. pull the primitive
   through the camera program and integrate/summarize it along the ray fiber.

2. **A local Schur-complement derivation for world tubes.** Under a local
   camera gauge, depth marginalization of a pulled-back spacetime Gaussian
   gives a UVT Gaussian-like footprint and conditional depth statistics used
   for support, visibility, and gradients.

3. **Event-certified gauge domains.** Instead of treating charts as ad hoc
   fitted patches, we use gauge domains that certify projection regularity,
   support validity, depth/order behavior, fallback conditions, and backward
   support.

4. **A lifted visibility gauge atlas.** Footprint traces are pushed down to
   `(u,v,t)` for support and shading, while visibility is compiled from lifted
   depth/order fields before ray-depth structure is discarded. Pairwise
   support-overlap predicates become sign certificates over gauge domains.

5. **A projective interval atlas implementation.** We implement interval
   compressed projective traces with Metal forward and direct backward paths
   for a STAR UVT feature-tube route. Visibility order and tile membership are
   compiled constants during the direct VJP, while trace coefficients, opacity,
   temporal opacity, spatial precision, and color remain differentiable.

6. **A sublinear frame-scaling evaluation protocol.** We report payload,
   trace-count, interval-entry, forward, backward, and timing ratios against
   per-frame replay, plus quality/media tethers to verify that the compiled
   route preserves the baseline renderer's output and gradients.

## 2. Related Work

**Gaussian splatting.** 3DGS introduced anisotropic 3D Gaussian primitives and a
visibility-aware rasterizer that supports real-time rendering and
optimization. Our work keeps this rasterization motivation but changes the
unit of compilation from one screen at one time to a sensor-time camera path.

**Dynamic Gaussian representations.** 4D-GS combines 3D Gaussians with 4D
neural voxels and lightweight deformation prediction. Deformable 3D Gaussians
learn a canonical Gaussian scene plus deformation field. Dynamic 3D Gaussians
track persistent Gaussians over time. Spacetime Gaussian Feature Splatting
adds temporal opacity and parametric motion/rotation to Gaussian primitives.
These methods primarily address the dynamic scene representation. We instead
target the repeated rendering work induced by known camera paths and many
temporal samples.

**Nonlinear cameras, rolling shutter, and ray-space splatting.** Gaussian
Splatting on the Move models blur and rolling shutter under natural camera
motion. 3DGUT replaces the EWA projection approximation with an unscented
transform to support nonlinear cameras, rolling shutter, and secondary rays.
Our method is complementary: sigma/projective projection helps define or test
gauge domains, while the trace atlas amortizes camera-path work over many
samples.

**Dynamic view-synthesis datasets.** Neural 3D Video, D-NeRF, HyperNeRF,
DyCheck, Technicolor-style light-field videos, and Google Immersive-style
captures provide public dynamic-scene benchmarks. Our paper should evaluate on
public data for comparison, but the central experiments must also include a
controlled synthetic trace suite where exact ray-fiber integration and
visibility events are known.

## 3. Method

### 3.1 Sensor-time base and camera-ray bundle

Let the sensor-time base be:

```text
B = Omega x T,      y = (u, v, tau).
```

Let the camera program define a ray bundle:

```text
pi: E_Gamma -> B,
pi^{-1}(y) = F_y.
```

The fiber `F_y` is the ray-depth domain over the sensor sample `y`. A camera
program maps bundle points into world spacetime:

```text
Gamma: E_Gamma -> M,      M = R^3 x R.
```

For a world primitive `w_i`, the invariant sensor-time trace is:

```text
Trace_Gamma[w_i] = pi_* Gamma^* w_i.
```

For a density-like primitive, this becomes:

```text
bar_rho_i(y) = integral_{F_y} rho_i(Gamma(e)) dmu_y(e).
```

This equation is the renderer's invariant object. It does not depend on a
particular depth coordinate or local chart.

### 3.2 Gauges and event-certified domains

A gauge is a local trivialization of the ray bundle over a domain `C_a`:

```text
chi_a: E_Gamma | C_a -> C_a x D_a,
chi_a(e) = (y, z_a).
```

In gauge `a`, the trace is:

```text
bar_rho_i^a(y)
  = integral_{D_a} rho_i(Gamma_a(y, z_a)) J_a(y, z_a) dz_a.
```

The Jacobian `J_a` is not optional. It is the measure correction that makes
ordinary depth, log depth, inverse depth, orbit-angle, and projective gauges
represent the same physical trace. In our current gauge-invariance artifact,
ordinary-depth and log-depth integration of the same revolving-camera
spacetime Gaussian agree to `3.50e-13` relative error with the Jacobian; without
it, the value error is at least `0.600`. The matching gradient artifact agrees
to `2.33e-12` relative error with the Jacobian; without it, gradient error is
at least `0.592`.

We use the term **gauge domain** instead of weak chart. A gauge domain certifies:

```text
projection denominator regularity
trace representation error
conservative support bounds
tile-time active set
conditional depth model
stable order, commutable order, or fallback
interval gates
backward support matching forward support
```

For revolving cameras, a projective/orbit gauge such as `r = tan(theta / 2)`
can make traces rational or low-order over longer intervals than naive frame
time. The domain still ends at real events: denominator zeros, behind-camera
transitions, near/far crossings, support entering/leaving tiles, order swaps,
and disocclusions.

**Scope of the current implementation.** The compiler and experiments cover
bounded, event-certified orbit segments inside one regular projective chart.
They do not implement chart transitions for complete `360°` or repeated
`720°` revolutions. We therefore make no full-orbit multi-gauge claim in this
paper; such a transition system is future work rather than an untested part of
the method.

### 3.3 Local Gaussian fiber pushforward

Let a spacetime Gaussian primitive be:

```text
rho_i(x) = a_i exp[-1/2 (x - m_i)^T Lambda_i (x - m_i)],
x in R^4.
```

In a local gauge, linearize the camera map around `(y0, z0)`:

```text
Gamma_a(y,z) ~= x0 + J eta,
eta = [delta_y, delta_z]^T.
```

Let `delta = m_i - x0` and `g = J^T Lambda_i delta`, partitioned as
`g = [g_y, g_z]^T`. Then the local exponent is
`eta^T H eta - 2 g^T eta + delta^T Lambda_i delta`.

Plugging the linearized map into the primitive gives a Gaussian over
`(y,z)` with precision:

```text
H = J^T Lambda_i J
  = [ H_yy  H_yz
      H_zy  H_zz ].
```

Marginalizing the fiber coordinate `z` yields the UVT precision:

```text
S = H_yy - H_yz H_zz^{-1} H_zy.
```

The conditional fiber/depth model is:

```text
z_hat_i(y) = z0 + H_zz^{-1}(g_z - H_zy (y - y0)),
Var(z | y) = H_zz^{-1}.
```

For a scalar fiber with `H_zz > 0`, an untruncated local fiber, and locally
constant fiber-measure factor `J_0`, the marginal also has amplitude:

```text
bar_rho_i(y)
  ~= J_0 a_i sqrt(2 pi / H_zz) exp[-1/2 q_y(delta_y)],

q_y(delta_y)
  = delta_y^T S delta_y
    - 2 (g_y - H_yz H_zz^{-1} g_z)^T delta_y
    + delta^T Lambda_i delta - g_z^T H_zz^{-1} g_z.
```

This local closed form must be replaced by certified quadrature or a residual
bound when fiber clipping or a varying Jacobian is not negligible.

Thus a pulled-back 4D world Gaussian becomes a 3D sensor-time footprint plus
conditional depth and uncertainty. These quantities are exactly what a
rasterizer needs for support, tile-time binning, visibility ordering, and
gradient propagation.

### 3.4 Trace atlas representation

The compiled atlas is:

```text
K_Gamma = { C_l, A_l, T_l, Pi_l, E_l }_{l=1}^L.
```

where:

```text
C_l       gauge domain / event cell in (u,v,t)
A_l       active primitive set
T_l       trace functions: alpha_i,l(y), c_i,l(y), z_i,l(y)
Pi_l      stable total order, partial order, or commutation certificate
E_l       error, support, fallback, and backward metadata
```

Rendering at `y in C_l` evaluates active traces and composites them in the
compiled order:

```text
I(y) = sum_m T_m(y) alpha_{pi_m,l}(y) c_{pi_m,l}(y).
```

The transmittance is:

```text
T_m(y) = product_{n<m} (1 - alpha_{pi_n,l}(y)).
```

A frame is a slice:

```text
I_k(u,v) = I(u,v,t_k).
```

Finite exposure is an integral through the atlas:

```text
I_k(u,v) = integral w_k(u,v,tau) I(u,v,tau) d tau.
```

Rolling shutter replaces `tau` with a row/time-coupled sensor program.

### 3.5 Visibility gauge atlas

Depth marginalization alone does not make alpha compositing linear in depth.
The footprint trace:

```text
pi_* Gamma^* w_i
```

is sufficient for support and shading, but visibility is a relation between
lifted traces before the ray-depth structure is fully marginalized. We
therefore compile a second object, a **visibility gauge atlas**:

```text
O_Gamma = { C_l, G_l, Delta_l, Pi_l, R_l }.
```

Here `G_l` is a local support-overlap graph; it contains only primitive pairs
whose sensor-time footprints overlap inside the same tile-time cell. `Delta_l`
stores certified depth/order predicates, `Pi_l` stores the induced total order
or partial-order DAG, and `R_l` stores commutation residuals and fallback
metadata. This avoids the all-pairs `N^2` problem: pairwise certification is
local in support overlap and event complexity, not global in primitive count.

For each primitive we keep conditional depth:

```text
z_hat_i(y),    sigma_z,i(y).
```

or a conservative lifted interval in a depth/order gauge:

```text
D_i(y) = [z_i^-(y), z_i^+(y)].
```

For support-overlapping pairs, define either a center-depth difference:

```text
Delta_ij(y) = z_i(y) - z_j(y)
```

or a conservative interval predicate:

```text
Delta_ij^-(y) = z_i^-(y) - z_j^+(y)
Delta_ij^+(y) = z_i^+(y) - z_j^-(y).
```

Then:

```text
if Delta_ij^+(y) < 0:  i definitely in front of j
if Delta_ij^-(y) > 0:  j definitely in front of i
otherwise:             split, commute, or fallback
```

The gauge should be chosen so these depth-difference fields are cheap to
certify. For orbit/projective cameras, ordinary frame time may make depth
curves high-curvature, while projective time, inverse depth, log depth, or
denominator gauges can make them rational, low-degree, or interval-friendly.

If unresolved pairs remain, we bound the effect of swapping two translucent
contributors:

```text
|Delta I_ij(y)| <= alpha_i(y) alpha_j(y) |c_i(y) - c_j(y)|.
```

Unresolved pairs below tolerance are marked commutable. Important unresolved
pairs induce an event boundary or fallback. Fallback is part of the theorem of
the implementation: hard regions can be rendered by local live sorting or a
reference path, while the rest of the atlas retains shared metadata.

The baseline-compatible theorem is:

```text
Within a domain C_l, if every support-overlapping pair has a certified order
or a certified commutation residual below epsilon, then compiled compositing
matches the baseline sorted-Gaussian renderer up to epsilon and trace
approximation error.
```

This claim is intentionally baseline-relative. It reproduces the chosen
Gaussian-splat compositing semantics. It does not claim that center-depth
alpha compositing is a physically exact solution of radiative transfer.

### 3.6 WorldFoam as lifted transmittance

The visibility gauge atlas preserves baseline Gaussian-splat semantics. A more
radical extension is to retain the lifted ray-fiber opacity field itself:

```text
sigma_l(y,z) = sum_i rho_i(Gamma_l(y,z)).
```

Instead of sorting primitive centers, one renders by Beer-Lambert
transmittance:

```text
tau(y,z) = integral_{z_front}^{z} sigma_l(y,s) ds
T(y,z)   = exp(-tau(y,z))
I(y)     = integral T(y,z) sigma_l(y,z) c_l(y,z) dz.
```

This **world foam** mode dissolves depth order into cumulative opacity along
the ray fiber. It is cleaner for physical transmittance, finite exposure, and
translucent ambiguity, but it is less directly baseline-compatible with
standard Gaussian splat alpha compositing. We therefore treat it as a separate
representation layer and a second paper direction: world tubes compile
primitive support and differentiable attributes; WorldFoam compiles lifted
opacity/transmittance for visibility.

### 3.7 Compiled adjoints

Inside a gauge domain with fixed support and visibility metadata, rendering is
differentiable with respect to trace parameters. For a primitive `i`, the
local derivative of compositing is:

```text
dI/dc_i     = T_i alpha_i,
dI/dalpha_i = T_i (c_i - I_behind,i).
```

The gradient of the loss is:

```text
dL/dtheta_i =
  sum_l integral_{C_l} A_l(y)^T dI(y)/dtheta_i dy,
```

where `A_l(y) = dL/dI(y)` is the image adjoint. The compiled implementation
uses interval Metal forward and direct VJP with topology, active intervals,
and visibility cells held as compiled constants. Trace coefficients, opacity,
temporal opacity, spatial precision, and colors remain differentiable.

## 4. Implementation

Our current implementation is a STAR UVT / projective interval backend. A trace
stores homogeneous/projective time coefficients, opacity, optional temporal
opacity coefficients, optional spatial precision, optional depth-affine terms,
and color. Tile-time cells store active intervals and visibility metadata. The
hot path packs accepted cells once into spatial tile bins and uses per-entry
`[active_start, active_stop)` checks in the Metal kernel.

The forward path is:

```text
render_projective_trace_cell_interval_atlas_metal(...)
```

The direct VJP path is:

```text
direct_backward_projective_trace_cell_interval_atlas_metal(...)
```

The trainer route calls:

```text
_render_projective_interval_feature_tubes_autograd(...)
```

via a custom autograd function:

```text
_ProjectiveCellIntervalBackward
```

Current limitations of this implementation:

```text
MPS/Metal backend
RGB / feature_dim=3 route for the projective interval trainer
compiled visibility/order and tile membership held fixed during direct VJP
fallback support present, but broad fallback-heavy scenes remain a stress case
STAR UVT support/visibility/composition quality is a separate active research lane
```

These are implementation limits, not limits of the bundle formulation.

## 5. Experiments

The final paper should separate four questions:

1. **Correctness:** Does the trace integral match dense ray/fiber evaluation?
2. **Sublinear scaling:** Does world-side work grow sublinearly with frame
   count or camera-family size?
3. **Renderer equivalence:** Does the compiled route preserve images, losses,
   media, and gradients relative to the baseline renderer?
4. **Usefulness:** Does the speed/memory benefit hold on public dynamic-scene
   workloads without excessive fallback?

### 5.1 Synthetic trace correctness

Controlled scenes of analytic 4D Gaussians with known camera programs:

```text
camera paths: static, dolly, orbit, fast orbit, rolling shutter
FOV: 30, 60, 90, 120 degrees
frames/samples: 4, 8, 16, 32, 64, 128
primitive anisotropy: isotropic, elongated spatial, elongated temporal
visibility: isolated, crossing pair, thin foreground occluder, dense translucent
```

Metrics:

```text
trace L1 / KL against dense fiber integration
image PSNR / SSIM / LPIPS
alpha error
depth-order error
fallback fraction
support over/under-coverage
gradient relative error
```

### 5.2 Frame-count scaling

Compare per-frame replay against compiled atlas for fixed camera paths:

```text
F = 4, 8, 16, 32, 64, 128
```

Metrics:

```text
payload bytes
trace count
tile/bin entries
interval entries
CPU compile time
GPU forward time
GPU backward time
total step time
peak memory
```

Current internal evidence:

```text
bounded-orbit F: 4, 8, 16, 32, 64, 128
fixed payload growth: 1.0x vs per-frame replay 32.0x
final fixed/replay payload ratio: 0.03125
final fixed/replay CPU compile ratio: 0.0477
final fixed/replay forward ratio: 0.181
final fixed/replay backward ratio: 0.392
trained interval-entry growth ratio: 0.148
final trained trace-count ratio: 0.1
final trained forward ratio: <= 0.266
final trained backward ratio: <= 0.094
fresh-process median no-first ratio: 0.565
fresh-process median projective-total ratio: 0.836
```

### 5.3 Camera-family scaling

For a low-dimensional camera family `Q x Omega x T`, compare one shared family
atlas against replaying one atlas per camera parameter sample:

```text
Q1: orbit phase offset
Q2: orbit phase + height
grid: 3x3, 5x5, 7x7
```

Current internal evidence:

```text
Q2 shared payload growth: 1.0x
Q2 replay payload growth: 64.0x
Q2 final payload ratio: 0.0625
Q2 final chart ratio: 0.015625
Q2 max UV fit residual: 0.111 px
```

### 5.4 Real-video renderer equivalence

Use source-distinct real videos and compare compiled projective interval
renderer against the cadence/per-frame route.

Metrics:

```text
loss curve delta
RGB loss curve delta
end PSNR delta
contact-sheet pixel delta
PNG hash match
gradient flags present
support rebins
stale refreshes
fallback fraction
```

Current internal evidence:

```text
10 source-distinct broad quality pairs
10 source-distinct broad media pairs
20 broad10 trainer case payloads
all cases projective interval main path
all renderer gradient flags present
zero support rebins / stale refreshes in accepted rows
compiled trainer replacement gap: 0
```

![Qualitative real-video tether for the accepted projective decisive-demo
artifact. Columns compare the target, compiled World Tubes render, and error
views at sampled times; the figure is a correctness/media tether rather than a
public-dataset quality claim.](research_notes/gauged_uvt_trace_atlas/paper/figures/real_video_equivalence.jpg)

### 5.5 Public dataset comparison

The paper should evaluate public data in two ways:

1. **In-representation ablation:** train or initialize our STAR UVT/projective
   primitives on public sequences, then compare frame-by-frame replay versus
   compiled atlas. This is the cleanest evaluation of the contribution.

2. **External dynamic-GS comparison:** compare speed/quality against existing
   dynamic Gaussian baselines where feasible, but present this as contextual
   comparison rather than the main theorem.

Recommended datasets:

```text
Neural 3D Video: real multiview dynamic scenes
D-NeRF: controlled synthetic dynamic scenes
HyperNeRF / DyCheck: monocular/topology stress tests
Technicolor-style light-field scenes: synchronized camera array stress
internal broad10 real-video set: engineering-only unless made reproducible
```

### 5.6 Finite exposure and rolling shutter

Compare:

```text
baseline per-shutter-sample rendering
compiled atlas sampled at same shutter samples
compiled atlas with adaptive temporal quadrature
high-sample reference
```

Metrics:

```text
quality vs high-sample reference
unique time samples
payload growth
forward/backward time
rolling row-time correctness
```

### 5.7 Visibility stress tests

Create scenes where the compiler should fail locally:

```text
crossing translucent slabs
thin foreground occluders
near-camera large splats
fast orbit around high-depth-variance geometry
disocclusion boundary
```

Metrics:

```text
fallback fraction
order-strata count
commutation-bound accepted pairs
image error near event boundaries
speed lost to fallback
```

## 6. Tables and figures

Two accepted figures are packaged with the manuscript: the real-video
equivalence contact sheet above and the partial progressive Coffee Martini
heldout-PSNR comparison below. They are derived from verifier-accepted source
artifacts. The remaining submission figures are:

1. **Concept figure:** per-frame dynamic GS replay vs world-tube trace atlas.
2. **Bundle diagram:** `B = Omega x T`, ray fibers, `Gamma`, pullback, pushforward.
3. **Schur complement diagram:** 4D primitive -> `(u,v,t,z)` Gaussian -> depth-marginalized UVT footprint.
4. **Gauge-domain/event diagram:** orbit camera with projective domains split by denominator/support/order events.
5. **System diagram:** compile, atlas, Metal interval forward, direct VJP.
6. **Scaling chart:** frame count vs payload/bin/forward/backward ratio.
7. **Camera-family chart:** Q-grid replay vs shared family atlas.
8. **Fallback stress chart:** fallback fraction vs visibility density.

Core tables:

```text
Table 1: synthetic correctness vs dense fiber reference
Table 2: frame-count scaling, internal + public sequences
Table 3: ablation of trace representation and gauge domains
Table 4: training/backward replacement and gradient errors
Table 5: public dynamic-scene comparison
Table 6: finite exposure / rolling shutter
Table 7: limitations and fallback-heavy cases
```

### 6.1 Certified correctness and theorem table

All rows below are generated from verifier-accepted JSON artifacts. Their scope
is bounded event-certified projective chart segments; they do not assert an
unimplemented full `360/720` multi-chart transition.

| Claim | Metric | Value | Acceptance |
|---|---|---:|---:|
| Fiber value is gauge invariant | max relative error | `3.50087e-13` | `<= 1e-10` |
| Fiber gradient is gauge invariant | max gradient relative error | `2.32523e-12` | `<= 1e-9` |
| Compiled atlas matches dense/replay image | max absolute image error | `0` | `<= 1e-5` |
| Unstratified interval exposes an order-crossing failure | raw crossing quality error | `0.186742` | `> 1e-5` (expected failure) |
| Visibility crossing is repaired by stratification | stratified crossing quality error | `0` | `<= 1e-5` |
| Finite exposure / rolling shutter forward parity | max Metal absolute error | `5.96046e-08` | `<= 1e-5` |
| Finite exposure / rolling shutter gradient parity | max Metal gradient relative error | `6.37738e-07` | `<= 1e-5` |
| Mixed fallback preserves gradients | max mixed gradient relative error | `7.40632e-07` | `<= 1e-5` |
| Bounded-orbit chart reuses payload at `F=128` | fixed/replay trace ratio | `0.03125` | `< 0.25` |
| Bounded-orbit compiled forward is faster at `F=128` | fixed/replay forward ratio | `0.181323` | `< 0.5` |
| Bounded-orbit compiled backward is faster at `F=128` | fixed/replay backward ratio | `0.392235` | `< 0.5` |

### 6.2 Exact same-representation frame scaling

The accepted `F={4,8,16,32,64,128}` experiment compares per-frame STAR replay
with the compiled projective atlas at identical representation settings. Fixed
payload growth is `1.0x` while replay grows `32.0x`; at `F=128`, fixed/replay
payload, compile, forward, and backward ratios are `0.03125`, `0.047677`,
`0.181323`, and `0.392235`. This is the central causal systems result. Public
quality rows test whether that compiler result survives real scene breadth;
they are not substitutes for this same-representation comparison.

### 6.3 Public comparison status

The shared progressive/fixed/global-shuffle Coffee Martini protocols, evidence
schema, and matrix generator are complete. Three clean-source progressive-512
runs (seeds 17, 29, and 43) completed for all representations on the full
300-frame sequence. The table reports final checkpoints and equal optimizer,
target-frame, and target-pixel budgets. Storage and parameter counts are not
matched: World Tubes shares temporal trace state, while dynamic 3DGS and
WorldFoam retain substantial per-frame state.

| Representation | Heldout PSNR | SSIM | LPIPS | L1 | Train wall (s) | Peak driver (GB) | Checkpoint (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| World Tubes | `5.9153 +/- 0.0053` | `0.03549` | `0.98305` | `0.45120` | `78.33 +/- 21.00` | `3.114` | `0.060` |
| WorldFoam | `5.6159 +/- 0.0083` | `0.00460` | `0.97054` | `0.47221` | `361.82 +/- 55.90` | `15.794` | `116.748` |
| Dynamic 3DGS | `4.9110 +/- 0.0001` | `0.28267` | `0.90228` | `0.52139` | `79.44 +/- 6.59` | `20.557` | `17.206` |

![Heldout PSNR for the three accepted full-300-frame progressive Coffee
Martini seeds. The figure does not include the missing fixed, sampler,
camera-triplet, scene-breadth, or D-NeRF controls.](research_notes/gauged_uvt_trace_atlas/paper/figures/coffee_progressive_heldout_psnr.png)

These are accepted evidence rows, but not a complete public comparison. The
absolute reconstruction quality is low and the metrics disagree: World Tubes
has the best PSNR/L1, whereas dynamic 3DGS has the best SSIM/LPIPS. The
pixel-matched fixed control, global-shuffle control, additional camera
triplets, two additional Neural3D scenes, and controlled D-NeRF row remain
mandatory before submission claims are frozen.

The next fixed-512 run was killed after severe unified-memory compression and
swap pressure destabilized the local workstation. Its partial outputs are
excluded. The runner now isolates representations in child processes and is
fail-closed on local MPS, but full-scale local execution remains unauthorized
until targets/rays/evaluation are streamed or a larger machine is used.

## 7. Discussion and limitations

The method is not a universal replacement for dynamic Gaussian scene
representations. It is a renderer/compiler layer for known or low-dimensional
camera programs. It is most useful when many temporal samples reuse a smooth
camera path or path family.

It is also not an information-theoretic claim that total work is sublinear in
materialized output pixels. The sharper claim is empirical and architectural:
in regimes where dynamic Gaussian training is dominated by world-side
projection, support, binning, visibility, and backward replay, compiling those
events into world tubes makes the dominant bottlenecks scale with trace/event
complexity instead of frame count. In our tested regime this yields sublinear
end-to-end training-time growth, while preserving a residual per-pixel shading
term.

Failure modes:

```text
fallback-heavy visibility chaos can erase speedups
very wide FOV / near-camera splats require more gauge domains
single random novel views do not amortize compile cost
current implementation is RGB/MPS/STAR-UVT scoped
current broad real-video evidence proves renderer equivalence, not general SOTA quality
world foam may be a better alpha/transmittance model, but it changes semantics
```

The main research claim is narrower and stronger:

```text
known camera programs expose repeated world-side rendering work;
camera-gauged world tubes compile that work into reusable sensor-time traces.
```

The hierarchy is:

```text
World Tubes:
  sublinear camera-path compilation for dynamic Gaussian-splat semantics.

Visibility Gauge Atlas:
  certified depth/order compilation for baseline-compatible alpha compositing.

World Foam:
  lifted opacity/transmittance compilation that avoids discrete depth sorting
  and moves toward volumetric ray-fiber transport.
```

## 8. Conclusion

We presented World Tubes in Gauged Camera Space, a trace-atlas renderer for
dynamic Gaussian splatting. By defining rendering as a camera-ray bundle
pushforward, deriving local Schur-complement UVT footprints, and compiling
event-certified gauge domains into interval tile-time metadata, the method
shares projection, support, binning, visibility, and backward work across time.
The current implementation demonstrates sublinear world-side scaling and a
compiled-adjoint training route on STAR UVT/projective interval traces. The
next step toward publication is a public benchmark suite that reproduces these
scaling claims on standard dynamic-view datasets and controlled visibility
stress tests.

## References To Cite

- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering,"
  SIGGRAPH 2023. https://arxiv.org/abs/2308.04079
- Wu et al., "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering,"
  CVPR 2024. https://arxiv.org/abs/2310.08528
- Yang et al., "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic
  Scene Reconstruction," 2023. https://arxiv.org/abs/2309.13101
- Luiten et al., "Dynamic 3D Gaussians: Tracking by Persistent Dynamic View
  Synthesis," 2023. https://arxiv.org/abs/2308.09713
- Li et al., "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View
  Synthesis," CVPR 2024. https://arxiv.org/abs/2312.16812
- Jiang et al., "Gaussian Splatting on the Move: Blur and Rolling Shutter
  Compensation for Natural Camera Motion," 2024. https://arxiv.org/abs/2403.13327
- Wu et al., "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian
  Splatting," CVPR 2025. https://arxiv.org/abs/2412.12507
- Li et al., "Neural 3D Video Synthesis from Multi-view Video," CVPR 2022.
  https://arxiv.org/abs/2103.02597
- Pumarola et al., "D-NeRF: Neural Radiance Fields for Dynamic Scenes," CVPR
  2021. https://arxiv.org/abs/2011.13961
- Park et al., "HyperNeRF: A Higher-Dimensional Representation for
  Topologically Varying Neural Radiance Fields," SIGGRAPH Asia 2021.
  https://hypernerf.github.io/
- Hou et al., "Sort-free Gaussian Splatting via Weighted Sum Rendering," ICLR
  2025. https://arxiv.org/abs/2410.18931
- Koo et al., "Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian
  Splatting," AAAI 2026. https://doi.org/10.1609/aaai.v40i7.37495
