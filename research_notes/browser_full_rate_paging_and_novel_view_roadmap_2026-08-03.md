# Browser Full-Rate Paging And Novel-View Roadmap

Date: 2026-08-03

## Context

The Coffee Martini browser demo looked materially better after the calibrated
camera correction, full-frame tiled training, anisotropic covariance fixes,
random training backgrounds, and the 8K growth reserve. Two problems remained:

1. the browser trained only 16 synchronized time samples from a 300-frame clip;
2. free-camera views still exposed unsupported translucent clouds and missing
   high-frequency structure.

This note records the implementation that makes the full timeline practical,
the temporal VJP bug found during that audit, the paper-backed floater control
added to the density score, and a falsifiable path for the remaining quality
work. The browser remains a systems/demo prototype and reuses the canonical
Python calibration and split contracts through one bundle adapter.

## Observed Facts

- The 18 Coffee Martini source videos each contain 300 frames at 30 fps.
- The clip duration is exactly 10.0 seconds. It is genuinely short; the old
  browser bundle was shorter only in temporal sampling density, not duration.
- The old atlases contained 16 synchronized indices distributed over the full
  clip.
- The active representation is trajectory-gated dynamic 3DGS: a base center,
  linear and optional harmonic center motion, temporal opacity gate, static
  mix, one 3D covariance, one RGB, and one opacity per splat.
- It is not native 4DGS, SpacetimeGS, or World Tubes.
- Training runs in a dedicated worker. Validation runs in another worker.
- The tiled trainer binds one target image on the GPU per optimizer step.
- The default fast backend uses the staged projection VJP and shared full-frame
  backward.
- A normal resident page has 17 train cameras x 16 times = 272 training pairs.
- The heldout `cam06` is never included in optimizer camera membership.

## Full-Rate Data Contract

The bundle now carries a `temporal_stream` contract:

```text
native_frame_count = 300
native_fps = 30
duration_seconds = 10.0
page_size = 16
page_count = 19
last_page_frame_count = 12
```

Each camera points to a same-origin 384x288 H.264 stream. The original 16-frame
RGBA8 atlas remains a fallback, so this is an extension of the thin browser
adapter rather than a second calibration or split format.

### Why pages are interleaved

Contiguous 16-frame pages would make a resident cycle cover only 0.53 seconds.
That can create long stretches where optimization sees one small motion phase.
The page planner therefore stratifies the 300 ordered frame indices into 16
timeline regions and takes one index from every region per page.

Consequences:

- every normal resident cycle spans almost the complete 10 seconds;
- all 300 exact source indices are visited after 19 pages;
- each frame appears exactly once per page rotation;
- source time, not page-local slot, drives the dynamic model;
- frame 0 maps to model time 0 and frame 299 maps to model time 1.

This is deterministic temporal coverage, not random frame sampling.

## Memory Derivation

Definitions:

```text
W, H       raster width and height
C          camera count = 18
F          native frames = 300
K          resident page frames = 16
B_rgba8    bytes per RGBA8 pixel = 4
B_bg       bytes per FP32 RGBA background pixel = 16
```

One decoded target frame is:

```text
frame_bytes = W * H * B_rgba8
```

The complete eager target corpus is:

```text
corpus_bytes = W * H * C * F * B_rgba8
```

Double-buffered target pages plus one camera-mean bank are:

```text
resident_bytes = 2 * W * H * C * K * B_rgba8
               + W * H * C * B_bg
```

Exact results:

| Raster | Eager target corpus | Two target pages | FP32 means | Bounded host total |
| --- | ---: | ---: | ---: | ---: |
| 96x72 | 142.383 MiB | 15.188 MiB | 1.898 MiB | 17.086 MiB |
| 384x288 | 2.225 GiB | 243.000 MiB | 30.375 MiB | 273.375 MiB |

The 18 encoded streams occupy 17,936,497 bytes (17.106 MiB) on disk. Encoded
size is not training memory: loss needs decoded linearized RGB values. Keeping
the raster output at 8 bit would quantize the model and gradient and is not
justified by 8-bit source files. The correct split is compact RGBA8 storage,
FP32 decode for loss and gradient math, and FP16 only for selected cold
checkpoints where parity has been measured.

## Nonblocking Paging Path

The page state machine has one current page and at most one prefetched page.

1. The UI asynchronously seeks and decodes the next page from each camera
   stream into an RGBA8 bank.
2. The training worker continues submitting bounded optimizer bursts. It does
   not await decode.
3. After one complete current-page camera/time cycle, the ready page is
   published through the existing shared dataset boundary.
4. The training worker validates identical raster size, camera identities,
   matrices, and train/heldout membership.
5. It replaces only the resident target bank and cycle-metric extent. Queue
   ordering puts new writes after old submissions without
   `queue.onSubmittedWorkDone`.
6. The validation worker receives the same page identity.

This prevents paging from becoming a periodic optimizer barrier. It does not
prove that decode has zero system impact: `HTMLVideoElement` and canvas decode
currently run asynchronously on the UI thread and can compete for CPU or UI
time. A dedicated WebCodecs worker is justified only if long-task and page
decode telemetry identify that as a real bottleneck.

## Progressive Resolution Marker

The dashed cyan/blue chart line marks the single 96x72 to 384x288 resolution
transition. Parameters, Adam moments, active topology, density statistics, and
global step survive the handoff. It is not:

- a validation event;
- densification or recycling;
- an LR reset;
- a temporal page boundary;
- a periodic synchronization pause.

## Temporal VJP Backtrack

### Prior belief

The fast staged backward was assumed to differentiate the same temporal opacity
gate as the forward renderer because live output moved and the aggregate loss
decreased.

Status: invalidated.

### Forward model

Let:

```text
f = temporal floor
k = exp(-0.5 * (t - t_center)^2 / sigma^2)
s = static mix in [0, 1]
g = f + (1 - f) * k
w = mix(g, 1, s) = g * (1 - s) + s
alpha = sigmoid(opacity_logit) * w * gaussian_2d
```

### Old staged reconstruction

The staged VJP used `w_old = g` and then attempted to reconstruct the dynamic
core from that value. This was inconsistent with the forward whenever `s > 0`.

At the static-heavy initialization `s=0.92`, if `g=0.20`:

```text
w_correct = 0.20 * 0.08 + 0.92 = 0.936
w_old / w_correct = 0.20 / 0.936 = 0.214
```

Thus opacity gradients could be under-scaled by almost 5x in this example, and
the derived temporal-center gradient could also be wrong. A decreasing loss did
not falsify this bug because color and position still had useful gradients.

### Correction

The VJP now reconstructs `w` with the exact forward mix and computes the
dynamic core directly as `(1 - f) * k`. The live parity fixture includes the
production `s=0.92` case plus more dynamic cases.

Observed parity after the fix:

```text
maximum RGB error             1.1920929e-7
objective absolute error      2.2585871e-7
active gradient families      9 / 9
tile overflow                 0
```

Scale LR also retains its former step-zero value but now follows geometry's
100x decay rather than color's 10x decay. This targets late footprint swelling,
not early mobility.

## Current Density Statistic

For each pixel covered by a splat, the backward computes the gradient of loss
with respect to the projected mean, `bar_mu_j`. The density statistic adds
`length(bar_mu_j)` before tile reduction.

Observed property:

```text
sum_j length(bar_mu_j) >= length(sum_j bar_mu_j)
```

The left side cannot cancel opposing pixel gradients. This implements the main
motivation of [AbsGS](https://arxiv.org/abs/2404.10484), though AbsGS sums
absolute x and y components separately and then takes their L2 norm. We should
call the browser statistic "AbsGS-style non-cancelling magnitude," not exact
AbsGS, until the separate-component ablation is run.

Because the browser sum contains one term per participating pixel, a large
footprint already contributes more evidence. That overlaps the pixel-aware
motivation of [Pixel-GS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf),
but it is not Pixel-GS's exact cross-view weighted-average formula.

## Pixel-GS Near-Camera Guard

Pixel-GS observes that near-camera primitives cover many pixels and receive
disproportionate density-growth evidence. The paper scales the density gradient
by:

```text
q(i, camera) = clip((z_camera / (gamma_depth * radius))^2, 0, 1)
gamma_depth = 0.37
radius = 1.1 * max_camera ||camera_center - mean_camera_center||
```

The browser now applies `q` only to the non-cancelling density statistic. It
does not multiply the world-space position gradient, covariance gradient,
appearance gradient, loss, or rendered alpha. This placement is an invariant:
the guard controls where capacity grows without changing the image VJP.

Current belief: this is a low-cost, plausible way to reduce newly spawned
near-camera floaters.

Confidence: medium on mechanism, low on Coffee Martini quality impact until a
matched on/off run completes.

Could be wrong if:

- most floaters are inherited from initialization rather than born by splits;
- floaters are at ordinary depth but unsupported across cameras;
- the camera-radius normalization is poorly matched to this bounded indoor rig;
- static appearance error, not topology, is the main novel-view failure.

Cheap falsification:

- same seed, camera/time order, resolution schedule, capacity, and step budget;
- guard on versus off;
- compare heldout PSNR/SSIM, deterministic orbit alpha/depth instability,
  near-camera splat count, tile pairs, and zero-overflow validity.

## Failure Branches

### Branch A: capacity is allocated to the wrong places

Hypothesis:
    Splits follow large gradient/alpha parents but do not target unsupported
    residuals with multi-view evidence.

Why it might be true:
    The model can fit calibrated views while free-camera alpha reveals clouds.

What would make it false:
    Residual/depth-guided relocation changes topology but not heldout/orbit
    quality.

Cheap test:
    Log per-splat contribution, residual ownership, depth, and view count over
    one deterministic camera cycle.

If supported:
    Add fixed-budget relocation toward high-error pixels with coarse depth and
    at least two-view support.

### Branch B: the footprint model aliases across scale

Hypothesis:
    The 0.3-pixel display sigma floor is insufficient when training and viewing
    scales differ.

Why it might be true:
    Zoomed views expose shimmer, needles, and high-frequency collapse.

What would make it false:
    A full Mip filter improves scale stability but not floaters or heldout
    detail.

Cheap test:
    Render a fixed checkpoint along a deterministic zoom path and measure
    low-pass error and alpha variation before changing training.

If supported:
    Implement both Mip-Splatting's 3D smoothing and determinant-compensated 2D
    filter, including the shared backward.

### Branch C: dynamic representation is under-parameterized

Hypothesis:
    One covariance/color/opacity over all time cannot model articulated motion
    and view-dependent appearance.

Why it might be true:
    Trajectory centers move, but shape and radiance remain static.

What would make it false:
    Static regions and heldout views remain poor at a fixed time even after
    topology and filtering improve.

Cheap test:
    Report static versus dynamic-region metrics and endpoint temporal support.

If supported:
    Compare a paper-faithful 4DGS or SpacetimeGS lane outside this prototype,
    then expose it through the same bundle/metric contract.

### Branch D: free orbit is extrapolating outside support

Hypothesis:
    The heldout calibrated camera is sound, while interactive orbit travels
    beyond the convex support of the training rig.

Why it might be true:
    Camera-corrected `cam06` improved sharply, but aggressive orbit still
    reveals clouds.

What would make it false:
    Floaters are equally strong on a bounded interpolation path inside the
    training camera hull.

Cheap test:
    Define two deterministic paths: hull interpolation and explicit
    extrapolation. Never combine their metrics.

If supported:
    Label extrapolation separately and optimize geometry with cross-view
    support rather than treating all orbit frames as validation.

### Branch E: initialization still leaks unsupported geometry

Hypothesis:
    The external 4,096-point cloud is train-visible but provenance-unverified,
    and its unsupported points seed later clouds.

What would make it false:
    A train-only verified cloud produces the same floater distribution under a
    matched topology schedule.

Cheap test:
    Compare the corrected known-pose train-only cloud against the external
    cloud at equal initial count and equal final capacity.

## Paper Intervention Matrix

| Paper | Relevant mechanism | Browser status | Next decision |
| --- | --- | --- | --- |
| [AbsGS](https://arxiv.org/abs/2404.10484) | Non-cancelling view-space density gradient | Similar scalar per-pixel magnitude is active | Test exact x/y absolute statistic only if topology diagnostics justify it |
| [Pixel-GS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf) | Pixel-aware growth and depth-scaled floater suppression | Depth guard implemented and toggleable; exact cross-view quotient absent | Run matched guard on/off |
| [3DGS-MCMC](https://arxiv.org/abs/2404.09591) | Fixed-count relocation and opacity regularization | Not implemented | Preferred bounded-capacity topology lane |
| [SpacetimeGS](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.html) | Error plus coarse-depth guided births; richer temporal parameters | Not implemented | Borrow allocation evidence before porting representation |
| [Mip-Splatting](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html) | 3D smoothing and 2D Mip filter | Only a 0.3-pixel sigma floor exists | Run deterministic scale-stability diagnostic, then implement full forward/backward |
| [DropoutGS](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_DropoutGS_Dropping_Out_Gaussians_for_Better_Sparse-view_Rendering_CVPR_2025_paper.html) | Gaussian dropout for sparse views | Not implemented | Optional ablation, not default with 17 train cameras |
| [4DGS](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html) | Deformation-based dynamic Gaussian representation | Not implemented in browser | Separate model baseline, not a shader toggle pretending parity |
| [Faster-GS](https://arxiv.org/abs/2602.09999) | Integrated raster/training optimizations with stability controls | Shared full-frame backward, compact projections, packed tape, bounded queues | Preserve parity and zero-overflow gates while optimizing |

## Ordered Ablation Plan

Every lane freezes seed, pair order, random background sequence, resolution
schedule, LR schedule, splat budget, and evaluation snapshots.

### A0: corrected systems baseline

- full 300-frame paging;
- exact staged temporal VJP;
- scale LR tied to geometry decay;
- current non-cancelling density score;
- Pixel-GS guard on;
- 4,096 initial / 8,192 capacity;
- progressive 96x72 to 384x288.

### A1: Pixel-GS guard off

Purpose: isolate the new floater intervention.

Promote guard only if heldout or bounded-orbit stability improves without a
meaningful train-detail or throughput regression.

### A2: exact AbsGS statistic

Replace `sum length(g_j)` with `length(sum abs(g_j.xy))`, preserving all other
settings. This tests whether component-wise accumulation matters beyond the
current non-cancelling scalar.

### A3: fixed-budget residual/depth relocation

At full capacity, choose low-contribution, low-opacity candidates and relocate
them toward high-residual pixels with coarse depth and multi-view support.
Reset moments for relocated slots. Keep active count fixed.

### A4: complete Mip-Splatting filter

Implement the forward and shared backward together. Compare at matched train
resolution and along fixed zoom paths. Do not call a forward-only opacity patch
complete.

### A5: conditional regularizers

- Gaussian dropout only if effective view support is empirically sparse.
- local rigidity only if temporal neighbor motion is inconsistent.
- view-dependent appearance only if geometry/alpha is stable but RGB remains
  view-dependent.

## Required Metrics

Quality:

- full-image train and heldout MSE, MAE, PSNR, SSIM;
- LPIPS in the offline evaluator before a research-baseline claim;
- static-region and motion-region metrics;
- low-pass error and high-frequency residual;
- deterministic hull-interpolation and extrapolation paths kept separate.

Geometry/support:

- rendered alpha mean and low-alpha pixel fraction;
- near-camera splat depth quantiles;
- per-splat visible-camera count;
- residual-weighted contribution;
- fixed-path alpha/depth temporal variation;
- active, dead, split, relocated, and pruned counts.

Systems validity:

- tile overflow must be zero;
- FP16 saturation must be zero or bounded by an explicit acceptance threshold;
- numerical parity after every renderer/backward change;
- memory plan and largest binding recorded;
- throughput promoted only when host preflight passes.

## Performance Preservation Rules

1. Keep optimizer submission independent of UI decode, validation, and metric
   mapping.
2. Keep one target frame GPU-resident; paging is a host data concern.
3. Add diagnostics to existing per-pixel/per-splat passes when possible.
4. Do not add a full per-pixel-per-splat tensor for density control.
5. Keep fixed capacity for browser memory predictability.
6. Preserve compact projection VJP records and packed FP16 checkpoints unless a
   parity or saturation gate fails.
7. Reject any quality result with tile overflow.
8. Never compare a contended live SPA rate with an isolated kernel artifact.

## Verification Completed

- Browser unit suite: 170 behavioral tests after Pixel-GS integration.
- Export adapter and 384 bundle contract: 23 Python tests before final rerun.
- All 18 streams probe as 384x288, 30 fps, 10 seconds, 300 frames.
- Live page visited native frames outside the old 16-frame atlas schedule.
- Live progressive handoff preserved state and continued at 384x288.
- Live parity passed the production static-mix fixture with all intended
  gradient families active.
- No new throughput number was promoted because the host contention preflight
  failed.

## Open Questions

1. Does Pixel-GS depth scaling reduce bounded-orbit alpha/depth instability on
   Coffee Martini, or only change where the 4,096 children are born?
2. Are current floaters primarily newly split children, external seed points,
   or ordinary splats stretched by the optimizer?
3. Does exact AbsGS component accumulation select materially different parents?
4. What fraction of each splat's opacity-weighted contribution is supported by
   two or more training cameras at the same time?
5. Can fixed-budget relocation improve heldout quality without raising tile
   occupancy or erasing the SfM scaffold?
6. Is high-frequency error dominated by topology, filtering, constant color,
   or the trajectory-only dynamic model?
7. At 384x288 and 30K capacity, does video decode ever become visible in
   optimizer completion timestamps on a quiet host?
8. Is a WebCodecs worker measurably better than asynchronous video/canvas decode
   after including its copy and synchronization costs?

## Current Decision

Treat A0 as the corrected browser systems baseline, not a paper baseline.
Run A1 first on a quiet host. If floaters remain but the near-camera count falls,
proceed to A3 rather than adding unrelated losses. If zoom instability dominates
with fixed parameters, prioritize A4. A native 4DGS or World Tubes browser lane
should remain a separate model experiment sharing the canonical data/evaluation
adapter, not a dropdown label over this trajectory 3DGS shader.
