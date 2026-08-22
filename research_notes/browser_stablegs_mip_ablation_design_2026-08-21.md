# Browser StableGS And Mip Ablation Design

Date: 2026-08-21

## Status

The browser trainer now has a source-complete, reset-time ablation stack for:

- the legacy covariance floor versus a determinant-compensated 2D Mip filter;
- coupled opacity versus dual geometry/appearance opacity;
- an auxiliary geometry-path color loss;
- prior-free paired-camera depth consistency;
- CPU-only multilayer-depth diagnostics for camera stress.

The implementation is isolated to `web/dynaworld_browser_trainer/`. It does
not enter the Python paper-trainer hierarchy, change the canonical dataset or
split contract, run browser SfM, or create a second camera representation.

The implementation is StableGS-inspired, not StableGS parity. In particular,
the browser uses seed-frustum overlap instead of COLMAP feature tracks, omits
the paper's homography-degeneracy test, and applies a one-sided source-depth
gradient on each periodic event rather than a simultaneous bidirectional loss.

## Trigger

Small orbit, dolly, and zoom changes exposed blurry translucent clouds even
when calibrated train and heldout views looked good. The existing camera stress
suite made the failure visible, but it changed no optimizer gradient. The next
step needed to be an actual train-time intervention with independent switches,
not another diagnostic-only score or a bundle of inseparable changes.

## Paper Mapping

### StableGS

[StableGS](https://arxiv.org/html/2503.18458) identifies a pseudo-equilibrium
where color errors cancel and opacity gradients vanish for floaters. Its main
architectural response is dual opacity:

```text
geometry opacity:   sigma_g(x) = alpha * G(x)
appearance opacity: sigma_a(x) = alpha_aux * sigma_g(x)
```

The geometry path receives strong depth pressure. The appearance path can
retain fine transparency, including glass, through `alpha_aux`. The paper also
uses an auxiliary geometry-color L1 term and bidirectional self-supervised
depth consistency. Its external monocular-depth prior is intentionally absent
here.

### Mip-Splatting

[Mip-Splatting](https://arxiv.org/abs/2311.16493) has two distinct mechanisms:

1. a 3D smoothing constraint derived from the maximum sampling frequency of
   the training cameras;
2. a determinant-compensated 2D Mip filter that replaces bare screen-space
   dilation.

Only item 2 is implemented. The UI and code call it `Mip 2D compensated` so it
cannot be mistaken for the complete paper.

### Pixel-GS

[Pixel-GS](https://arxiv.org/abs/2403.15530) motivates the existing
near-camera scaling of the non-cancelling density statistic. That control is
orthogonal to this ablation stack: it changes topology-growth evidence, while
dual opacity and paired depth change the rendered objective.

## Frozen Baseline

The baseline tuple is:

```text
pixelFilterMode       = legacy-floor
opacityModel          = coupled
geometryColorWeight   = 0
crossViewDepth        = false
```

At that tuple:

- the 24-float splat ABI is unchanged;
- lane 11 remains ignored padding;
- no geometry packet, geometry checkpoint, reference raster, or depth buffer
  is allocated;
- the generated raster and VJP use the previous opacity/filter equations;
- CPU validation ignores lane 11 as well.

This exact baseline is the control for every quality and speed comparison.

## 2D Mip Math

Let `C` be the unfiltered projected 2D covariance in height-normalized screen
coordinates. Let the pixel filter variance be:

```text
s = (0.3 / H)^2
C_f = C + s I
```

The compensated peak multiplier is:

```text
rho = clamp(sqrt(det(C) / det(C_f)), 0, 1)
```

The filtered Gaussian is evaluated with `C_f`, while peak opacity is multiplied
by `rho`. Broad splats have `rho ~= 1`; subpixel splats are widened without
gaining artificial integrated mass.

The staged one-per-splat VJP differentiates both `C_f` and `rho`. For an
incoming scalar derivative `bar_rho`, the determinant contribution is:

```text
dL/dC += 0.5 * bar_rho * rho
         * (d log det(C) / dC - d log det(C_f) / dC)
```

The preview renderer uses the same compensation at display resolution. A
forward-only patch would make live output and training optimize different
functions, so that route is rejected.

## Dual Opacity Math

Each splat keeps the existing base opacity logit `o` and reuses the old padding
lane `harmonicPad.w` as material logit `m_raw`:

```text
alpha_base = sigmoid(o)
alpha_aux  = sigmoid(m_raw + log(99))
```

The bias makes an old zero-filled checkpoint initialize to `alpha_aux = 0.99`.
The learned lane is bounded to `[-16, 8]`, so its effective biased opacity can
range from approximately `1.1e-5` to nearly one. This wider lower range is
important for glass; a `[-8, 8]` lane would bottom out near 3.2% after adding
the compatibility bias.

For temporal weight `w(t)`, filter compensation `rho`, and 2D Gaussian value
`G_2d(p)`:

```text
alpha_geometry   = alpha_base * w(t) * rho * G_2d(p)
alpha_appearance = alpha_aux * alpha_geometry
```

Both paths retain front-to-back source-over transmittance. A softmax over
splats is not used because it would replace, rather than regularize, this
visibility law.

## Fused Forward And Backward

When a geometry objective is enabled, one tile-sorted forward loop carries two
states:

```text
(C_app, T_app)
(C_geo, T_geo, M_geo)
```

where `M_geo = sum(T_i * alpha_geometry_i * z_i)` is the geometric depth first
moment. The geometry path does not launch a second source-camera raster.

Checkpoint-block backward replays the same sorted block once and accumulates:

- appearance RGB and opacity derivatives through `alpha_appearance`;
- geometry-color derivatives through `alpha_geometry`;
- expected-depth derivatives through `M_geo / (1 - T_geo)`;
- one projected mean/conic/peak/color/material record per splat.

The existing staged update then applies the expensive 3D projection/covariance
VJP once per splat. The projected-gradient record remains 12 floats and the
splat record remains 24 floats. Mean alpha for densification is read from
`screen1.z`; the material-opacity gradient uses the otherwise spare
`colorPad.w`, so dual opacity does not widen either record.

The modes are shader specializations generated at reset time. Pixel-filter and
opacity choices therefore add no per-contribution mode switch to the baseline
shader. Geometry-enabled variants have their own packet/checkpoint layout.

## Objective

The fused objective is:

```text
L = 0.8 * L1(C_app, C_gt)
  + 0.2 * DSSIM(C_app, C_gt)
  + lambda_color * L1(C_geo, C_gt)
  + I_depth(step) * lambda_depth * L_pair_depth
```

The 11x11 Gaussian DSSIM remains only on the appearance path. Geometry color
uses L1, matching the role of StableGS's auxiliary scaffold loss without
doubling the SSIM workspace.

The UI defaults are deliberately control-preserving:

```text
lambda_color = 0
lambda_depth = 0.05
depth cadence = every 8 steps
cross-view depth = off
```

`0.05` matches StableGS's reported depth-consistency weight, but it is not a
claim that the browser approximation has the same effective scale.

## Prior-Free Paired Depth

On an active cadence event, the source geometry depth map is already available
from the fused forward. A reference camera is projected and tile-sorted on the
same WebGPU command encoder, and a geometry-only expected-depth map is rendered.

For source pixel `p`:

1. unproject `p` using source intrinsics and source expected depth;
2. transform the point through the known canonical camera matrices;
3. project it into the selected reference camera;
4. bilinearly sample reference expected depth;
5. compare reprojected reference-frame `z` with sampled reference depth.

The robust residual is:

```text
r = (z_reprojected - z_reference) / geometryScale
L_pair_depth = mean(sqrt(r^2 + 0.01^2))
```

Only source depth receives a gradient in the current event. Reference depth and
the bilinear sample coordinate are stop-gradient. Source cameras and reference
cameras rotate through the schedule, so both directions are sampled over time,
but this is not equivalent to StableGS's simultaneous bidirectional equation.

All reference passes share the training queue submission. `trainStep` performs
no `mapAsync`, `onSubmittedWorkDone`, metric readback, or validation wait.

## Pair Selection Contract

StableGS selects pairs using more than 30 co-visible COLMAP features, a
16-60 degree relative rotation, and a homography inlier fraction below 0.8.

The browser bundle does not carry feature tracks or pairwise homographies. It
therefore precomputes a conservative approximation from the canonical seed
cloud:

```text
minimum co-visible seed points = 30
minimum seed co-visible fraction = 0.25
relative rotation = [16, 60] degrees
train cameras only
```

Candidates are ordered near a 30-degree rotation. If a small custom bundle has
no qualifying candidate, the best-overlap pair is retained and the fallback
count is exposed in `memoryPlan.geometryPairSelection`.

Important limitations:

- frustum-visible seeds are not verified feature-track co-visibility;
- there is no homography-degeneracy rejection;
- the checked-in external seed cloud has unverified construction provenance;
- heldout cameras are never pair candidates or optimizer targets.

The seed cloud is used only to choose pairs. It is not a depth prior in the
loss.

## Glass Contract

The dual-opacity path can express the StableGS glass behavior:

- geometry opacity can make a bottle an opaque structural surface for depth
  consistency;
- auxiliary opacity can make the final appearance path transparent again;
- geometry-color loss keeps the base scaffold attached to observed RGB.

This does not make the browser renderer a complete glass model. It still has:

- one view-independent RGB per splat;
- no refraction law or environment transport;
- no explicit front/back surface identity;
- no normal or de-lighting prior;
- expected-depth ambiguity when several transparent layers have similar mass.

The implementation should therefore reduce the floater/transparency conflict,
not be advertised as physically correct refractive reconstruction.

## Multilayer Diagnostics

CPU validation now estimates, for stressed rays:

- the fraction with two opacity-supported depth groups separated by at least
  15% in relative depth;
- the weaker group's share of opacity mass.

These are diagnostic only. They do not collapse a multimodal ray to a single
surface or add a loss. That distinction matters for glass: a generic
distortion or single-depth penalty can erase valid front/back/transmitted
layers while removing floaters.

## Memory Model

The static resource estimator reports the following for 8,192 capacity splats,
packed projection VJP, packed application checkpoints, a 4,096-entry tile cap,
and the complete geometry stack:

| Raster | Fast baseline | Geometry stack | Largest geometry binding |
| --- | ---: | ---: | ---: |
| 96x72 | 23.45 MiB | 53.14 MiB | 27.0 MiB geometry checkpoints |
| 384x288 | 191.13 MiB | 278.67 MiB | 108.0 MiB geometry checkpoints |

At 384x288 the geometry variant increases checkpoint stride from 32 to 64 so
both application and geometry bindings stay beneath the portable 128 MiB
storage-binding floor. These are allocator-model outputs, not measured GPU
residency or bandwidth claims.

The main optional costs are:

- 64-byte per-pixel geometry packet instead of the 16-byte baseline gradient;
- one 16-byte geometry checkpoint per retained checkpoint block;
- reference raster projections, tile lists, and an 8-byte depth/coverage map
  when paired depth is enabled;
- reference-camera passes only on the selected cadence.

## UI Ablation Matrix

All controls are reset-time controls on `tiled3d-fast`:

| ID | Pixel filter | Opacity | Geometry color | Paired depth | Question |
| --- | --- | --- | ---: | --- | --- |
| A0 | legacy | coupled | 0 | off | Exact fast baseline |
| A1 | Mip 2D | coupled | 0 | off | Scale/zoom filtering only |
| A2 | legacy | dual | 0.05 | off | Decoupling and scaffold only |
| A3 | legacy | dual | 0.05 | every 8 | Prior-free floater pressure |
| A4 | Mip 2D | dual | 0.05 | every 8 | Combined candidate |
| A5 | Mip 2D | dual | 0.05 | every 4 | Cadence/cost sensitivity |

Do not promote A4 merely because it contains more mechanisms. A1-A3 are needed
to identify whether improvements come from sampling stability, dual opacity,
or actual cross-view geometry pressure.

## Falsification Plan

### Hypothesis H1: floaters are trapped opacity equilibria

Support:
    A2 or A3 reduces orbit near-alpha and multilayer mass without sacrificing
    calibrated-view PSNR/SSIM.

Weakening result:
    material opacity changes but physical-pose stress remains unchanged.

Next action if weakened:
    inspect topology support and relocation rather than increasing depth weight.

### Hypothesis H2: zoom artifacts are sampling-rate artifacts

Support:
    A1 improves deterministic optical zoom/shift PSNR and scale consistency.

Weakening result:
    A1 changes only blur while zoom/shift PSNR and stress remain flat.

Next action if weakened:
    test Mip-Splatting's separate 3D smoothing constraint.

### Hypothesis H3: prior-free pair depth is enough for Coffee Martini

Support:
    A3 improves heldout and physical-camera stress with no external prior.

Weakening result:
    pair residual falls while heldout/orbit geometry becomes smoother or worse.

Next action if weakened:
    add the missing pair-quality/occlusion contract before increasing model
    complexity. Do not jump directly to a monocular prior.

### Hypothesis H4: the stack remains practically fast

Support:
    alternating A0/A4 timing has stable host-qualified throughput and the
    cadence spike matches the predicted periodic reference cost.

Weakening result:
    non-depth steps regress materially, indicating geometry packet/checkpoint
    bandwidth dominates even when reference passes are absent.

Next action if weakened:
    gate source geometry checkpointing to active depth steps when
    `lambda_color=0`, or create a raster-only reference projection shader that
    omits the unused reference VJP packet.

## Verification Status

Verified locally:

- JavaScript syntax for browser modules and tests;
- 194 browser unit/contract tests;
- exact baseline defaults and CPU baseline equivalence;
- compensated-filter determinant behavior;
- dual appearance/geometry separation;
- pair rotation and heldout exclusion;
- no train-step GPU readback/synchronization call;
- resource estimates and storage-binding preflight;
- clean `git diff --check`.

Not yet verified in this work chunk:

- compilation of every possible specialized WGSL combination on the live Apple adapter;
- forward/backward finite differences for the complete geometry variant;
- matched quality ablations A0-A5;
- host-qualified throughput for the geometry stack;
- glass on a dataset with scored transparent geometry.

No `BASELINES.md` row should be added until those runtime artifacts exist.

### 2026-08-22 headless runtime follow-up

The selected full candidate and exact control were initialized in headless
Chrome/Dawn on the Apple adapter at 8,192 splats and 96x72. Live WGSL
validation found two source-only-test blind spots before measurement:

1. checkpoint-block backward emitted an unused projection-VJP helper against
   the raster-only packet, which does not contain `cameraPointValid`;
2. reference-depth tile-count atomics were declared with read-only storage
   access, which WGSL forbids for atomic types.

Both were corrected and the matched run completed with finite loss, zero tile
overflow, and zero projection-VJP FP16 saturation. Under a heavily contended
host, the baseline measured 499.2 steps/s and the full stack measured 476.9
steps/s, a diagnostic 4.5% wall-throughput reduction. The candidate's four
rounds were stable (CV 0.042); the control was not (CV 0.195). Preflight also
reported roughly 70% existing Apple GPU utilization, high CPU load, and high
swap, so the artifact is explicitly non-promotable. The resource gate blocked
the reversed-order mate. This is compile/trainability evidence and a rough cost
bound, not a benchmark or a quality result.

The temporary artifact is
`/private/tmp/dynaworld-stablegs-geometry-headless.json`; it is intentionally
not a retained baseline artifact. No PSNR/SSIM conclusion follows from the
64-step kernel timing run because its reported objective includes the new
regularizers and the runner does not perform heldout image validation.

## Decision

Keep all new controls opt-in until matched evidence exists. The implementation
is valuable now because it converts the camera-stress finding into falsifiable
optimizer interventions while preserving an exact control. The next work is
measurement and parity, not another renderer family.
