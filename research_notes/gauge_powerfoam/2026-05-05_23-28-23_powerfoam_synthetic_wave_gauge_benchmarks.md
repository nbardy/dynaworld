# PowerFoam Synthetic Wave/Gauge Benchmarks

Date: 2026-05-05 23:28:23 +0700

Scope: proposal note only. This does not claim full official PowerFoam on Metal.
It assumes the current local state: fast/trainable Metal core and local 4K gates
exist; the official CUDA/Warp fixture is absent on this Mac; paper acceptance is
still blocked by low heldout metrics. The goal here is to define small synthetic
scenes that can tell real transported motion from per-frame repainting.

## Common Harness

Use deterministic scenes with analytic state, cameras, masks, and flow. Each
scene should run in two sizes:

- local gradient/debug size: 128x128 or 192x192, 8-16 frames, 2 train cameras,
  1-2 heldout cameras, <= 4k cells/texels where possible.
- 4K stress size: 3840x2160, 2-4 frames, same analytic scene, no long training;
  accept only if the saved verifier reports forward/backward timing and memory.

Train-camera loss alone is not an acceptance signal. A scene only passes if it
meets the analytic metric and the heldout metric. For every dynamic scene, log
both a normal run and a repaint control:

```text
motion-enabled:
    geometry/texels/frames may be transported by the intended dynamic variables

repaint-only control:
    freeze geometry/support/transport; allow per-frame color/material update
```

The benchmark should reject a method when repaint-only gets similar train loss
but fails analytic transport or heldout behavior.

Common metrics:

```text
render_l1 / psnr / ssim
alpha_iou and alpha_mass_error
analytic_flow_epe_px
transport_consistency_l1
identity_repaint_score = train_psnr - heldout_psnr plus transport residual
finite_difference_grad_relerr
4k_forward_ms / 4k_backward_ms / peak_memory_mb
```

Default pass thresholds unless a scene tightens them:

```text
train_psnr >= 30 dB
heldout_psnr >= 27 dB
heldout_ssim >= 0.92
alpha_iou >= 0.95
flow_epe <= 0.75 px at 128-192px debug size
transport_consistency_l1 <= 0.03 in [0,1] color/feature units
finite_difference_grad_relerr <= 2e-2 for active scalar parameters
4k_forward_ms does not regress by > 15% vs current saved local 4K core
4k_backward_ms does not regress by > 20% vs current saved local 4K core
```

## Benchmark Matrix

| Scene | Main Failure It Separates | Expected Metrics | Acceptance Thresholds | Tests |
| --- | --- | --- | --- | --- |
| Advected checker | transported material vs color repaint | checker phase error, flow EPE, heldout PSNR, repaint gap | phase error <= 0.04 cycles; flow EPE <= 0.5 px; heldout PSNR >= 30 dB; repaint-only heldout <= motion run - 4 dB | forward, backward, heldout |
| Sinusoidal height wave | moving geometry vs fixed support with changing texture | height amplitude/phase, normal error, silhouette alpha IoU | amp error <= 5%; phase error <= pi/24; normal cosine >= 0.98; alpha IoU >= 0.96 | forward, backward, 4K, heldout |
| Area-preserving stretch | gauge/cell deformation without density gain/loss | Jacobian determinant, mass conservation, texture pullback error | mean abs(det J - 1) <= 0.02; mass drift <= 1%; pullback L1 <= 0.025 | forward, backward |
| Occlusion/visibility handoff | correct front-to-back ownership vs blended repaint | visibility event timing, front-cell id accuracy, disocclusion error | event time error <= 0.5 frame; active-owner accuracy >= 98%; disocclusion L1 <= 0.04 | forward, backward, heldout |
| Rotating tangent frame | gauge covariance under local frame rotation | equivariance residual, frame-angle recovery, tangent-vector error | scalar render invariant delta <= 1e-3; frame angle error <= 2 deg; tangent vector cosine >= 0.99 | forward, backward |
| Foam birth/death source budget | source/sink dynamics vs opacity hacks | source budget residual, alpha mass trajectory, creation/deletion localization | cumulative budget residual <= 3%; alpha mass RMSE <= 0.03; source localization IoU >= 0.9 | forward, backward, heldout |
| Shallow-water pulse | coupled height/velocity transport vs local flicker | wave speed, energy drift, reflected-pulse timing, heldout temporal SSIM | speed error <= 5%; energy drift <= 5%; reflection timing <= 1 frame; temporal SSIM >= 0.9 | forward, backward, 4K, heldout |
| Negative controls | catches metric leakage and appearance shortcuts | should fail targeted analytic metrics while some RGB loss may look good | no negative control may pass the full scene gate; if it does, the gate is invalid | forward, backward, heldout |

## Scene Specs

### 1. Advected Checker

Analytic state:

```text
domain: square patch in z=0, viewed by 2 train cameras and 1 oblique heldout
texture: c(u,v,t) = checker(u - vx*t, v - vy*t)
motion: constant tangential velocity, no normal displacement
```

Why it matters: a fixed lattice can repaint checker colors per frame and get
good source-view RGB, but it cannot preserve phase under heldout view and time
unless material is transported.

Expected metrics:

- phase error from cross-correlation against analytic checker phase.
- analytic flow EPE in image pixels.
- heldout PSNR/SSIM at intermediate and extrapolated frames.
- repaint gap: motion-enabled heldout PSNR minus repaint-only heldout PSNR.

Acceptance:

- phase error <= 0.04 checker cycles.
- flow EPE <= 0.5 px at debug size.
- heldout PSNR >= 30 dB and SSIM >= 0.95.
- repaint-only control must be at least 4 dB worse on heldout or have
  transport_consistency_l1 >= 0.08.

Tests: forward correctness, backward through transport/color samples, heldout
generalization. It is not primarily a 4K benchmark.

### 2. Sinusoidal Height Wave

Analytic state:

```text
h(x,y,t) = A sin(kx*x + ky*y - omega*t)
normal = normalize([-dh/dx, -dh/dy, 1])
appearance = weak checker or lambertian ramp tied to the surface
```

Why it matters: it forces geometric displacement, changing normals, and
silhouette/alpha changes. Repainting a flat support should fail phase and normal
metrics even if RGB partially matches.

Expected metrics:

- recovered height amplitude and phase from rendered depth/normal diagnostic.
- normal cosine error on visible surface.
- alpha IoU against analytic silhouette.
- heldout PSNR/SSIM from an oblique camera.
- 4K forward/backward timing on 2-4 frames.

Acceptance:

- amplitude error <= 5%.
- phase error <= pi/24.
- mean normal cosine >= 0.98.
- alpha IoU >= 0.96.
- heldout PSNR >= 28 dB and SSIM >= 0.93.
- 4K timing does not regress beyond common thresholds.

Tests: forward, backward, 4K, heldout generalization.

### 3. Area-Preserving Stretch

Analytic state:

```text
u' = s(t) * u
v' = v / s(t)
det J = 1
texture/material follows the pullback, total opacity/mass is conserved
```

Why it matters: gauge-like deformation should not invent density. This catches
models that solve stretch by changing opacity or color rather than transporting
mass and chart coordinates.

Expected metrics:

- local Jacobian determinant error.
- total alpha/mass drift.
- pullback texture L1 under inverse deformation.
- gradient check on stretch parameter and density/radius parameters.

Acceptance:

- mean abs(det J - 1) <= 0.02; p95 <= 0.05.
- total alpha/mass drift <= 1%.
- pullback texture L1 <= 0.025.
- finite-difference gradient relative error <= 2e-2.

Tests: forward geometry/material consistency and backward sensitivity. Heldout
is optional here unless the implementation already supports quick heldout views.

### 4. Occlusion/Visibility Handoff

Analytic state:

```text
two crossing sheets or foam ribbons
front object passes behind/through another with known depth ordering
disoccluded checker patch becomes visible at a known frame
```

Why it matters: true motion must hand off visibility and alpha ownership. A
repaint model can smear, blend, or keep the wrong support active.

Expected metrics:

- active front-cell/object id accuracy where analytic depth gap is large.
- event time error for occlusion and disocclusion.
- alpha IoU on newly visible regions.
- heldout L1/PSNR on disoccluded pixels only.
- backward gradient sign: moving the front sheet in depth should reduce the
  visibility error in the expected direction.

Acceptance:

- active-owner accuracy >= 98% on pixels with depth gap >= 2 cell radii.
- event time error <= 0.5 frame.
- disocclusion alpha IoU >= 0.93.
- disocclusion region L1 <= 0.04.
- gradient sign agreement >= 95% for depth-offset finite differences.

Tests: forward compositing, backward visibility gradients, heldout
generalization.

### 5. Rotating Tangent Frame

Analytic state:

```text
same surface and scalar material, but local tangent frames rotate by theta(t)
scalar fields should render invariantly
tangent-vector fields should rotate covariantly
```

Why it matters: this is the gauge benchmark. It distinguishes benign local
frame rotation from physical motion and catches frame-dependent artifacts.

Expected metrics:

- scalar-render invariance under tangent-frame rotation.
- tangent-vector equivariance residual after applying SO2 frame action.
- gradient consistency with respect to frame angle.
- wrong-frame negative control residual.

Acceptance:

- scalar render delta <= 1e-3 mean absolute RGB/feature units.
- tangent vector cosine >= 0.99 after covariant transport.
- frame angle recovery error <= 2 degrees where the vector field is nonzero.
- finite-difference frame-angle gradient relative error <= 2e-2.
- wrong-frame negative control must exceed 5x the equivariant residual.

Tests: forward gauge covariance and backward through frame parameters. It is not
a 4K stress case unless a frame-rotation kernel is suspected to be slow.

### 6. Foam Birth/Death Source Budget

Analytic state:

```text
foam density rho evolves with known source S and sink lambda:
rho(t+dt) = rho(t) + dt*S(x,t) - dt*lambda*rho(t)
color is fixed; opacity follows rho
```

Why it matters: creation and deletion should be explainable by a source budget,
not hidden as arbitrary opacity/color changes everywhere.

Expected metrics:

- cumulative source/sink budget residual.
- alpha mass trajectory RMSE.
- source localization IoU.
- false birth/death mass outside analytic source/sink regions.
- heldout temporal consistency.

Acceptance:

- cumulative budget residual <= 3%.
- alpha mass RMSE <= 0.03 normalized to peak alpha mass.
- source localization IoU >= 0.9.
- outside-source false birth/death mass <= 5% of total created/deleted mass.
- heldout temporal SSIM >= 0.92.

Tests: forward, backward through density/source parameters, heldout
generalization.

### 7. Shallow-Water Pulse

Analytic state:

```text
small-amplitude shallow-water pulse on a plane:
h(x,t) = h0 + a exp(-||x - ct||^2 / sigma^2)
velocity aligned with propagation; optional reflective wall
```

Why it matters: it gives a fluid-like height/velocity coupling without requiring
a full solver. The model should learn a transported wave packet with coherent
speed and energy, not frame-local flicker.

Expected metrics:

- pulse center speed and direction error.
- energy drift for height+velocity proxy.
- reflected-pulse timing if a wall is enabled.
- heldout temporal SSIM and flow EPE.
- 4K forward/backward timing on a short sequence.

Acceptance:

- pulse speed error <= 5%; direction error <= 3 degrees.
- energy drift <= 5% over the sequence when no sink is present.
- reflected-pulse timing error <= 1 frame.
- temporal SSIM >= 0.9 and flow EPE <= 0.75 px.
- 4K timing does not regress beyond common thresholds.

Tests: forward, backward, 4K, heldout generalization.

### 8. Negative Controls

Run these against every scene before treating the metric as useful:

```text
static-support repaint:
    freeze positions/radii/frames/transport; allow per-frame color or opacity

wrong-flow sign:
    reverse velocity or phase direction

time-shuffled frames:
    train on the right images with incorrect temporal order

heldout-camera leak:
    deliberately include heldout view in the train set to verify the metric moves

frame-randomized gauge:
    rotate tangent frames without applying the correct covariant action

mass-leak source:
    allow unconstrained density birth everywhere
```

Acceptance:

- static-support repaint may reach train PSNR but must fail transport, phase,
  ownership, or heldout thresholds.
- wrong-flow and time-shuffled controls must fail flow/phase/timing thresholds.
- heldout-camera leak should improve heldout metrics; if it does not, the
  metric/camera setup is broken.
- frame-randomized gauge must fail the tangent-vector equivariance threshold
  while scalar render invariance remains a separate check.
- mass-leak source must fail the source localization and budget thresholds.

Tests: gate validity for forward, backward, and heldout metrics. A scene whose
negative controls pass should be removed or tightened before it informs the
PowerFoam acceptance path.

## Rollout Order

1. Implement advected checker and rotating tangent frame first. They are the
   fastest tests for repainting and gauge mistakes.
2. Add sinusoidal height wave and occlusion handoff next. These are the minimum
   geometric-motion tests.
3. Add area-preserving stretch and foam birth/death once replay metadata can
   expose mass and source terms cleanly.
4. Add shallow-water pulse last. It is the closest to the desired wave/fluid
   story but has the most ways to fail for non-renderer reasons.
5. Only after debug-size gates pass, run the 4K variants for sinusoidal height
   wave and shallow-water pulse.

Decision rule: if a synthetic scene fails heldout while source-view loss passes,
do not launch a paper-scale DeepView sweep. First classify the failure as
transport, visibility, gauge, source-budget, or 4K throughput.
