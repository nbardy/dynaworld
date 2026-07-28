# Browser Plateau Convergence Research

## Scope And Claim Boundary

This is a read-only audit of the current browser trainer and relevant primary
3DGS/dynamic-GS sources. No runtime code was changed. The browser remains a
demo/prototype using the canonical exported data contract, not a new paper
trainer lane. Paper-backed practices below are separated from hypotheses about
this simplified WGSL implementation.

## Current Browser Contract: Observed Facts

Files inspected:

- `web/dynaworld_browser_trainer/trainerWebGpu3d.js`
- `web/dynaworld_browser_trainer/app.js`
- `web/dynaworld_browser_trainer/dataset.js`
- `web/dynaworld_browser_trainer/coffee_martini_multicam.json`
- `web/dynaworld_browser_trainer/README.md`
- `src/train/export_dynaworld_browser_bundle.py`
- `research_notes/data_contract.md`
- `research_notes/renderer_lane_taxonomy.md`
- `BASELINES.md` and the matched Coffee Martini report

Current data/model:

- Coffee Martini exports `cam04` and `cam09` for training and `cam06` strictly
  for validation, at 8 synchronized times and `96x72` pixels.
- The local Coffee Martini directory contains 18 camera videos (`cam00` through
  `cam20` with several absent indices), but the browser bundle intentionally
  exports only the canonical 2-train/1-heldout split.
- Initialization is 768 XYZRGB rows exported from the existing Ex4DGS SfM
  `input.ply`, normalized by median anchor-camera depth. It is not random and
  does not use heldout pixels, but it is also not a browser-run COLMAP solve.
- The slider cannot exceed 768 and `maintainDensity()` returns zero. Therefore
  there is no birth, clone, split, prune, relocation, or opacity-reset stage.
- Each primitive has a 3D center, one scalar radius, linear velocity, one
  harmonic displacement vector, one fixed RGB color, one opacity logit, one
  temporal center, and a `harmonicStatic.w` mixture. There is no anisotropic
  covariance, rotation, SH/view-dependent appearance, or learned deformation
  field.
- Both UI modes use the same rasterizer and backward. `World Tubes-style` adds
  one sinusoidal displacement to linear motion; `Dynamic splats-style` omits
  it. Neither is parity with native STAR UVT/World Tubes nor canonical Dynamic
  3D Gaussians/4DGS.
- A step samples 96 individual rays by default. Each ray independently samples
  a train camera, time, and pixel, with effective defaults 90% motion, 8%
  static, and 2% uniform. Thus one step already normally contains rays from
  both train cameras; an explicit `K=2` camera loop would mainly guarantee
  balance, not introduce new supervision.
- The loss is sampled RGB MSE plus browser-specific alpha support/cleanup
  terms. The displayed SSIM is a sparse global-luma proxy used for validation,
  not the standard local-window SSIM and not a training loss.
- Adam uses constant per-group rates at UI scale 1.25: position/radius
  `4.375e-4`, color `1.875e-3`, opacity `1.0e-3`, and motion/time `2.5e-4`.
  There is no warmup or decay. Adam is dense over all 768 splats.
- Training and rendering composite in fixed parameter-index order. There is no
  per-camera or per-time depth sort. The objective and preview agree with each
  other, but neither is a visibility-correct 3D Gaussian compositor.

## Critical Code-Level Finding: Temporal Support Is Nearly Disabled

`makeInitialSplats()` sets `harmonicStatic.w = 0.92` for every splat. The update
shader explicitly preserves this value, so it never learns. The temporal gate
is:

```text
dynamicGate = floor + (1-floor) exp(-0.5 dt^2 / sigma^2)
gate = mix(dynamicGate, 1, staticMix)
```

At the default `sigma=0.30`, `floor=0.09`. At the maximum initial distance from
the common temporal center (`dt=0.5`), `dynamicGate ~= 0.317`, but:

```text
gate = 0.08 * 0.317 + 0.92 * 1 = 0.945
```

Therefore every primitive remains at least about 94.5% temporally active across
the whole eight-frame clip. Narrowing sigma from 0.30 to 0.26 changes that edge
gate only from about 0.945 to about 0.938. This makes the current Temporal
Support control and auto-narrow schedule almost inert. It also forces dynamic
content to be explained primarily by linear/sinusoidal center motion while
preventing short-lived support from specializing.

This is a code-derived fact, not a paper claim. It is the cheapest and highest
priority plateau hypothesis to falsify.

## Ranked Causes

### 1. Fixed, Nearly Global Temporal Occupancy

Confidence: high.

Evidence: the exact calculation above. The current motion is not merely broad;
all splats are hard-wired to be 92% static. This can explain quick early color
and coarse-motion fitting followed by a plateau on details that need different
support at different times.

Falsification: matched runs with static mixture `{0.0, 0.5, 0.92}`, keeping
sigma, seed, samples, cameras, and wall-clock budget fixed. Log train/heldout
PSNR, true local-window SSIM, motion-region PSNR, alpha coverage, and active
count. Do not combine this with densification in the first test.

### 2. Visibility-Incorrect Fixed Compositing Order

Confidence: high.

Evidence: current WGSL blends by array index for every camera and time. The
original 3DGS method explicitly uses visibility-aware tile sorting and alpha
blending. A single global order cannot generally be back-to-front for cam04,
cam09, cam06, and moving centers simultaneously.

Falsification: add a correctness-only depth-sorted reference for a small ray
set, then compare its predictions and gradients to fixed order. Measure pixel
error specifically where two or more splats have alpha above 0.05. If errors
cluster there, optimizer tuning cannot repair the renderer mismatch.

### 3. Fixed Capacity And No Density Control

Confidence: high that it limits detail; medium that it is the first plateau
cause.

Evidence: the model starts and ends at 768 isotropic splats. The original 3DGS
paper reports final captured-scene representations on the order of 1-5 million
Gaussians and treats interleaved adaptive density control as a core component.
Dynamic 3D Gaussians reports roughly 200-300k primitives per scene. These
counts are not directly transferable to a `96x72x8` demo, but 768 is plainly a
tiny budget and cannot relocate dead capacity because `maintainDensity()` is a
no-op.

Important nuance: repo evidence does not justify “more always wins.” The
browser's earlier 384/512/768 tests favored 768 under the current sampler, but
the PowerFoam baseline contains cases where retaining all available cells was
worse than a selected subset. Capacity should be tested together with useful
support, not by blindly duplicating points.

Falsification: fixed-capacity runs at 384/768/1536/3072, and separately a
fixed-budget relocation run at 768. Match both optimizer steps and wall time.
If relocation beats 1536, placement is the bottleneck; if quality scales with
count, capacity is.

### 4. Isotropic Footprints And Constant Appearance

Confidence: medium-high.

Evidence: one radius cannot represent oriented thin surfaces or adapt its
projected ellipse by view. One RGB triplet cannot model view-dependent effects
or camera exposure. Anisotropic covariance and view-dependent SH are core 3DGS
parameters, not optional paper embellishments. Coffee Martini is a real
multicamera scene where these omissions matter more than in a synthetic toy.

Falsification: first add screen-space 2D anisotropy (two scales plus angle) to
isolate footprint capacity; only then consider full projected 3D covariance.
Separately test degree-0 RGB against a tiny view-direction basis or per-camera
affine exposure. Avoid changing both geometry and appearance in one arm.

### 5. Too Few Camera Views For Heldout Geometry

Confidence: medium-high for novel-view quality, low for source-view plateau.

Evidence: only two train cameras supervise a real dynamic scene, while 18
camera videos are local. The heldout camera lies between the two selected train
views, which is a sensible smoke split, but two-view ambiguity remains severe.
The repo already has a clean train4 Coffee Martini route; its feature
triangulation becomes denser than train2, although that particular weak
initializer still does not solve heldout quality.

Falsification: preserve cam06 as heldout and export nested train sets of 2, 4,
8, and 17 cameras using the canonical loader/export adapter. Keep total rays
per step fixed first. Then compare a balanced `K`-camera sampler at the same ray
budget. More cameras should improve heldout geometry; merely grouping the same
two cameras per step should mostly reduce gradient variance.

### 6. Objective Mismatch: Pixel MSE Without Patch Structure

Confidence: medium.

Evidence: canonical 3DGS and official Dynamic 3D Gaussians train with
`0.8*L1 + 0.2*(1-SSIM)`. The current 96 independent rays cannot compute local
11x11-window SSIM. MSE encourages conditional means and can flatten fine
structure, especially with an underexpressive motion model.

Falsification: sample small contiguous patches and add true local DSSIM with
weight 0.2, comparing against L1-only and current MSE at matched ray/pixel and
wall-clock budgets. The current global-luma SSIM proxy must not be used as the
training implementation or as evidence of parity.

### 7. Constant And Poorly Calibrated Parameter-Group LRs

Confidence: medium.

Evidence: browser rates were chosen through short interactive probes, not a
matched schedule ablation. Canonical 3DGS uses different rates by parameter and
an exponential position decay (`1.6e-4 -> 1.6e-6` over 30k steps), while the
official Dynamic 3D Gaussians code uses scene-scaled position LR, color
`2.5e-3`, opacity `5e-2`, scale/rotation `1e-3`, and Adam epsilon `1e-15`.
Browser units, minibatch normalization, and parameterization differ, so copying
those numbers literally would be invalid. Still, browser opacity LR is about
25-50x lower numerically and all rates remain constant for 60k+ steps.

Falsification: log per-group update/parameter norms, saturation fractions, and
Adam second moments. Sweep `{0.5,1,2,4}x` by group, not one global slider. Then
test a short position/motion warm phase followed by exponential decay. A rate
is too low if normalized updates are negligible while residual gradients stay
high; too high if parameters hit clamps or heldout quality oscillates.

### 8. Motion Basis Is Too Small

Confidence: medium.

Evidence: one linear trajectory plus at most one global-frequency sinusoid per
splat is not canonical Dynamic 3D Gaussians, 4D-GS, or native World Tubes.
Official dynamic methods use per-time persistent trajectories plus rigidity,
learned deformation fields, or temporal opacity with richer parametric
motion/rotation. Coffee Martini contains nonrigid and partially transient
motion.

Falsification: only after sorting and temporal occupancy are corrected, compare
linear, linear+harmonic, piecewise control points, and the native browserable
World Tubes approximation at equal parameter count. Today the two menu labels
are not a meaningful representation ablation because they differ by only one
sine term.

## Paper-Backed Practices

### Original 3D Gaussian Splatting

The paper and official implementation support:

- initialization from SfM sparse points for captured scenes;
- anisotropic covariance and visibility-aware sorted alpha compositing;
- `0.8*L1 + 0.2*(1-SSIM)` training loss;
- densification from step 500 through 15k, every 100 iterations, using average
  projected-position gradient threshold `2e-4`;
- clone small high-gradient Gaussians and split large high-gradient Gaussians;
- prune opacity below 0.005 and later oversized Gaussians;
- reset opacity every 3000 iterations;
- parameter-specific rates and exponential position-LR decay.

These are strong priors for experiments, not constants that can be transplanted
unchanged into normalized WGSL units.

### Dynamic 3D Gaussians

The official code optimizes an initial timestep for 10k iterations, performs
3DGS-style densification during that initial stage, then trains each later
timestep for 2k iterations. It initializes later centers/rotations by linear
extrapolation, freezes opacity and scale after the first timestep, and adds
local rigidity, rotation consistency, isometry, floor, static-background, and
soft color-consistency losses. Its image loss is also `0.8*L1 + 0.2*DSSIM`.

This supports testing temporal regularization, but not blindly adding
acceleration loss to the current global parametric trajectory. The persistent
per-timestep representation and neighborhood graph are materially different.

### 4D-GS And Spacetime Gaussians

The CVPR 2024 4D-GS implementation uses a 3k coarse/static phase, 14k total
iterations, batch size 4, density control through 10k, and spatial/temporal
plane regularizers. It predicts Gaussian deformation from a 4D encoder rather
than using one linear velocity.

Spacetime Gaussian Feature Splatting adds temporal opacity and parametric
motion/rotation, uses sparse per-frame SfM points for Neural 3D Video, and
explicitly samples new Gaussians from training error plus coarse depth. Its
Coffee Martini configs enable densification and use a dedicated scene recipe.

Together these support temporal occupancy, staged optimization, and
error-guided spawning as serious experiments. They do not prove that the
browser's current support guard or 0.30 sigma is optimal.

### Budgeted/Fast Density Control

Taming 3DGS is directly relevant to a browser budget: it uses contribution
priors and constructive, score-guided growth toward an exact primitive budget,
reporting 4-5x model-size and training-time reductions while preserving
competitive quality. FastGS similarly emphasizes multi-view-consistent
densification and targeted pruning. This is a better conceptual fit than
unbounded original-3DGS growth.

## Recommended Experiment Order

Each row should report train and heldout MSE/L1/PSNR/true local SSIM, motion
region metrics, active count, alpha/radius quantiles, update norms, steps/s,
wall time to quality thresholds, and GPU memory. Keep cam06 validation-only.

1. **Temporal occupancy A/B:** static mixture 0.92 versus 0.5 versus 0.0. This
   is the cheapest high-information test and should precede any schedule work.
2. **Depth-order correctness gate:** compare fixed order with a small sorted
   reference. Do not tune around an invalid compositor.
3. **Capacity versus placement:** 384/768/1536/3072 fixed counts plus
   fixed-budget 768 relocation. Match wall time as well as steps.
4. **Budgeted density control:** every 256-512 steps, prune low-contribution or
   low-opacity splats and respawn/split them at high-residual, high-gradient
   train rays; reset optimizer moments for recycled slots. Keep a strict max
   count for WebGPU.
5. **Camera breadth:** canonical 2/4/8/17-train exports with cam06 held out.
   First keep total rays fixed; then test balanced `K` cameras per update.
6. **Patch loss:** current MSE versus L1 versus `0.8 L1 + 0.2 DSSIM`, using
   contiguous patches and standard local-window SSIM.
7. **Anisotropy:** scalar radius versus 2D ellipse, then projected 3D
   covariance. Record whether thin/high-residual regions improve.
8. **Per-group LR/schedule:** instrument before sweeping; test group-specific
   multipliers and position/motion decay rather than only the global slider.
9. **Representation ablation:** after shared renderer correctness, compare a
   genuinely distinct World Tubes path and dynamic-GS path. The current
   sine-on/sine-off menu is insufficient for a paper-style ablation.

## What Not To Do First

- Do not simply increase the global LR. It cannot create temporal locality,
  correct visibility, anisotropy, or missing primitives.
- Do not add DSSIM to isolated random rays; SSIM is a local-window statistic.
- Do not import original 3DGS thresholds verbatim without normalizing for scene
  scale, image resolution, ray count, and gradient convention.
- Do not use all 18 cameras without preserving a heldout camera and the
  canonical split metadata.
- Do not call the current mode toggle a World Tubes versus Dynamic 3DGS
  ablation. Both share the same simplified representation and backward.
- Do not interpret the sparse 12x12 global-luma SSIM proxy as paper SSIM.

## Current Best Working Theory

The plateau is most likely a compound representation/rendering ceiling:
hard-wired near-static temporal support and fixed visibility order prevent the
optimizer from expressing the correct dynamic scene; 768 isotropic constant-
color splats then cap remaining spatial/view detail. Two-camera supervision
amplifies the heldout ambiguity. Constant LR and MSE-only training probably
affect convergence speed and sharpness, but they are secondary until the first
three constraints are falsified.

Confidence: high that 768 alone is not the whole explanation; medium-high in
the ordering above. A clean temporal-mixture A/B and sorted-reference residual
test could change this ranking quickly.

## Primary Sources

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field
  Rendering*: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Official 3DGS implementation and optimization defaults:
  https://github.com/graphdeco-inria/gaussian-splatting
- Luiten et al., *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View
  Synthesis*: https://arxiv.org/abs/2308.09713
- Official Dynamic 3D Gaussians implementation:
  https://github.com/JonathonLuiten/Dynamic3DGaussians
- Wu et al., *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*:
  https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html
- Official 4D-GS implementation: https://github.com/hustvl/4DGaussians
- Li et al., *Spacetime Gaussian Feature Splatting for Real-Time Dynamic View
  Synthesis*: https://arxiv.org/abs/2312.16812
- Official Spacetime Gaussians implementation:
  https://github.com/oppo-us-research/SpacetimeGaussians
- Mallick et al., *Taming 3DGS: High-Quality Radiance Fields with Limited
  Resources*: https://arxiv.org/abs/2406.15643
- Official Taming 3DGS implementation:
  https://github.com/humansensinglab/taming-3dgs
- Ren et al., *FastGS: Training 3D Gaussian Splatting in 100 Seconds*:
  https://openaccess.thecvf.com/content/CVPR2026/papers/Ren_FastGS_Training_3D_Gaussian_Splatting_in_100_Seconds_CVPR_2026_paper.pdf
