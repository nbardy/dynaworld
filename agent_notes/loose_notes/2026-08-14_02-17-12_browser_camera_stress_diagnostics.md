# Browser camera-stress diagnostics

Date: 2026-08-14

## Prompt and observed failure

The browser result fit a calibrated train camera reasonably well, but a small
orbit or dolly on another train camera and the heldout camera revealed broad,
translucent floaters. The requested constraint was explicit: diagnose and
regularize this without monocular or foundation-model depth priors.

## Audit

There was no automated small-zoom or small-pose gate. Exact calibrated-view
PSNR/SSIM and manual orbit inspection existed, but manual orbit mixed optical
resampling, physical camera motion, and unsupported extrapolation.

The shader audit found a plausible topology cause. Densification parent utility
combines non-cancelling screen gradient, a `4 * mean_alpha` term, and motion
gradient. The Pixel-GS near-camera guard scales only the screen-gradient term.
Large, close, alpha-heavy splats can therefore remain attractive parents. Once
the 4,096-to-8,192 growth schedule fills capacity, no pruning or relocation
repairs poor assignments.

## Implemented

- Added an arbitrary-camera CPU snapshot path with opacity-aware contributor
  depth mean/std, near-depth contribution, and giant-footprint contribution.
- Added deterministic optical perturbations: `1.05x` zoom and `+/-1.5%`
  principal-point shifts. Their targets are real captured frames transformed by
  exact crop/resampling, so PSNR is meaningful.
- Added deterministic physical perturbations: `+/-3%` dolly, `+/-1.5%` lateral
  translation, and `+/-2 degrees` orbit. Since no captured target exists, these
  report risk indicators only and never fake PSNR. The giant-footprint signal
  uses each splat's opacity-aware support bounding rectangle rather than an
  exact ellipse-area integral.
- Runs the bounded 48-pixel-high stencil in the existing validation worker on
  two train cameras plus heldout. The WebGPU optimizer does not wait for it.
- Added four UI readouts: optical PSNR, near alpha, giant-footprint alpha, and
  normalized contributor depth spread, each train / heldout.

## Verification

- Browser trainer test suite: 175/175 passed after the final integration
  assertion was added.
- Explicit calibrated-camera rendering matches the old indexed camera path.
- Optical and physical semantics, target masking, geometry telemetry, and UI
  integration have behavioral/source-contract coverage.
- Live Apple WebGPU smoke: the run advanced from step 0 to 4,352 in five
  seconds at 96x72 (the UI reported 472 completed steps/s), crossed the
  progressive 384x288 transition, and was paused at step 14,096. The initial
  stress fields populated, and the trained validation changed optical PSNR
  from `6.8/7.0 dB` to `22.1/24.1 dB` while near-alpha and giant-footprint
  contribution both fell. The 8K/384 full CPU validation reported 56.2 seconds;
  optimization continued while it ran because validation is a separate worker.

## Paper map and next experiment

The highest-value prior-free next lane is contribution-aware support plus
fixed-budget relocation: TrimGS-style transmittance/footprint contribution for
victim evidence, and 3DGS-MCMC-style relocation/moment reset at fixed capacity.
Require multiview support before spawning or relocating. Mip-Splatting remains
the right optical-scale filter but should not be expected to fix wrong-depth
geometry. 2DGS ray-depth distortion is a stronger subsequent geometry
regularizer. DropoutGS is a cheap ablation, not yet justified as a default with
17 train cameras.

Do not merge this prototype diagnostic into the Python trainer hierarchy. It
uses the existing browser bundle calibration and heldout split unchanged.
