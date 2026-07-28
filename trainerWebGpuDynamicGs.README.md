# Bounded Browser Dynamic 3DGS

`trainerWebGpuDynamicGs.js` is an isolated browser baseline module. It does not
participate in the Python trainer hierarchy and is not the SPA's existing
World-Tubes-like mode.

## Implemented contract

- calibrated browser-bundle cameras and view-major RGBA targets
- explicit `(frame, splat)` state; no shared temporal trajectory
- world-space 3D means, anisotropic log scales, quaternion rotations, RGB and opacity
- exact pinhole projection of the 3D covariance to a 2D conic
- ascending camera-depth ordering for every sampled ray
- front-to-back alpha compositing
- Adam updates of per-frame RGB and opacity logits
- analytic RGB/opacity VJP checked against central finite differences

Geometry, scale, and rotation are fixed in this first bounded version. That is
intentional: claiming trainable geometry while omitting covariance/projection
derivatives would be less honest than exposing the restricted parameter set.

## Matched-cost knobs and omissions

The benchmark reports `splatCount`, `samplesPerStep`, frame count, train-camera
count, state bytes, sampled pixels/s, and primitive evaluations/s. Match those
before comparing with another representation. The hard demo bounds are 32
splats and 256 sampled rays per step.

This is a correctness-first tiny baseline, not fast-mac parity. It has no tile
binning, global radix sort, transmittance early exit, SH appearance,
densification/pruning, SSIM loss, temporal regularization, geometry VJP, or
heldout-camera training. State scales linearly with frame count, as a
conventional per-frame baseline should.

Run the CPU contract and gradient checks with:

```bash
npm test --prefix web/dynaworld_browser_trainer
```

Serve the repo and open `benchmarkWebGpuDynamicGs.html` to run the WebGPU gate.
