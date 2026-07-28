# Affine STAR UVT Browser Prototype

`trainerWebGpuStar.js` is the standalone browser-prototype implementation of
the affine STAR UVT subset used by **World Tubes in Gauged Camera Space**. It
does not replace or modify the existing browser dynamic-splat trainer.

## Matched Contract

The packed state follows `star_uvt_v0`:

- `ma = (u, v, t)` in pixel-center coordinates and centered frame time
- symmetric `q_uvt = (q_uu, q_uv, q_ut, q_vv, q_vt, q_tt)`
- conditional depth `depth0 + depth_beta dot (a - ma)`
- direct opacity and RGB

For each sampled sensor-time point `a = (x + 0.5, y + 0.5, t)`, support is
selected by `opacity * exp(-0.5 * (a-ma)^T q_uvt (a-ma))`. Supported tubes are
stable-sorted by conditional depth, composited front-to-back with source-over
alpha, and terminated at the configured transmittance threshold. Support and
order are detached, matching the discrete boundary in the Metal reference.

`compileCameraSpaceWorldTubes()` consumes camera-space linear world tubes and a
fixed pinhole camera. It evaluates the pinhole projection at the reference time
and uses its first derivative to form an affine screen trace. General moving
cameras and large perspective-depth motion require the projective STAR path and
are intentionally outside this affine module.

The WGSL adjoint reverses the same source-over sequence and accumulates gradients
from every sampled pixel and time into the single time-shared tube state. The
GPU optimizer updates `ma`, all six `q_uvt` values, RGB, and opacity. Conditional
depth participates in ordering but is not optimized because its derivative is
zero inside a fixed order stratum. Training submission itself performs no map,
readback, or queue wait; `readParams()`, `readGradients()`, and
`gradientCheck()` are explicit diagnostic synchronization points.

## Deliberate Omissions

- UVT tile binning, sparse interval atlases, and tile-capacity overflow handling
- projective/rational traces, camera-family gauges, and moving-camera compilation
- order-event certificates, interval splitting, and certified fallback paths
- finite exposure and rolling shutter
- gradients through discrete support, discrete order, or conditional depth
- Metal fixed-point/atomic reductions and production optimizer parity
- densification, pruning, perceptual losses, and dataset/UI integration

These omissions keep this a tiny executable affine subset, not a browser claim
of native Metal or paper-trainer parity.

## Matched-Cost Knobs

Use the same `tubeCount`, `sampleCount`, `frameCount`, optimizer-step count, and
`alphaThreshold` when comparing this module with another representation. Report
gradient/reduction work and any diagnostic readback separately. The included
benchmark uses two tubes, 144 samples over four times, and reports the synchronized
wall time only after a batch of nonblocking train submissions.

## Verification

CPU analytic gradients are checked against central finite differences for every
optimized parameter family:

```bash
node --test web/dynaworld_browser_trainer/tests/trainerWebGpuStar.test.mjs
```

The browser benchmark compiles the WGSL, compares the GPU shared adjoint against
the same finite-difference reference, then checks that a tiny optimization run
reduces loss:

```text
http://127.0.0.1:8080/web/dynaworld_browser_trainer/benchmarkWebGpuStar.html
```
