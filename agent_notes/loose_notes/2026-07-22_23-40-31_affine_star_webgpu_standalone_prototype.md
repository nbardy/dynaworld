# Standalone affine STAR UVT WebGPU prototype

## Scope

Added a standalone affine STAR UVT / World Tubes browser module without wiring
it into the existing SPA or Python trainer hierarchy. The contract was reduced
from `third_party/fast-mac-gsplat/variants/star_uvt_v0`: pixel-center UVT
coordinates, centered frame time, symmetric `q_uvt`, conditional depth,
alpha-threshold support, stable detached depth order, source-over compositing,
and reverse compositing adjoints.

## Files

- `web/dynaworld_browser_trainer/trainerWebGpuStar.js`
- `web/dynaworld_browser_trainer/trainerWebGpuStar.README.md`
- `web/dynaworld_browser_trainer/tests/trainerWebGpuStar.test.mjs`
- `web/dynaworld_browser_trainer/benchmarkWebGpuStar.html`
- `web/dynaworld_browser_trainer/benchmarkWebGpuStar.js`

## Verification

The Node reference tests passed four behavioral gates: camera-space affine
compilation and positive-definite UVT precision, depth-order compositing,
analytic adjoint versus central finite differences for every optimized family,
and explicit taxonomy/omission metadata.

The in-app WebGPU benchmark used 2 tubes, 144 sampled rays, 4 centered times,
and a 16x16 fixture. The final run reported:

- GPU shared adjoint max absolute error: `4.2159808799624443e-7`
- GPU shared adjoint max relative error: `4.428477008665581e-4`
- loss: `9.169110592786613e-4 -> 4.1137367934228775e-4`
- final/initial loss ratio: `0.4486516714782757`
- 200-step synchronized batch: `16.68 ms`, about `11,990 steps/s`

The throughput number is only a tiny-fixture implementation smoke. It is not a
Metal comparison and should not be promoted as one.

## Boundaries

The module omits UVT tiling/interval atlases, projective traces, camera-family
gauges, event certificates/fallbacks, finite exposure, rolling shutter,
gradients through support/order/depth, Metal reduction parity, and production
optimizer behavior. `trainStep()` has no readback or queue wait; diagnostics
are explicit synchronization methods.
