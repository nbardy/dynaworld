# Browser Compact Targets, FP16 VJP, And 30K / 384x288 Gate

Date: 2026-07-31

## Scope

This session closed the highest-value browser systems work from the 30K / 384
execution board without creating a second calibration, split, or model
contract. The browser remains a trajectory-gated dynamic 3DGS prototype. It is
not World Tubes, native spacetime Gaussians, or a replacement for the Python
paper trainers.

The work covered four linked questions:

1. Can the calibrated multicamera target bank stop occupying FP32 memory in
   every JavaScript context?
2. Can dormant reserve splats avoid dense clear and Adam traffic?
3. Can the cold projection VJP packet use packed FP16 storage without changing
   FP32 arithmetic or silently saturating?
4. Can the tiled full-frame path actually finish a complete 30K-splat,
   384x288 camera-time cycle inside WebGPU limits?

## Precision Decision: 8-Bit Targets, Floating-Point Training

The source PNGs are 8-bit, so preserving target RGB as exact bytes is lossless
with respect to the checked-in data. That does **not** imply an 8-bit training
raster or integer loss.

- Host target RGB stays byte-packed.
- RGB is decoded as exact `byte / 255` when sampled or paged onto the GPU.
- The alpha byte stores normalized motion-loss weight as `byte / 127`, giving
  the required `[0, 2]` range.
- The selected tiled target page is decoded to RGBA32F before forward/loss.
- Predicted pixels, L1, 11x11 Gaussian SSIM, adjoints, Adam state, and updates
  stay floating point.
- The display canvas may quantize for presentation; it is not the training
  surface.

Quantizing predictions before loss would create a piecewise-constant objective
whose ordinary derivative is zero almost everywhere. A straight-through
estimator could be tested as a deliberately biased robustness ablation, but it
is not a sound default. FP16 storage for selected intermediates remains a
useful lane because values are decoded before arithmetic; integer raster/loss
is a different and much riskier intervention.

## Compact Frame-Bank Implementation

The dataset now exposes explicit frame-bank formats:

- `rgba8unorm-rgb+weight-u8x127/v1`
- `rgba32float/v1` as a reference/debug path

Calibrated camera atlases decode sequentially into the final bank instead of
building per-camera FP32 banks and concatenating them. When the isolated local
server exposes `SharedArrayBuffer`, allocation happens directly in shared
storage, and the main, training, and validation contexts retain views into the
same immutable backing. FP32 backgrounds stay separate and shared.

At 384x288 with 18 cameras and 16 frames:

```text
compact frames             121.500 MiB
FP32 backgrounds            30.375 MiB
compact scaled bank total  151.875 MiB
former FP32 total          516.375 MiB
savings                    364.500 MiB
```

The full benchmark resource estimate includes the checked-in 96x72 source,
the scaled bank, and one sequential decode transient: 161.789 MiB.

### Failure Found During Parity

Independently rounding every normalized motion weight changed the per-frame
mean away from exactly one. The tiled objective divides by pixel count under
that invariant, so this produced a real objective mismatch even though each
individual byte error was small.

The final encoder starts from nearest-byte rounding and applies the minimum
additional-error one-byte corrections needed to make the exact encoded sum
`127 * pixelCount`. This restored the objective contract. Compact live parity
then measured zero RGB decode error, at most one FP32 ULP of weight decode
error near 2.0, objective error `3.86e-8`, and all 9 gradient families passing.

## Sparse Active-Prefix Optimizer

The fixed capacity remains contiguous, but clear and Adam dispatch only over
the exact active prefix. A partial final split is represented exactly, and a
newly activated tail starts with clean gradients, moments, and density state.
This removes provably useless traffic while preserving dense Adam semantics for
every active splat. It is not visibility-sparse Adam; once all 30K slots are
active, update traffic is dense by design.

## Packed-FP16 Projection VJP

The split projection path keeps the 32-byte raster-hot record and offers two
cold VJP packets:

- FP32 reference: 80 bytes per capacity splat
- packed FP16 storage: 48 bytes per capacity splat

Arithmetic and projected-gradient atomics remain FP32. The packed path uses
`pack2x16float` / `unpack2x16float`, so it does not depend on native FP16
arithmetic. World-space variances are normalized by `geometryScale^2` before
packing and restored before the VJP. Current and cumulative NaN/half-range
saturation counters are part of benchmark validity.

Correctness passed with bit-identical forward pixels/objective, relative
gradient L2 `2.87e-4`, cosine `0.99999996`, zero sign flips, and zero
saturation on the live anisotropic fixture. Packed target plus packed VJP also
passed all 9 gradient-family gates.

Timing did not justify a speed claim:

```text
8K control-first:    packed 1.0089x throughput, 0.9969x GPU-time speedup
8K candidate-first:  packed 0.9959x throughput, 0.9997x GPU-time speedup
30K control-first:   packed 1.0209x throughput, 1.0706x GPU-time speedup
30K candidate-first: packed 0.9873x throughput, 0.8870x GPU-time speedup
```

All four host environments failed promotion, and the 30K reversed run began at
97% Apple GPU utilization. The result is therefore a proven memory reduction
of 256 KiB at 8K and 937.5 KiB at 30K, with throughput currently neutral and
unproven. Do not make packed VJP the performance default until quiet-host,
reversed-order pairs reproduce.

## 30K / 384x288 Full-Cycle Result

Artifact:
`web/dynaworld_browser_trainer/benchmark_results/runs/2026-07-31/19-44-18_packed_30k_384x288_full_cycle_diagnostic.json`

The full-cycle diagnostic completed the mechanical gate successfully:

```text
raster                         384x288 (110,592 pixels)
splats                         30,000 active / 30,000 capacity
train camera-time pairs        272 / 272 complete
steps/s                        239.302 diagnostic
median timestamped GPU span    4.369058 ms
finite final loss              0.381745
tile overflow                  0 current / 0 cumulative
FP16 saturation                0 current / 0 cumulative
visible splats                 22,220
maximum tile occupancy ever    749 / 4,096 capacity
WebGPU buffers                 205.09 MiB
largest storage binding        108.00 MiB / 128 MiB portable floor
round throughput CV            0.0426
```

The profile shows the remaining GPU center of gravity clearly: backward was
2.350 ms median, versus 0.411 ms SSIM statistics, 0.394 ms SSIM gradient,
0.329 ms forward, 0.251 ms sort, 0.179 ms projection, and 0.184 ms update.
The compact target decode was only 0.0069 ms median.

This is a valid mechanical systems result but not a promotable performance
claim. Swap occupied about half of physical memory, and the strict host gate
failed. Its four rounds also declined monotonically from 253.1 to 225.3
steps/s, despite remaining inside the diagnostic 10% CV threshold. Preview,
validation, and topology maintenance were excluded, and there was no
reversed-start repeat. More importantly, the benchmark uses nearest-neighbor
expansion of the checked-in 96x72 target tensor. It exercises real 384 raster
dimensions and memory traffic, but it does not contain true 384-frequency
training detail.

## SPA Integration Smoke

The real worker-backed SPA loaded the compact shared dataset and initialized
the fast tiled full-frame backend with 4,096 active / 8,192 capacity splats.
It ran past 5,000 steps at roughly 704 steps/s on the checked-in 96x72 bundle,
with finite loss, `0 / 0` tile overflow, nonblank three-camera live previews,
and no console errors. The browser correctly reports 6,912 pixels per tiled
step and disables sampled-ray-only controls.

The distinction between the backend controls is intentional:

- **Fast tiled full-frame:** tile-binned, camera-depth-sorted rasterization of
  every pixel, with shared projected gradients and `0.8 L1 + 0.2 (1-SSIM)`.
- **Direct tiled reference:** the same full-frame objective and raster contract
  with the slower direct 3D VJP retained as a correctness control.
- **Sampled rays control:** the older 96-ray all-splats path. It is useful for
  regression comparison, but its steps/s are not comparable with a full image.

At 96x72 the full tiled step covers 6,912 pixels versus 96 sampled rays. At
384x288 it covers 110,592 pixels versus the same 96-ray control.

## Verification

- browser unit/contract suite: 139 passed, 0 failed
- live compact RGBA8 + FP32 VJP parity: all 9 gradient families pass
- live compact RGBA8 + packed-FP16 VJP parity: all 9 families pass
- backend smoke: fast tiled, direct tiled, and sampled control all finite
- real SPA worker smoke: finite loss, nonblank renders, no console errors
- scoped `git diff --check`: clean
- local server: HTTP 200 with COOP/COEP isolation headers

## What Remains

1. Generate and reload the canonical real-detail 384x288 browser bundle on a
   quiet host through `export_dynaworld_browser_bundle.py`.
2. Repeat FP32-versus-packed VJP pairs under the strict host gate with both
   start orders; keep FP32 as the default until the speed result reproduces.
3. Run real-detail 384 quality comparisons at fixed wall time, including
   low-pass and high-frequency residuals, at least two seeds, and another
   scene before claiming convergence improvement.
4. Profile the 2.35 ms backward phase further. Its share is now much larger
   than target decode, Adam, projection, or sorting, so it is the next credible
   kernel optimization lane.
5. Keep sampled rays only as a labeled control; do not use its nominal
   steps/second to advertise full-frame training speed.
