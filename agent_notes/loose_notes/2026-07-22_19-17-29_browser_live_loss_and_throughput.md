# Browser Live Loss And Throughput

## Scope

Kept the WebGPU trainer inside the browser-demo lane. No browser abstractions
were added to the Python trainer hierarchy and the canonical multicamera bundle
contract remains unchanged.

## Findings

- The WebGPU result was live, but the animation loop rasterized the full canvas
  after every optimizer step.
- The calibrated multicamera bundle selects the heavier 3D trainer and defaults
  to 768 splats, so it should not be compared directly with older 512-splat
  source-view traces.
- The main avoidable regression was `DynamicSplatWebGpu3dTrainer.readLoss`:
  every 16 steps it copied all parameters and ran CPU train plus heldout grid
  evaluation. This mislabeled deterministic grid evaluation as sample loss and
  stalled optimizer throughput.
- The monolithic 3D shader was at Apple WebGPU's storage-binding limit. Splitting
  gradient generation from reduction made a dedicated sample-loss buffer
  admissible and removed the need to overload per-splat statistics.

## Changes

- Each sampled-ray workgroup writes one stochastic MSE value. Loss readback
  copies and averages only the active sample range, without reading parameters
  or invoking CPU validation.
- Full train/heldout validation now uses a 12x12 sparse grid on a 256-step
  cadence and can be disabled for train-loop isolation.
- A separate train pump submits eight-step multicamera batches while target and
  WebGPU preview paints are capped near 30 FPS. Training is no longer capped by
  `requestAnimationFrame`.
- Added a log-scale history chart for sample-loss EMA and deterministic
  train-grid MSE.
- Raised the interactive default learning-rate scale from `1.0x` to `1.25x`.
  This is deliberately marked as unablated and remains live-adjustable.
- Normalized points, camera translations, radii, and clamps into a median-depth
  gauge. Initial projections are invariant to `8.7e-8`, while the same Adam
  world-coordinate step produces `17.43x` larger projected motion on this
  Coffee Martini bundle.
- The first shared-tape rewrite removed `samples * splats * splats` arithmetic
  but serialized all samples through one workgroup. The user-observed 7.3
  steps/s exposed the occupancy failure.
- Replaced that dispatch with one workgroup per sampled ray plus a second
  per-splat reduction/Adam pass. This retains the exact shared under/suffix
  tape, remains `O(samples * splats)`, and avoids non-portable float atomics.
- Added train and heldout loss, PSNR, and SSIM-proxy histories. Quality charts
  update only on the existing sparse validation cadence.
- Added `Live Preview` and `Full Metrics` toggles. Turning both off leaves the
  batched train pump plus a four-byte sample-loss synchronization every 256
  steps as the in-app isolation path.

## Verification

- JavaScript syntax checks pass for `app.js`, `dataset.js`, and both WebGPU
  trainers.
- `git diff --check` passes.
- Naga 30 validates the sample-gradient, reduction/update, and render WGSL
  modules.
- The sample-gradient shader uses 15,408 bytes of workgroup storage, below the
  WebGPU guaranteed 16 KiB minimum, with 256 invocations and six storage
  bindings. The reduction/update pass uses seven storage bindings.
- A direct bundle check confirms geometry scale `0.0573641`, initial projection
  invariance below `1e-7`, and a `17.43x` projected optimizer-step multiplier.
- Repo-root HTTP serving returns `200` for the trainer route and updated assets.
- A real Apple WebGPU browser smoke loaded without warnings/errors, exercised
  all three charts, and observed about 584-824 steps/s at 96 samples and 768
  splats. After 19,328 rapidly submitted/synchronized steps it reported train
  `18.7 dB / 0.870` and heldout `11.7 dB / 0.279`. These are smoke observations,
  not canonical baseline rows.

## Metal comparison follow-up

- There was no saved native row matching the browser contract of 96 sampled
  rays and 768 splats. Existing fast-mac trainer rows are full-raster and often
  use 8,192 or 65,536 splats, so raw steps/s is not comparable.
- A fresh diagnostic using `fast_mac_project3d_benchmark.py` at 96x96's square
  case interface with 768 splats measured v9 Metal projection plus full-raster
  forward/backward at `7.7983 ms` for 9,216 pixels. The direct rectangular v9
  projected-raster probe at 96x72 measured `7.1824 ms` mean / `6.9336 ms`
  median for 6,912 pixels; it omits v9's 3D projection VJP.
- The browser's observed 584-824 steps/s is about `1.21-1.71 ms` for 96 sampled
  rays. Normalized by trained pixels, that is roughly `12.6-17.8 us/ray`, while
  the rectangular native raster probe is about `1.04 us/pixel`: Metal remains
  roughly `12-17x` more efficient per pixel. This is not an API-only WebGPU tax.
  The browser still projects every splat for every sampled ray; native Metal
  bins contributors by tile and traverses only active support.

## WebGPU performance worker follow-up

- Added a synchronized browser-local harness at
  `web/dynaworld_browser_trainer/benchmarkWebGpu3d.html`. It reports the
  96-sample/768-splat hot loop separately from the usable default that includes
  loss synchronization, preview opportunities, and 12x12 train/heldout metrics.
- The two-pass sampled-ray path materializes `96 * 768 * 64 = 4,718,592` bytes
  of per-sample gradients each step. That traffic and all-splats-per-ray
  projection remain the structural gap versus Metal active-support traversal.
- Two low-risk GPU changes were measured and rejected rather than promoted:
  reusing taped alpha/premultiplied color measured `1066.7` steps/s versus
  `1098.2` for the prior shader in the same harness, and putting gradient plus
  update dispatches in one compute pass measured `1035.6` steps/s. Both were
  reverted. There is no claimed kernel speedup from this pass.
- Isolation measurements varied from `742.5` to `1098.2` steps/s and usable
  default measurements varied from `409.9` to `720.9` steps/s while another
  trainer tab was active. Keep the saved `584-824` UI observation as the exact
  browser comparison and treat these harness numbers as contention-sensitive.
- The step-256 freeze came from the CPU metric evaluator: 6912 grid rays times
  768 splats, with temporary center, camera-point, color, target, and error
  arrays in the inner loop. The evaluator now uses scalar arithmetic with no
  per-splat arrays while preserving the same ray set and formulas.
- Cooperative row yields were also measured and rejected after the allocation
  fix: blocking validation took `145.8ms` wall with a `33.4ms` maximum observed
  animation-frame gap, while yielding took `146.5ms` and produced a `67.7ms`
  gap on the same run. Metrics agreed (`14.808304` train PSNR and `10.6539`
  heldout PSNR). The quality sweep remains enabled; results are not hidden.
