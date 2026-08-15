# Browser 8K Splat Capacity

## Goal

Double the tiled browser trainer from 4,096 to 8,192 global splats without
reintroducing the sampled-ray storage explosion or exceeding portable WebGPU
workgroup storage.

## What changed

- Raised the shared browser render/model limit to 8,192.
- Kept the sampled-ray control capped at 2,048 because its per-ray depth-order
  and gradient tape remain intentionally independent from the tiled backend.
- Stopped the tiled backend from preflighting the sampled gradient slab it does
  not allocate.
- Expanded tile sort keys from 12 to 13 splat-ID bits, preserving IDs through
  8,191 while reducing positive-depth precision from 20 to 19 bits.
- Separated global capacity from tile-local sort capacity. The global model can
  hold 8,192 splats; each tile still has a 4,096-contributor bitonic sort bound
  so the workgroup array remains below the common 32 KiB limit.
- Preserved `Tile Overflow` as the fail-observable guard for scenes that exceed
  that local bound.
- Left the visible default at all 4,096 checked-in SfM seeds. Another 4,096
  slots start hidden and are filled by opacity-preserving splits from step 600
  through step 26,100, avoiding exact visible duplicates at initialization.

## Why not an 8K tile-local sort

An `array<u32, 8192>` consumes 32 KiB before shared counters and alignment.
That is not portable across the target browser devices and would double the
large pair/checkpoint reservations even though normal tile occupancy is much
smaller than global model size. A bounded local sort plus an explicit overflow
counter is the tighter systems tradeoff.

## Verification

- Browser unit suite: 82 passed, 0 failed.
- Live in-app Apple WebGPU smoke: `4096 -> 8192` initialized, compiled, and
  reached step 1,218 with 112 split activations, 4,201 active splats, finite
  loss, and zero tile overflow.
- Three all-active 8K benchmark runs completed with finite loss and zero
  overflow, exercising splat IDs above 4,095.

The split count initially appeared to remain zero because it was published only
by the asynchronous full validation worker. Once that worker returned, the
step-1,218 snapshot reported the expected seven 16-slot growth events. The
regular loss message now carries this cheap CPU-side counter too, without a GPU
wait; a clean final run displayed 80 activations at step 1,248 before another
full validation.

At 96x72 with packed-FP16 checkpoints, 11x11 Gaussian SSIM, 32 warmup steps,
and 128 GPU-drained measured steps, the three-repeat median changed from 350.2
steps/s at 4,096 active splats to 188.3 steps/s at 8,192. That is 53.8%
throughput retention, or a 46.2% lower step rate. The complete artifact is
`web/dynaworld_browser_trainer/benchmark_results/2026-07-30_tiled_8k_apple_m4.json`.

## Important limitation

The all-active 8K benchmark repeats the 4,096 seed bank to create a worst-case
systems stress. It proves capacity, ID correctness, finite optimization, and
zero overflow for this workload. It does not prove improved reconstruction.
The canonical hidden-slot growth path needs a matched convergence run before
8K can be credited with recovering high-frequency detail.

The final live SPA smoke reported 170.5 completed steps/s at step 1,248. Even
before every hidden slot becomes visible, projection, clear, and update work
use the allocated 8K capacity; the isolated 188.3 steps/s result is therefore a
better expectation for this build than the older 4K-capacity live rate.
