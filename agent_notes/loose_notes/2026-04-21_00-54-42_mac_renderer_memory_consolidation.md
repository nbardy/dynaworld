# Mac Renderer Memory Consolidation

## Context

This note consolidates the Mac Gaussian splat renderer work after the raw
Metal/MLX import, the Torch+Metal `fast-mac-gsplat` handoffs, the v3 shader
iteration, the Taichi-compatible Mac fork, the native Taichi batch work, and
the final cross-renderer benchmark matrix.

The purpose is not to replace the lower-level notes. It is to give future
agents a durable field map:

- what renderer variants exist,
- what was actually measured,
- which bugs were real,
- where speedups came from,
- what numbers are safe to quote,
- what traps make benchmark claims misleading,
- and what to do next.

## Current Repositories And Commits

Dynaworld parent branch:

```text
codex/register-taichi-mac-submodule
8bd1900 Update renderer README benchmark pointers
f0b3885 Update fast Mac splat benchmark docs
a86c3f5 Add Mac renderer stack benchmark notes
7f26301 Update Taichi 4K benchmark docs pointer
29d74a9 Update Mac renderer repo pointers
```

Renderer submodules:

```text
third_party/fast-mac-gsplat
  736bb3f Document renderer stack percent speedups
  56dab1f Document Torch Taichi fast-mac benchmark matrix
  477cd33 Link Taichi-compatible Mac renderer
  efe39d8 Add direct Torch reference benchmark
  87fced7 Optimize v3 backward tile order reuse
  fc802d3 Add v3 Metal gsplat variant
  338c0f9 Initial fast Mac gsplat Metal renderer

third_party/taichi-splatting
  1d7b50f Document Taichi and fast-mac speedups
  4bd458a Clarify 4K pure-Metal results and Torch limit
  c73625b Document Mac renderer choice and Torch speedups
  3851e15 Add native batched Metal rasterization
  37542bc Document fastest Mac path and future notes
  13f28b6 Add Mac Gaussian splatting search terms
```

Public repos:

- `https://github.com/nbardy/fast-mac-gsplat`
- `https://github.com/nbardy/taichi-gsplat-differentiable-render-mac`

## Renderer Inventory

### Raw MLX/Metal Projected Rasterizer

Location:

```text
shader_experiments/raw_metal_fast_rasterizer/
src/benchmarks/raw_metal_mlx_bridge.py
```

Status:

- Imported from the first chief-scientist handoff.
- Integrated into Dynaworld's optional speed and accuracy benchmark surface.
- Forward-only hot path through MLX custom Metal kernel.
- Accuracy-checked against a CPU float64 packed-2D Torch reference.
- Useful as an early proof that the raw Metal raster math was viable on Mac.
- Not the cleanest training path because it routes through MLX and bridge
  overhead rather than a direct Torch custom op.

Important fixes:

- MLX kernel names must be valid Metal/C identifiers. The original cache-key
  names contained punctuation like `=`, `.`, and `|`, causing Metal compile
  failure before the shader math ran.
- `mx.eval(...)` returns `None` in the tested MLX path. Evaluate first, then
  read `.item()` from the original array.

Accuracy:

- 18/18 packed-2D cases passed across 16/32px and 4/8/16 splats.
- Worst image error was below about `1e-7`.
- Gradient errors were around `1e-9` in the checked packed/features gradients.

Throughput:

- `64x64/G=1024` forward-only smoke: about `8.3 ms`, including bridge overhead.
- First-use `4096x4096/G=65536` forward-only smoke: about `940.8 ms`, dominated
  by setup/compile effects.
- Warmed `4096x4096/G=65536` forward-only smoke: about `86.1 ms`, including
  Torch/NumPy/MLX bridge and output conversion.

Interpretation:

The raw MLX renderer proved that custom Metal rasterization was worth pursuing,
but the direct Torch extension path became the better engineering target because
it keeps the training/autograd stack in PyTorch.

### fast-mac-gsplat v2

Location:

```text
third_party/fast-mac-gsplat/
```

Status:

- Direct Torch custom op backed by pure Metal kernels.
- Works on MPS tensors.
- Implements forward and backward for projected 2D splats.
- Validated as a low-overhead, high-throughput baseline.
- Generally best for low-res bootstrap, small splat counts, and forward-heavy
  smoke tests.

Important fixes:

- Packaging and setup needed repo-relative extension sources.
- Torch custom-op schemas with fixed tuple returns need fixed C++ tuple return
  types, not `std::vector<torch::Tensor>`.
- The original shader read tightly packed Torch `[G,3]` tensors as
  `device float3*`. Metal `float3` has 16-byte alignment/stride while Torch
  rows are 12 bytes, corrupting all but the first row. The durable pattern is
  flat `device float*` plus manual `idx * 3 + channel` loads.
- Backward initially repeated the tile-local sort. v2 now saves tile-local
  sorted IDs in forward and reuses them in backward.

Accuracy:

- Tiny CPU reference check matched forward/backward at roughly `1e-8`.
- Small v2/v3 MPS image parity checks hit max diff `0.0` when using the same
  projected 2D inputs.

Measured character:

- Low launch/staging overhead.
- Strong at `64-128px` and hundreds of splats.
- Still loses to v3 at high splat count when backward and tile overlap dominate.

### fast-mac-gsplat v3

Location:

```text
third_party/fast-mac-gsplat/variants/v3/
```

Status:

- Extracted from the v3 chief-scientist handoff.
- Source artifact preserved under `source_artifacts/`.
- Built and reference-checked locally.
- Kept side by side with v2 because it changes kernel structure enough that the
  old baseline remains valuable.

Architectural changes relative to v2:

- 256-thread tile groups, matching one thread per pixel for 16x16 tiles.
- Stages Gaussian parameters in threadgroup memory.
- Has overflow-tile fallback machinery.
- Reduces global gradient pressure with tile-local SIMD/threadgroup reductions.
- After our patch, forward writes sorted tile-local order so backward can skip
  duplicate local sort in the no-overflow fast path.

Accuracy:

Reference check after saved-order changes:

```text
image max error: 5.960464477539063e-08
means grad max error: 2.4010660126805305e-10
conics grad max error: 9.313225746154785e-10
colors grad max error: 9.313225746154785e-10
opacities grad max error: 1.862645149230957e-09
```

Measured character:

- More overhead than v2 on tiny scenes.
- Stronger at large scenes and especially backward-heavy workloads.
- The current training-scale choice among the pure Metal variants.

### Taichi-Compatible Mac Renderer

Location:

```text
third_party/taichi-splatting/
```

Public repo:

```text
https://github.com/nbardy/taichi-gsplat-differentiable-render-mac
```

Status:

- Fork of `uc-vision/taichi-splatting` with an Apple Silicon path.
- Uses PyTorch/MPS for tile/depth ordering fallback and Taichi/Metal for
  raster/backward.
- Correct and useful when the surrounding stack needs Taichi compatibility.
- Now has native batch rendering through `rasterize_batch`.

Important fixes and discoveries:

- Upstream `taichi-splatting==0.13.0` is not installable in the local Python
  3.11 `uv` environment; the vendored fork is the source of truth.
- CUDA sort/cumsum fallbacks had to be replaced for MPS/Metal.
- The original fast Taichi kernels use CUDA-only SIMT block/warp intrinsics;
  the Metal path needed separate `metal_reference` / `pixel_reference` kernels.
- Taichi's public `parallel_sort` works on Taichi fields, not Torch tensors, and
  `PrefixSumExecutor` rejects Metal. Native Metal sorting needed new kernels,
  not just a backend switch.
- `taichi_field`, `bucket_taichi`, and `ordered_taichi` sort experiments are
  correct in narrow cases but not performance winners yet.
- Timing mixed PyTorch/Taichi needs `TaichiQueue.run_sync(ti.sync)`, not only
  `torch.mps.synchronize()`.
- Passing MPS slice views like `packed[b]` through the single-image Taichi
  autograd function can trip PyTorch version-counter checks. Native batch avoids
  this train-loop hazard by keeping `[B, G, ...]` tensors intact.

Accuracy:

- Small packed-2D CPU float64 reference checks matched forward and gradients at
  about `1e-8`.
- Native batch vs loop CPU check:
  - forward image max diff `0.0`
  - alpha max diff `0.0`
  - packed grad max diff `3.7e-9`
  - feature grad max diff `0.0`
- MPS native batch backward produced finite packed/features gradients.

Measured character:

- Much faster than direct Torch.
- Slower than the pure Metal fast-mac variants when maximum raster throughput is
  the goal.
- Valuable because it integrates with Taichi and has a native batch API now.

## Current Renderer Decision Model

Use this as the short policy until a new benchmark changes it:

```text
Need Taichi compatibility / native Taichi batch?
    Use taichi-gsplat-differentiable-render-mac.

Bootstrapping training at 64-128px with hundreds of splats?
    Use fast-mac v2 if a pure Metal/Torch single-frame path is acceptable.

Training or stress testing with many splats, larger footprints, or backward
dominating wall time?
    Use fast-mac v3.

Need a correctness baseline?
    Use direct Torch only at small sizes.

Need historical raw shader experiments?
    Keep raw MLX/Metal in shader_experiments, but do not treat it as the
    current training path.
```

The key crossover is overhead vs occupancy:

- small scenes are launch/staging/abstraction overhead dominated,
- large scenes are bandwidth, tile occupancy, reverse scan, and atomic-gradient
  pressure dominated.

That is why v2 beats v3 at tiny scale while v3 wins the serious backward cases.

## Benchmark Methodology Invariants

For the latest stack benchmark:

- Synthetic projected 2D circular Gaussians.
- Shared inputs across direct Torch, Taichi, v2, and v3.
- Same `x + 0.5, y + 0.5` pixel-center convention for packed-2D correctness
  comparisons.
- Direct Torch sorts by detached depth and loops over splats.
- Taichi uses native `rasterize_batch` when `B > 1`.
- fast-mac v2/v3 currently loop over `B` frames because their public APIs are
  single-image.
- Large direct Torch is skipped because `H * W * G` becomes absurd.
- `% faster = (baseline_ms / renderer_ms - 1) * 100`.

Benchmark traps:

- Speedup factors are only meaningful when the baseline implementation is named.
  The same Taichi absolute time can look like `7x` or `30x` depending on the
  direct Torch reference.
- First pass includes compilation/pipeline setup for several stacks. Warmed
  numbers are the real steady-state numbers.
- Warmup does not mean "different splats"; it means the same workload is run
  before timing so compilation/cache/setup effects do not dominate. The kernels
  still recompute output buffers; there is no evidence of result caching.
- Mixed Taichi/PyTorch timing must sync both Taichi and MPS.
- Background CPU jobs can inject host-side noise into launch-heavy small cases.
- Direct Torch at 4K/64K is not a useful comparator:
  `4096 * 4096 * 65536 ~= 1.1e12` pixel-splat evaluations.

## Latest Cross-Renderer Timing Matrix

### Small Batch: 64x64, B=4, G=128

| Case | Mode | Torch | Taichi | v2 | v3 | Best |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sigma 1-5 | forward | `126.618 ms` | `14.759 ms` | `12.861 ms` | `20.065 ms` | v2 |
| sigma 3-8 | forward | `145.510 ms` | `17.993 ms` | `15.916 ms` | `27.048 ms` | v2 |
| sigma 1-5 | fwd+bwd | `685.026 ms` | `35.858 ms` | `21.535 ms` | `26.991 ms` | v2 |
| sigma 3-8 | fwd+bwd | `593.915 ms` | `36.881 ms` | `33.096 ms` | `51.422 ms` | v2 |

Percent conclusions:

- Taichi faster than Torch: `+709%` to `+1810%`.
- Best fast-mac faster than Torch: `+814%` to `+3081%`.
- Best fast-mac faster than Taichi: `+11%` to `+67%`.

### Small Batch: 128x128, B=4, G=128

Use the forward rerun for forward-only because it had more warmup/iters and less
host noise.

| Case | Mode | Torch | Taichi | v2 | v3 | Best |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sigma 1-5 | forward | `83.627 ms` | `11.961 ms` | `11.050 ms` | `16.823 ms` | v2 |
| sigma 3-8 | forward | `97.898 ms` | `16.517 ms` | `11.517 ms` | `16.688 ms` | v2 |
| sigma 1-5 | fwd+bwd | `702.222 ms` | `47.335 ms` | `24.254 ms` | `28.558 ms` | v2 |
| sigma 3-8 | fwd+bwd | `717.170 ms` | `66.641 ms` | `28.428 ms` | `39.775 ms` | v2 |

Percent conclusions:

- Taichi faster than Torch: `+493%` to `+1384%`.
- Best fast-mac faster than Torch: `+657%` to `+2795%`.
- Best fast-mac faster than Taichi: `+8%` to `+134%`.

### Small Single Frame: 128x128, B=1, G=512

| Case | Mode | Torch | Taichi | v2 | v3 | Best |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sigma 1-5 | forward | `107.720 ms` | `18.254 ms` | `4.007 ms` | `5.099 ms` | v2 |
| sigma 3-8 | forward | `113.577 ms` | `14.365 ms` | `3.680 ms` | `5.209 ms` | v2 |
| sigma 1-5 | fwd+bwd | `612.363 ms` | `44.268 ms` | `8.769 ms` | `8.578 ms` | v3 |
| sigma 3-8 | fwd+bwd | `636.652 ms` | `43.138 ms` | `6.236 ms` | `6.858 ms` | v2 |

Percent conclusions:

- Taichi faster than Torch: `+490%` to `+1376%`.
- Best fast-mac faster than Torch: `+2588%` to `+10109%`.
- Best fast-mac faster than Taichi: `+290%` to `+592%`.

### Large Stress: 1024x1024, B=1, G=65,536

Direct Torch was skipped.

| Case | Mode | Taichi | v2 | v3 | Best |
| --- | --- | ---: | ---: | ---: | --- |
| sigma 1-5 | forward | `54.323 ms` | `14.410 ms` | `9.717 ms` | v3 |
| sigma 3-8 | forward | `63.798 ms` | `16.919 ms` | `18.117 ms` | v2 |
| sigma 1-5 | fwd+bwd | `352.500 ms` | `41.148 ms` | `20.238 ms` | v3 |
| sigma 3-8 | fwd+bwd | `1081.152 ms` | `119.657 ms` | `39.510 ms` | v3 |

Percent conclusions:

- Best fast-mac faster than Taichi: `+277%` to `+2636%`.
- v3 is the large backward winner.
- Taichi's backward degrades hard under wider splat overlap.

### 4K fast-mac v2 vs v3

| Case | v2 | v3 | v3 faster than v2 |
| --- | ---: | ---: | ---: |
| 4096x4096, G=65,536, sigma 1-5, forward | `15.506 ms` | `12.410 ms` | `+25%` |
| 4096x4096, G=65,536, sigma 3-8, forward | `24.935 ms` | `13.702 ms` | `+82%` |
| 4096x4096, G=65,536, sigma 1-5, fwd+bwd | `70.654 ms` | `47.872 ms` | `+48%` |
| 4096x4096, G=65,536, sigma 3-8, fwd+bwd | `134.162 ms` | `60.738 ms` | `+121%` |

The practical headline is that v3 can do 4K projected splatting with 65,536
splats in about `12-14 ms` forward and `48-61 ms` forward+backward on these
synthetic cases. That is the result that made the path feel unexpectedly good.

## Backward Cost Model

Observed fact:

- Backward can be several times slower than forward even when forward is
  extremely fast.
- The ratio depends strongly on tile overlap and splat footprint.

Known contributors:

1. recomputing the forward alpha/transmittance chain,
2. reverse scanning per pixel,
3. global atomic accumulation into per-Gaussian gradients,
4. repeated parameter loads across tile pixels,
5. local tile sorting if not saved/reused,
6. overflow fallback if pathological tiles exceed fast caps.

What was fixed:

- v2 no longer repeats tile-local sort in backward.
- v3 now saves sorted tile-local IDs too.
- v3 saved-order patch made forward `1-2 ms` slower at 4K/64K because of ID
  writeback, but improved forward+backward:
  - sigma 1-5: `52.460 ms -> 47.872 ms` (`-8.7%`)
  - sigma 3-8: `65.687 ms -> 60.738 ms` (`-7.5%`)
- Estimated backward-only improved about `12-14%`.

Current belief:

- Sorting was a real backward tax but is not the only tax.
- Once sorted IDs are saved, the next major costs are bandwidth and atomic
  gradient pressure, especially with wider splats.
- v3's tile-local reduction and parameter staging are why it wins large
  backward cases even though it can lose tiny forward cases.

Falsification tests:

- Add counters for total tile pairs, max tile count, mean tile count, overflow
  tiles, and gradient atomic pressure proxies.
- Benchmark real projected training distributions, not only uniform random
  projected 2D splats.
- Run randomized renderer order with multiple seeds on an idle machine.
- Build 8x8 / 16x16 / 32x32 tile variants only after counters show tile size is
  a meaningful knob.

## Tile Size And Batch Support

Tile size:

- v2/v3 are specialized around 16x16 tiles.
- In v3 that maps cleanly to 256 threads, one per tile pixel.
- This is not currently a casual runtime hyperparameter. Changing it touches
  shader constants, threadgroup memory, local sort/reduction layout, caps, and
  wrapper validation.
- A compiled-variant study is the right way to test 8x8 or 32x32.

Batch:

- Taichi supports native batch through `rasterize_batch`.
- fast-mac v2/v3 currently expose single-image projected raster APIs.
- The stack benchmark loops v2/v3 over batch and reports total and per-frame
  time.
- A true batched fast-mac API is an obvious next engineering target if training
  wants multi-frame raster in one call.

Device contract:

- fast-mac custom ops need MPS tensors because the Metal extension operates via
  PyTorch's MPS/Metal path.
- CPU tensors are for small references and correctness checks.
- The early scaffold failure mode "intentionally throws" meant the code reached
  the placeholder C++/ObjC++ op but did not yet do MPS tensor to `MTLBuffer`
  interop, command submission, and aux buffer plumbing. The later fastpath
  handoffs implemented those pieces.

## Correctness And Convention Traps

Pixel centers:

- Raw MLX/Metal and Taichi packed-2D reference use `x + 0.5`.
- Some older Dynaworld dense/sparse training renderers sampled integer grid
  coordinates.
- Throughput comparisons can show expected image diffs until sampling
  conventions are unified.

Depth:

- fast-mac treats depth as a piecewise-constant ordering key.
- Depth gradients are zero.
- This is standard enough for the current projected rasterizer, but any future
  differentiable sorting/order work must be explicit.

Packed Gaussian layout:

- Taichi packed 2D row is `[mean_x, mean_y, axis_x, axis_y, sigma_x, sigma_y,
  opacity]`.
- fast-mac projected inputs are `means2d [G,2]`, `conics [G,3]`, `colors [G,3]`,
  `opacities [G]`, `depths [G]`.
- Do not use Metal `float3*` for Torch `[G,3]`.

Overflow/caps:

- v2 has a compile-time tile-pair cap.
- v3 has fast cap plus overflow fallback machinery.
- No-overflow synthetic cases are well measured; deliberately pathological
  center-cluster overflow tests remain important.

## How To Quote Results Safely

Safe statements:

- Direct Torch projected splatting is useful for correctness and small baselines
  but not a serious high-res renderer.
- Taichi Mac is much faster than direct Torch and now supports native batch.
- fast-mac v2/v3 are faster than Taichi in every measured "best fast-mac"
  scenario.
- Taichi can beat v3 alone on small batched cases, but v2 was faster in those
  same rows.
- v2 is best for low-res bootstrap/smoke; v3 is best for large/backward-heavy
  projected raster.
- Benchmark claims should include absolute milliseconds, baseline
  implementation, warmup/iters, batch handling, and sync details.

Unsafe statements:

- "Taichi is 30x faster than Torch" without naming the Torch baseline.
- "v3 is always faster than v2" without mentioning tiny scenes.
- "4K/64K vs Torch speedup" with no Torch number. The direct baseline is not
  meaningful at that scale.
- "Warm means cached output." Warm only means compilation/setup/cache effects
  are excluded from timed iterations.

## Open Questions

1. What does the renderer ranking look like on real projected TokenGS/DUSt3R
   distributions instead of uniform projected 2D synthetic splats?
2. How much does a native batched fast-mac API improve training throughput
   relative to Python looping over frames?
3. Is v3 overflow fallback correct and fast enough on deliberately pathological
   tile-overflow scenes?
4. Does tile size 8x8 or 32x32 beat 16x16 on Apple GPUs for real scenes?
5. Can gradient accumulation be reorganized further to reduce global atomic
   pressure without blowing memory?
6. Should fast-mac offer a forward-only inference path that does not write back
   sorted tile IDs, recovering the `1-2 ms` forward-only cost added for faster
   backward?
7. Can fp16/bfloat16 parameter staging help without accuracy problems? Taichi
   needs separate kernels for fp16; it is not a simple cast.
8. How should camera/scene normalization and projected splat radii be controlled
   so benchmarks match training distributions?

## Recommended Next Work

Near term:

1. Keep docs honest by including absolute ms and `% faster` together.
2. Add benchmark metadata capture: active renderer commit, PyTorch version,
   macOS version, MPS availability, warmup/iters, and sync method.
3. Add a fast-mac native batch API if training remains batch-of-frames oriented.
4. Add a v3 forward-only no-sorted-ID writeback mode for inference/smoke tests.
5. Run the stack benchmark on an idle machine with randomized renderer order.

Medium term:

1. Real training-distribution renderer benchmark after the Taichi train adapter
   changes are isolated from unrelated dirty trainer work.
2. Tile occupancy and atomic-pressure counters in v2/v3.
3. Overflow stress harness.
4. Compiled tile-size variants if counters justify it.
5. Investigate global atomic reduction alternatives for backward.

Long term:

1. Full 3D camera projection + raster path integration for fast-mac, not only
   projected 2D inputs.
2. Decide whether Taichi remains the compatibility backend only or gets a fused
   threadgroup-memory kernel to close the gap.
3. Build a stable renderer selection layer in Dynaworld with explicit contracts
   for device, batch shape, pixel convention, and depth ordering.
