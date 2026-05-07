# Fast-Mac Feature Kernel Catalog

Date: 2026-05-07

Scope: F32/F64 feature splatting on local MPS, mainly
`B=16,G=8192,F=32` with trainer-like `GSP_CHUNK=64,GSP_FAST_CAP=2048`.
Stable baselines stayed untouched; all kernel experiments were copied forks.

## Current Best Read

The largest safe win is still `v6_refined_features_f32_reduce`: it fixes the
big F-channel color-gradient atomic bottleneck without changing render
semantics. The best next trainable candidate is `f32_gradcache` or a combined
`gradcache + zero_bg` fork. The best semantic speed knob is raising
`alpha_threshold`, but that needs a heldout-quality A/B before use in training.

`transmittance_threshold` is top-p-like per pixel, but backward mostly pays the
tile max depth prefix, so it is not the main lever unless we change backward
scheduling.

## Improvement Catalog

Percentages are total-step/raster-kernel timing deltas inside the measured
microbenchmarks unless otherwise noted.

| Attempt | Baseline -> candidate | Main measured row | Speed read | Status |
| --- | ---: | --- | ---: | --- |
| `f32_reduce` atomics | `1001.9ms -> 449.2ms` | 512px B16/G8192/F32 | **55.2% faster total**, **64.3% faster backward** | keeper fork |
| `f32_accum` local output accum | `449.2ms -> 388.1ms` | 512px B16/G8192/F32 | **13.6% faster total**, **36.7% faster forward** | shape-dependent |
| Frozen splat features/colors | `449.2ms -> 267.7ms` | 512px B16/G8192/F32 | **40.4% faster total**, **44.9% faster backward** | useful for camera-only/frozen-feature runs, changes training |
| `f32_gradcache` | `431.7ms -> 388.1ms` | 512px B16/G8192/F32 latest local confirm | **10.1% faster total** | live opt-in candidate |
| `f32_gradcache` | `326.5ms -> 286.8ms` | 256px B16/G8192/F32 latest local confirm | **12.2% faster total** | live opt-in candidate |
| Alpha threshold `1/64` | `431.7ms -> 360.4ms` | 512px B16/G8192/F32 | **16.5% faster total** | quality A/B needed |
| Alpha threshold `1/64` | `326.5ms -> 261.5ms` | 256px B16/G8192/F32 | **19.9% faster total** | quality A/B needed |
| Transmittance threshold `1e-2` | `431.7ms -> 404.5ms` | 512px B16/G8192/F32 | **6.3% faster total** | weaker; tile max stop remains |
| `zero_bg` tail skip | `610.7ms -> 556.7ms` | 512px B16/G8192/F32 active off | **8.8% faster total** | bounded fork, safe version keeps explicit pixel zero/write |
| `zero_bg` tail skip | `386.0ms -> 361.1ms` | 256px B16/G8192/F32 active off | **6.5% faster total** | bounded fork |
| `fixedbin` no-overflow IDs | `855.4ms -> 501.8ms` | 512px B16/G8192/F32 synthetic target row | **41.3% faster total** vs reduce in that matrix | narrow, higher trainer memory |
| `fixedbin` trainer fixed-render | `725.4ms -> 696.1ms` | 256px multicam fixed-render | **4.0% faster total** | mixed; not a memory fix |
| Compact lookup K=4 | `225.9ms -> 96.9ms` | 128px B16/G8192/F32 prototype | **57.1% faster**, **31.9% lower sampled current memory** | architectural prototype |
| Compact lookup K=8 | `179.6ms -> 117.9ms` | 128px B16/G8192/F32 prototype | **34.3% faster**, **24.5% lower sampled current memory** | architectural prototype |
| Compact lookup K=16 | `184.1ms -> 155.6ms` | 128px B16/G8192/F32 prototype | **15.5% faster**, **29.9% lower sampled current memory** | architectural prototype |
| `block4` dot/reduction fusion | `437.6ms -> 547.3ms` | 512px B16/G8192/F32 | **25.1% slower** | negative |
| `f64_accum64` | `646.4ms -> 682.4ms` | 256px B16/G8192/F64 vs best existing | **5.6% slower** | negative |
| `f64_accum64` | `542.7ms -> 567.0ms` | 512px B8/G8192/F64 vs best existing | **4.5% slower** | negative |

## Contributor Counts

Sampled per-pixel contributor counts show why hard top-3/top-4 is risky:

| Shape / threshold | Tile candidates mean / p95 | Checked prefix mean / p95 | Alpha contributors mean / p95 | Early-stop fraction |
| --- | ---: | ---: | ---: | ---: |
| 512px default | 62.9 / 78 | 62.9 / 78 | 26.6 / 36 | 0.0% |
| 512px alpha `1/64` | 50.4 / 64 | 50.4 / 64 | 18.8 / 26 | 0.0% |
| 512px trans `1e-2` | 62.9 / 78 | 61.9 / 77 | 26.2 / 35 | 10.2% |
| 256px default | 241.3 / 286 | 182.3 / 246 | 77.9 / 101 | 89.3% |
| 256px alpha `1/64` | 194.1 / 230 | 149.5 / 199 | 56.2 / 72 | 87.5% |

Key point: `alpha_threshold` reduces binned candidates and actual contributors.
`transmittance_threshold` can stop pixels earlier, but backward still pays the
tile max stop prefix.

## CUDA Port Notes

The Metal kernel now has explicit `CUDA port note` comments at the mechanisms
that matter:

- `alpha_support_params(...)`: bin-time support cutoff. CUDA needs the same
  support pruning to reproduce the alpha-threshold speed/quality tradeoff.
- `eval_alpha(...)`: per-pixel alpha cutoff. CUDA should keep this paired with
  the bin-time cutoff.
- `reduce_atomic_add_feature_grads(...)`: the main `f32_reduce` topology:
  warp/simd reduce, block/threadgroup reduce, then one global atomic per
  splat/channel/tile.
- backward `tile_stop_counts`: current backward uses tile max depth prefix.
  A real top-p CUDA speedup needs per-pixel prefix handling or a different
  block schedule; simply raising transmittance will not remove most backward
  work.

Do not port the failed paths blindly:

- staging full `[chunk,F]` colors into shared/threadgroup memory increased
  pressure and lost
- `block4` reuse increased backward time
- naive F64 private accumulator enlarged per-thread state and lost

## Next Experiments

1. Run heldout-quality A/B for `alpha_threshold`: default, `1/128`, `1/96`,
   `1/64`. This is the most direct speed knob but changes support.
2. Copy a combined `f32_gradcache + zero_bg` fork and benchmark 256/512 B16
   plus trainer fixed-render.
3. Prototype per-pixel backward stop handling. This is the real top-p path,
   but it needs a different scheduling design so barriers remain safe.
4. Wire compact lookup into trainer fixed-render. It is the biggest remaining
   speed/memory idea if quality holds.
5. Only after those, try a very narrow F32-specialized vector/unrolled backward
   path. The `block4` failure says this should be measured immediately and
   killed if it regresses.

## Primary Artifacts

- Long-form performance log:
  `research_notes/fast_mac_feature_metal_performance.md`
- Session note:
  `agent_notes/loose_notes/2026-05-07_14-26-25_feature_contributor_thresholds.md`
- Contributor profiler:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/benchmarks/profile_contributors.py`
- Threshold benchmark surface:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/benchmarks/benchmark_mps.py`
- Experimental Metal comments:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/csrc/metal/gsplat_v6_refined_features_f32_reduce_kernels.metal`
