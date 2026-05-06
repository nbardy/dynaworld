# Fast-Mac Feature Splatting Metal Performance Notes

Date: 2026-05-07

## Artifact Map

- Stable baseline, do not mutate for experiments:
  `third_party/fast-mac-gsplat/variants/v6_refined_features`
- Main F32 atomic-reduction fork:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce`
- Rejected / not promoted staging fork:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_stage`
- Accumulation experiment / synthetic train-path candidate:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_accum`
- F64 local-accumulation pressure test / not promoted:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64`
- F32 backward grad-cache trainer-timing candidate:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_gradcache`
- F32 block4 dot/reduction fusion pressure test / not promoted:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_block4`
- F32 fixed-cap binning no-overflow experiment / target-row timing candidate:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_fixedbin`
- Compact-basis feature lookup prototype:
  `third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment`
- Safe benchmark runner:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/benchmarks/benchmark_matrix.py`
- Opt-in trainer dispatch:
  `render.fast_mac.feature_variant = "v6_refined_features_f32_reduce"`,
  `"v6_refined_features_f32_accum"`,
  `"v6_refined_features_f32_gradcache"`, or
  `"v6_refined_features_f32_fixedbin"`

No checked-in trainer config points at the new experimental forks yet. Keep it
that way until a larger trainer phase trace and heldout-quality parity check
pass.

## Current Measurements

Local MPS, `512x512`, `B=16`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`, `batch_strategy=flatten`, `active_policy=off`.
These are short local probes, so treat trends as stronger than exact values.

Important correction: early safe-runner rows forced `GSP_CHUNK=32` and
`GSP_FAST_CAP=512`. That is useful as a fallback-pressure stress case, but it is
not the trainer-like default. Dynaworld configs and the RGB refined benchmark
use `max_fast_pairs=2048`; the safe matrix runner now defaults to
`GSP_CHUNK=64,GSP_FAST_CAP=2048`.

Trainer-like runtime cap, `GSP_CHUNK=64`, `GSP_FAST_CAP=2048`:

| Shape | Variant | colors trainable | forward ms | backward ms | total mean ms | Artifact |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 128px B16/G8192/F3 | RGB `v6_refined` | true | 12.1 | 15.5 | 27.6 | direct JSON stdout |
| 128px B16/G8192/F3 | stable `v6_refined_features` | true | 12.2 | 28.8 | 40.9 | `2026-05-07_128_b16_g8192_f3_feature_cap2048.jsonl` |
| 128px B16/G8192/F3 | `v6_refined_features_f32_reduce` | true | 11.8 | 16.9 | 28.7 | `2026-05-07_128_b16_g8192_f3_feature_cap2048.jsonl` |
| 128px B16/G8192/F32 | stable `v6_refined_features` | true | 25.6 | 171.3 | 196.9 | `2026-05-07_128_b16_g8192_f32_feature_cap2048.jsonl` |
| 128px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 26.1 | 67.4 | 93.5 | `2026-05-07_128_b16_g8192_f32_feature_cap2048.jsonl` |
| 128px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 24.3 | 66.9 | 91.2 | `2026-05-07_128_b16_g8192_f32_accum_cap2048_stdout.jsonl` |
| 512px B16/G8192/F32 | stable `v6_refined_features` | true | 88.2 | 913.7 | 1001.9 | `2026-05-07_512_b16_g8192_f32_baseline_vs_reduce_cap2048.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 123.0 | 326.2 | 449.2 | `2026-05-07_512_b16_g8192_f32_baseline_vs_reduce_cap2048.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 77.8 | 310.3 | 388.1 | `2026-05-07_512_b16_g8192_f32_accum_cap2048_stdout.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | false | 87.9 | 179.8 | 267.7 | `2026-05-07_512_b16_g8192_f32_reduce_cap2048_freeze_colors.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | false | 78.4 | 178.3 | 256.7 | `2026-05-07_512_b16_g8192_f32_accum_cap2048_freeze_colors_stdout.jsonl` |
| 128px B16/G8192/F64 | stable `v6_refined_features` | true | 87.7 | 590.7 | 678.4 | `2026-05-07_128_f64_b8_b16_variant_matrix.jsonl` |
| 128px B16/G8192/F64 | `v6_refined_features_f32_reduce` | true | 95.8 | 244.8 | 340.6 | `2026-05-07_128_f64_b8_b16_variant_matrix.jsonl` |
| 128px B16/G8192/F64 | `v6_refined_features_f32_accum` | true | 91.3 | 274.4 | 365.6 | `2026-05-07_128_f64_b8_b16_variant_matrix.jsonl` |
| 256px B16/G8192/F64 | stable `v6_refined_features` | true | 219.8 | 1801.8 | 2021.6 | `2026-05-07_256_f64_b16_variant_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_reduce` | true | 332.2 | 892.8 | 1224.9 | `2026-05-07_256_f64_b16_variant_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_accum` | true | 254.3 | 710.4 | 964.6 | `2026-05-07_256_f64_b16_variant_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_reduce` | false | 269.1 | 376.8 | 645.9 | `2026-05-07_256_f64_b16_freeze_colors_variant_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_accum` | false | 270.9 | 357.0 | 628.0 | `2026-05-07_256_f64_b16_freeze_colors_variant_matrix.jsonl` |
| 512px B4/G8192/F64 | stable `v6_refined_features` | true | 79.3 | 674.6 | 754.0 | `2026-05-07_512_f64_b4_variant_matrix.jsonl` |
| 512px B4/G8192/F64 | `v6_refined_features_f32_reduce` | true | 90.9 | 321.8 | 412.7 | `2026-05-07_512_f64_b4_variant_matrix.jsonl` |
| 512px B4/G8192/F64 | `v6_refined_features_f32_accum` | true | 159.0 | 461.3 | 620.3 | `2026-05-07_512_f64_b4_variant_matrix.jsonl` |
| 512px B8/G8192/F64 | stable `v6_refined_features` | true | 165.8 | 1448.2 | 1614.0 | `2026-05-07_512_f64_b8_variant_matrix.jsonl` |
| 512px B8/G8192/F64 | `v6_refined_features_f32_reduce` | true | 222.1 | 668.8 | 890.8 | `2026-05-07_512_f64_b8_variant_matrix.jsonl` |
| 512px B8/G8192/F64 | `v6_refined_features_f32_accum` | true | 197.3 | 640.3 | 837.6 | `2026-05-07_512_f64_b8_variant_matrix.jsonl` |
| 256px B16/G8192/F32 | stable `v6_refined_features` | true | 116.5 | 1012.2 | 1128.7 | `2026-05-07_256_f32_b16_b32_variant_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 153.4 | 466.7 | 620.1 | `2026-05-07_256_f32_b16_b32_variant_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 123.4 | 441.7 | 565.1 | `2026-05-07_256_f32_b16_b32_variant_matrix.jsonl` |
| 256px B32/G8192/F32 | stable `v6_refined_features` | true | 215.7 | 1901.3 | 2117.0 | `2026-05-07_256_f32_b16_b32_variant_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_reduce` | true | 249.1 | 753.7 | 1002.9 | `2026-05-07_256_f32_b16_b32_variant_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_accum` | true | 244.6 | 805.1 | 1049.7 | `2026-05-07_256_f32_b16_b32_variant_matrix.jsonl` |
| 512px B16/G8192/F64 | stable `v6_refined_features` | false/profile | 613.9 | 0.0 | 613.9 | `2026-05-07_512_f64_b16_profile_forward.jsonl` |
| 512px B16/G8192/F64 | `v6_refined_features_f32_reduce` | false/profile | 734.4 | 0.0 | 734.4 | `2026-05-07_512_f64_b16_profile_forward.jsonl` |
| 512px B16/G8192/F64 | `v6_refined_features_f32_accum` | false/profile | 706.2 | 0.0 | 706.2 | `2026-05-07_512_f64_b16_profile_forward.jsonl` |
| 512px B16/G8192/F64 | `v6_refined_features_f32_reduce` | false | 675.2 | 1277.2 | 1952.3 | `2026-05-07_512_f64_b16_freeze_colors_variant_matrix.jsonl` |
| 512px B16/G8192/F64 | `v6_refined_features_f32_accum` | false | 896.4 | 915.1 | 1811.5 | `2026-05-07_512_f64_b16_freeze_colors_variant_matrix.jsonl` |
| 512px B16/G8192/F64 | `v6_refined_features_f32_reduce` | true | 767.9 | 1512.9 | 2280.8 | `2026-05-07_512_f64_b16_trainable_forks.jsonl` |
| 512px B16/G8192/F64 | `v6_refined_features_f32_accum` | true | 786.3 | 1642.8 | 2429.1 | `2026-05-07_512_f64_b16_trainable_forks.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_reduce` | true | 162.3 | 492.2 | 654.5 | `2026-05-07_256_f64_b16_accum64_confirm_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_accum` | true | 162.7 | 483.7 | 646.4 | `2026-05-07_256_f64_b16_accum64_confirm_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f64_accum64` | true | 200.1 | 482.3 | 682.4 | `2026-05-07_256_f64_b16_accum64_confirm_matrix.jsonl` |
| 512px B8/G8192/F64 | `v6_refined_features_f32_reduce` | true | 152.5 | 390.2 | 542.7 | `2026-05-07_512_f64_b8_accum64_matrix.jsonl` |
| 512px B8/G8192/F64 | `v6_refined_features_f32_accum` | true | 131.0 | 436.9 | 567.9 | `2026-05-07_512_f64_b8_accum64_matrix.jsonl` |
| 512px B8/G8192/F64 | `v6_refined_features_f64_accum64` | true | 159.5 | 407.5 | 567.0 | `2026-05-07_512_f64_b8_accum64_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 84.2 | 252.9 | 337.1 | `2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 70.3 | 243.9 | 314.3 | `2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 70.9 | 202.8 | 273.7 | `2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_reduce` | true | 135.7 | 449.3 | 585.0 | `2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_accum` | true | 132.2 | 461.7 | 593.8 | `2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 144.7 | 464.5 | 609.2 | `2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 123.3 | 403.7 | 527.0 | `2026-05-07_512_f32_b16_gradcache_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 125.1 | 471.3 | 596.4 | `2026-05-07_512_f32_b16_gradcache_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 157.8 | 439.3 | 597.2 | `2026-05-07_512_f32_b16_gradcache_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_block4` | true | 86.7 | 291.3 | 377.9 | `2026-05-07_256_f32_b16_b32_block4_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_block4` | true | 145.2 | 547.6 | 692.8 | `2026-05-07_256_f32_b16_b32_block4_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 91.6 | 346.0 | 437.6 | `2026-05-07_512_f32_b16_block4_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 77.4 | 315.7 | 393.1 | `2026-05-07_512_f32_b16_block4_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 92.0 | 277.7 | 369.7 | `2026-05-07_512_f32_b16_block4_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_block4` | true | 103.8 | 443.5 | 547.3 | `2026-05-07_512_f32_b16_block4_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 94.5 | 328.8 | 423.3 | `2026-05-07_512_f32_b16_gradcache_confirm_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 85.0 | 327.9 | 412.9 | `2026-05-07_512_f32_b16_gradcache_confirm_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 98.3 | 293.0 | 391.3 | `2026-05-07_512_f32_b16_gradcache_confirm_matrix.jsonl` |
| 128px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 66.1 | 147.1 | 213.2 | `2026-05-07_128_f32_b16_fixedbin_smoke_matrix.jsonl` |
| 128px B16/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 64.3 | 132.8 | 197.1 | `2026-05-07_128_f32_b16_fixedbin_smoke_matrix.jsonl` |
| 128px B16/G8192/F32 | `v6_refined_features_f32_fixedbin` | true | 55.6 | 121.1 | 176.7 | `2026-05-07_128_f32_b16_fixedbin_smoke_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 147.1 | 346.1 | 493.2 | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 117.0 | 403.0 | 520.0 | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 128.3 | 369.0 | 497.3 | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B16/G8192/F32 | `v6_refined_features_f32_fixedbin` | true | 129.4 | 394.0 | 523.4 | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_reduce` | true | 245.2 | 758.2 | 1003.4 | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_accum` | true | 221.6 | 826.4 | 1048.0 | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 218.6 | 750.3 | 968.9 | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B32/G8192/F32 | `v6_refined_features_f32_fixedbin` | true | 214.4 | 782.2 | 996.6 | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 208.5 | 646.9 | 855.4 | `2026-05-07_512_f32_b16_fixedbin_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 150.5 | 556.0 | 706.5 | `2026-05-07_512_f32_b16_fixedbin_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_gradcache` | true | 194.8 | 521.6 | 716.4 | `2026-05-07_512_f32_b16_fixedbin_matrix.jsonl` |
| 512px B16/G8192/F32 | `v6_refined_features_f32_fixedbin` | true | 124.3 | 377.5 | 501.8 | `2026-05-07_512_f32_b16_fixedbin_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_reduce` | true | 184.5 | 595.3 | 779.8 | `2026-05-07_256_f64_b16_fixedbin_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_accum` | true | 164.1 | 561.9 | 726.0 | `2026-05-07_256_f64_b16_fixedbin_matrix.jsonl` |
| 256px B16/G8192/F64 | `v6_refined_features_f32_fixedbin` | true | 158.2 | 559.9 | 718.1 | `2026-05-07_256_f64_b16_fixedbin_matrix.jsonl` |

Read: the earlier `F=3` feature overhead was mostly a runtime-cap/fallback
artifact, not an inherent generic-channel tax. With `64/2048`, F3 feature
forward matches RGB. Stable F3 backward still pays extra feature-gradient
handling; the reduction fork recovers RGB-class F3 backward. At F32, the
reduction fork's color-gradient reduction remains the main backward win, while
the accumulation fork became the fastest early synthetic train path under the
corrected cap because it also reduces direct-forward output traffic. The later
fixedbin fork beats it on the target `512px/B16/F32` row, so treat both as
shape-dependent timing candidates rather than defaults. At F64, `f32_accum`
wins the batch-heavy `256px/B16` and `512px/B8` rows, while
`f32_reduce` wins the high-resolution lower-batch `512px/B4` row. Treat
accumulation as shape-dependent: it helps when dense feature output traffic is
dominant, but private/thread array pressure can erase the win. The true F64
local-accumulation fork (`f64_accum64`) passed correctness gates but did not
beat the existing forks on `256px/B16/F64` or `512px/B8/F64`; raising private
accumulation from 32 to 64 floats mostly increased forward cost. The F32
grad-cache fork is noisy but still live: it loses one early `512px/B16/F32`
matrix, then wins the later `512px/B16/F32` confirm and same-window trainer
fixed-render gate. The `block4` fusion fork is a clear negative; reusing
`grad_features` loads in 4-channel blocks increased backward enough to lose all
tested rows. The `f32_fixedbin` fork is a real but narrow host/binning
candidate: it removes the exact-size bin allocation sync and wins the target
synthetic `512px/B16/F32` row, but the fixed ID buffer is large and the win is
not monotonic across `256px` or trainer-fixed-render rows. Keep it opt-in and
no-overflow-only.

Profile sanity check, 128px/B16/G8192/F32 forward-only with f32_reduce:

| `GSP_CHUNK` | `GSP_FAST_CAP` | forward ms | overflow tiles | max pairs/tile | mean pairs/tile | Artifact |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 512 | 833.9 | 970 | 1135 | 875.7 | `2026-05-07_128_b16_g8192_f32_cap_profile.jsonl` |
| 32 | 2048 | 23.8 | 0 | 1135 | 875.7 | `2026-05-07_128_b16_g8192_f32_cap_profile.jsonl` |
| 64 | 512 | 833.1 | 970 | 1135 | 875.7 | `2026-05-07_128_b16_g8192_f32_cap_profile.jsonl` |
| 64 | 2048 | 23.5 | 0 | 1135 | 875.7 | `2026-05-07_128_b16_g8192_f32_cap_profile.jsonl` |

Read: the cap, not chunk size, caused the catastrophic forward path. `cap=512`
overflowed most tiles (`970/1024`) for this synthetic shape and fell into the
slow path. `cap=2048` eliminated overflow.

Legacy fallback-pressure rows, `GSP_CHUNK=32`, `GSP_FAST_CAP=512`:

| Variant | iters | forward ms | backward ms | total mean ms | total median ms | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 3 | 216.6 | 1712.1 | 1928.7 | 1919.7 | baseline direct feature atomics |
| `v6_refined_features_f32_reduce` | 5 | 180.2 | 535.3 | 715.4 | 704.4 | current best total |
| `v6_refined_features_f32_stage` | 3 | 183.2 | 755.8 | 938.9 | 941.9 | reject for train total |
| `v6_refined_features_f32_accum` | 5 | 168.9 | 610.1 | 779.1 | 749.5 | forward improves, total not better |

Forward-only eval on the same shape:

| Variant | iters | forward mean ms | forward median ms | Read |
| --- | ---: | ---: | ---: | --- |
| `v6_refined_features_f32_reduce` | 5 | 202.6 | 202.6 | baseline for eval path |
| `v6_refined_features_f32_stage` | 5 | 276.7 | 271.1 | staging hurts/does not help enough |
| `v6_refined_features_f32_accum` | 5 | 168.0 | 165.4 | real eval-forward win |

Interpretation:

- The big training win is still the atomic-reduction fork. It attacks
  `g_colors` atomics in backward and is the only fork that clearly improves
  total train-path raster timing.
- After the no-color-gradient allocation cleanup in `f32_reduce`, a same-shape
  post-build probe measured `568.9ms` total / `446.0ms` backward with trainable
  colors and `362.2ms` total / `242.5ms` backward with `--freeze-colors`. Treat
  these as current-session timing, not a replacement for the cross-fork table.
- Feature staging is not worth promoting. It can slightly improve one forward
  slice, but the larger threadgroup-memory footprint worsens backward/total.
- Thread-local feature accumulation looked eval-only under the bad cap, but
  under the corrected `64/2048` cap it beats `f32_reduce` on the synthetic
  fwd+bwd pressure rows. It still needs trainer dispatch and fixed-render phase
  timing before promotion.

Synthetic `128x128`, `B=16`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`, `batch_strategy=flatten`, `warmup=1`, `iters=3`:

| Variant | forward ms | backward ms | total mean ms | total median ms | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 664.8 | 310.2 | 974.9 | 978.4 | `2026-05-07_128_b16_g8192_f32_baseline_vs_reduce.jsonl` |
| `v6_refined_features_f32_reduce` | 656.0 | 275.2 | 931.2 | 930.2 | `2026-05-07_128_b16_g8192_f32_baseline_vs_reduce.jsonl` |

Read: at smaller resolution under the old cap, the reduction fork still helps
backward, but the total synthetic win is only about `4.5%`. This row is now
best interpreted as a fallback-cap stress case, not the primary comparison.

Frozen-feature synthetic rerun on the same 128px shape:

| Variant | colors trainable | forward ms | backward ms | total mean ms | Artifact |
| --- | --- | ---: | ---: | ---: | --- |
| `v6_refined_features_f32_reduce` | true | 794.6 | 261.1 | 1055.7 | `2026-05-07_128_b16_g8192_f32_reduce_trainable_rerun.jsonl` |
| `v6_refined_features_f32_reduce` | false | 852.5 | 150.2 | 1002.7 | `2026-05-07_128_b16_g8192_f32_reduce_freeze_colors.jsonl` |

Read: the forward timing drifted between adjacent local runs, so do not compare
these forward numbers against the earlier 128px cap512 table. The backward
delta is the meaningful signal: skipping feature/color gradients still removes a
large fraction of backward work, matching the 512px observation.

Multicam trainer phase, checked-in 128px DeepView 2-train/1-heldout F32 config
with `16` frames, `8192` splats, cached V-JEPA features, `warmup=1`,
`iters=2`:

| Variant | total mean ms | raster fwd ms | autograd backward total ms | Artifact |
| --- | ---: | ---: | ---: | --- |
| current config, `v5_features` | 439.9 | 25.8 | 245.3 | `multicam128_f32_v5_features_warm1_iters2.json` |
| opt-in `v6_refined_features_f32_reduce` | 403.5 | 22.0 | 241.4 | `multicam128_f32_f32_reduce_warm1_iters2.json` |

Warmed one-iteration detached backward probes on the same trainer shape:

| Variant | raster backward probe ms | loss/colorize probe ms | project probe ms | model probe ms | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| current config, `v5_features` | 74.0 | 100.9 | 61.6 | 96.7 | `multicam128_f32_v5_features_breakdown_warm1.json` |
| opt-in `v6_refined_features_f32_reduce` | 81.5 | 101.1 | 59.0 | 50.9 | `multicam128_f32_f32_reduce_breakdown_warm1.json` |

Read: the 128px trainer step is directionally faster with the fork, but the
detached raster-backward probe alone is not faster. At this trainer scale,
colorize/loss, projection, and model backward are large enough that raster
microbench wins should not be promoted without full phase traces and quality
parity.

Multicam trainer phase, same DeepView split at 256px, `16` frames, `8192`
splats, cached V-JEPA features:

| Variant | warmup/iters | total mean ms | total median ms | raster fwd ms | autograd backward total ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `v5_features` current config | 1/2 cache hit | 844.3 | 844.3 | 69.5 | 589.0 | `multicam256_f32_v5_features_warm1_iters2_cachehit.json` |
| stable `v6_refined_features` | 1/2 | 842.6 | 842.6 | 71.6 | 588.4 | `multicam256_f32_v6_refined_features_warm1_iters2.json` |
| `v6_refined_features_f32_reduce` | 1/2 | 883.1 | 883.1 | 78.8 | 562.5 | `multicam256_f32_f32_reduce_warm1_iters2.json` |
| stable `v6_refined_features` | 2/4 | 959.1 | 980.3 | 73.6 | 661.6 | `multicam256_f32_v6_refined_features_warm2_iters4.json` |
| `v6_refined_features_f32_reduce` | 2/4 | 1188.4 | 1147.8 | 89.5 | 656.0 | `multicam256_f32_f32_reduce_warm2_iters4.json` |

Warmed one-iteration detached backward probes at 256px:

| Variant | raster backward probe ms | loss/colorize probe ms | project probe ms | model probe ms | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 151.0 | 595.5 | 95.3 | 294.9 | `multicam256_f32_v6_refined_features_breakdown_warm1.json` |
| `v6_refined_features_f32_reduce` | 116.6 | 391.6 | 63.3 | 76.1 | `multicam256_f32_f32_reduce_breakdown_warm1.json` |

Read: the fork improves the isolated raster-backward probe at 256px, but the
full-step trainer timing is noisy and not a clear win over `v5_features` or
stable `v6_refined_features`. Do not promote the fork as a default trainer
variant from these phase traces. The next trainer-side timing tool should reduce
sample/encode/model noise or reuse the same decoded graph before making a
renderer-selection decision.

Seeded fixed-render graph, same 256px multicam config, `seed=0`, `warmup=2`,
`iters=4`. This mode samples/decodes once, then each measured iteration clones
detached Gaussian leaves and times only project/raster/loss/backward. It
excludes sample, encode, model backward, regularizers, optimizer, and W&B:

| Variant | colors trainable | total mean ms | total median ms | raster fwd ms | autograd backward total ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `v5_features` current config | true | 672.1 | 672.3 | 65.8 | 526.5 | `multicam256_f32_v5_features_fixed_render_seed0_warm2_iters4.json` |
| stable `v6_refined_features` | true | 674.4 | 673.3 | 66.7 | 527.9 | `multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4.json` |
| `v6_refined_features_f32_reduce` | true | 659.5 | 658.9 | 67.7 | 511.5 | `multicam256_f32_f32_reduce_fixed_render_seed0_warm2_iters4.json` |
| `v6_refined_features_f32_accum` | true | 665.2 | 665.5 | 73.7 | 511.8 | `multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4.json` |
| `v6_refined_features_f32_reduce` | false | 608.0 | 608.9 | 68.7 | 457.7 | `multicam256_f32_f32_reduce_fixed_render_freeze_colors_seed0_warm2_iters4.json` |
| `v6_refined_features_f32_accum` | false | 611.4 | 609.4 | 76.9 | 453.3 | `multicam256_f32_f32_accum_fixed_render_freeze_colors_seed0_warm2_iters4.json` |

Later same-session rerun after adding `f32_gradcache` trainer dispatch, same
seed/warmup/iters:

| Variant | total mean ms | total median ms | raster fwd ms | autograd backward total ms | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 1133.8 | 1122.8 | 89.7 | 888.6 | `multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json` |
| `v6_refined_features_f32_reduce` | 1165.6 | 1113.1 | 87.5 | 923.5 | `multicam256_f32_f32_reduce_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json` |
| `v6_refined_features_f32_accum` | 1060.0 | 1073.4 | 93.8 | 807.6 | `multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json` |
| `v6_refined_features_f32_gradcache` | 1008.0 | 966.4 | 82.6 | 765.6 | `multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4.json` |

Read: once model/sample jitter is removed, fork rankings are still sensitive to
same-session load. The first fixed-render window favored `f32_reduce` by a small
amount and showed the frozen-color path is materially faster (`~9.9%` total
versus stable), which supports camera-only or frozen-splat follow-up runs. The
later same-session rerun favored `f32_gradcache` by about `11%` total versus
stable and about `5%` versus `f32_accum`. This is still a timing gate only;
default promotion still needs heldout-quality parity.

Later same-session rerun after adding `f32_fixedbin`, same seed/warmup/iters:

| Variant | total mean ms | total median ms | raster fwd ms | autograd backward total ms | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 725.4 | 718.8 | 69.4 | 572.2 | `multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_accum` | 696.9 | 691.5 | 76.3 | 537.2 | `multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_gradcache` | 814.7 | 828.0 | 79.3 | 638.4 | `multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_fixedbin` | 696.1 | 687.7 | 68.8 | 544.4 | `multicam256_f32_f32_fixedbin_fixed_render_seed0_warm2_iters4.json` |

Read: fixedbin ties `f32_accum` and modestly beats stable in this fixed-render
trainer window, but it is not a clear replacement for all shapes. It remains a
targeted no-overflow candidate until a train/heldout-quality run confirms it.

Sampled-MPS-memory fixed-render rerun, same 256px multicam config,
`seed=0`, `warmup=1`, `iters=2`, `--memory-sample-interval-ms 1.0`. This
adds a background sampler around each measured iteration. It is still not a
Metal hardware capture, but it catches transient `torch.mps` allocation
pressure better than reading memory only after synchronized backward:

| Variant | total mean ms | raster fwd ms | autograd backward total ms | sampled peak current bytes | sampled peak driver bytes | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 673.9 | 68.0 | 525.8 | 1412745984 | 2529247232 | `multicam256_f32_v6_refined_features_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_fixedbin` | 649.8 | 68.5 | 500.6 | 1700988160 | 2529247232 | `multicam256_f32_f32_fixedbin_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_accum` | 670.2 | 75.5 | 513.0 | 1412745984 | 2529247232 | `multicam256_f32_f32_accum_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_reduce` | 667.4 | 75.8 | 510.5 | 1636237568 | 2529247232 | `multicam256_f32_f32_reduce_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_gradcache` | 649.3 | 67.0 | 502.3 | 1412745984 | 2529247232 | `multicam256_f32_f32_gradcache_fixed_render_sampled_memory_seed0_warm1_iters2.json` |

Read: `f32_gradcache` is the cleanest timing/memory candidate in this bounded
trainer-path row: it ties fixedbin for speed while keeping sampled current
allocation at the stable level. `f32_fixedbin` still has a real timing win, but
the larger fixed ID buffer shows up as higher sampled current allocation in the
trainer graph. `f32_reduce` also costs extra sampled memory here. This weakens
the case for fixedbin as a memory-pressure fix even though it remains useful as
a host/binning timing experiment.

Benchmark-only render/loss microbatch probe, same 256px multicam config,
`seed=0`, `warmup=1`, `iters=2`, sampled memory enabled. This uses
`--fixed-render-temporal-chunk-size` and `--fixed-render-backward-mode chunked`
to split each 16-frame view into chunks and backprop each chunk immediately.
It does not change trainer behavior yet:

| Variant | temporal chunk | chunks | total mean ms | raster fwd ms | autograd backward total ms | sampled peak current bytes | sampled peak driver bytes | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | full | 2 | 673.9 | 68.0 | 525.8 | 1412745984 | 2529247232 | `multicam256_f32_v6_refined_features_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| stable `v6_refined_features` | 8 | 4 | 770.9 | 93.7 | 579.4 | 567055872 | 1300316160 | `multicam256_f32_v6_refined_features_fixed_render_chunk8_chunked_backward_sampled_memory_seed0_warm1_iters2.json` |
| stable `v6_refined_features` | 4 | 8 | 910.1 | 147.3 | 628.8 | 365686016 | 1216430080 | `multicam256_f32_v6_refined_features_fixed_render_chunk4_chunked_backward_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_gradcache` | full | 2 | 649.3 | 67.0 | 502.3 | 1412745984 | 2529247232 | `multicam256_f32_f32_gradcache_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_gradcache` | 8 | 4 | 1257.7 | 149.2 | 930.4 | 567056896 | 1300316160 | `multicam256_f32_f32_gradcache_fixed_render_chunk8_chunked_backward_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_gradcache` | 4 | 8 | 1528.1 | 170.3 | 1090.7 | 365686016 | 1216446464 | `multicam256_f32_f32_gradcache_fixed_render_chunk4_chunked_backward_sampled_memory_seed0_warm1_iters2.json` |

Read: chunked render/loss backward is the strongest memory lever measured so
far. Chunk size 8 cuts sampled current allocation by about `60%` (`1.41 GB` to
`0.57 GB`). The stable row pays about `14%` wall time at chunk size 8 and a
larger hit at chunk size 4. The shared-background rerun made `f32_gradcache`
chunked backward much slower even though memory fell by the same amount, so do
not combine "gradcache is best batched" with "chunked is best memory" without a
fresh same-session trainer run. This is exactly the dense-surface pathology:
smaller frame chunks reduce `[T,H,W,F]` graph residency more than any current
kernel fork, but extra project/raster/loss/backward launches cost time and
interact with kernel variant rankings.

Fixed-render output parity, same 256px seeded clip:

| Baseline | Candidate | max feature diff | max alpha diff | max RGB diff | loss diff | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | `v6_refined_features_f32_reduce` | 0.0 | 0.0 | 0.0 | 0.0 | `multicam256_v6_vs_f32_reduce_fixed_render_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_accum` | 0.0 | 0.0 | 0.0 | 0.0 | `multicam256_v6_vs_f32_accum_fixed_render_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_gradcache` | 0.0 | 0.0 | 0.0 | 0.0 | `multicam256_v6_vs_f32_gradcache_fixed_render_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_fixedbin` | 0.0 | 0.0 | 0.0 | 0.0 | `multicam256_v6_vs_f32_fixedbin_fixed_render_parity_seed0.json` |
| stable `v6_refined_features`, heldout | `v6_refined_features_f32_reduce` | 0.0 | 0.0 | 0.0 | 0.0 | `multicam256_heldout_v6_vs_f32_reduce_fixed_render_parity_seed0.json` |
| stable `v6_refined_features`, heldout | `v6_refined_features_f32_accum` | 0.0 | 0.0 | 0.0 | 0.0 | `multicam256_heldout_v6_vs_f32_accum_fixed_render_parity_seed0.json` |
| stable `v6_refined_features`, heldout | `v6_refined_features_f32_gradcache` | 0.0 | 0.0 | 0.0 | 0.0 | `multicam256_heldout_v6_vs_f32_gradcache_fixed_render_parity_seed0.json` |
| stable `v6_refined_features`, heldout | `v6_refined_features_f32_fixedbin` | 0.0 | 0.0 | 0.0 | 0.0 | `multicam256_heldout_v6_vs_f32_fixedbin_fixed_render_parity_seed0.json` |

Read: these parity rows catch forward-output drift on the trainer path without
running a long train. They do not replace heldout-quality training parity, but
they are a useful pre-flight gate before spending W&B/GPU time.

Fixed-render gradient parity, same seeded 128px train graph:

| Baseline | Candidate | Target | max feature diff | max alpha diff | loss diff | max sequence grad diff | max colorize grad diff | Artifact |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | `v6_refined_features_f32_reduce` | train | 0.0 | 0.0 | 0.0 | 8.15e-10 | 0.0 | `multicam128_train_v6_vs_f32_reduce_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_reduce` | heldout | 0.0 | 0.0 | 0.0 | 8.15e-10 | 0.0 | `multicam128_heldout_v6_vs_f32_reduce_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_accum` | train | 0.0 | 0.0 | 0.0 | 4.07e-10 | 0.0 | `multicam128_train_v6_vs_f32_accum_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_accum` | heldout | 0.0 | 0.0 | 0.0 | 9.60e-10 | 0.0 | `multicam128_heldout_v6_vs_f32_accum_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_gradcache` | train | 0.0 | 0.0 | 0.0 | 8.15e-10 | 0.0 | `multicam128_train_v6_vs_f32_gradcache_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_gradcache` | heldout | 0.0 | 0.0 | 0.0 | 7.86e-10 | 0.0 | `multicam128_heldout_v6_vs_f32_gradcache_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_fixedbin` | train | 0.0 | 0.0 | 0.0 | 8.15e-10 | 0.0 | `multicam128_train_v6_vs_f32_fixedbin_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `v6_refined_features_f32_fixedbin` | heldout | 0.0 | 0.0 | 0.0 | 1.14e-09 | 0.0 | `multicam128_heldout_v6_vs_f32_fixedbin_fixed_render_grad_parity_seed0.json` |

Read: the gradient gate compares decoded sequence gradients (`xyz`, `scales`,
`quats`, `opacities`, `rgbs`) and colorize-MLP parameter gradients on the
trainer render/loss path. It validates the forked backward math at 128px, but
still does not prove trained heldout quality or long-run optimizer behavior.

## Metal Performance Model

Apple's compute docs model a kernel as threads in a grid, grouped into
threadgroups that can share memory. A 16x16 tile maps naturally to 256 threads,
which is the shape these fast-mac kernels use.

SIMD groups matter. Metal exposes a pipeline `threadExecutionWidth`, and Apple
recommends making threadgroup dimensions multiples of that width. The current
fast-mac forks still assume `GSP_SIMD_WIDTH=32`, matching observed M-series
behavior but not a portable contract.

The reduction fork uses a two-level reduction pattern: each SIMD group first
does `simd_sum`, lane 0 writes a partial into threadgroup scratch, then SIMD
group 0 reduces those group partials before issuing one atomic per output
field. This is the core difference between direct per-pixel/per-channel
feature atomics and the forked color-gradient path.

Threadgroup memory is only a win when it reduces global traffic or contention.
Apple's `setThreadgroupMemoryLength` docs also make the hard limit explicit:
all static and dynamic threadgroup allocations must fit the device/pipeline
limits. For this codebase, staging a `[GSP_CHUNK, F]` feature slab is not free;
it competes with `shared_ids`, chunk params, reduction scratch, and occupancy.
The `f32_reduce` fast-backward scratch budget includes `shared_ids[GSP_FAST_CAP]`,
`sh_means`, `sh_conics`, `sh_opacities`, `partial0/1/2`, and
`partial_features`. The rejected staging fork adds `sh_colors[GSP_CHUNK,
GSP_FEATURE_CAP]`, which is the concrete reason the "save global color reads"
idea can lose on Metal.

Barriers must remain uniform across the threadgroup. The reverse backward loops
already carry a comment about this: per-pixel stop state may mask contribution,
but it must not change how many barriers a thread executes.

Occupancy is not a goal by itself. Apple's Xcode occupancy docs say low
occupancy may mean resource exhaustion, small grids, or too-simple shaders, and
high occupancy can still be inefficient if cache/memory traffic thrashes. For
F32 feature splatting, use timing plus memory/counter captures before trusting
an occupancy-only interpretation. The code knobs that feed occupancy here are
static threadgroup scratch, private/thread arrays such as `feature_accum`,
barrier density, tile/threadgroup size, and active-tile grid size.

The host ABI is deliberately simple. `meta_i32` and `meta_f32` are device
tensors passed as constant buffers; `MetaF32.bg` is padded to
`GSP_FEATURE_CAP`; and `MetaI32.reserved0` currently signals "skip color
gradients". The bridge uses direct `setArg` calls for each buffer rather than
Metal argument buffers, so shader variants should keep argument lists stable
unless the host binding is updated in the same fork.

The feature lookup prototype validates one algebraic escape hatch without new
compositing math: if per-splat F-dimensional features can be represented as
`feature_weights @ lookup` with compact dimension K, then the rasterizer can
splat K channels with zero compact background and reconstruct
`features = compact @ lookup + (1 - alpha) * background` afterward. A tiny MPS
parity check matched direct full-feature rendering and gradients to MPS noise.
This is not yet the true sparse-ID kernel; the current ID-shaped helper
densifies IDs to `[G,K]`, and the final `[H,W,F]` tensor still exists after the
lookup. A bounded synthetic probe at `128px/G2048/F32` showed lookup timing wins
for `K in {4,8,16}` at both `B=4` and `B=16`, but sampled MPS allocation was
mixed. A follow-up background-sampled run at `128px/B16/G8192/F32` showed lookup
still faster and lower sampled current allocation for K=4/8/16; this is better
evidence than the after-backward allocation read, but still not a Metal hardware
memory capture.

Relevant official docs:

- https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)
- https://developer.apple.com/documentation/metal/compute-passes
- https://developer.apple.com/documentation/metal/setting-resource-storage-modes
- https://developer.apple.com/metal/capabilities/
- https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon
- https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy

Official-doc implications for these forks:

- Treat `threadExecutionWidth` and `maxTotalThreadsPerThreadgroup` as runtime
  pipeline facts, not constants from one machine. The feature tables report
  theoretical limits, but Apple also documents that the actual threadgroup max
  is pipeline-specific. The current kernels use 16x16 threadgroups and assume
  `GSP_SIMD_WIDTH=32`; keep that assumption guarded by local validation before
  generalizing to every Apple GPU family.
- Divergence inside a SIMD group is expensive by construction because divergent
  branches serialize the paths for that SIMD group. In this rasterizer, the
  dangerous divergence is not just `if (contributes)`: it is branch structure
  around barrier count, overflow/fallback paths, and per-feature loops. Any new
  fork must keep barrier execution uniform across all threads in a threadgroup.
- Threadgroup memory is a shared resource with alignment and per-pipeline
  limits. The feature-table row for threadgroup-memory alignment is 16 bytes,
  and the table notes that imageblock/threadgroup allocations share a total
  budget on relevant families. That matches the measured staging result:
  adding `[GSP_CHUNK,F]` scratch can reduce global reads while still losing
  occupancy or resident threadgroups.
- `dispatchThreads(..., threadsPerThreadgroup:)` supports arbitrary-sized grids
  on devices with nonuniform threadgroups. That helps boundary handling, but it
  does not solve the real pressure here because our hot path is one thread per
  dense output pixel and the feature dimension lives inside each thread's loop
  and private state.
- Apple silicon defaults many resources to shared storage, while GPU-only
  temporary resources should prefer private or memoryless storage when the app
  owns allocation. In this PyTorch extension most hot inputs/outputs are tensor
  buffers owned by PyTorch, so storage-mode tuning is mainly relevant for
  fork-owned temporary buffers such as fixedbin IDs or future scratch buffers.
- Compute encoders can encode multiple dispatches in one pass, but the encoder
  itself is not the place to hide Python/PyTorch graph pressure. The measured
  trainer bottleneck is mostly `autograd_backward_total`, dense feature
  surfaces, and MPS allocations; a lower-level Metal command-encoding trick is
  only useful if it removes an allocation, readback, or dispatch from that
  graph.

## Bottlenecks

1. Feature-gradient atomics:
   Stable `v6_refined_features` issues per-pixel/per-channel atomics into
   `g_colors`. The reduction fork collapses those atomics with SIMD/threadgroup
   reductions and is the biggest measured win. This does not remove all
   atomics: binning still uses tile-count/cursor atomics, and backward still
   accumulates means/conics/opacities across tiles. The fork specifically
   reduces the most expensive F-channel color-gradient atomics.

   This also explains the corrected `F=3` feature-path red flag. With the
   trainer-like `64/2048` cap, feature F3 forward has RGB-class speed. Stable
   feature F3 backward remains slower because it uses generic per-channel
   gradient handling, while RGB `v6` and the reduction fork use `float3`
   reduction-style arithmetic.

2. Dense feature surfaces:
   `B=16,H=W=512,F=32` is about `536.9 MB` for one dense fp32 feature image
   tensor. Output, grad output, alpha, loss graphs, and retained activations
   make batch much more expensive than the same projected Gaussian count in
   one batch item.

3. Forward global output traffic:
   The generic feature forward writes the dense output inside every
   Gaussian/channel contribution. The accumulation fork reduces those writes
   and improves eval-forward, but thread-local storage/reg pressure makes the
   win shape-dependent. It wins several F64 batch-heavy rows and loses the
   `512px/B4/F64` row.

4. Threadgroup-memory pressure:
   The staging fork showed the failure mode directly: staging `[chunk,F]`
   features lowers some device reads but costs enough occupancy/scratch that
   training total regresses.

5. Active-tile overhead:
   Active mode is not globally faster. It can pay dense output/background
   initialization and sparse launch overhead. Promote only from profile fields
   such as active fraction, overflow count, and stop ratios.

6. CPU/GPU synchronization:
   Avoid `.item()`, `.tolist()`, readback-heavy hardware paths, and ad hoc
   per-step CPU parsing in the training path. Use benchmark scripts that
   synchronize only around measured forward/backward windows. The fixedbin fork
   is the current no-overflow test of this thesis: it removes the exact-length
   `binned_ids` sizing sync at the cost of a fixed per-tile ID buffer.

7. Memory coalescing and layout:
   Feature tensors are laid out as `[B,H,W,F]` for dense images and `[BG,F]` for
   splat colors. Each lane walks contiguous channels for its own pixel/splat,
   but adjacent lanes reading the same feature channel are separated by `F`
   floats. That access shape matters for `grad_features`, `out_features`, and
   repeated `colors` reads. The accumulation fork reduces global output writes,
   but it does not change the underlying dense image layout.

## Safe Benchmark Contract

Use the safe runner for exploratory sweeps:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/benchmarks/benchmark_matrix.py \
  --dry-run --variants f32_reduce,f32_accum,f32_fixedbin --include-stable-baseline --backward
```

Rules:

- Start with `128px`, `G=1024`, `B=1`, `F=32`, `warmup>=1`, `iters=2`,
  `GSP_CHUNK=64`, and `GSP_FAST_CAP=2048` unless deliberately stress-testing
  overflow/fallback behavior.
- Use `--timeout-s` for every matrix run.
- For `F>64`, pass `--feature-caps` explicitly and start at tiny sizes. The
  compile-time `GSP_FEATURE_CAP` controls `MetaF32.bg` padding and reduction
  scratch, so increasing it is a shader-resource change, not just a data-size
  change.
- Avoid `overflow_stress` unless deliberately testing overflow fallback.
- Run benchmarks sequentially; do not launch multiple GPU benchmarks in
  parallel.
- For the known pressure probe, `512px`, `B=16`, `G=8192`, `F=32`,
  `medium_sigma_3_8`, `iters<=5` has been safe locally. Do not jump from there
  to 4K/64K without an explicit reason.
- Always compare an experiment against stable and the current best fork in the
  same session where possible.
- Do not lower `GSP_FAST_CAP` to `512` for primary throughput claims; that can
  force a fallback-heavy regime and make feature/RGB comparisons misleading.
- Treat `warmup=0` output as shader-compile smoke only, not performance data.

Batch launch model: one Metal threadgroup renders one tile. `flatten` increases
the grid size in one launch across batch tiles, `serial` repeats bin/render
launches per batch item, and `auto` chunks by configured tile/Gaussian limits.
Batch size can therefore hurt through both dense `[B,H,W,F]` tensors and launch
strategy, not just Gaussian count.

## Profiling Workflow

The local `--profile` path reports logical raster stats such as pair counts,
overflow tiles, and active-tile fractions. It is not a Metal counter capture.
Before promoting another kernel, pair benchmark JSON with an Xcode GPU/Metal
capture when possible and check occupancy, memory bandwidth, cache behavior,
atomic pressure, threadgroup utilization, and whether threadgroup scratch or
private arrays are reducing resident threadgroups. Use local profile rows to
choose which shapes to capture; do not treat them as hardware-counter evidence.

Useful saved smoke artifacts:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_safe_matrix_dry_run.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_safe_matrix_warm_smoke.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_safe_batch_strategy_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_baseline_vs_reduce.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_freeze_colors_matrix_dry_run.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_default_cap2048_dry_run.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_default_cap2048_smoke.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_reduce_trainable_rerun.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_reduce_freeze_colors.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f3_feature_baseline_vs_reduce.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f3_feature_cap2048.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_feature_cap2048.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_accum_cap2048_stdout.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_cap2048.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_baseline_vs_reduce_cap2048.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_cap2048_freeze_colors.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_accum_cap2048_stdout.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_accum_cap2048_freeze_colors_stdout.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_cap_profile.jsonl`
- `benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_current_warm4.json`
- `benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_f32_reduce_warm4.json`
- `benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_current_breakdown1.json`
- `benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_f32_reduce_breakdown1.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_v5_features_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_f32_reduce_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_v5_features_breakdown_warm1.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_f32_reduce_breakdown_warm1.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v5_features_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v5_features_warm1_iters2_cachehit.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_breakdown_warm1.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_breakdown_warm1.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_f32_reduce_fixed_render_smoke.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v5_features_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_freeze_colors_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_freeze_colors_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_reduce_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_accum_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_heldout_v6_vs_f32_reduce_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_heldout_v6_vs_f32_accum_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_reduce_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_reduce_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_accum_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_accum_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_f64_b8_b16_variant_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b8_variant_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_variant_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_freeze_colors_variant_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b4_variant_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b8_variant_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_variant_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b16_profile_forward.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b16_freeze_colors_variant_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b16_trainable_forks.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_accum64_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_accum64_confirm_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b8_accum64_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_gradcache_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_block4_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_block4_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_gradcache_confirm_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_f32_b16_fixedbin_smoke_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_fixedbin_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_fixedbin_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_128_b4_g2048_f32_k4_8_16.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_128_b16_g2048_f32_k4_8_16.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_sampled_peak_128_b16_g2048_f32_k4_8_16.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_sampled_peak_128_b16_g8192_f32_k4_8_16.jsonl`
- `benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_gradcache_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_heldout_v6_vs_f32_gradcache_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_gradcache_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_gradcache_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json`
- `benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_fixedbin_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_heldout_v6_vs_f32_fixedbin_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_fixedbin_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_fixedbin_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_fixedbin_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_fixedbin_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_chunk8_chunked_backward_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_chunk4_chunked_backward_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_chunk8_chunked_backward_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_chunk4_chunked_backward_sampled_memory_seed0_warm1_iters2.json`

Small batch-strategy smoke, `128x128`, `G=1024`, `F=32`,
`case=medium_sigma_3_8`, `warmup=1`, `iters=2`, fwd+bwd:

| B | Strategy | Total mean ms | Forward ms | Backward ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | flatten | 42.4 | 29.4 | 13.0 |
| 1 | serial | 44.5 | 29.6 | 15.0 |
| 4 | flatten | 39.0 | 17.2 | 21.7 |
| 4 | serial | 83.1 | 52.8 | 30.3 |
| 16 | flatten | 109.5 | 35.7 | 73.8 |
| 16 | serial | 350.2 | 228.9 | 121.3 |

Read: for small warmed cases, flatten beats serial decisively once `B > 1`.
This does not contradict the high-pressure memory concern; it says serial
launch overhead is not the fix at small resolution.

Tiny trainer phase trace, checked-in 64px F32 config, `warmup=2`, `iters=4`:

| Variant | Total mean ms | Raster fwd ms | Autograd backward total ms | Read |
| --- | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 363.0 | 17.3 | 172.5 | noisy baseline |
| `v6_refined_features_f32_reduce` | 292.3 | 13.2 | 150.1 | directionally better total |

One-iteration detached backward probes were inconsistent at this tiny size:
stable reported raster backward `13.2ms`, while f32_reduce reported `20.9ms`.
Conclusion: the opt-in dispatch is runtime-valid, but this 64px trainer probe is
not a sufficient promotion gate. Use a larger phase trace before claiming trainer
speedup.

## Next Forks To Try

1. Accumulation fork trainer dispatch:
   Done as an opt-in trainer variant. The 256px fixed-render gate did not beat
   `f32_reduce`, so keep it as a synthetic/eval candidate rather than the
   primary trainer candidate.

2. Zero-background active fill:
   In active F32 mode, avoid repeated full-channel fills when background is all
   zero. This should be isolated from direct fast mode.

3. No-color-gradient allocation cleanup:
   Done in `v6_refined_features_f32_reduce`: when Python marks color gradients
   skipped, C++ returns an empty `g_colors` tensor and passes a 1-float
   placeholder to the kernel. This is useful for camera-only or frozen-feature
   follow-up runs. The safe matrix runner supports this path with
   `--freeze-colors` and skips stable-baseline rows rather than mutating the
   stable benchmark script.

4. Trainer microbatch/framewise backward:
   Benchmark-only probe is now in `src/benchmarks/trainer_phase_benchmark.py`
   via `--fixed-render-temporal-chunk-size` and
   `--fixed-render-backward-mode chunked`. At 256px, chunk size 8 reduced
   sampled current allocation by about `60%`. The stable row paid an `~14%`
   timing cost; `f32_gradcache` lost its batched-mode speed edge in the
   shared-background chunked rerun. This should be wired into the real multicam
   trainer only after a parity/quality smoke, because it changes backward
   accumulation order and must preserve shared train-background semantics.

5. Better trainer-phase isolation:
   Done in `src/benchmarks/trainer_phase_benchmark.py` via
   `--fixed-render-graph`, optional `--fixed-render-freeze-colors`, and
   opt-in `--memory-sample-interval-ms`. Use this before comparing renderer
   variants in the trainer context; full-step timing remains useful, but it
   includes sample/encode/model/optimizer noise.

6. Fixed-render parity gate:
   Done in `src/benchmarks/fixed_render_variant_parity.py`. Run it before long
   trainer jobs to verify feature/alpha/RGB/loss parity between stable and a
   fork. The current gate covers 256px train/heldout forward parity and 128px
   train/heldout gradient parity for `f32_reduce`, `f32_accum`, `f32_gradcache`,
   and `f32_fixedbin`.

7. Backward grad-feature cache / `float4` channel blocks:
   Tried as `v6_refined_features_f32_gradcache` for `F<=32`. It passes local
   feature/alpha gates and wins `256px/B16/F32`, but loses `256px/B32/F32` and
   `512px/B16/F32`. The private 32-float cache is not a promotion path for the
   current target. If revisiting, try a lower-register `float4` block cache
   rather than a full F32 vector.

8. F64 local accumulation:
   Tried as `v6_refined_features_f64_accum64`. It raises
   `GSP_LOCAL_ACCUM_CAP` to `64` and passes F64 gradient/alpha gates, but it did
   not beat existing forks on `256px/B16/F64` or `512px/B8/F64`. Treat the
   naive cap-64 private-array version as a negative result. If revisiting, try
   a two-pass/two-block design that keeps private pressure lower rather than a
   single 64-float thread array.

9. Active/overflow accumulation:
   Open. `f32_accum` optimizes direct fast forward only. Active and overflow
   forward paths still mutate global `out_features` per contribution. Port
   local accumulation there only after profile rows show active/overflow paths
   matter for the target trainer shape.

10. Fixed-cap fast binning:
    Done as `v6_refined_features_f32_fixedbin` for the no-overflow fast path.
    It trades an exact-size `binned_ids` sync/allocation for a fixed
    `[tile_count, max_fast_pairs]` int32 buffer and raises on overflow rather
    than falling back. It wins the target synthetic `512px/B16/F32` row and
    ties `f32_accum` on one 256px trainer fixed-render graph, but the sampled
    memory rerun shows higher current allocation than stable/gradcache. It also
    loses some `256px` rows and costs about `128 MiB` for IDs at
    `512px/B16/tile16/cap2048`. Keep it opt-in until heldout-quality training
    proves the no-overflow path is safe over optimizer time.

11. Compact-basis feature lookup:
    Prototype built as `v6_feature_lookup_experiment`. It passes a tiny MPS
    direct-vs-lookup parity check for features, alpha, loss, and gradients
    through means/conics/compact weights/lookup/opacities. A bounded synthetic
    benchmark at `128px/G2048/F32` showed timing wins for K=4/8/16 at B=4 and
    B=16. A sampled-peak follow-up at `128px/B16/G8192/F32` showed timing wins
    and lower sampled current allocation for K=4/8/16, but the final
    `[B,H,W,F]` tensor still exists and this is not an Xcode/Metal memory
    capture. Do not wire it into trainer dispatch until a fixed-render trainer
    profile shows the branch solves the actual F32 pressure.
