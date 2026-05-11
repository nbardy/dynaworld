# Fast-Mac Feature Kernel Catalog

Date: 2026-05-07

Scope: F32/F64 feature splatting on local MPS, mainly
`B=16,G=8192,F=32` with trainer-like `GSP_CHUNK=64,GSP_FAST_CAP=2048`.
Stable baselines stayed untouched; all kernel experiments were copied forks.

## Current Best Read

The largest safe kernel win is still the `f32_reduce` family: it fixes the big
F-channel color-gradient atomic bottleneck without changing render semantics.
As of 2026-05-08, the best opt-in trainer candidate is
`v9_features_gradcache_zero_bg`, a copied `f32_gradcache` fork with the
zero-feature-background tail skip added. It has trainer fixed-render parity
against stable `v6_refined_features` and is slightly faster than plain
`f32_gradcache` at 512px in the direct benchmark, but the win is incremental,
not a new speed class.

The best semantic speed knob now has a stable-training answer for the current
256px goodset F32 `v5_features` setup: `alpha_threshold = 1/128` improved
heldout PSNR/SSIM/L1 over default and cut W&B runtime by 29.4%. `1/96` was
faster and still above default quality, but lower than `1/128`; `1/64` was the
fastest but source-view quality dropped and heldout PSNR was essentially
default. This validates `1/128` for the stable goodset trainer, not automatic
promotion of any experimental kernel fork.

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
| Alpha threshold `1/64` | `431.7ms -> 360.4ms` | 512px B16/G8192/F32 | **16.5% faster total** | synthetic speed win; trainer A/B says too aggressive for promotion |
| Alpha threshold `1/64` | `326.5ms -> 261.5ms` | 256px B16/G8192/F32 | **19.9% faster total** | fastest stable-trainer A/B row, but not best quality |
| Transmittance threshold `1e-2` | `431.7ms -> 404.5ms` | 512px B16/G8192/F32 | **6.3% faster total** | weaker; tile max stop remains |
| `zero_bg` tail skip | `610.7ms -> 556.7ms` | 512px B16/G8192/F32 active off | **8.8% faster total** | bounded fork, safe version keeps explicit pixel zero/write |
| `zero_bg` tail skip | `386.0ms -> 361.1ms` | 256px B16/G8192/F32 active off | **6.5% faster total** | bounded fork |
| `v9 gradcache + zero_bg` | `367.4ms -> 364.0ms` | 512px B16/G8192/F32 active off | **0.9% faster total**, **2.9% faster forward** vs `f32_gradcache` | parity-safe combined candidate |
| `v9 gradcache + zero_bg` | `291.2ms -> 291.5ms` | 256px B16/G8192/F32 active off | tied/slightly slower | no 256px win in direct matrix |
| `v10 hostmeta` | `273.2ms -> 271.2ms` | 256px B16/G8192/F32 same-session vs v9 | **0.7% faster total** | small bridge sync win |
| `v10 hostmeta` | `367.9ms -> 366.1ms` | 512px B16/G8192/F32 same-session vs v9 | **0.5% faster total** | small bridge sync win |
| `v11 hostmeta+fixedbin` | `273.2ms -> 270.5ms` | 256px B16/G8192/F32 same-session vs v9 | **1.0% faster total**, **3.5% faster forward** | best direct row, no-overflow only |
| `v11 hostmeta+fixedbin` | `367.9ms -> 364.0ms` | 512px B16/G8192/F32 same-session vs v9 | **1.0% faster total**, **4.0% faster forward** | best direct row, no-overflow only |
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

## 2026-05-08 Stable Trainer Alpha-Threshold A/B

Setup: 256px DeepView `03_Dog` goodset F32 feature splatting, train
`camera_0006`/`camera_0014`, heldout `camera_0005`,
`feature_variant = "v5_features"`, 8192 splats, 16 frames, 250 steps. Only
`render.alpha_threshold`, `render.fast_mac.alpha_threshold`, and label/checkpoint
fields differed across configs. Real W&B runs logged both multicam diagnostic
video grids.

| Threshold | Config suffix | W&B | Runtime | Speedup vs default | Heldout PSNR / SSIM / L1 | Train PSNR mean | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `1/255` | `alphaab_alpha1_255` | [`j9fkocvj`](https://wandb.ai/nbardy/dynaworld/runs/j9fkocvj) | 26.01 min | 0.0% | 12.7536 / 0.1716 / 0.1729 | 19.4952 | default control |
| `1/128` | `alphaab_alpha1_128` | [`hru1yv0t`](https://wandb.ai/nbardy/dynaworld/runs/hru1yv0t) | 18.37 min | 29.4% | **13.6248 / 0.1922 / 0.1561** | 19.4875 | promote for this stable setup |
| `1/96` | `alphaab_alpha1_96` | [`dsq6u3wq`](https://wandb.ai/nbardy/dynaworld/runs/dsq6u3wq) | 16.60 min | 36.2% | 13.2942 / 0.1838 / 0.1599 | 20.4087 | faster, still above default, but lower heldout than `1/128` |
| `1/64` | `alphaab_alpha1_64` | [`obclxj4w`](https://wandb.ai/nbardy/dynaworld/runs/obclxj4w) | 14.27 min | 45.1% | 12.7667 / 0.1806 / 0.1712 | 18.2031 | fastest, but source quality drops and heldout PSNR is near default |

Qualitative read from the final `Multicam_Feature_GT_Render_ByCamera_Grid_Video`
and `Multicam_GT_Splat_Alpha_Feature_Grid_Video` artifacts: all rows are still
blurry and under-detailed, but `1/128` and `1/96` preserve the same camera-row
structure as default without obvious support holes. `1/64` keeps broad coverage
but loses source-view fidelity and looks more smeared, matching the train PSNR
drop. Use heldout quality as selector: `1/128` is the safe stable-training
threshold; `1/96` is a speed/quality tradeoff candidate; `1/64` remains a
throughput stress point rather than a promoted default.

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

1. Prototype per-pixel backward stop handling. This is the real top-p path,
   but it needs a different scheduling design so barriers remain safe.
2. Wire compact lookup into trainer fixed-render. It is the biggest remaining
   speed/memory idea if quality holds.
3. Try a longer trainer A/B with `v11_features_gradcache_zero_bg_hostmeta_fixedbin`
   and a no-overflow profile guard. It is the best current opt-in shader, but
   the fixedbin memory/no-overflow tradeoff should be proven over optimizer
   time before config promotion.
4. Only after those, try a very narrow F32-specialized vector/unrolled backward
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
- Stable-trainer alpha A/B configs:
  `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_{255,128,96,64}.jsonc`
- Stable-trainer alpha A/B W&B runs:
  [`j9fkocvj`](https://wandb.ai/nbardy/dynaworld/runs/j9fkocvj),
  [`hru1yv0t`](https://wandb.ai/nbardy/dynaworld/runs/hru1yv0t),
  [`dsq6u3wq`](https://wandb.ai/nbardy/dynaworld/runs/dsq6u3wq),
  [`obclxj4w`](https://wandb.ai/nbardy/dynaworld/runs/obclxj4w)
- v9 combined fork note:
  `agent_notes/loose_notes/2026-05-08_16-55-00_v9_feature_shader_benchmark.md`
- Alpha/background contract note:
  `agent_notes/loose_notes/2026-05-08_16-25-46_alpha_bg_bleed_features.md`
- v9 direct benchmark artifacts:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-08_feature_variants_B16_G8192_F32_{256,512}_active_off.jsonl`
- v9 trainer fixed-render parity:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-08_fixed_render_parity_v6_refined_features_vs_v9_features_gradcache_zero_bg_256.json`
- v10/v11 direct and trainer artifacts:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-08_v9_v10_v11_{256,512}_B16_G8192_F32.jsonl`,
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-08_fixed_render_parity_v9_vs_v{10,11}_*_256_train.json`,
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-08_trainer_fixed_render_v{9,10,11}_*_256_seed0_warm1_iters3.json`
- Experimental Metal comments:
  `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/csrc/metal/gsplat_v6_refined_features_f32_reduce_kernels.metal`
