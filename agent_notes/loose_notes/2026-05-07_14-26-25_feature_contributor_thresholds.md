# Feature Splatting Contributor Counts And Thresholds

## Context

Follow-up after the F32 feature-splatting backward work. The user asked three
specific things:

- check the average K/contributor count
- verify whether we already have an opacity/top-p-like threshold
- try the next fork/measurement path, and explain whether "fusing downstream
  projection/loss" is a real optimization

Kept stable renderer variants untouched. The only code change was benchmark and
diagnostic tooling under the already-experimental
`v6_refined_features_f32_reduce` fork.

## Subagent Findings

- Existing profiles report tile-level candidate and stop-prefix metrics, not
  exact per-pixel contributor counts. `mean_stop_count` is the tile max prefix
  used by backward, not the average number of alpha-passing splats.
- Existing pruning already has two knobs:
  - `alpha_threshold`: absolute low-alpha support cutoff before binning and
    per-pixel `eval_alpha` rejection.
  - `transmittance_threshold`: front-to-back alpha early-out, top-p-like only
    in the compositing sense.
- Backward is still channel-scaled after the reduction fork because geometry
  terms need a dot over `grad_features[pix, :] * colors[g, :]`, and trainable
  splat features need `g_colors[g, f]`. The current `f32_reduce` fork reduced
  global atomic contention, but it did not remove the F loop.
- The "fuse downstream projection/loss" idea is not a drop-in optimization for
  current configs because `FeatureToColor` uses `pre_norm` and sigmoid around a
  `Conv2d(F, 3, 1)`. Projection commutes with rasterization only for an affine
  colorizer without pixelwise normalization/nonlinear hidden layers. Treat that
  as a new compact/linear architecture, not a safe kernel optimization.
- The already-created `v6_refined_features_f32_gradcache` fork is the live
  version of the "cache/reuse grad_features" idea. The block4 and stage forks
  remain negative examples: extra threadgroup/private pressure can erase the
  intended channel-load savings.

## Tooling Added

- `variants/v6_refined_features_f32_reduce/benchmarks/benchmark_mps.py`
  now exposes:
  - `--alpha-threshold`
  - `--transmittance-threshold`
- Added
  `variants/v6_refined_features_f32_reduce/benchmarks/profile_contributors.py`.
  It samples pixels, reuses the same synthetic inputs and binned candidate list,
  and estimates:
  - tile candidates
  - checked depth prefix
  - actual alpha-passing contributors
  - final transmittance
  - early-stop fraction

This is still a sampled CPU-side diagnostic, not a full-frame Metal histogram.
It is sufficient for the current K-distribution question without adding a new
kernel.

## Commands

Contributor profiles:

```bash
PYTHONDONTWRITEBYTECODE=1 GSP_TILE_SIZE=16 GSP_CHUNK=64 GSP_FAST_CAP=2048 \
  .venv/bin/python third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/benchmarks/profile_contributors.py \
  --height 512 --width 512 --gaussians 8192 --batch-size 16 --feature-dim 32 \
  --case medium_sigma_3_8 --seed 0 --samples 2048 --sample-seed 123 \
  --batch-strategy flatten --json
```

Timing profiles:

```bash
PYTHONDONTWRITEBYTECODE=1 GSP_TILE_SIZE=16 GSP_CHUNK=64 GSP_FAST_CAP=2048 \
  .venv/bin/python third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/benchmarks/benchmark_mps.py \
  --height 512 --width 512 --gaussians 8192 --batch-size 16 --feature-dim 32 \
  --case medium_sigma_3_8 --seed 0 --batch-strategy flatten --active-policy off \
  --warmup 1 --iters 3 --backward --profile --json
```

Also reran `v6_refined_features_f32_gradcache` at 256/512 B16/G8192/F32 to
confirm the existing fork remains useful on the same local machine.

## Results

Sampled contributor counts, `B=16`, `G=8192`, `F=32`, 2048 sampled pixels:

| Shape / threshold | tile candidates mean / p95 | checked prefix mean / p95 | alpha contributors mean / p95 | early-stop fraction |
| --- | ---: | ---: | ---: | ---: |
| 512px default | 62.9 / 78 | 62.9 / 78 | 26.6 / 36 | 0.0% |
| 512px alpha `1/64` | 50.4 / 64 | 50.4 / 64 | 18.8 / 26 | 0.0% |
| 512px trans `1e-2` | 62.9 / 78 | 61.9 / 77 | 26.2 / 35 | 10.2% |
| 256px default | 241.3 / 286 | 182.3 / 246 | 77.9 / 101 | 89.3% |
| 256px alpha `1/64` | 194.1 / 230 | 149.5 / 199 | 56.2 / 72 | 87.5% |

Timing, same synthetic row:

| Shape / threshold | forward ms | backward ms | total mean ms | profile mean stop |
| --- | ---: | ---: | ---: | ---: |
| 512px default | 100.4 | 331.3 | 431.7 | 62.9 |
| 512px alpha `1/64` | 78.3 | 282.1 | 360.4 | 50.3 |
| 512px trans `1e-2` | 93.5 | 310.9 | 404.5 | 62.9 |
| 256px default | 85.7 | 240.8 | 326.5 | 224.5 |
| 256px alpha `1/64` | 63.1 | 198.5 | 261.5 | 183.0 |

Gradcache confirmation:

| Shape | reduce total / fwd / bwd | gradcache total / fwd / bwd |
| --- | ---: | ---: |
| 512px B16/G8192/F32 | 431.7 / 100.4 / 331.3 | 388.1 / 95.4 / 292.6 |
| 256px B16/G8192/F32 | 326.5 / 85.7 / 240.8 | 286.8 / 74.0 / 212.8 |

## Read

The K count is not tiny. At the 512px target row the average sampled pixel has
about 27 real contributors, while 256px has about 78. That makes a hard
"top 3-4 splats" approximation risky unless it is its own quality experiment.

`transmittance_threshold` is the knob closest to top-p, but it is not the best
speed lever for backward because the backward pass uses the tile max prefix. On
the 512px row, raising it to `1e-2` gave only a modest speedup and did not move
profile mean stop. `alpha_threshold` is more direct: it reduces support before
binning, contributor counts, and stop counts. The `1/64` row is a plausible
short heldout-quality A/B.

For #3, the existing gradcache fork remains the best current forked candidate
for the grad-feature reuse idea. A new vectorized F32-specialization fork is
still plausible, but prior `block4` and staging failures argue for keeping it
very narrow. Do not replace stable configs with any of these until heldout
quality parity passes.

## Artifacts

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_sampled_contributors_default.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_sampled_contributors_alpha1_64.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_sampled_contributors_trans1e_2.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_b16_g8192_f32_reduce_sampled_contributors_default.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_b16_g8192_f32_reduce_sampled_contributors_alpha1_64.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_threshold_default_timing.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_threshold_alpha1_64_timing.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_threshold_trans1e_2_timing.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_b16_g8192_f32_reduce_threshold_default_timing.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_b16_g8192_f32_reduce_threshold_alpha1_64_timing.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_gradcache_confirm_timing.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_b16_g8192_f32_gradcache_confirm_timing.json`

## Closeout Catalog

Added curated catalog:
`research_notes/fast_mac_feature_kernel_catalog_2026-05-07.md`.

Most important percent reads:

- `f32_reduce` vs stable at 512px B16/G8192/F32:
  `1001.9ms -> 449.2ms`, `55.2%` faster total and `64.3%` faster backward.
- `f32_accum` vs `f32_reduce` on the same row:
  `449.2ms -> 388.1ms`, `13.6%` faster total.
- `f32_gradcache` latest local confirm:
  `431.7ms -> 388.1ms` at 512px and `326.5ms -> 286.8ms` at 256px,
  about `10-12%` faster.
- alpha threshold `1/64`:
  `431.7ms -> 360.4ms` at 512px and `326.5ms -> 261.5ms` at 256px,
  about `16-20%` faster, but it changes support and needs heldout-quality A/B.
- transmittance `1e-2`:
  `431.7ms -> 404.5ms`, about `6%` faster; weaker because tile max stop remains.
- compact lookup prototype:
  K=4/8/16 at 128px B16/G8192/F32 measured `57%`, `34%`, and `15%` faster,
  with lower sampled current memory, but it is not trainer-wired.

CUDA-port breadcrumbs were added as comments in the experimental Metal kernel:

- `alpha_support_params(...)`: bin-time alpha cutoff
- `eval_alpha(...)`: per-pixel alpha cutoff
- `reduce_atomic_add_feature_grads(...)`: warp/threadgroup/global atomic
  reduction topology
- backward `tile_stop_counts`: tile max stop prefix caveat for top-p style
  pruning
