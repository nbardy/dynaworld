# STAR UVT cached-bin sidecar gate

Date: 2026-05-19

## Goal

Test whether a small sidecar optimization helps the feature STAR UVT path:
reuse the forward tile bins in backward instead of clearing/binning the same
tubes again inside `direct_atomic_feature_backward`.

This is the concrete follow-up to the plan note that `feature_direct_fixedbin`
is still only a validity/fallback surface and that the next real shader work
needs a fixedbin/sidecar or two-pass feature-gradient path.

## Code changes

- Added `render_features_with_bins` to `star_uvt_v0`.
  - Returns the normal feature forward outputs plus `tile_tube_ids` and
    `tile_depths`.
- Added `direct_atomic_feature_backward_with_bins`.
  - Consumes forward `tile_counts`, `tile_tube_ids`, `tile_depths`, and
    `tile_unstable`.
  - Skips the backward `clear_tiles + bin_tubes` stage.
  - Reuses the existing feature backward Metal kernel for gradients.
- Added Python/autograd mode plumbing:
  - `gradcache_cached_bins` maps to the same Metal mode bits as `gradcache`.
  - `feature_uvt.render_mode=feature_direct_gradcache_cached_bins` is accepted
    by the first-class feature trainer.
- Added benchmark config:
  - `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_cachedbins_chunk2_8192t_prenorm_2step.jsonc`

## Validation commands

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode gradcache --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 --timing-warmup 2 \
  --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_same_session_before_cachedbins_64f_256_32768_f32.json

PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode gradcache_cached_bins --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 --timing-warmup 2 \
  --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_cachedbins_64f_256_32768_f32.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_cachedbins_chunk2_8192t_prenorm_2step.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_2step.jsonc
```

## Results

Synthetic direct-kernel same-session gate, 64f/256px/32768t/F32:

- Plain `gradcache`: pass, total `1697.4ms`, forward `629.4ms`, backward
  `1068.0ms`.
- `gradcache_cached_bins`: pass, total `1544.0ms`, forward `608.2ms`, backward
  `935.8ms`.
- Interpretation: cached bins are a real isolated renderer-backward win here,
  about `12.4%` on backward and `9.0%` total in this noisy same-session pair.

First-class trainer gate, 64f/512px/8192t/F32/chunk2/pre-norm, 2 steps:

- `feature_direct_gradcache_cached_bins`: pass, zero overflow, loss
  `0.33874 -> 0.33432`, mean step `16196.4ms`, forward `2763.8ms`,
  colorize/loss `2211.9ms`, backward `10241.4ms`.
- Same-session plain `feature_direct_gradcache` control: pass, zero overflow,
  loss `0.33874 -> 0.33432`, mean step `16210.3ms`, forward `3022.2ms`,
  colorize/loss `2145.8ms`, backward `9675.8ms`.
- Interpretation: no first-class end-to-end win. The cached-bin row is
  correctness-valid but effectively tied on step time and slower on measured
  backward in this noisy 2-step trainer run.

## Decision

Do not promote cached bins as the trainer default. Keep it as a diagnostic
sidecar mode and as evidence that rebinnning is not the whole 512px problem.
The larger bottleneck remains the dense image-space `FeatureToColor`/loss VJP
plus per-channel feature-gradient accumulation.

Next shader work should be a true fixedbin/sidecar implementation that changes
the accumulation topology, or a faster image-space VJP/handoff. Merely reusing
forward bins trims the isolated renderer benchmark but does not move the current
first-class trainer enough.
