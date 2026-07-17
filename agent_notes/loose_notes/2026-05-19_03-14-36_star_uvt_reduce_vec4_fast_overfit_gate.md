# STAR UVT Reduce-Vec4 Fast Overfit Gate

## Goal

Repeat the core STAR UVT feature-shader plan in docs, fill missing routing
details, and execute the next gate: test whether the single-pass
`feature_direct_gradcache_reduce_vec4` path should be used in the first-class
512px feature-tube overfit route.

The end goal for this lane is a fast, trainable 64-frame/512px STAR UVT
feature-tube path that can use precomputed feature targets without the projected
feature-raster backward explosion, then scale to the prepared larger manifests.
The current quality gate is still same-clip source-view overfit against RGB
STAR before treating feature STAR as a replacement.

## Commands

Fresh default-pre-norm 2-step controls:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_2step_rerun_20260519.jsonc
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_reduce_vec4_chunk2_8192t_2step.jsonc
```

Fresh no-pre-norm 2-step controls:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_2step_rerun_20260519.jsonc
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_2step.jsonc
```

Fresh no-pre-norm 20-step media gate:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_20step_media_rerun_20260519.jsonc
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_20step_media.jsonc
```

## Results

All rows passed: loss decreased, gradients reached tube features and geometry,
tile overflow was zero, and no mode fallback was required.

| row | pre_norm | steps | step s | backward s | forward s | color/loss s | media s | end loss | end PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `feature_direct_gradcache` | false | 20 | 2.858 | 1.327 | 1.053 | 0.344 | 1.660 | 0.32053 | 4.941 |
| `feature_direct_gradcache_reduce_vec4` | false | 20 | 2.491 | 1.184 | 0.911 | 0.287 | 1.360 | 0.32053 | 4.941 |
| `feature_direct_gradcache` | true | 2 | 7.825 | 5.181 | 1.479 | 0.762 | n/a | 0.33432 | 4.758 |
| `feature_direct_gradcache_reduce_vec4` | true | 2 | 7.690 | 5.088 | 1.444 | 0.724 | n/a | 0.33432 | 4.758 |
| `feature_direct_gradcache` | false | 2 | 2.449 | 1.153 | 0.794 | 0.230 | n/a | 0.33764 | 4.715 |
| `feature_direct_gradcache_reduce_vec4` | false | 2 | 2.298 | 1.062 | 0.824 | 0.243 | n/a | 0.33764 | 4.715 |

Artifacts:

- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_2step_rerun.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_2step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_2step_rerun.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_2step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_media_rerun.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_fast_overfit_reduce_vec4_summary.md`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_rerun_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_rerun_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_20step_sbs.mp4`
- offline W&B runs:
  `wandb/offline-run-20260519_031141-rxvl61x2` and
  `wandb/offline-run-20260519_031257-csd8vwc8`

## Decision

Use `feature_direct_gradcache_reduce_vec4` plus `colorize.pre_norm=false` as
the current fast feature-tube diagnostic. It is now wired into:

```bash
src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-fast
```

Do not promote this as the source-view quality baseline. The same-clip Gate 4
bracket still says RGB STAR reaches about `12.44` PSNR after 20 steps while
feature STAR is still around `4.99` PSNR. This row is a speed script for the
feature-tube path, not proof that feature tubes are good enough.

## Next Plan

1. Keep the fast feature script pointed at no-pre-norm reduce-vec4 while doing
   short shader/objective probes.
2. Do not run 512px/32768t feature tubes until either feature-gradient
   accumulation or image-space VJP changes again.
3. The next shader implementation should be a true fixedbin/tile-slot
   feature-gradient path with compact scalar contribution weights/prefixes, or
   a faster image-space VJP/handoff that avoids dense F32 image-gradient
   backprop.
4. The next quality implementation should change feature targets, feature
   initialization, or decoder/objective structure; simple pre-norm removal,
   identity decode, hidden-64 dense decode, and gain tweaks are already
   recorded as speed-only or negative quality moves.
