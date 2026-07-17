# STAR UVT No-Pre-Norm 20-Step Media A/B

## Why

The 2026-05-18 backward split showed that 512px STAR UVT feature training is
dominated by the image-space `FeatureToColor`/loss VJP, and the 2-step
no-pre-norm row made that cost much smaller. That was still too short to
promote as a usable fast overfit setting, so this gate ran a matched 20-step
media A/B.

## Commands

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_prenorm_20step_media.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_20step_media.jsonc
```

Both configs used the same source clip, seed, `64f`, `512px`, `8192` tubes,
`F32`, `feature_direct_gradcache`, `chunk2`, `tile_t=2`, and `tile_capacity=128`.
Both W&B runs were logged offline:

- pre-norm: `wandb/offline-run-20260519_001546-tcdeuod8`
- no-pre-norm: `wandb/offline-run-20260519_002101-laprra78`

## Results

| row | pass | loss | PSNR | mean step | mean backward | mean forward | color/loss | overflow | max/p95/p99 tile |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| pre-norm | `true` | `0.33874 -> 0.31742` | `4.701 -> 4.984` | `11097.8ms` | `7070.1ms` | `2070.7ms` | `1397.4ms` | `0` | `37 / 22 / 26` |
| no-pre-norm | `true` | `0.33817 -> 0.32053` | `4.709 -> 4.941` | `7365.8ms` | `3370.3ms` | `2457.1ms` | `1100.9ms` | `0` | `36 / 22 / 26` |

Media artifacts:

- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_20step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_sbs.mp4`

Benchmark artifacts:

- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json`

## Interpretation

No-pre-norm remains a real speed lever, especially on backward
(`7070.1ms -> 3370.3ms`), but it is not a quality promotion: after 20 steps the
default pre-norm row has slightly lower loss and higher PSNR. The next STAR UVT
feature move should not declare no-pre-norm as the default 512px feature script
unless a longer or better-tuned gate closes that quality gap.

The result strengthens the split diagnosis: colorizer/loss simplification helps
whole-step speed, but the remaining cost is still large enough that optimized
fixedbin/two-pass feature-gradient accumulation or a native image-space VJP is
still needed before 512px/32768t feature rows make sense.

## Validation

- `py_compile` passed for:
  - `research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py`
  - `research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py`
  - `src/train/train_star_uvt_feature_overfit.py`
- `trainer_entry_for_config` loaded all `28`
  `src/train_configs/star_uvt_feature*.jsonc` configs through
  `train_star_uvt_feature_overfit`.
- Artifact sanity passed for both 20-step JSON rows: `pass=true`, loss
  decreased, gradient flow present, tile overflow `0`, no-pre-norm faster than
  pre-norm on step and backward timing.
- Media sanity passed: both contact sheets open as `4110x1026` images and both
  MP4 paths exist with nonzero size.
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json`
  contains both 20-step media rows.
- `agent_notes/key_learnings.md` remains at `195` lines.
- `git diff --check` passed.
- Trailing-whitespace scan passed for the changed docs/configs.
