# STAR UVT Gate 4 Same-Clip Quality Bracket

## Why

The feature STAR UVT path had speed diagnostics and no-pre-norm A/B evidence,
but the quality claim was still unproven because the strong RGB STAR rows were
on a different clip. Gate 4 needed a same-clip source-overfit bracket before
any feature-tube path could be treated as a replacement.

## Runs

Same clip and shape for quality rows:

- source: `test_data/test_video_384_128_6fps.mp4`
- crop: `center_square`
- frames: `64`
- resolution: `512px`
- tubes: `8192`
- steps: `20`

Commands:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_rgb_testvideo_64f_512_directatomic_8192t_20step_media.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_directatomic_chunk2_8192t_prenorm_20step_media.jsonc
```

Existing matched feature rows used in the bracket:

- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_prenorm_20step_media.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_20step_media.jsonc`

Report command:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/gate4_quality_bracket_report.py \
  --rgb-star-json outputs/benchmarks/2026-05-19_star_uvt_rgb_testvideo_64f_512px_8192t_directatomic_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_directatomic_prenorm_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_20step_media.json \
  --feature-json outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_media.json \
  --renderer-csv outputs/benchmarks/2026-05-19_renderer_scaling_report.csv \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.json \
  --out-md outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md
```

## Results

| row | pass | loss | PSNR | mean step | backward | render/fwd | overflow |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| RGB STAR direct_atomic | `true` | `0.33084 -> 0.05696` | `12.444` | `681.1ms` | - | `138.8ms` render bench | - |
| feature direct_atomic pre-norm | `true` | `0.33874 -> 0.31742` | `4.701 -> 4.984` | `12349.9ms` | `7946.0ms` | `2168.1ms` | `0` |
| feature gradcache pre-norm | `true` | `0.33874 -> 0.31742` | `4.701 -> 4.984` | `11097.8ms` | `7070.1ms` | `2070.7ms` | `0` |
| feature gradcache no-pre-norm | `true` | `0.33817 -> 0.32053` | `4.709 -> 4.941` | `7365.8ms` | `3370.3ms` | `2457.1ms` | `0` |

`feature_meets_rgb_psnr=false`. The best feature row reaches `4.984` PSNR,
while RGB STAR reaches `12.444` PSNR on the same 20-step bracket.

Speed-only references from the regenerated renderer scaling report:

- dynamic GSplat RGB v8 512px/32768: `693.0ms` total, `541.3ms` backward
- projected F32 fixedbin 512px/32768: `5920.8ms` total, `4010.8ms` backward
- STAR RGB direct_fixedpoint 512px/32768: `507.1ms` total/backward kernel row

These are synthetic speed references, not same-source quality rows.

## Artifacts

- `outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md`
- `outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.json`
- `outputs/benchmarks/2026-05-19_renderer_scaling_report.md`
- `outputs/benchmarks/2026-05-19_renderer_scaling_report.csv`
- `outputs/benchmarks/2026-05-19_star_uvt_rgb_testvideo_64f_512px_8192t_directatomic_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_directatomic_prenorm_20step_media.json`

Media:

- `outputs/media/2026-05-19_star_uvt_rgb_testvideo_64f_512px_8192t_directatomic_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_rgb_testvideo_64f_512px_8192t_directatomic_20step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_directatomic_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_directatomic_prenorm_20step_sbs.mp4`

## Interpretation

Gate 4 blocks feature promotion. Feature STAR can train and no-pre-norm is
faster, but the current F32 feature-image plus simple colorizer objective is
nowhere near RGB STAR source-overfit quality on the same clip. The next feature
work should improve the feature decoder/objective or training contract before
claiming a replacement path. Renderer work is still necessary for 512px scale,
but renderer-only speed cannot fix this quality gap.

## Validation

- `py_compile` passed for:
  - `research_experiments/renderer_scaling_report.py`
  - `research_experiments/star_uvt_feature_tubes/gate4_quality_bracket_report.py`
  - `research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py`
  - `src/train/train_star_uvt_feature_overfit.py`
  - `src/train/train_star_uvt_video_overfit.py`
- `trainer_entry_for_config` loaded all `33` `src/train_configs/star_uvt*.jsonc`
  configs: `29` feature-overfit configs and `4` RGB video-overfit configs.
- Gate 4 artifact sanity passed: all quality rows report `pass=true`, RGB PSNR
  is greater than feature PSNR, and `feature_meets_rgb_psnr=false`.
- Media sanity passed for the new RGB and feature-direct-atomic contact sheets
  and side-by-side MP4s.
- `agent_notes/key_learnings.md` remains at `195` lines.
- `git diff --check` passed.
- Trailing-whitespace scan passed for the changed docs/configs/scripts.
