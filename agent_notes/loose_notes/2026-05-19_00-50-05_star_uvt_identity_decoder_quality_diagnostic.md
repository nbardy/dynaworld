# STAR UVT Identity Decoder Quality Diagnostic

Date: 2026-05-19

## Why

Gate 4 showed that the 512px STAR UVT feature path is far below RGB STAR on the
same source clip. The first 20-step no-pre-norm A/B proved that
`FeatureToColor` pre-norm is a large speed cost, but default pre-norm still had
slightly better PSNR. The next cheap hypothesis was whether sigmoid/pre-norm was
also suppressing quality.

## Command

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_identity_no_prenorm_20step_media.jsonc
```

The run used W&B offline mode.

## Config

- Source: `test_data/test_video_384_128_6fps.mp4`
- Frames/resolution: `64f`, `512px`, center-square crop
- Tubes/features: `8192t`, `F32`
- Renderer: `feature_direct_gradcache`
- Frame chunk size: `2`
- Tile capacity: `128`
- Colorizer: `activation=identity`, `pre_norm=false`, `hidden_dim=null`,
  `weight_init=kaiming`, `weight_init_gain=1.0`
- Steps/lr: `20`, `0.02`

## Result

Artifact:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_media.json
outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_sbs.mp4
wandb/offline-run-20260519_005005-4fllf7ea
```

Numbers:

- `pass=true`
- Loss: `0.3474775581 -> 0.3244628217`
- PSNR: `4.5907 -> 4.8884`
- Mean step: `2536.56ms`
- Mean render forward: `936.24ms`
- Mean colorize/loss: `306.03ms`
- Mean backward: `1173.45ms`
- Tile overflow: `0`
- Tile max/p95/p99: `36/21/25`
- Fixedbin eligible: `true`
- Gradients seen for tube feature, opacity, precision, UV/T center, velocity,
  and colorizer.

## Interpretation

Identity/no-pre-norm is the fastest 512px feature row recorded so far:
`2.54s/step` and `1.17s` backward. It is still not a quality fix. It ends below
both sigmoid/no-pre-norm (`4.941` PSNR) and default pre-norm (`4.984` PSNR), and
all feature rows remain far below RGB STAR direct-atomic (`12.444` PSNR) in the
same Gate 4 bracket.

This closes the simple decoder-unclamp hypothesis. Removing pre-norm/sigmoid can
make the feature row faster, but the next quality step needs evidence around
feature initialization, decoder capacity, objective shape, or a better
feature-to-RGB training contract. The shader-speed plan still matters, but the
fastest current decoder does not make STAR feature tubes a source-overfit
replacement for RGB STAR.

## Follow-up State

Regenerated after this row:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json
outputs/benchmarks/2026-05-19_renderer_scaling_report.md
outputs/benchmarks/2026-05-19_renderer_scaling_report.csv
outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md
outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.json
```

Gate 4 remains failed for feature promotion:

- RGB STAR best PSNR: `12.444`
- Best feature PSNR: `4.984`
- Fastest feature step: `2536.56ms`
- `feature_meets_rgb_psnr=false`
