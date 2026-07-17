# STAR UVT Pre-Norm Gain-2 Colorizer Init Diagnostic

Date: 2026-05-19

## Why

After identity/no-pre-norm and hidden-64 decoder capacity both failed as
practical quality fixes, the next cheap Gate 4 hypothesis was colorizer
initialization. `FeatureToColor` notes that pre-norm unit-variance inputs may
need sigmoid gain around `2`, while the current 512px pre-norm rows used gain
`4`.

## Command

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_prenorm_gain2_20step_media.jsonc
```

W&B ran offline at:

```text
wandb/offline-run-20260519_012514-17yv8gz3
```

## Config

- Source: `test_data/test_video_384_128_6fps.mp4`
- Frames/resolution: `64f`, `512px`, center-square crop
- Tubes/features: `8192t`, `F32`
- Renderer: `feature_direct_gradcache`
- Frame chunk size: `2`
- Tile capacity: `128`
- Colorizer: `hidden_dim=null`, `activation=sigmoid`, `pre_norm=true`,
  `weight_init=kaiming`, `weight_init_gain=2.0`
- Steps/lr: `20`, `0.02`

## Result

Artifacts:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_media.json
outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_sbs.mp4
```

Numbers:

- `pass=true`
- Loss: `0.3384382436 -> 0.3171853276`
- PSNR: `4.7052 -> 4.9869`
- Mean step: `14119.50ms`
- Mean render forward: `2567.77ms`
- Mean colorize/loss: `2047.33ms`
- Mean backward: `8913.38ms`
- Tile overflow: `0`
- Tile max/p95/p99: `37/22/26`
- Fixedbin eligible: `true`

## Interpretation

Gain `2` is not the missing quality bridge. It reaches essentially the same
feature PSNR as hidden-64 (`4.987`) and only slightly above gain-4 linear
pre-norm (`4.984`), while being slower than gain-4 linear pre-norm
(`14.12s/step`, `8.91s` backward versus `11.10s/step`, `7.07s` backward).

Gate 4 remains failed. The next quality work should not spend more time on
simple colorizer activation/norm/gain/capacity changes. Move to feature
objective/target representation, feature initialization with a stronger prior,
or a cheaper image-space/native VJP boundary paired with shader work.

## Follow-up State

Regenerated:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json
outputs/benchmarks/2026-05-19_renderer_scaling_report.md
outputs/benchmarks/2026-05-19_renderer_scaling_report.csv
outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md
outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.json
```

Gate 4 after this row:

- RGB STAR best PSNR: `12.444`
- Best feature PSNR: `4.987`
- Fastest feature step: `2536.56ms`
- `feature_meets_rgb_psnr=false`
