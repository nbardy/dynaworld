# STAR UVT Hidden-64 Decoder Capacity Diagnostic

Date: 2026-05-19

## Why

The identity/no-pre-norm diagnostic closed the simple decoder-unclamp hypothesis:
it made the row fast but worse. The next bounded Gate 4 quality hypothesis was
whether the linear per-pixel F32-to-RGB decoder was simply too weak. This run
kept the same 64f/512px/8192t Gate 4 setup and added one hidden 1x1 layer.

## Command

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_hidden64_prenorm_20step_media.jsonc
```

W&B ran offline at:

```text
wandb/offline-run-20260519_011003-dq5mb9f4
```

## Config

- Source: `test_data/test_video_384_128_6fps.mp4`
- Frames/resolution: `64f`, `512px`, center-square crop
- Tubes/features: `8192t`, `F32`
- Renderer: `feature_direct_gradcache`
- Frame chunk size: `2`
- Tile capacity: `128`
- Colorizer: `hidden_dim=64`, `activation=sigmoid`, `pre_norm=true`,
  `weight_init=kaiming`, `weight_init_gain=4.0`
- Steps/lr: `20`, `0.02`

## Result

Artifacts:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_media.json
outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_sbs.mp4
```

Numbers:

- `pass=true`
- Loss: `0.3386945073 -> 0.3171556294`
- PSNR: `4.7019 -> 4.9873`
- Mean step: `19179.60ms`
- Mean render forward: `2229.25ms`
- Mean colorize/loss: `2415.34ms`
- Mean backward: `13769.29ms`
- Tile overflow: `0`
- Tile max/p95/p99: `37/22/26`
- Fixedbin eligible: `true`

## Interpretation

Hidden-64 technically becomes the best feature PSNR in the same-clip bracket,
but only by a tiny margin: `4.987` versus `4.984` for the linear pre-norm
feature row. It is much slower: `19.18s/step` and `13.77s` backward, versus
`11.10s/step` and `7.07s` backward for linear pre-norm gradcache, and versus
`2.54s/step` and `1.17s` backward for the lower-quality identity/no-pre-norm
diagnostic.

This is a negative practical result for naive dense decoder capacity. The next
Gate 4 quality work should move toward feature initialization, feature objective
shape, target representation, or a cheaper image-space/native VJP boundary
rather than scaling a per-pixel dense decoder.

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
