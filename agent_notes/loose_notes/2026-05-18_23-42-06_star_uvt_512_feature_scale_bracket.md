# STAR UVT 512px Feature Scale Bracket

Date: 2026-05-18 23:42 Asia/Ho_Chi_Minh

## Goal

Execute the documented 512px STAR UVT feature-tube scale gate before attempting
any 32768-tube 512px row. The previous 512px evidence was only
`64f/512px/2048t/F32/chunk2`, which passed but already cost about `4s/step`.

## Small Fix Before The Gate

While re-reading the vec4 reducer, I found a boundary-pixel guard bug:
`direct_atomic_feature_backward` kept invalid tile threads alive for scalar
reduction barriers, but not for the new vec4 reduction flag. This did not affect
the 256/512 power-of-two rows, but could deadlock or misbehave on non-multiple
image sizes. Fixed the guard to keep invalid threads alive when either scalar or
vec4 reduction is active:

```text
if (!pixel_valid && !reduce_feature_grad_atomic && !reduce_feature_grad_atomic_vec4) return;
```

Boundary smoke:

```text
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_boundary_3f_130px_256t_f32.json
pass: true
3f/130px/256t/F32 vec4 timing: 10.45ms total / 6.03ms backward
```

## New Configs

- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_4096t_2step.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_2step.jsonc`

Both use `feature_direct_gradcache`, `64f`, `512px`, `F32`, frame chunk size
`2`, tile cap `128`, and require no tile overflow.

## Commands

```bash
rtk env PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 WANDB_MODE=offline .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_4096t_2step.jsonc
```

```bash
rtk env PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 WANDB_MODE=offline .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_2step.jsonc
```

Reports regenerated:

```bash
rtk env PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py \
  --result-jsons outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_*.json \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_scale_summary.json \
  --out-md outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_scale_report.md
```

```bash
rtk env PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/renderer_scaling_report.py \
  --out-md outputs/benchmarks/2026-05-18_renderer_scaling_report.md \
  --out-csv outputs/benchmarks/2026-05-18_renderer_scaling_report.csv
```

## Results

4096 tubes:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_4096t_f32_chunk2_gradcache_2step.json
pass: true
loss: 0.34277 -> 0.34051
PSNR: 4.65 -> 4.68
mean step: 6456.4ms
mean forward: 1220.7ms
mean color/loss: 472.2ms
mean backward: 4208.9ms
overflow: 0
max tile / p95 tile: 18 / 9
```

8192 tubes:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_2step.json
pass: true
loss: 0.33874 -> 0.33432
PSNR: 4.70 -> 4.76
mean step: 7937.3ms
mean forward: 1385.8ms
mean color/loss: 1223.4ms
mean backward: 4882.8ms
overflow: 0
max tile / p95 tile: 33 / 17
```

## Interpretation

512px support has plenty of tile headroom at 4096/8192 tubes under this
real-video setup. The blocker is not fixed-bin eligibility; it is total step
time, with backward still dominant and colorize/loss becoming material at
8192 tubes.

Do not spend a 512px/32768t row yet. The 8192-tube row is already nearly
`8s/step`; 32768 would mostly prove that the known feature-backward/colorize
bottleneck is expensive. The next aligned work is still an optimized fixedbin
feature backward or a two-pass/sidecar feature-gradient accumulation path.

## Docs Updated

- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `PROJECT_INDEX.md`
- `README.md`
- `agent_notes/key_learnings.md`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_scale_report.md`
- `outputs/benchmarks/2026-05-18_renderer_scaling_report.md`
