# STAR UVT Target-Grid 20-Step Media Gate

Date: 2026-05-19 05:20 +0700

## Goal

Run the next target-grid gate after the 5-step timing check: a longer
64f/512px/8192t/F32 overfit with media outputs, using the same real V-JEPA
target-grid loss route.

## Config

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_lr005_20step_media.jsonc`

Key settings:

- `feature_target.materialization=target_grid`
- `feature_target.rgb_loss_weight=0.0`
- `token_grid_shape=[32,16,16]`
- `feature_direct_gradcache_reduce_vec4`
- 64 frames, 512px, 8192 tubes, F32, chunk size 2
- W&B offline run: `wandb/offline-run-20260519_052006-rbulvmdt`

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_lr005_20step_media.jsonc
```

## Result

- output:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_20step_media.json`
- pass: `true`
- feature-target loss: `0.999935 -> 0.997425`
- target grid: `[32,32,16,16]`, `1.0MiB`
- target load/prep: `240.42ms`
- mean step: `1451.19ms`
- mean render forward: `629.67ms`
- mean target/loss: `37.46ms`
- mean backward: `722.06ms`
- last step: `1401.71ms`, `706.36ms` backward
- zero tile overflow; max tile count `33`, p95 `17`
- media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_20step_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_20step_sbs.mp4`

## Interpretation

The target-grid objective does overfit monotonically for 20 steps and keeps the
small-memory timing profile. The target/loss bucket is no longer the bottleneck:
it averages only `37.46ms` versus `722.06ms` backward and `629.67ms` render.

Do not treat this as visual-quality promotion. `rgb_loss_weight=0.0`, the
colorizer is not trained, and the reported RGB PSNR (`0.000283 -> 0.011198`) is
only the incidental random colorizer/media path. The next visual gate needs an
explicit RGB auxiliary objective, a trained/frozen feature-to-RGB probe, or a
different evaluation path for V-JEPA-space quality.

## Updated Reports

- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  now includes the 20-step target-grid media row.
- `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
  now records that the 20-step row passes but is not RGB quality evidence.
- `BASELINES.md`, `README.md`, `PROJECT_INDEX.md`, `EXPERIMENTS.md`,
  `TODO/README.md`, the feature-tube README, the fast-shader plan, and
  `agent_notes/key_learnings.md` were updated with the caveat.
