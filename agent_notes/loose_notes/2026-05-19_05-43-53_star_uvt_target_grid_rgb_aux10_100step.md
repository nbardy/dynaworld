# STAR UVT Target-Grid RGB-Aux10 100-Step Probe

Date: 2026-05-19 05:43 +0700

## Goal

After RGB-aux1 and RGB-aux10 barely moved RGB PSNR at 20 steps, test whether
the target-grid visual path needs schedule length rather than only a larger RGB
weight.

## Config

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux10_lr005_100step_media.jsonc`

Key settings:

- `feature_target.materialization=target_grid`
- `feature_target.loss_weight=1.0`
- `feature_target.rgb_loss_weight=10.0`
- 64 frames, 512px, 8192 tubes, F32, chunk size 2
- W&B offline run: `wandb/offline-run-20260519_054353-fdchromb`

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux10_lr005_100step_media.jsonc
```

## Result

- output:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_100step_media.json`
- pass: `true`
- total loss: `4.381647 -> 4.048905`
- feature target loss: `0.999935 -> 0.964670`
- RGB loss: `0.338171 -> 0.308424`
- RGB PSNR: `4.709 -> 5.109`
- colorizer gradients present
- target grid: `[32,32,16,16]`, `1.0MiB`
- target load/prep: `129.91ms`
- mean step: `1876.37ms`
- mean render forward: `580.07ms`
- mean target/loss: `43.08ms`
- mean backward: `1032.85ms`
- last step: `1814.33ms`, `993.45ms` backward
- zero tile overflow; max tile count `43`, p95 `28`
- media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_100step_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_100step_sbs.mp4`

## Interpretation

Schedule length matters. The 100-step aux10 run improves RGB PSNR by `0.400dB`
and moves feature loss much more than the 20-step probes:

- aux10 20-step RGB PSNR: `4.709 -> 4.750`
- aux10 100-step RGB PSNR: `4.709 -> 5.109`
- aux10 20-step feature loss final: `0.997547`
- aux10 100-step feature loss final: `0.964670`

This is still not a quality promotion. It remains far below the RGB STAR
same-clip bracket and does not solve the visual gap. The next target-grid visual
gate should be warm-started or should use a trained/frozen feature-to-RGB probe.
Keep pure `target_grid` as the speed/memory diagnostic.

## Updated Reports

- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  now includes the 100-step aux10 row.
- `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
  records the schedule-length read.
- `BASELINES.md`, `README.md`, `PROJECT_INDEX.md`, `EXPERIMENTS.md`,
  `TODO/README.md`, the feature-tube README, the fast-shader plan, and
  `agent_notes/key_learnings.md` were updated.
