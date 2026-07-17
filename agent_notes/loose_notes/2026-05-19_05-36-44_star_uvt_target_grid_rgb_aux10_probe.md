# STAR UVT Target-Grid RGB-Aux10 Probe

Date: 2026-05-19 05:36 +0700

## Goal

Test whether the weak RGB improvement from the RGB-aux1 target-grid probe was
mainly a loss-weighting issue. This repeats the same 64f/512px/8192t target-grid
V-JEPA route with `rgb_loss_weight=10.0`.

## Config

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux10_lr005_20step_media.jsonc`

Key settings:

- `feature_target.materialization=target_grid`
- `feature_target.loss_weight=1.0`
- `feature_target.rgb_loss_weight=10.0`
- 64 frames, 512px, 8192 tubes, F32, chunk size 2
- W&B offline run: `wandb/offline-run-20260519_053644-otp89ki7`

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux10_lr005_20step_media.jsonc
```

## Result

- output:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_20step_media.json`
- pass: `true`
- total loss: `4.381647 -> 4.347160`
- feature target loss: `0.999935 -> 0.997547`
- RGB loss: `0.338171 -> 0.334961`
- RGB PSNR: `4.709 -> 4.750`
- colorizer gradients present
- target grid: `[32,32,16,16]`, `1.0MiB`
- target load/prep: `158.48ms`
- mean step: `1996.93ms`
- mean render forward: `605.77ms`
- mean target/loss: `51.61ms`
- mean backward: `1089.39ms`
- zero tile overflow; max tile count `33`, p95 `18`
- media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_20step_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_20step_sbs.mp4`

## Interpretation

RGB-aux10 is only a marginal RGB improvement over RGB-aux1:

- aux1 RGB PSNR: `4.709 -> 4.746`
- aux10 RGB PSNR: `4.709 -> 4.750`
- aux1 feature loss final: `0.997336`
- aux10 feature loss final: `0.997547`

So the missing visual lever is not simply a larger RGB scalar. The next visual
probe should be longer or warm-started, or it should use a trained/frozen
feature-to-RGB probe. Keep pure `target_grid` as the speed/memory diagnostic and
keep RGB-aux1/10 as controls.

## Updated Reports

- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  now includes RGB-aux10.
- `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
  records RGB-aux10 as a weak negative control.
- `BASELINES.md`, `README.md`, `PROJECT_INDEX.md`, `EXPERIMENTS.md`,
  `TODO/README.md`, the feature-tube README, the fast-shader plan, and
  `agent_notes/key_learnings.md` were updated.
