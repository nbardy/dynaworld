# STAR UVT Scheduled Balance 800->1000 Continuation

Date: 2026-05-19

## Goal

Test whether the probe-emphasis visual gain can be kept while recovering the
V-JEPA target-grid feature alignment that drifted in the 600->800 run.

## Code/Config Changes

- `FeatureTargetWeightStage` now includes `rgb_probe_loss_weight`.
- `feature_target.weight_schedule` can now schedule the frozen RGB-probe loss
  as well as feature loss and RGB aux loss.
- Result rows now include `step_rgb_probe_loss_weights`.
- Focused tests cover the new scheduled probe-weight field.
- New config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.jsonc`.

## Run

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.jsonc
```

Inputs:

- resume checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_800step_after_resume.pt`
- schedule:
  - global `800-900`: feature loss `1.0`, RGB-probe loss `10.0`
  - global `900-1000`: feature loss `0.5`, RGB-probe loss `20.0`
- W&B offline run:
  `wandb/offline-run-20260519_080000-n79f13p3`

Artifacts:

- result JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.json`
- checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_1000step_after_resume.pt`
- media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_sbs.mp4`

## Results

- `pass=false`
- `resume_loaded=true`
- `resume_optimizer_loaded=true`
- `resume_checkpoint_steps=200`
- `tile_overflow_sum=0`
- feature loss: `0.703862 -> 0.643852`
- RGB-probe PSNR: `21.428 -> 21.382`
- RGB-probe loss: `0.007197 -> 0.007275`
- mean timing:
  - step `1308.1ms`
  - render `543.5ms`
  - target-grid loss prep `15.5ms`
  - RGB-probe loss `27.7ms`
  - backward `667.6ms`

## Read

This is a useful negative/partial result. The alignment-heavy first half
successfully recovers the feature target, and the balanced second half brings
probe loss back down from the worst point inside the run, but the final probe
loss is still worse than the run start. The row is correctly nonpassing on the
probe-loss-decrease gate.

The tradeoff is real: static probe emphasis gains visual decodability while
hurting V-JEPA feature alignment, and simple two-stage catch-up recovers
alignment while giving back probe quality. The next gate should use either a
more adaptive ratio or a native image-space VJP that better matches the frozen
decoder without discarding the target-grid feature constraint.

## Validation

- `py_compile` passed for the STAR trainer and focused tests before this run.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py -q`
  passed: `10 passed`.
