# STAR UVT Recover Schedule Resume100 From 1100

Date: 2026-05-19

## Goal

Follow the passing 1000->1100 feature0.5/probe40 Pareto row with a short
alignment-recovery schedule. The question was whether we could recover the
feature-target drift without giving back much of the frozen RGB-probe visual
gain.

## Config

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.jsonc`

Key settings:

- resume checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_1100step_after_resume.pt`
- `train.steps=100`
- `train.global_step_offset=1100`
- schedule:
  - global 1100-1150: `feature=1.0`, `rgb_probe=20.0`
  - global 1150-1200: `feature=0.75`, `rgb_probe=30.0`
- renderer: `feature_direct_gradcache_reduce_vec4`, `frame_chunk_size=2`
- target: 64 frames, 512px, 8192 tubes, F32, no pre-norm, cached V-JEPA target grid

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.jsonc
```

## Result

Output:

`outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.json`

Checkpoint:

`outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_1200step_after_resume.pt`

Media:

- `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_sbs.mp4`

Offline W&B:

`wandb/offline-run-20260519_082831-15uzmgdw`

Metrics:

- `pass=false`
- global steps `1100 -> 1200`
- total loss `0.789153 -> 0.677388`
- feature target loss `0.656765 -> 0.635093`
- RGB-probe loss `0.006619 -> 0.006702`
- RGB-probe PSNR `21.792 -> 21.738`
- mean step `1520.9ms`
- mean render forward `588.3ms`
- mean feature target loss `19.5ms`
- mean RGB-probe loss `41.4ms`
- mean backward `795.4ms`
- tile overflow `0`
- max tile count `61`, p95 `40`, p99 `47`

## Read

This is a useful partial/negative row. It proves the feature drift from the
1000->1100 Pareto objective is recoverable quickly: the first 50-step
feature-heavy stage drops feature loss from `0.656765` to `0.631313`, then the
second stage gives some alignment back while restoring much of the probe PSNR.
End state is better aligned than the 1100 checkpoint (`0.635093` vs
`0.656728`) and still visually close (`21.738` vs `21.789`), but it is
nonpassing because probe loss does not decrease end-to-end.

The lesson is still objective balance, not plumbing: alternate schedules can
move along the feature/probe frontier, but none yet keeps both the best feature
alignment (`~0.635-0.644`) and the best probe quality (`>=21.79`, much less the
same-grid `23.401` oracle).

## Next

The next gate should either:

1. test a shorter probe-recovery step from the 1200 checkpoint, such as
   `feature=0.75`, `rgb_probe=40`, and stop when probe PSNR recovers without
   feature loss exceeding the 1000-step `0.644` bracket; or
2. prototype a native image-space VJP / scalar fixedbin feature-gradient path,
   because schedule tuning alone is now tracing a clear tradeoff frontier.
