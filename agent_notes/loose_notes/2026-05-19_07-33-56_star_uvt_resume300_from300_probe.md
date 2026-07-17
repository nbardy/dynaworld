# STAR UVT Frozen RGB-Probe Resume300 From 300 Gate

## Context

The 300-step frozen RGB-probe STAR target-grid row was the current keeper
diagnostic: feature loss `0.999935 -> 0.811652`, probe PSNR
`13.985 -> 16.560`, zero overflow, and offline W&B `jhv2lgdj`. After adding
trainer checkpoint/resume support, I reran the 300-step keeper with checkpoint
output enabled, then resumed for another 300 local steps.

## Configs

Checkpoint-producing 300-step rerun:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.jsonc
```

Resume continuation:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.jsonc
```

Both use the same frozen hidden64 target-grid feature-to-RGB checkpoint:

```text
outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt
```

## Results

300-step checkpoint/no-media rerun:

```text
W&B offline: wandb/offline-run-20260519_071506-1pqbr6xw
result: outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.json
checkpoint: outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step.pt
pass=true
feature loss: 0.999935 -> 0.811652
RGB-probe loss / PSNR: 0.039944 -> 0.022079 / 13.985 -> 16.560
mean step/render/feature-target/probe/backward:
  1268.0ms / 530.2ms / 17.1ms / 31.0ms / 632.8ms
tile_overflow_sum=0
```

Resume300-from300 media run:

```text
W&B offline: wandb/offline-run-20260519_072156-vtti65kr
result: outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.json
checkpoint: outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_600step_after_resume.pt
media:
  outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_contact.jpg
  outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_sbs.mp4
pass=true
resume_loaded=true
resume_optimizer_loaded=true
resume_checkpoint_steps=300
feature loss: 0.810827 -> 0.655366
RGB-probe loss / PSNR: 0.022001 -> 0.010271 / 16.576 -> 19.884
mean step/render/feature-target/probe/backward:
  1439.5ms / 569.3ms / 19.2ms / 41.1ms / 733.7ms
tile_overflow_sum=0
```

## Interpretation

The frozen-probe objective is not just a short-run artifact. It keeps improving
through 600 local steps without tile overflow. The resumed row nearly reaches the
standalone full-video upsample number (`20.073` PSNR), but that is not the
strict same-metric oracle: the standalone target-grid probe's grid PSNR is
`23.401`, and the integrated probe is still below that.

The current conclusion is therefore narrower and stronger:

- decodability is not the blocker
- checkpoint/resume plumbing is not the blocker
- the target-grid objective is viable and cheap enough for more probes
- the remaining gap is schedule/objective/native-VJP, especially if we want the
  same-grid oracle quality or dataset-scale training

## Follow-Up

Next run should not be another identical continuation unless we need a curve
tail. Better gates:

- a scheduled resume segment that changes the frozen-probe/feature-loss balance,
  with explicit global-step semantics if schedule boundaries matter
- native-VJP or a tighter probe loss that avoids dense image-gradient overhead
- dataset-scale promotion only after the same-grid oracle gap is much smaller
