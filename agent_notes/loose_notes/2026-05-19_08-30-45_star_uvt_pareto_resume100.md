# STAR UVT Pareto Resume100 From 1000

Date: 2026-05-19

## Goal

Continue the 64f/512px/8192t STAR V-JEPA target-grid frozen-probe lane from the
scheduled-balance 1000-step checkpoint and test whether a constant Pareto
objective can keep the recovered V-JEPA feature alignment while regaining visual
probe PSNR.

## Config

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.jsonc`

Key settings:

- resume checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_1000step_after_resume.pt`
- `train.steps=100`
- `train.global_step_offset=1000`
- `train.resume_optimizer=true`
- `feature_target.materialization=target_grid`
- `feature_target.loss_weight=0.5`
- `feature_target.rgb_probe_loss_weight=40.0`
- renderer: `feature_direct_gradcache_reduce_vec4`, `frame_chunk_size=2`
- target: 64 frames, 512px, 8192 tubes, F32, no pre-norm, cached V-JEPA token grid

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.jsonc
```

## Result

Output:

`outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.json`

Checkpoint:

`outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_1100step_after_resume.pt`

Media:

- `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_sbs.mp4`

Offline W&B:

`wandb/offline-run-20260519_081453-mv04a29b`

Metrics:

- `pass=true`
- global steps `1000 -> 1100`
- total loss `0.612785 -> 0.593314`
- feature target loss `0.643823 -> 0.656728`
- RGB-probe loss `0.007272 -> 0.006624`
- RGB-probe PSNR `21.384 -> 21.789`
- mean step `1461.3ms`
- mean render forward `571.5ms`
- mean feature target loss `17.1ms`
- mean RGB-probe loss `37.3ms`
- mean backward `766.2ms`
- tile overflow `0`
- max tile count `57`, p95 `37`, p99 `45`

## Read

This is a passing combined-loss Pareto row, but not an oracle-closing answer.
Compared with the scheduled 800->1000 balance row, it regains probe PSNR
(`21.382 -> 21.789`) and keeps overflow at zero, but feature loss drifts back
up (`0.643823 -> 0.656728`). The row strengthens the conclusion that the bridge
problem is objective balance/native VJP: the cached target is decodable, resume
works, and the renderer is not the immediate blocker for this lane.

## Docs Updated

- `README.md`
- `PROJECT_INDEX.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `agent_notes/key_learnings.md`
- `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py`
- `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py`

## Next

Do not spend more time proving cached V-JEPA target plumbing. The next useful
STAR UVT target-grid gate should either:

1. use a better objective schedule that prevents feature drift while keeping
   probe PSNR above 21.8, or
2. prototype a native image-space VJP / scalar fixedbin feature-gradient path so
   the same visual objective can be run at larger dataset scale without dense
   feature-image backward cost.
