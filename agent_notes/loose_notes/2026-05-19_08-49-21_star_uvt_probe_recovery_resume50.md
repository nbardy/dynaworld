# STAR UVT Probe-Recovery Resume50

Date: 2026-05-19

## Goal

Test whether the 1100->1200 feature-recovery checkpoint can regain frozen
RGB-probe quality without immediately destroying the recovered V-JEPA target
alignment.

## Run

- Config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.jsonc`
- Result:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.json`
- Resume checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_1200step_after_resume.pt`
- Output checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_1250step_after_resume.pt`
- Media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_sbs.mp4`
- Offline W&B:
  `wandb/offline-run-20260519_083906-e4dl4qgy`

## Result

- Pass: true.
- Global steps: `1200 -> 1250`.
- Objective: constant `feature_target.loss_weight=0.75`,
  `rgb_probe_loss_weight=40.0`.
- Mean timing: `1523.5ms/step`, `580.7ms` render, `18.7ms` feature-target loss,
  `41.4ms` RGB-probe loss, `807.1ms` backward.
- Feature loss: `0.635066 -> 0.638799`.
- Frozen RGB-probe loss / PSNR: `0.006700 -> 0.006413` /
  `21.740 -> 21.929`.
- Tile overflow: `0`.

## Interpretation

The short feature0.75/probe40 continuation is a useful probe-recovery row: it
restores probe PSNR above the 1100-step Pareto row while keeping the V-JEPA
feature loss much better than the 800-step probe-emphasis drift state. It does
not solve objective balance. The immediate tradeoff remains clear: probe PSNR
can be pushed back up quickly, but even this short recovery raises feature loss
from `0.635066` to `0.638799`.

## Docs Updated

- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
- `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
- `README.md`
- `PROJECT_INDEX.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `agent_notes/key_learnings.md`

## Validation

- `python -m py_compile` passed for the STAR feature trainer, both report
  generators, and `tests/test_star_uvt_feature_target_adapter.py`.
- Focused pytest passed: `10 passed in 2.40s`.
- Report invariants passed: comparison report has `25` rows, the
  feature0.75/probe40 row is `pass`, the audit pass flag is true, global steps
  are `1200 -> 1250`, and tile overflow is `0`.
- `git diff --check` passed.
- Touched-file trailing-whitespace scan passed.
- `pgrep -fl train_star_uvt_feature_overfit.py` found no active STAR feature
  trainer process.
