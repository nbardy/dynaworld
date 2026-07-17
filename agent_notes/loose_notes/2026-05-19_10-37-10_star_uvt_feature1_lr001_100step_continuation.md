# STAR UVT Feature1 LR001 100-Step Continuation

Date: 2026-05-19

## Goal

Run the real 100-step continuation from the 1300-step STAR UVT feature1/probe40
checkpoint with effective `lr=0.001`, media, and checkpoint output. The prior
20-step LR gate showed effective `lr=0.001` removes the early global-step 1318
spike, but it did not prove the schedule remains better over the same 1300->1400
horizon as the older `lr=0.005` row.

## Run

Config:

- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume100_from1300_checkpoint_media.jsonc`

Output:

- `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume100_from1300_checkpoint_media.json`
- checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_1400step_after_resume.pt`
- contact sheet:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume100_from1300_contact.jpg`
- side-by-side:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume100_from1300_sbs.mp4`
- W&B offline run:
  `wandb/offline-run-20260519_103022-ndt9b3fc`

Generated comparison report:

- `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.json`
- generator:
  `research_experiments/star_uvt_feature_tubes/star_uvt_feature1_lr001_continuation_report.py`

## Result

The effective-`lr=0.001` 100-step continuation passes:

- loaded/effective optimizer LR: `[0.005] -> [0.001]`
- loss: `0.886537 -> 0.880942`
- feature loss: `0.632124 -> 0.630549`
- probe PSNR: `21.965 -> 22.034`
- mean timing: `1463.8ms/step`, `571.8ms` render, `778.4ms` backward
- zero tile overflow, tile max/p95/cap `63/43/128`

Compared with the older `lr=0.005` 1300->1400 row:

- `lr=0.001` improves final probe PSNR by `+0.056`.
- `lr=0.001` is faster by `-226.5ms/step` and `-131.2ms` backward.
- `lr=0.005` still has better final feature loss by `0.003421`.
- `lr=0.005` still has slightly better final weighted loss by `0.000191`.
- Both have transient objective jumps. `lr=0.005` jumps at `1354->1355`
  (`+0.026093` loss). `lr=0.001` avoids the early `1318` jump but later jumps
  at `1377->1378` (`+0.015224`) and then recovers by the final step.

## Read

Effective `lr=0.001` is not a full replacement quality schedule. It is safer
around the early 1318 cliff, faster in this run, and better for the frozen-probe
visual score, but it gives up feature-target alignment versus the older
`lr=0.005` 100-step row.

The right next quality gate is a real LR schedule or checkpoint selection
around these transient jumps, not just "lower LR forever." The speed gate is
unchanged: native VJP/scalar fixedbin remains the real renderer-backward work.
