# STAR UVT Feature1 LR Schedule Gate

Date: 2026-05-19 10:55 +07

## Goal

Close the immediate quality-schedule question from the feature1/probe40
1300-step checkpoint: after fixing optimizer-LR resume, test whether a simple
late LR drop can suppress transient objective jumps without giving up the
current frozen-probe quality path.

## What Changed

- Added optimizer LR schedule support to
  `src/train/train_star_uvt_feature_overfit.py`.
- The trainer now records `optimizer_lr_schedule` and per-step `step_lrs` in
  result JSONs.
- Added schedule config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_to_lr00025_resume100_from1300_trace.jsonc`.
- Added late-spike trace config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_to_lr00025_resume88_from1300_late_spike_trace.jsonc`.
- Added report generator:
  `research_experiments/star_uvt_feature_tubes/star_uvt_feature1_lr_schedule_report.py`.

## Runs

Baseline rows used for comparison:

- Static `lr=0.005`, 1300->1400:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.json`.
- Static effective `lr=0.001`, 1300->1400:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume100_from1300_checkpoint_media.json`.

New rows:

- Scheduled `lr=0.001` until global step `1375`, then `0.00025` until `1400`:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_to_lr00025_resume100_from1300_trace.json`.
- Diagnostic late trace with the same schedule, stopped at `1388`:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_to_lr00025_resume88_from1300_late_spike_trace.json`.

W&B offline dirs:

- Static lr005: `wandb/offline-run-20260519_091629-2ouws83u`.
- Static lr001: `wandb/offline-run-20260519_103022-ndt9b3fc`.
- Scheduled 100-step: `wandb/offline-run-20260519_104219-8xepopkv`.
- Late trace: `wandb/offline-run-20260519_105122-80ju90o1`.

## Result

The simple LR drop is not a promotion.

- It passes mechanically and proves schedule plumbing.
- It removes the static-lr001 `1377->1378` jump.
- It moves a same-scale jump to `1385->1386`.
- It ends worse than static lr001 on the 100-step comparison:
  - weighted loss: `0.881602` vs `0.880942`
  - feature loss: `0.630803` vs `0.630549`
  - probe PSNR: `22.027` vs `22.034`
  - timing: `1506.9ms/step`, `807.2ms` backward vs `1463.8ms/step`,
    `778.4ms` backward

The 88-step late trace intentionally stops just after the spike and is expected
to fail the quality pass bit. Its value is attribution: `26/32` chunks worsen
at `1385->1386`, with summed weighted-loss delta `0.015248`; the largest chunk
is frame `0` with delta `0.001802`.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.md`.

## Handoff

- Current quality/default path remains the static effective-lr001 media
  checkpoint when probe PSNR and speed matter, and the static lr005 checkpoint
  when final feature/weighted loss matters.
- Do not use the `0.001 -> 0.00025` schedule as the next training default.
- The next quality gate should be checkpoint selection or a schedule keyed to
  measured transient recovery, not a blind lower-LR continuation.
- The next speed gate is unchanged: native VJP/scalar fixedbin feature
  gradients. LR tuning does not remove the 800ms-class backward.
