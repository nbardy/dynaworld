# STAR UVT Frozen RGB-Probe 300-Step Extension

Date: 2026-05-19

## Goal

After the 100-step frozen target-grid feature-to-RGB STAR gate moved probe PSNR
to `14.641`, run the same objective for 300 steps to see whether it keeps
closing the standalone decoder oracle gap.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.jsonc
```

Offline W&B directory:

```text
wandb/offline-run-20260519_064930-jhv2lgdj/
```

## Artifacts

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.json
outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_sbs.mp4
```

## Result

The gate passed.

```text
total loss:          1.399375 -> 1.032446
feature target loss: 0.999935 -> 0.811652
frozen probe loss:   0.039944 -> 0.022079
frozen probe PSNR:   13.985 -> 16.560
mean step:           1355.06 ms
render forward:      551.65 ms
feature target loss: 18.53 ms
probe loss:          37.47 ms
backward:            680.57 ms
target load/prep:    130.46 ms
tile overflow:       0
fixedbin eligible:   true
```

## Read

The frozen-probe objective keeps working beyond the 100-step gate. It is no
longer just plumbing: the probe PSNR trajectory is `13.985 -> 14.060` at
20 steps, `14.641` at 100 steps, and `16.560` at 300 steps. Step cost stays
in the same band (`1.22s`, `1.27s`, `1.36s`) with zero overflow.

This still is not a final visual quality promotion. The standalone hidden64
feature-to-RGB probe on the cached target grid reaches `20.073` full-video PSNR
and `23.401` grid PSNR. The next useful gate is either a longer/scheduled
frozen-probe objective, or a native VJP/scalar tile-slot route that lets this
feature objective scale without dense image-gradient costs.

Reports regenerated after the run:

```text
outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md
outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md
```
