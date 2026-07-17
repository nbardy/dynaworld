# STAR UVT Frozen RGB-Probe 100-Step Follow-Up

Date: 2026-05-19

## Goal

Repeat the 20-step frozen target-grid feature-to-RGB STAR integration gate long
enough to see whether the probe objective actually moves, not just whether the
decoder can be loaded and frozen inside the STAR target-grid trainer.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.jsonc
```

Offline W&B directory:

```text
wandb/offline-run-20260519_063726-3f4hm6wq/
```

## Artifacts

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.json
outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_100step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_100step_sbs.mp4
outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt
```

## Result

The gate passed.

```text
total loss:          1.399375 -> 1.313538
feature target loss: 0.999935 -> 0.970035
frozen probe loss:   0.039944 -> 0.034350
frozen probe PSNR:   13.985 -> 14.641
mean step:           1268.39 ms
render forward:      531.65 ms
feature target loss: 17.18 ms
probe loss:          31.24 ms
backward:            630.37 ms
target load/prep:    198.28 ms
tile overflow:       0
fixedbin eligible:   true
```

## Read

This is a positive diagnostic compared with the 20-step frozen-probe gate:
probe PSNR moved `+0.655 dB` instead of `+0.075 dB`, and feature loss reached
`0.970035`. It is also cheaper than the 100-step RGB-aux10 row
(`1.268s/step` versus `1.876s/step`) while moving the visual probe more.

It is not yet a quality promotion. The standalone hidden64 feature-to-RGB probe
trained directly on the cached target grid reaches `20.073` full-video PSNR and
`23.401` grid PSNR, so the STAR feature renderer/objective still leaves a large
oracle gap. The next useful target-grid gate is a longer or scheduled
frozen-probe objective, or a native VJP/scalar tile-slot feature-gradient route
that lets the same target run scale without paying dense image-gradient costs.

The generated comparison and bridge-audit reports were regenerated after this
run:

```text
outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md
outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md
```
