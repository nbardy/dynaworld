# STAR UVT Compact RGB-Grid Bridge Gate

Date: 2026-05-20 01:52 +07

## Why

The selected compact target-area route was the practical visual helper but
failed scale-up (`6.023` dense RGB PSNR, sparse/streaked media). The previous
trainable RGB-grid bridge proved the output colorizer can train cheaply through
the fast target-grid sparse VJP path, but RGB-grid alone also failed dense media
(`5.657` full RGB PSNR).

This gate tested the natural combination: compact target-area support plus
`feature_target.rgb_grid_loss_weight=40`.

## What Ran

Added config:
`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_rgbgrid40_from1500_lr001_50step_media.jsonc`

Command:

```bash
PYTHONPATH=src/train rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_rgbgrid40_from1500_lr001_50step_media.jsonc
```

W&B offline run: `t51dowf2`.

## Result

Mechanics pass:

- pass `true`
- tile overflow `0`
- colorizer gradients seen
- raw STAR feature/geometry gradients seen

Quality/timing:

- mean/no-first/last step `1647.92/1617.37/1518.41ms`
- mean/no-first/last backward `892.43/874.70/865.49ms`
- feature target loss worsens `0.625418 -> 0.630296`
- RGB-grid PSNR improves `22.028 -> 22.237`
- frozen RGB-probe PSNR improves `22.028 -> 22.112`
- sparse visual PSNR improves `5.678 -> 5.760`
- dense full RGB PSNR is `5.720`

Media read: dense contact remains a grid/sparkle pattern on a dark background;
probe contact is still a blurred low-frequency reconstruction.

## Decision

Reject as scale-up route. The combination improves all sampled/low-frequency
visual metrics while worsening feature alignment and staying below the compact
route on dense full RGB (`5.720` vs `6.023`). This says the failure is not just
missing low-frequency colorizer stabilization. Future visual gates need a
stronger visibility-aware/dense-support bridge or output-basis change.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_compact_rgbgrid40_visual_bridge_gate.md`
