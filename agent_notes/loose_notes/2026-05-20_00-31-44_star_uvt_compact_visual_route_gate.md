# STAR UVT Compact Visual Route Gate

Date: 2026-05-20

## Why

After promoting exact full-cell8 native vec4 W^T as the full-support baseline,
the remaining question was what route should actually be used for a fast
single-video visual overfit. The full-support baseline is exact but slower and
lower quality than the older compact target-area row. I added a dated
current-build compact config and pointed the helper's `star-feature-512-visual`
mode at it so old 2026-05-19 artifacts are not overwritten.

## Command

```bash
WANDB_MODE=offline rtk ./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh \
  star-feature-512-visual
```

Offline W&B run: `wo20rbow`.

## Results

The helper route passes and writes fresh 2026-05-20 JSON/media/checkpoint
artifacts.

- weighted loss: `1.14673298 -> 1.12264495`
- feature target loss: `0.62541795 -> 0.62534487`
- sparse visual loss: `0.27053779 -> 0.24752884`
- RGB probe PSNR: `22.027719 -> 22.045176`
- sparse visual PSNR: `5.677720 -> 6.063742`
- final full RGB PSNR: `6.023322`
- tile overflow: `0`, max/p95/cap `69/48/128`
- mean step/backward: `930.62/581.32ms`
- mean sparse visual loss/backward: `379.74/125.71ms`

This beats the full-cell8 native vec4 W^T 50-step gate on both speed and dense
RGB endpoint (`930.62ms`, `6.023` full RGB versus `3359.20ms`, `5.732` full
RGB).

## Decision

Use `star-feature-512-visual` as the practical fast visual overfit route for
now. Keep `star-feature-512-native-fullcell` as the exact full-support native
shader baseline and parity target.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_compact_target_area_visual_route_gate.md`
