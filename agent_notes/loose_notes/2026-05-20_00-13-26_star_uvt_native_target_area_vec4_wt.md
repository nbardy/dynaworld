# STAR UVT Native Target-Area Vec4 W^T

Date: 2026-05-20

## Why

Row-major W^T proved that simple memory-ordering was not enough: it preserved
gradients but slowed the full path. The next exact W^T test was vectorizing the
feature-channel reconstruction in `float4` blocks while keeping the same
hidden64 math and full gradients.

## What Changed

- Added native modes `target_area_star_only_vec4_wt` and
  `target_area_recompute_only_vec4_wt`.
- Added trainer modes `native_hidden_target_area_star_only_vec4_wt` and
  `native_hidden64_target_area_star_only_vec4_wt`.
- Added the matched 5-step trainer config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_nativehidden_vec4wt_from1500_lr001_5step_media.jsonc`.
- Rebuilt `star_uvt_v0`.

## Results

Tiny F4/F32 parity passes. F32 feature-gradient max error is `2.33e-10`.

Same-build direct-kernel timing:

| Mode | 256 total/backward ms | 512 total/backward ms |
|---|---:|---:|
| canonical full | `874.52 / 675.89` | `2990.25 / 2408.12` |
| vec4 full | `866.33 / 642.21` | `2411.31 / 1804.71` |
| vec4 full repeat | n/a | `2358.69 / 1832.82` |
| canonical recompute-only | `789.51 / 586.64` | `2849.91 / 2305.17` |
| vec4 recompute-only | `753.79 / 518.32` | `2227.60 / 1635.75` |

Trainer smoke passed as offline W&B run `ct79s3ii`. It kept the endpoint class
the same (`0.625451` feature loss, `22.029` probe PSNR, `5.648` full RGB, zero
overflow), and sparse visual backward was slightly lower than the older
canonical artifact (`1963.54ms` vs `2056.10ms`). It was not a whole-step
promotion yet: mean step was `4071.01ms` versus the older canonical artifact's
`3495.95ms`, with more time showing up in target-area forward/loss.

## Decision

Keep vec4 W^T as a positive exact kernel gate and an opt-in trainer mode. Do not
make it the default full-support native visual-VJP mode until a same-window
trainer A/B confirms total step time. The next useful gate is trainer A/B or a
follow-up that attacks target-area forward/loss, because the direct backward
path improved but the trainer total did not yet.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_gate.md`
