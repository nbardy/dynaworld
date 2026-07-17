# STAR UVT Compact Native Star-Only Diagnostic

Date: 2026-05-20

## Why

After selecting compact target-area as the practical visual overfit route, I
checked whether the newly promoted native target-area vec4 W^T path could be
used directly on compact support. This is an important gap check because the
existing native target-area modes are `star_only`; they do not train the
colorizer.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_nativehidden_vec4wt_from1500_lr001_5step_diagnostic.jsonc
```

Offline W&B run: `qd9edudt`.

## Result

The diagnostic passes mechanically with zero overflow, but it is rejected:

- `colorizer_grad_required=false`, `colorizer_grad_seen=false`
- mean/no-first step: `2265.02/1870.42ms`
- mean/no-first backward: `957.31/904.40ms`
- mean sparse visual loss/backward: `360.08/413.12ms`
- sparse visual loss: `0.27053779 -> 0.27001877`
- feature target loss worsens: `0.62541795 -> 0.62543344`

By comparison, the compact autograd visual route is `930.62ms` mean step and
`581.32ms` mean backward while training the colorizer and reaching `6.023` full
RGB.

## Decision

Do not route compact visual overfit through native star-only target-area. The
next native port must preserve colorizer gradients and beat the compact
autograd route; otherwise stay with `star-feature-512-visual`.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_compact_native_staronly_diagnostic.md`
