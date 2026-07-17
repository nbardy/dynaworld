# STAR UVT Vec4 W^T 50-Step Gate

Date: 2026-05-20

## Why

The current-build A/B promoted exact vec4 W^T as the preferred full-support
native target-area star-only VJP mode, but that was still a 5-step gate. I ran a
50-step promoted-mode media/checkpoint gate to see if it stays stable, warms to
a better steady timing regime, and improves quality enough to change the next
plan.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_nativehidden_vec4wt_from1500_lr001_50step_media.jsonc
```

Offline W&B run: `78q76ryx`.

## Results

The run passes with `require_loss_decrease=true`, gradient flow, and zero tile
overflow.

- loss: `1.14298183 -> 1.13633765`
- feature target loss: `0.62541795 -> 0.62528551`
- sparse visual loss: `0.26678663 -> 0.26128818`
- RGB probe PSNR: `22.027719 -> 22.045302`
- sparse visual PSNR: `5.738359 -> 5.828803`
- final full RGB PSNR: `5.731537`
- tile max/p95/cap: `69/47/128`

Mean timing is `3359.20ms` step, `2934.35ms` backward,
`984.82ms` sparse visual loss, and `1862.80ms` sparse visual backward. Last-step
timing is `3072.12ms` step and `2689.69ms` backward.

## Interpretation

This confirms the promoted mode is not just a 5-step artifact. It is the
full-support native target-area baseline now.

It is still not the overfit answer. The earlier compact target-area64 50-step
row is much faster (`1103ms` mean step) and gets better dense RGB (`6.023`
PSNR). Full-cell8 exact support is useful as a correctness/speed baseline, but
we still need either a cheaper exact support path or a better objective/support
choice for dense visual quality.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_50step_gate.md`

## Script routing follow-up

I updated `src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh`
so the next agent does not have to infer the selected route from notes:

- `star-feature-512-fast`: fastest batched sparse-forward V-JEPA target-grid
  diagnostic.
- `star-feature-512-visual`: current compact target-area visual overfit route.
- `star-feature-512-native-fullcell`: promoted exact full-support native
  target-area vec4 W^T baseline.

This keeps the speed diagnostic, visual overfit route, and exact full-support
shader baseline separate.
