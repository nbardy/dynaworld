# STAR UVT Native Hidden Trainer Gate

Date: 2026-05-19

## Goal

Wire the sparse cached-bin hidden sigmoid-MSE native kernel into the
first-class STAR UVT feature trainer, then compare it against the existing
manual hidden64 star-only sparse visual VJP path on the same real 64f/512px
checkpoint.

## Implementation

- Added `native_hidden_star_only` and `native_hidden64_star_only` to
  `sparse_visual.loss_vjp_mode`.
- Restricted native sparse visual VJP to `loss_basis=pixel` and
  `loss_weight=1.0`.
- Plumbed `total_loss_elems` into
  `direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins` through
  `meta_i32.reserved1`, so chunked trainer calls normalize each chunk against
  the full step denominator.
- In the trainer sparse-visual loop, the native branch calls the fused hidden
  cached-bin backward directly and feeds returned STAR gradients to
  `torch.autograd.backward(...)`; colorizer parameter gradients remain omitted.
- Added matched 5-step configs:
  - `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_pixel64_manualhidden_staronly_from1500_lr001_5step_media.jsonc`
  - `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_pixel64_nativehidden_staronly_from1500_lr001_5step_media.jsonc`

## Result

Report:
`outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_nativehidden_trainer_gate.md`

Warm means exclude step 0.

| mode | step ms | sparse render ms | sparse loss ms | sparse backward ms | sparse loss+backward ms | final sparse loss |
|---|---:|---:|---:|---:|---:|---:|
| manual hidden64 star-only | `405.97` | `70.57` | `45.53` | `67.71` | `113.25` | `0.271377709` |
| native hidden64 star-only | `403.83` | `71.62` | `0.00` | `116.27` | `116.27` | `0.271377741` |

The native trainer row passes and matches the manual endpoint
(`3.26e-08` end sparse-loss difference), but it does not speed up this pixel64
trainer case. Loss+backward is `0.974x` versus manual, and step time is
effectively tied (`1.005x`). Both rows see STAR gradients, see no colorizer
gradients, and have zero tile overflow.

## Read

The standalone sparse native kernel result was real, but trainer promotion is
support-dependent. At `262,144` selected pixels/step, the manual loss-side VJP
is only ~`45.5ms` warm, so folding it into a heavier native STAR reverse does
not win end-to-end. The expensive row remains full-cell/target-area support,
where Python hidden VJP costs seconds. Next work should port compact dense
visual gradients or visibility/prefix tape for that basis, not keep tuning the
pixel64 native branch.

## Validation

- `py_compile` passed for the trainer and native wrapper.
- The `star_uvt_v0` extension rebuilt cleanly.
- Tiny parity after the normalization patch passed:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_sigmoid_mse_native_tiny_parity_after_norm_patch.json`
- Manual trainer gate passed:
  `wandb/offline-run-20260519_223813-fy65puoi`
- Native trainer gate passed:
  `wandb/offline-run-20260519_223828-tj1tm2ce`
