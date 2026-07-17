# STAR UVT Native Colorizer Vec4 Target-Area VJP Gate

Date: 2026-05-20 00:56 +07

## Goal

Implement the missing compact native target-area VJP variant that returns
colorizer parameter gradients, then test whether it beats the compact autograd
visual route.

## Code Changes

- Native op now returns four extra tensors from
  `direct_atomic_feature_sparse_hidden_target_area_backward_with_bins`:
  `grad_hidden_weight`, `grad_hidden_bias`, `grad_output_weight`,
  `grad_output_bias`.
- Added mode bit `128` via `target_area_colorizer_vec4_wt`.
- Added trainer mode `native_hidden64_target_area_colorizer_vec4_wt`.
- Trainer now writes returned native colorizer gradients onto the
  `FeatureToColor` conv params in the native target-area branch.

## Build

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Build passed.

## Tiny Parity

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 4,32 \
  --hidden-dim 64 \
  --backward-mode target_area_colorizer_vec4_wt \
  --timing-repeat 3 \
  --timing-warmup 1 \
  --timing-size 64 \
  --timing-frames 8 \
  --timing-tubes 256 \
  --tile-capacity 256 \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_vec4_wt_tiny_gate.json
```

Pass. Tiny parity checked loss, STAR geometry/feature grads, and all four
colorizer parameter-gradient tensors:

- F4 max errors: feature `8.73e-11`, hidden weight `1.16e-10`, hidden bias
  `5.82e-10`, output weight `2.91e-10`, output bias `9.31e-10`
- F32 max errors: feature `2.33e-10`, hidden weight `1.75e-10`, hidden bias
  `1.63e-09`, output weight `4.66e-10`, output bias `1.86e-09`
- no tile overflow in the timing row

## Trainer Gate

Config:
`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_nativecolorizer_vec4wt_from1500_lr001_5step_diagnostic.jsonc`

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_nativecolorizer_vec4wt_from1500_lr001_5step_diagnostic.jsonc
```

W&B offline run: `6hocfst2`

Output JSON:
`outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_nativecolorizer_vec4wt_from1500_lr001_5step_diagnostic.json`

## Result

Rejected.

- `colorizer_grad_required=true`
- `colorizer_grad_seen=true`
- zero tile overflow
- mean/no-first step: `2738.75/2312.82ms`
- mean/no-first backward: `1474.24/1429.12ms`
- mean sparse visual loss/backward: `403.70/871.46ms`
- weighted loss: `1.146733 -> 1.153926`
- feature loss: `0.625418 -> 0.626795`
- RGB-probe PSNR: `22.0277 -> 21.8596`

The comparison report now includes compact autograd, compact native star-only,
compact manual hidden64, and compact native colorizer vec4 W^T:
`outputs/benchmarks/2026-05-20_star_uvt_compact_visual_vjp_gate.md`.

## Decision

The native colorizer-gradient return path is correct, but returning gradients is
not enough. It adds expensive per-pixel atomics into tiny colorizer tensors and
lands slower than both compact autograd and manual hidden64 while preserving the
same bad first-5-step quality movement as manual hidden64.

Keep `star-feature-512-visual` / compact autograd as the practical single-video
visual overfit route. The next native attempt needs a reducer or different
support/objective that removes the colorizer-gradient atomic envelope, not just
a new returned-gradient ABI.
