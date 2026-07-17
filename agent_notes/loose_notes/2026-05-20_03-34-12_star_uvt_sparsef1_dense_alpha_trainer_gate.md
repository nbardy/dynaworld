# STAR UVT Sparse-F1 Dense-Alpha Trainer Gate

## Context

The alpha-only visibility profile showed that sparse-pixel F1 alpha render plus
cached F1 backward is correct and faster than dense F32 alpha rendering. This
follow-up wires that implementation into the trainer as an opt-in dense-alpha
render mode and benchmarks it against the previous dense F32 dense-alpha gate.

This is not a new visual-quality hypothesis. It is an implementation hardening
step for the current plan: if alpha-only visibility diagnostics continue, they
should not pay dense F32 image cost.

## Code/config changes

- `src/train/train_star_uvt_feature_overfit.py`
  - added `dense_alpha.render_mode`
  - supported values: `dense_f32`, `sparse_f1`
  - default remains `dense_f32`
  - `sparse_f1` uses `render_uvt_feature_alpha_all_pixels_with_bins` plus
    `direct_atomic_feature_backward_cached_bins` with a dummy F1 feature tensor
- `tests/test_star_uvt_feature_target_adapter.py`
  - validates `dense_alpha.render_mode`
- Added config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_densealpha075_sparsef1_from1500_lr001_5step_media.jsonc`

The sparse-F1 config sets `train.require_loss_decrease=false` so the speed row
can complete cleanly, but the trainer row still records `pass=false` because
loss and dense-alpha loss do not decrease.

## Commands

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py
```

```bash
PYTHONPATH=src/train rtk uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py -q
```

```bash
PYTHONPATH=src/train rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_densealpha075_sparsef1_from1500_lr001_5step_media.jsonc
```

## Results

Matched dense F32 alpha row:

- mean step/backward `2558.6/1114.2ms`
- dense-alpha render/loss/backward `834.5/124.6/858.9ms`
- weighted loss `1.271702 -> 1.284505`
- dense alpha loss `0.395507 -> 0.397107`
- feature loss `0.625418 -> 0.626814`
- RGB-probe PSNR `22.028 -> 21.861`
- dense RGB PSNR `5.647`
- pass false

Sparse-F1 alpha trainer row:

- mean step/backward `873.3/370.0ms`
- dense-alpha render/loss/backward `276.0/22.0/303.7ms`
- weighted loss `1.271702 -> 1.284505`
- dense alpha loss `0.395507 -> 0.397107`
- feature loss `0.625418 -> 0.626814`
- RGB-probe PSNR `22.028 -> 21.861`
- dense RGB PSNR `5.647`
- zero overflow
- pass false
- W&B offline run `wandb/offline-run-20260520_032737-0uqvhxwe`

Speed ratios:

- step `2.93x`
- backward `3.01x`
- dense-alpha render `3.02x`
- dense-alpha backward `2.83x`

## Interpretation

This is a clean speed implementation win and a clean quality non-promotion.
The cheap sparse-F1 path reproduces the dense F32 alpha endpoint while removing
most of the alpha-only overhead. That means the remaining STAR UVT visual
blocker is not "dense alpha was too expensive"; it is that same-support dense
alpha is the wrong objective/support bridge.

Future alpha-only diagnostics should use `dense_alpha.render_mode=sparse_f1`.
Do not launch the 300-video scale lane from this result.

## Artifacts

- `outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_sparsef1_trainer_gate.md`
- `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_densealpha075_sparsef1_from1500_lr001_5step_media.json`
- `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_densealpha075_sparsef1_from1500_lr001_5step.pt`
- `outputs/media/2026-05-20_star_uvt_feature_targetgrid_densealpha075_sparsef1_from1500_lr001_5step_contact.jpg`
- `outputs/media/2026-05-20_star_uvt_feature_targetgrid_densealpha075_sparsef1_from1500_lr001_5step_probe_contact.jpg`
