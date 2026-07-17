# STAR UVT Manual Hidden64 Sparse Visual VJP

Date: 2026-05-19

## Context

The full-cell8 target-area gate proved that Python-side dense visual support is
both slow and quality-negative. The dominant measured cost was sparse visual
loss construction: `5702.60ms` of a `7526.73ms/step` mean. Before starting a
larger Metal fork, I added a manual hidden64 colorizer VJP so the loss-side
math is explicit and parity-tested.

## Implementation

`src/train/train_star_uvt_feature_overfit.py` now supports:

`sparse_visual.loss_vjp_mode=manual_hidden64`

This mode supports the current selected sparse visual setup:

- `FeatureToColor`
- `hidden_dim=64`
- `pre_norm=false`
- `view_condition=none`
- exact `GELU`
- final `sigmoid` or identity activation

It manually computes the hidden64 colorizer forward, target-area mean loss,
per-pixel `grad_feature_values`, `grad_alpha_values`, and colorizer parameter
gradients. The sparse-pixel Metal backward still consumes the resulting per-pixel
feature/alpha gradients, so this is a loss-side VJP scaffold, not a fused
visibility/prefix kernel.

Focused test coverage now includes a target-area parity test against the old
autograd path:

`tests/test_star_uvt_feature_target_adapter.py::test_sparse_visual_manual_hidden64_vjp_matches_autograd_target_area`

The focused suite passed with `29 passed`.

## Benchmark

Config:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_manualvjp_from1500_lr001_5step_media.jsonc`

W&B offline run:

`wandb/offline-run-20260519_204335-dllwz35x`

Result JSON:

`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_manualvjp_from1500_lr001_5step_media.json`

Report:

`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_manualvjp_from1500.md`

## Result

Compared with the autograd full-cell8 row:

- mean step improves `7526.73ms -> 6413.96ms`
- mean backward improves `6569.12ms -> 4990.62ms`
- sparse visual loss construction improves `5702.60ms -> 3803.65ms`
- sparse visual render regresses `456.50ms -> 617.23ms`
- sparse visual backward regresses `746.57ms -> 1026.08ms`
- endpoint quality is effectively identical and still bad:
  `5.722436` dense full RGB PSNR
- pass remains `false`

## Read

The manual VJP is worth keeping as a parity target and lower-overhead scaffold.
It proves a meaningful part of the cost was Torch autograd/materialization
overhead through the hidden64 colorizer. It does not rescue full dense support:
the row remains far too slow and quality-negative.

Next work should move the manual VJP shape into a native fused
visibility/prefix or RGB/loss/gradient path. More Python-side dense support
experiments are unlikely to change the promotion decision.
