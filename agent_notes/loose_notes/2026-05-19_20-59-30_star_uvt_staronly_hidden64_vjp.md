# STAR UVT Star-Only Hidden64 VJP Diagnostic

## Goal

Check whether the full-cell8 target-area sparse visual row is slow mainly
because it accumulates trainable hidden64 colorizer parameter gradients, or
because the STAR-facing feature/alpha VJP itself is expensive.

## Code Change

- Added `sparse_visual.loss_vjp_mode=manual_hidden64_star_only` in
  `src/train/train_star_uvt_feature_overfit.py`.
- The mode reuses the manual hidden64 VJP and returns the same STAR
  `grad_feature_values` and `grad_alpha_values`, but skips colorizer parameter
  gradient accumulation.
- Updated the gradient-flow requirement so colorizer gradients are not required
  for this diagnostic mode.
- Added a focused parity test proving star-only loss, feature gradients, and
  alpha gradients match joint manual hidden64 while colorizer grads stay absent.
- Split the sparse visual loss VJP profiler so the old combined
  `conv1_param_feature_grad_ms` bucket is separated into exact GELU backward,
  conv1 parameter grad, and conv1 feature grad.

## Benchmark

Config:
`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_staronly_from1500_lr001_5step_media.jsonc`

Command:
`PYTHONPATH=src/train .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_staronly_from1500_lr001_5step_media.jsonc`

W&B: offline run `b6d3hfza` at
`wandb/offline-run-20260519_205651-b6d3hfza`.

Artifacts:

- JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_staronly_from1500_lr001_5step_media.json`
- Report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_staronly_from1500.md`
- Split joint profile:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_split_fullstep.md`
- Split star-only profile:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_staronly_subphase_profile_fullstep.md`

## Result

Compared with full-cell8 autograd and joint manual hidden64:

| Metric | Autograd | Joint manual | Star-only manual |
| --- | ---: | ---: | ---: |
| Mean step | `7526.7ms` | `6414.0ms` | `5801.7ms` |
| Sparse visual loss construction | `5702.6ms` | `3803.6ms` | `3405.1ms` |
| Sparse visual backward | `746.6ms` | `1026.1ms` | `1027.7ms` |
| Dense full RGB PSNR | `5.722` | `5.722` | `5.648` |
| Probe PSNR | `21.860` | `21.860` | `22.029` |
| Sparse visual PSNR | `5.822` | `5.822` | `5.746` |

The star-only row passes mechanically because colorizer gradients are no longer
part of the contract, but it is not a quality promotion. Dense RGB gets worse,
and sparse visual PSNR barely moves.

## Updated Bottleneck Read

The old profile bucket hid too much. After splitting it:

- Joint manual loss VJP full-step extrapolation: `3649.0ms`.
- Star-only manual loss VJP full-step extrapolation: `2977.1ms`.
- Exact GELU backward: `1435.4ms` joint, `1343.8ms` star-only.
- fc1 forward matmul: `888.7ms` joint, `751.0ms` star-only.
- Target-area loss/grad: only `129.0ms` joint, `118.1ms` star-only.
- Conv1 parameter grad is `269.7ms` joint and nearly zero when skipped, but
  removing it does not make the dense-support route viable.

So the next useful implementation is not "freeze colorizer" and not another
support shuffle. It is a fused or simplified hidden64 RGB/loss VJP boundary,
ideally fused into sparse-pixel backward or a visibility/prefix-tape path so we
do not materialize dense hidden tensors and dense per-pixel feature gradients in
Python.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train_star_uvt_feature_overfit.py research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q`
  passed with `30 passed`.
- Full 5-step star-only benchmark completed with zero tile overflow.
