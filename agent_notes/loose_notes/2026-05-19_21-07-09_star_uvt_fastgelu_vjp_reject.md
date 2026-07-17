# STAR UVT Fast-GELU Hidden64 VJP Rejection

## Goal

Test whether the exact-GELU backward bucket identified by the split manual
hidden64 profile is actionable by replacing the exact derivative with the
sigmoid-GELU derivative in the manual sparse visual VJP.

## Change

- Added `sparse_visual.loss_vjp_mode=manual_hidden64_fastgelu`.
- The mode keeps the exact hidden64 forward pass (`F.gelu`) and keeps colorizer
  parameter gradients, but uses the derivative of `x * sigmoid(1.702x)` for the
  hidden activation VJP.
- Added a focused test that verifies the fast derivative formula, same forward
  loss, same alpha gradients, same conv2 grads, and intentionally changed
  conv1/feature VJP.
- Extended the loss VJP profiler to accept any `manual_hidden64*` mode and to
  record the selected GELU gradient mode.

## Commands

Profile:

```bash
PYTHONPATH=src/train .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_fastgelu_from1500_lr001_5step_media.jsonc \
  --repeat 1 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_fastgelu_subphase_profile_fullstep
```

Benchmark:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_fastgelu_from1500_lr001_5step_media.jsonc
```

W&B: offline run `ulbcygz9` at
`wandb/offline-run-20260519_210709-ulbcygz9`.

## Results

- Mean step: `6252.1ms` versus exact manual `6414.0ms`.
- Sparse visual loss construction: `3416.7ms` versus exact manual `3803.6ms`.
- Sparse visual backward: `1130.2ms` versus exact manual `1026.1ms`.
- Profiled loss VJP total: `4525.0ms`, worse than split exact manual
  `3649.0ms`.
- GELU-backward bucket: `1401.6ms`, barely below exact manual `1435.4ms`.
- Final dense full RGB PSNR: `5.722`, unchanged and still bad.
- Feature target loss and probe PSNR reproduce the same nonpassing endpoint as
  exact manual (`0.626812`, `21.860` probe PSNR).

## Read

Fast-GELU derivative substitution is rejected. It creates a small end-to-end
trainer timing change but does not materially remove the profiled GELU-backward
cost, and it increases nearby matmul/grad buckets. More importantly, quality is
unchanged and still negative.

The next implementation should not spend more time on scalar derivative
approximations. It should move the hidden64 RGB/loss VJP boundary: fuse it into
the sparse-pixel backward or a visibility/prefix path, or replace the dense
hidden64 visual objective with a compact objective that avoids dense hidden
activation tensors entirely.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train_star_uvt_feature_overfit.py research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q`
  passed with `31 passed`.
