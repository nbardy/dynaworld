# STAR UVT RGB-Grid Low-Frequency Bridge Gate

Date: 2026-05-20 01:48 +07

## Why

The selected compact target-area visual route is fast enough for local work but
fails the dense visual-quality gate (`6.023` full RGB PSNR, sparse/streaked
media). The next planned experiment was to stop rearranging sparse visual support
and try a lower-frequency RGB bridge tied to the target-grid oracle.

## What Changed

Added `feature_target.rgb_grid_loss_weight` to
`src/train/train_star_uvt_feature_overfit.py`.

The new path trains the actual output `FeatureToColor` on the target-grid RGB
surface while keeping the fast `analytic_sparse_grid_forward_batched` sparse VJP
route. The helper uses a detached local grid, runs the trainable colorizer, takes
autograd gradients for the grid and colorizer parameters, manually accumulates
colorizer grads, and packs the grid gradient back through the sparse target-grid
VJP.

New checked-in config:

- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_rgbgrid40_feature1_probe40_from1500_lr001_50step_media.jsonc`

## Validation

- `PYTHONPATH=src/train rtk uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q`
- Result: `33 passed`

## Run

Command:

```bash
PYTHONPATH=src/train rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_rgbgrid40_feature1_probe40_from1500_lr001_50step_media.jsonc
```

W&B: offline run `n9bk0n0j`.

Result JSON:
`outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_rgbgrid40_feature1_probe40_from1500_lr001_50step_media.json`

Report:
`outputs/benchmarks/2026-05-20_star_uvt_rgb_grid_lowfreq_bridge_gate.md`

## Result

Mechanics pass:

- pass `true`
- zero tile overflow
- colorizer gradients seen
- STAR raw feature/geometry/opacity gradients seen
- mean/last step `353.13/268.25ms`
- no-first step `289.91ms`
- mean/last backward `130.24/108.12ms`

Quality fails:

- feature loss worsens `0.625418 -> 0.630230`
- trainable RGB-grid PSNR improves `22.028 -> 22.248`
- frozen RGB-probe PSNR improves `22.028 -> 22.114`
- dense full RGB PSNR is only `5.657`, below the selected compact route's
  `6.023` and far below RGB STAR's same-clip `12.444`
- dense contact sheet remains sparse/streaked

## Read

This is a useful negative. The actual colorizer can now be trained through the
fast target-grid sparse VJP, and the cost after warmup is tiny. But optimizing
the low-frequency 16x16 RGB-grid surface does not force coherent 512px dense
media. It can improve the metric it sees while harming feature-target alignment
and dense render quality.

Do not scale this route. The next visual experiment needs stronger dense or
visibility-aware support, or a model/output basis that cannot hide sparse
high-resolution failures behind a good low-resolution grid.
