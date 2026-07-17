# STAR UVT Patch Sparse Visual Support Gate

## Why

The mixed target-grid/probe plus stratified64 sparse visual VJP gate preserved
the target-grid objective and improved sparse visual sample loss, but dense full
RGB stayed at `6.024`. The next planned gate was to change visual support/basis
instead of remixing the same objectives.

This gate keeps the sparse visual pixel count fixed and changes only the support
shape: `stratified_grid [64,64,64]` becomes `stratified_patch_grid
[64,32,32]` with `patch_shape=[2,2]`, still `262,144` pixels/step.

## Implementation

- Added `sparse_visual.pixel_source=stratified_patch_grid` to
  `src/train/train_star_uvt_feature_overfit.py`.
- Added optional `sparse_visual.patch_shape=[height,width]`.
- Added `_stratified_patch_indices(...)` and extended
  `_sparse_visual_pixel_ids_for_chunk(...)`.
- Added a focused unit test proving chunk-local contiguous patch ids.
- Added config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_patch2x2_from1500_lr001_50step_media.jsonc`.

## Run

Command:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_patch2x2_from1500_lr001_50step_media.jsonc
```

W&B offline run:
`wandb/offline-run-20260519_194552-odmaokfj`

## Result

- Pass bit: `true`
- Total loss: `1.147087 -> 1.116758`
- Feature target loss: `0.625418 -> 0.625532` (`feature_target_loss_decreased=false`)
- Frozen probe PSNR: `22.028 -> 22.038`
- Sparse visual PSNR: `5.672 -> 6.179`
- Final dense full RGB loss / PSNR: `0.251182 / 6.000`
- Mean step/backward/render: `619.54 / 290.72 / 88.88 ms`
- Mean sparse visual render/loss/backward: `89.71 / 100.95 / 101.47 ms`
- Sparse visual pixels: `262,144` per step
- Tile overflow: `0`

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patch2x2_from1500.md`

JSON:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patch2x2_from1500_lr001_50step_media.json`

## Read

This is a negative support-basis result. Patch support improves the sparse
sample objective and is faster than the mixed stratified64 gate at the same
pixel count, but it does not improve dense visual quality. Dense full RGB PSNR
falls to `6.000`, below mixed stratified64 `6.024`, joint sparse visual `6.025`,
and colorizer-only stratified `6.132`.

The pass bit only means total loss, sparse visual loss, gradient flow, and tile
overflow checks passed. It should not be read as feature-objective or visual
promotion because the feature-target component worsened.

The next gate should stop rearranging sparse RGB support and move to a denser
visual basis: downsampled dense visual loss without full F32 image backward, or
a compact visibility/prefix tape that carries denser visual gradients back to
tubes.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train_star_uvt_feature_overfit.py src/train/train.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q`
  - `23 passed`
- Config preflight:
  - `TrainerEntry(module='train_star_uvt_feature_overfit', runner='run_training')`
  - patch-grid ids: `torch.int32`, `8192` unique pixels per 2-frame chunk.
- Regenerated comparison report:
  - `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  - `39` rows.
