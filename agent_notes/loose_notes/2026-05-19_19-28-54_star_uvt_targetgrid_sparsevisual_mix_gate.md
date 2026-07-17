# STAR UVT Target-Grid Sparse-Visual Mixed Gate

Date: 2026-05-19

## Question

The prior gates showed that sparse visual RGB-only training is not enough:
colorizer-only stratified rendered pixels reached only `6.132` dense full-video
PSNR, frozen sparse visual VJP was worse at `5.739`, and joint STAR+colorizer
sparse visual VJP recovered only to `6.025`. This gate tested whether mixing
native sparse visual VJP with the selected target-grid feature/probe objective
fixes that quality problem.

## Implementation

`src/train/train_star_uvt_feature_overfit.py` now has an optional
`sparse_visual` section. When enabled, each optimizer step first runs the
existing target-grid/probe objective, then runs a native sparse visual RGB pass:

```json
"sparse_visual": {
  "enabled": true,
  "loss_weight": 1.0,
  "pixel_source": "stratified_grid",
  "sample_grid_shape": [64, 64, 64]
}
```

The visual path samples a full-resolution stratified lattice, renders only those
pixels with cached bins, computes sparse RGB loss through the trainable
`FeatureToColor`, accumulates colorizer gradients through local autograd, and
pushes feature/alpha gradients back into STAR with
`direct_atomic_feature_sparse_pixels_backward_cached_bins`.

## Run

- Config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_mix_from1500_lr001_50step_media.jsonc`
- Command:
  `PYTHONPATH=src/train .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_mix_from1500_lr001_50step_media.jsonc`
- W&B:
  `wandb/offline-run-20260519_192854-lzrfgv62`

## Result

- Pass: `true`
- Feature loss: `0.625418 -> 0.625363`
- RGB-probe PSNR: `22.028 -> 22.045`
- Sparse visual loss: `0.271902 -> 0.249118`
- Sparse visual PSNR: `5.656 -> 6.036`
- Final full RGB loss / PSNR: `0.249817 / 6.024`
- Mean step/render/backward: `964.01 / 110.25 / 473.66 ms`
- Mean sparse visual render/loss/backward: `122.71 / 166.14 / 165.91 ms`
- Last step/render/backward: `611.62 / 81.41 / 290.89 ms`
- Target-grid sparse source pixels: `65,536` per step
- Sparse visual pixels: `262,144` per step
- Tile overflow: `0`

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_mix_from1500.md`

JSON:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_mix_from1500_lr001_50step_media.json`

## Read

This is a mechanics pass but a quality negative. Mixing target-grid
feature/probe supervision with sparse visual RGB VJP preserves the target-grid
line and improves sparse visual sample loss, but it does not improve dense RGB
quality. Full RGB PSNR is `6.024`, effectively tied with the joint sparse
visual-only gate (`6.025`) and still below the colorizer-only stratified
rendered-feature diagnostic (`6.132`).

The result rules out the simple hypothesis that target-grid/probe supervision
would make sparse RGB-only visual VJP become a dense-quality bridge. The
remaining issue is visual support/basis: sampled sparse RGB gradients still do
not make the full-resolution feature/alpha field visually coherent.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train_star_uvt_feature_overfit.py src/train/train.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q`
  - `22 passed`
- Regenerated
  `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.{json,md}`
  with `38` rows and the mixed gate row present.
- JSON invariants passed for mixed gate loss decreases, gradients, dense PSNR
  comparison, sparse visual pixel count, and comparison row count.
- `agent_notes/key_learnings.md` remains at `199` lines.
- `git diff --check` passed after doc sync.
