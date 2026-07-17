# STAR UVT Phased Target-Area Sparse Visual Gate

Date: 2026-05-19 20:22:41 Asia/Ho_Chi_Minh

## Goal

After target-area64 rejected selected-patch target bias, test whether the real
problem is fixed sparse support. The new gate keeps per-step support fixed at
`1,048,576` sparse rendered visual pixels but cycles the `2x2` patch position
through a `4x4` subcell schedule inside each `8x8` target-area cell.

## Implementation

Added `sparse_visual.pixel_source="stratified_patch_grid_phase"` to
`src/train/train_star_uvt_feature_overfit.py`.

The new source:

- uses the same patch-grid shape and target-area loss as target-area64
- requires `target_size` divisible by the sample grid shape
- requires `patch_shape * patch_phase_shape` to fit inside one stratified cell
- records `sparse_visual_patch_phase_shape` and
  `sparse_visual_patch_phases` into the result JSON

For the 512px / `64x64` grid / `2x2` patch case, each cell is `8x8`, so a
`4x4` phase schedule can visit all non-overlapping `2x2` subpatches over 16
steps. Because this continuation starts at global step `1500`, the first phase
is `[3,0]`, then `[3,1]`, `[3,2]`, `[3,3]`, `[0,0]`, and so on.

Added focused tests for:

- phase patch ids shifting within cells without overlap
- row-major phase cycling
- existing target-area loss behavior still passing

## Run

Config:
`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_phase2x2_from1500_lr001_50step_media.jsonc`

Command:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_phase2x2_from1500_lr001_50step_media.jsonc
```

W&B:
`wandb/offline-run-20260519_202241-807f3ahx`

Result JSON:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_from1500_lr001_50step_media.json`

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_from1500.md`

## Result

| Metric | Value |
| --- | --- |
| Pass | `true` |
| Feature target loss | `0.6254179478 -> 0.6253407598` |
| Frozen RGB-probe PSNR | `22.027719 -> 22.045462` |
| Sparse visual PSNR | `5.694432 -> 6.076919` |
| Final dense full RGB loss / PSNR | `0.2500763834 / 6.019273` |
| Mean step / backward / render | `1169.16 / 694.99 / 97.72 ms` |
| Sparse visual render / loss / backward | `113.12 / 435.34 / 160.07 ms` |
| Sparse visual pixels / loss cells | `1,048,576 / 262,144` |
| Tile overflow | `0` |

Compared with target-area64, phase cycling slightly improves sparse visual PSNR
but is slower and lowers dense full RGB PSNR. Media inspection still shows
sparse/high-frequency structure in the main contact sheet and blurred frozen
probe output.

## Read

This rejects fixed support position as the main explanation. We now have three
negative sparse-support/basis gates after the mixed sparse visual row:

- patch2x2 improves sparse samples but worsens feature/dense RGB
- target-area64 improves sparse sample PSNR but leaves dense RGB unchanged
- phase2x2 target-area improves sparse sample PSNR a little more but lowers
  dense RGB

The next implementation should be a true visibility/prefix tape or stronger
compact dense visual-gradient path, not another sparse support shuffle.

## Validation

- `python -m py_compile src/train/train_star_uvt_feature_overfit.py src/train/train.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q`
  passed with `28 passed`.
- Config preflight verified `patch_phase_shape=[4,4]`, first phases
  `[3,0]`, `[3,1]`, `[3,2]`, `[3,3]`, `[0,0]`, `[0,1]`, `[0,2]`, `[0,3]`,
  and `32,768` sparse visual pixels per 2-frame chunk.
