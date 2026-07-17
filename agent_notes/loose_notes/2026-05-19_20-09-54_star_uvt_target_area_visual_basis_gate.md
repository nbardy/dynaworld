# STAR UVT Target-Area Sparse Visual Basis Gate

Date: 2026-05-19 20:09:54 Asia/Ho_Chi_Minh

## Goal

Test whether the patch-mean64 sparse visual gate was limited by using sampled
patch pixels as the RGB target. The alternative here keeps the same sparse
rendered patch support, but compares each local patch-mean prediction against a
true area-downsampled dense RGB target cell.

## Implementation

Updated `src/train/train_star_uvt_feature_overfit.py` with a new
`sparse_visual.loss_basis="target_area_mean"` mode:

- `pixel` still compares selected rendered RGB samples to selected RGB targets.
- `patch_mean` still compares local rendered patch means to local selected-target
  patch means.
- `target_area_mean` compares local rendered patch means to
  `F.interpolate(..., mode="area")` target RGB cells at the same
  `[frames, grid_h, grid_w, 3]` layout.

The trainer now computes local chunk frame ids once for sparse visual chunks and
passes the dense target chunk only when `target_area_mean` needs it. The full
step still samples `1,048,576` rendered sparse visual pixels and reduces them to
`262,144` loss cells.

Added test coverage in `tests/test_star_uvt_feature_target_adapter.py`:

- dense area-downsampled target equality drives zero loss and zero gradient when
  the rendered patch mean already matches the dense area target
- chunk-local frame ids match the selected render-frame window

## Run

Config:
`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_from1500_lr001_50step_media.jsonc`

Command:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_from1500_lr001_50step_media.jsonc
```

W&B:
`wandb/offline-run-20260519_200954-8xbz5xhg`

Result JSON:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_from1500_lr001_50step_media.json`

Report:
`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_from1500.md`

## Result

| Metric | Value |
| --- | --- |
| Pass | `true` |
| Feature target loss | `0.6254179478 -> 0.6253448725` |
| Frozen RGB-probe PSNR | `22.027719 -> 22.045176` |
| Sparse visual PSNR | `5.677720 -> 6.063742` |
| Final dense full RGB loss / PSNR | `0.2498433292 / 6.023322` |
| Mean step / backward / render | `1103.13 / 664.36 / 87.08 ms` |
| Sparse visual render / loss / backward | `108.34 / 424.62 / 145.20 ms` |
| Sparse visual pixels / loss cells | `1,048,576 / 262,144` |
| Tile overflow | `0` |

Compared with patch-mean64, target-area64 is slightly faster and the sparse
visual sample PSNR is slightly higher. Dense RGB PSNR and media are unchanged.

## Read

This rejects selected-patch target bias as the reason sparse visual VJP is not
promoting dense RGB quality. The missing piece is not another sparse support
shuffle; it is a denser visual-gradient representation that can still fit the
fast STAR UVT route. The next implementation fork should be visibility/prefix
tape or a compact dense visual-gradient path.

## Validation

- `python -m py_compile src/train/train_star_uvt_feature_overfit.py src/train/train.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q`
  passed with `26 passed`.
- Config preflight verified `loss_basis=target_area_mean`, `1,048,576` sparse
  visual pixels, and `262,144` loss cells.
