# STAR UVT Patch-Mean Visual Basis Gate

Date: 2026-05-19 19:56:39

## Why

The mixed target-grid/probe plus sparse visual VJP gate preserved the token/probe
objective but left dense RGB at `6.024` PSNR. The same-pixel-count patch2x2
support gate was faster and improved sampled sparse visual PSNR, but it worsened
feature loss and dense RGB. That rejected support layout alone.

The next question was whether a denser visual basis helps: sample local `2x2`
patches on a denser `64x64` lattice, pool each patch to a local mean, and train
that low-frequency visual cell loss while keeping the selected sparse-forward
batched target/probe path.

## Implementation

- Added `sparse_visual.loss_basis`.
  - `pixel` keeps the previous sparse visual RGB loss unchanged.
  - `patch_mean` reshapes `stratified_patch_grid` samples into
    `[local_frames, grid_h, patch_h, grid_w, patch_w, 3]` and averages over the
    local patch axes before loss.
- Added `sparse_visual_loss_sample_counts` and
  `mean_sparse_visual_loss_sample_count` to the result row.
- Added focused unit coverage for patch-mean pooling and loss-cell counting.
- Added config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_patchmean64_from1500_lr001_50step_media.jsonc`.

## Run

Command:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_patchmean64_from1500_lr001_50step_media.jsonc
```

W&B offline run:

```text
wandb/offline-run-20260519_195639-g8eg20l8
```

Result:

- pass: `true`
- total loss: `1.147896 -> 1.123807`
- feature target loss: `0.625418 -> 0.625345`
- frozen probe PSNR: `22.028 -> 22.045`
- sparse visual PSNR: `5.659 -> 6.043`
- final full RGB loss / PSNR: `0.249843 / 6.023`
- mean step / backward / render: `1124.58 / 703.48 / 92.49 ms`
- sparse visual render/loss/backward:
  `113.60 / 446.49 / 157.83 ms`
- sparse visual pixels: `1,048,576` per step
- sparse visual patch-mean loss cells: `262,144` per step
- tile overflow: `0`

Report:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patchmean64_from1500.md
```

## Read

Patch-mean64 is better than patch2x2 as an objective gate: it restores
feature/probe movement and brings dense RGB back to the mixed stratified64
range. But it is not a quality promotion. It costs `1.125s/step`, and media
still shows sparse high-frequency structure in the main render plus a blurred
frozen-probe view.

This suggests the next lever is not another sparse sample pattern. The current
path needs a compact dense visual gradient representation or a visibility/prefix
tape so denser visual supervision can reach the tubes without full dense F32
image backward.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train_star_uvt_feature_overfit.py src/train/train.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q`
  - `24 passed`
- Preflight: `32,768` unique sparse pixels per 2-frame chunk, `1,048,576`
  pixels/step, `262,144` patch-mean cells/step.
- Comparison regenerated with `40` rows:
  `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.{json,md}`.
