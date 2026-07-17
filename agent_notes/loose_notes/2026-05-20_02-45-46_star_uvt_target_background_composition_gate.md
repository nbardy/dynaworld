# STAR UVT Target-Background Composition Gate

Date: 2026-05-20

## Goal

After alpha-to-one, phase-covered alpha, and target-aware black-hole penalties
failed, test a different axis: change sparse visual composition itself. The
question was whether the target-background oracle from the dense diagnostic can
be turned into training signal.

## What Changed

- Added `sparse_visual.composition`, default `black`.
- Added `target_background` composition:
  `target + alpha * (pred_rgb - target)`.
- Kept native sparse visual VJP modes on black composition only.
- Added unit coverage for the empty-alpha behavior.
- Added two configs:
  - `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_targetbg_from1500_lr001_50step_media.jsonc`
  - `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_targetbg_alpha1_from1500_lr001_50step_media.jsonc`

## Runs

Target-background:

```bash
PYTHONPATH=src/train rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_targetbg_from1500_lr001_50step_media.jsonc
```

Target-background plus alpha:

```bash
PYTHONPATH=src/train rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_targetbg_alpha1_from1500_lr001_50step_media.jsonc
```

Report:

- `outputs/benchmarks/2026-05-20_star_uvt_target_background_composition_gate.md`

Dense diagnostic:

- `outputs/benchmarks/2026-05-20_star_uvt_targetbg_alpha1_dense_alpha_diagnostic.md`

W&B offline:

- `wandb/offline-run-20260520_024028-hz1bp404`
- `wandb/offline-run-20260520_024344-owjahu3m`

## Result

Target-background alone mechanically passes but is not a visual promotion:

- Feature loss improves `0.625418 -> 0.624822`.
- Probe PSNR improves `22.028 -> 22.045`.
- Sparse visual PSNR improves `25.779 -> 27.759`.
- Dense full RGB is only `5.666`, worse than compact black-background `6.023`.
- Dense diagnostic: forced-alpha PSNR improves to `14.891`, oracle composition
  to `27.443`, but alpha `>0.1` falls to `40.8%`.

Target-background plus alpha is also rejected:

- Alpha loss improves `0.752440 -> 0.738556`.
- Sparse visual PSNR improves `25.779 -> 27.354`.
- Feature loss worsens `0.625418 -> 0.626734`.
- Probe PSNR drops `22.028 -> 21.901`.
- Dense full RGB is only `5.748`.
- Dense diagnostic: forced-alpha PSNR is `14.899`, oracle composition is
  `27.105`, and alpha `>0.1` is `43.1%`.

## Read

The target-background idea is informative because it separates color/content
from coverage. It makes the forced-alpha/oracle image much better, but the
black-background render still fails. Adding sampled alpha-to-one recovers the
old alpha coverage range but loses feature/probe quality and does not recover
dense RGB.

The next bridge has to create dense visibility while preserving the improved
target-background color path. Another scalar same-support alpha penalty is not
the right next move.
