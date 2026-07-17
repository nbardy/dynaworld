# STAR UVT Alpha Coverage Gate

## Context

The dense alpha diagnostic showed that compact, RGB-grid, and compact+RGB-grid
STAR UVT feature routes are coverage/visibility/composition limited. Forced
alpha raises dense RGB PSNR substantially, but the actual black-background
composite stays near `5.7-6.0` PSNR and alpha `>0.1` covers only about
`41.5-43.5%` of pixels.

This gate tested the simplest direct fix: add a sparse visual alpha loss on the
same compact target-area support.

## Code

Added optional config fields to `src/train/train_star_uvt_feature_overfit.py`:

- `sparse_visual.alpha_loss_weight`
- `sparse_visual.alpha_target`

The implementation computes alpha MSE on the sampled sparse visual pixels,
adds the weighted alpha gradient to the sparse-pixel feature backward, records
alpha-loss metrics, and requires alpha-loss decrease when the weight is
positive. Native sparse visual VJP modes reject this option for now because the
alpha loss is implemented in the non-native sparse-pixel VJP path.

Test coverage:

- `_sparse_visual_alpha_loss_and_grad` gradient scaling
- config normalization for `alpha_loss_weight` and `alpha_target`

## Run

Config:

```bash
PYTHONPATH=src/train rtk .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_alpha1_from1500_lr001_50step_media.jsonc
```

W&B offline run:

```text
wandb/offline-run-20260520_020623-fecnvxnh
```

Result JSON:

```text
outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_alpha1_from1500_lr001_50step_media.json
```

Diagnostic:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case compact=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc \
  --case alpha1=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_alpha1_from1500_lr001_50step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_alpha1_dense_alpha_diagnostic.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_alpha1_dense_alpha_diagnostic.md \
  --date 2026-05-20
```

## Result

The route fails as a visual improvement.

- pass: `false`
- zero tile overflow
- total loss decreases `1.899173 -> 1.871243`
- feature target worsens `0.625418 -> 0.627071`
- RGB probe PSNR worsens `22.028 -> 21.900`
- sparse visual PSNR improves `5.678 -> 6.061`
- sparse alpha loss improves `0.752440 -> 0.738210`
- dense full RGB PSNR is `6.018`
- mean step/backward: `1777.13/936.65ms`
- last step/backward: `2116.17/1033.52ms`

The dense alpha diagnostic confirms the local alpha objective did not improve
actual coverage:

| route | normal PSNR | forced-alpha PSNR | target-bg oracle PSNR | alpha > 0.1 |
| --- | --- | --- | --- | --- |
| compact | `6.023` | `11.450` | `20.149` | `43.5%` |
| alpha1 | `6.018` | `11.426` | `20.181` | `43.1%` |

## Takeaway

Naive sampled alpha-to-one is not the missing bridge. It optimizes the sampled
alpha metric, but dense coverage and dense media do not move. The next STAR UVT
visual gate should target coverage support/composition directly: phase-covered
visibility, dense/rotating coverage verification, black-hole/compositing loss,
or a model/support change that makes full-frame alpha coverage possible.
