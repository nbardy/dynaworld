# STAR UVT Phase Alpha Coverage Gate

Date: 2026-05-20

## Goal

Test whether the failed compact alpha-to-one result was caused by sampling the
same sparse support forever. The follow-up used phase-cycled compact
target-area support plus `sparse_visual.alpha_loss_weight=1.0`, so the visual
loss visits all `2x2` phases over time while also pushing sampled alpha to one.

## Commands

```bash
PYTHONPATH=src/train rtk .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_phase2x2_alpha1_from1500_lr001_50step_media.jsonc
```

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case compact=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc \
  --case phase=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_phase2x2_from1500_lr001_50step_media.jsonc \
  --case alpha1=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_alpha1_from1500_lr001_50step_media.jsonc \
  --case phase_alpha1=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_phase2x2_alpha1_from1500_lr001_50step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_dense_alpha_diagnostic.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_dense_alpha_diagnostic.md \
  --date 2026-05-20
```

## Result

The phase alpha gate is rejected.

- W&B offline run: `wandb/offline-run-20260520_021443-s7eym2r0`
- Result JSON:
  `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_alpha1_from1500_lr001_50step_media.json`
- Human report:
  `outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_coverage_gate.md`
- Dense alpha diagnostic:
  `outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_dense_alpha_diagnostic.md`
- Checkpoint:
  `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_alpha1_from1500_lr001_50step.pt`

Key numbers:

- total weighted loss: `1.897462 -> 1.871930`
- feature target loss: `0.625418 -> 0.626961`
- frozen RGB probe PSNR: `22.028 -> 21.904`
- sparse visual PSNR: `5.694 -> 6.072`
- sampled alpha loss: `0.751768 -> 0.739891`
- dense full RGB PSNR: `6.014`
- mean/last step: `1312.31/1075.31ms`
- mean/last backward: `746.10/638.90ms`
- zero tile overflow

Dense alpha diagnostic:

| Case | Normal PSNR | Forced-alpha PSNR | Oracle PSNR | Alpha `>0.1` |
| --- | ---: | ---: | ---: | ---: |
| compact | `6.023` | `11.450` | `20.149` | `43.5%` |
| phase | `6.019` | `11.360` | `19.925` | `43.5%` |
| alpha1 | `6.018` | `11.426` | `20.181` | `43.1%` |
| phase_alpha1 | `6.014` | `11.342` | `19.971` | `43.0%` |

## Learning

Same-support alpha pressure was not the issue, and phase-covered sparse support
does not fix it either. The sampled alpha objective can improve on its own
support while the dense alpha field remains under-covered and dense RGB gets
worse. Future work should target visibility/composition directly: dense or
prefix alpha coverage, target-background composition, black-hole penalties, or
a stronger model/support bridge. Do not spend the next cycle on another sparse
support shuffle.
