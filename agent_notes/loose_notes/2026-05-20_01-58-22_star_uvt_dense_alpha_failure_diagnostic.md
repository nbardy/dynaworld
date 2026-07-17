# STAR UVT Dense Alpha Failure Diagnostic

Date: 2026-05-20 01:58 +07

## Why

The compact target-area route, RGB-grid route, and compact+RGB-grid route all
failed dense media even when sampled/grid metrics improved. The next actionable
question was whether the dense failure is mostly low alpha/visibility coverage
or bad feature-to-RGB content everywhere.

## What Ran

Added:
`research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py`

Command:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case compact=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc \
  --case rgbgrid=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_rgbgrid40_feature1_probe40_from1500_lr001_50step_media.jsonc \
  --case compact_rgbgrid=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_rgbgrid40_from1500_lr001_50step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_failure_diagnostic.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_failure_diagnostic.md \
  --date 2026-05-20
```

## Result

| Case | Normal PSNR | Forced alpha=1 PSNR | Target-background oracle PSNR | alpha>0.1 |
| --- | ---: | ---: | ---: | ---: |
| compact | `6.023` | `11.450` | `20.149` | `43.5%` |
| rgbgrid | `5.657` | `14.548` | `25.562` | `41.5%` |
| compact_rgbgrid | `5.720` | `14.616` | `25.505` | `43.1%` |

## Read

The dense STAR UVT feature failures are strongly coverage/visibility limited.
The feature-to-RGB content is not perfect, but forcing alpha to one gives a
large jump and compositing over target background gives an oracle-quality jump.
That means the next visual gate should not be another grid colorizer loss or
sparse support reshuffle. It should directly supervise or change alpha,
visibility, background/composition, or the support basis that creates coherent
512px coverage.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_failure_diagnostic.md`
