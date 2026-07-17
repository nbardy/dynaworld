# STAR UVT Black-Hole Coverage Gate

Date: 2026-05-20

## Goal

Close the next visual-quality question after the phase-covered alpha retry:
does STAR UVT mainly fail because sparse visual training leaves black holes in
target-energy regions, and can an explicit target-aware low-alpha penalty fix
that without changing the renderer support?

## What Changed

- Added `sparse_visual.black_hole_loss_weight` to the STAR UVT feature overfit
  trainer.
- Added `_sparse_visual_black_hole_loss_and_grad(...)`, an analytic alpha loss
  and gradient on the configured sparse visual basis:
  `(1 - alpha)^2 * target_rgb_energy`.
- Added a focused unit test proving bright target pixels produce stronger
  negative alpha pressure than black target pixels.
- Added the 50-step compact target-area64 black-hole config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_blackhole4_from1500_lr001_50step_media.jsonc`.

## Run

Command:

```bash
PYTHONPATH=src/train rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_blackhole4_from1500_lr001_50step_media.jsonc
```

Artifacts:

- Result:
  `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_blackhole4_from1500_lr001_50step_media.json`
- Report:
  `outputs/benchmarks/2026-05-20_star_uvt_blackhole4_coverage_gate.md`
- Dense diagnostic:
  `outputs/benchmarks/2026-05-20_star_uvt_blackhole4_dense_alpha_diagnostic.md`
- W&B offline run:
  `wandb/offline-run-20260520_022828-yt5fu9wz`

## Result

Rejected.

- Total weighted loss decreased `2.196880 -> 2.161469`.
- Sparse visual PSNR improved `5.678 -> 6.059`.
- Black-hole loss improved `0.262537 -> 0.256889`.
- Feature target loss worsened `0.625418 -> 0.627272`.
- Frozen RGB probe PSNR worsened `22.028 -> 21.890`.
- Dense full RGB PSNR stayed bad at `6.014`.
- Mean/last step was `1197.45/1152.90ms`; mean/last backward was
  `711.30/692.11ms`.
- Zero tile overflow; tile max/p95 was `68/48` against cap `128`.

Dense alpha diagnostic:

| Case | Normal PSNR | Forced-alpha PSNR | Target-background oracle PSNR | Alpha `>0.1` |
| --- | ---: | ---: | ---: | ---: |
| compact | `6.023` | `11.450` | `20.149` | `43.5%` |
| alpha1 | `6.018` | `11.426` | `20.181` | `43.1%` |
| phase_alpha1 | `6.014` | `11.342` | `19.971` | `43.0%` |
| blackhole4 | `6.014` | `11.428` | `20.191` | `43.0%` |

## Read

The black-hole loss is another same-support alpha pressure. It improves the
sampled scalar it directly optimizes, but it does not create dense coverage.
That makes the coverage diagnosis stronger, not weaker: alpha-to-one,
phase-covered alpha, and target-aware empty-pixel pressure all leave dense alpha
coverage around `43%` and dense RGB around `6.014-6.018` PSNR.

The next visual experiment should stop modifying sparse support penalties and
change the visibility/composition path itself: dense or prefix-style alpha
coverage, target-background composition during training, a full-image coverage
prior, or a model/support bridge that makes rendered alpha dense before the
larger 300-video run.
