# STAR UVT Patch4 Support And Alpha Sweep Gate

Date: 2026-05-20 03:03:18

## Goal

Continue the STAR UVT fast feature-shader plan from the target-background gate.
The blocker was no longer raw shader speed or colorizer decodability: the dense
diagnostic showed strong forced-alpha/oracle RGB but weak black-background
renders. This chunk asks whether a simple alpha scale or denser sparse support
can bridge that gap.

## What changed

- Extended
  `research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py`
  with post-render alpha gain and alpha floor sweeps.
- Cleaned the manual sparse visual RGB VJP path so target-background
  composition reuses the already-computed RGB instead of recomputing the
  colorizer.
- Added parity coverage for manual hidden/linear target-background VJPs in
  `tests/test_star_uvt_feature_target_adapter.py`.
- Added the patch4 pilot config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_patch4_targetbg_alpha1_from1500_lr001_20step_media.jsonc`.

## Commands And Results

Alpha sweep diagnostic:

```bash
PYTHONPATH=src/train rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case compact=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc \
  --case targetbg=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_targetbg_from1500_lr001_50step_media.jsonc \
  --case targetbg_alpha1=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_targetbg_alpha1_from1500_lr001_50step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_alpha_sweep_dense_diagnostic.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_alpha_sweep_dense_diagnostic.md
```

Key read: target-background checkpoints still have weak real dense RGB
(`5.666-5.748`), forced-alpha/oracle remain strong (`14.891-14.899` and
`27.105-27.443`), and `16x` posthoc alpha gain reaches only `8.337-8.592`.
Alpha floor reaches the forced-alpha result, which means the missing operation
is closer to dense support/visibility than to scalar gain.

Patch4 support pilot:

```bash
PYTHONPATH=src/train rtk .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_patch4_targetbg_alpha1_from1500_lr001_20step_media.jsonc
```

The run wrote artifacts, then exited `1` because `require_loss_decrease` failed:

- W&B offline run: `wandb/offline-run-20260520_025323-r553enpn`
- Result JSON:
  `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_patch4_targetbg_alpha1_from1500_lr001_20step_media.json`
- Checkpoint:
  `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_patch4_targetbg_alpha1_from1500_lr001_20step.pt`
- Support: `25%` of dense pixels, zero tile overflow, max/p95 `68/46`
- Weighted loss: `1.631071 -> 1.637982`
- Feature loss: `0.625418 -> 0.626858`
- RGB probe PSNR: `22.028 -> 21.878`
- Sparse visual PSNR: `26.319 -> 27.251`
- Alpha loss: `0.752542 -> 0.749687`
- Dense full RGB PSNR: `5.698`
- Mean/last step: `2664.37/2373.03ms`
- Mean/last backward: `1906.63/1781.93ms`

Patch4 dense diagnostic:

```bash
PYTHONPATH=src/train rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case compact=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc \
  --case targetbg_alpha1=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_targetbg_alpha1_from1500_lr001_50step_media.jsonc \
  --case patch4_targetbg_alpha1=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_patch4_targetbg_alpha1_from1500_lr001_20step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_patch4_alpha_sweep_dense_diagnostic.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_patch4_alpha_sweep_dense_diagnostic.md
```

Patch4 dense read: normal `5.698`, forced alpha `14.767`, target-background
oracle `26.646`, best alpha gain `8.414 @ 16x`, best alpha floor `14.767 @
1.0`, and alpha `>0.1` on `41.5%` of pixels.

## Conclusion

This is real progress because it narrows the failure. The missing piece is not
V-JEPA target plumbing, target-grid decodability, compact target-background
composition, scalar alpha pressure, multiplicative opacity gain, or merely
sampling `4x4` support. The next useful branch is a real dense
visibility/support mechanism: first test raw-opacity-bias render sweeps, then
port the winning idea into a fused dense-alpha/support VJP or compact
visibility/prefix tape that does not drag the feature target backward.
