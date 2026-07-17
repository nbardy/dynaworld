# STAR UVT Raw-Opacity Bias Gate

Date: 2026-05-20 03:20:41

## Goal

Continue the STAR UVT visual-support diagnosis after the negative
target-background, alpha, and patch4 gates. The specific question here was:
would adding a logit-space bias to tube opacity before rasterization expand
dense support enough to recover the target-background color path?

## What changed

- Added `--raw-opacity-biases` to
  `research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py`.
- The diagnostic now rerenders optional raw-opacity-bias sweeps and records
  `best_raw_opacity_bias`, `best_raw_opacity_bias_psnr`, and full per-bias
  alpha coverage stats.
- The sweep is opt-in so routine dense diagnostics do not become much slower.

## Command

```bash
PYTHONPATH=src/train rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case compact=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc \
  --case targetbg_alpha1=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_targetbg_alpha1_from1500_lr001_50step_media.jsonc \
  --case patch4_targetbg_alpha1=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_patch4_targetbg_alpha1_from1500_lr001_20step_media.jsonc \
  --raw-opacity-biases=-2,-1,0,1,2,3,4 \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_raw_opacity_bias_dense_diagnostic.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_raw_opacity_bias_dense_diagnostic.md
```

## Results

- Compact: normal `6.023`, forced-alpha `11.450`, oracle `20.149`, best
  posthoc alpha gain `7.861 @ 16x`, best alpha floor `14.203 @ 0.75`, best
  raw-opacity bias `6.194 @ +4`.
- Target-background alpha1: normal `5.748`, forced-alpha `14.899`, oracle
  `27.105`, best posthoc alpha gain `8.592 @ 16x`, best alpha floor
  `14.899 @ 1.0`, best raw-opacity bias `5.926 @ +4`.
- Patch4 target-background alpha1: normal `5.698`, forced-alpha `14.767`,
  oracle `26.646`, best posthoc alpha gain `8.414 @ 16x`, best alpha floor
  `14.767 @ 1.0`, best raw-opacity bias `5.871 @ +4`.

Coverage barely moves under the best raw-opacity bias:

- Compact alpha `>0.1`: `43.5% -> 46.5%`.
- Target-background alpha1 alpha `>0.1`: `43.1% -> 45.8%`.
- Patch4 target-background alpha1 alpha `>0.1`: `41.5% -> 44.2%`.

## Conclusion

This rejects plain opacity-bias scheduling as the next major branch. The
failure is not just that opacity logits are too low. Dense visual recovery needs
the STAR feature route to change where support exists or how visibility is
supervised, not another scalar opacity/gain/alpha-pressure trick.

The next concrete gate should be a trainable dense-alpha/support objective or a
visibility/prefix-tape design that can apply dense support gradients without
the Python/Torch dense hidden64 penalty.
