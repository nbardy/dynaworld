# Strong-init 16f matrix results

Ran:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh strong-matrix
```

This is the focused control for whether the recent weak conditioned cells were mostly losing because they lacked the stronger RGB/uniform/token/head initialization used by the best same-source lane.

## New 250-step runs

| run | variant | backend | Eval/Loss | SSIM | PSNR | temporal adj L1 ratio | camera adj rot deg | camera adj trans | camera rot mean deg |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pg5iwhks` | learned time orbit path | local | 0.1451 | 0.3192 | 16.7569 | 0.0323 | 0.0075 | 0.00046 | 3.0755 |
| `ffsx3u65` | learned time orbit path | V-JEPA HF | 0.1464 | 0.3014 | 16.7387 | 0.0298 | 0.0053 | 0.00077 | 8.3159 |
| `i33lyr0w` | unconditioned tokens | none | 0.1310 | 0.3947 | 17.3957 | 0.0701 | 0.0160 | 0.00106 | 5.0391 |

## Prior weak/old controls from local W&B summaries

| run | variant | Eval/Loss | SSIM | temporal adj L1 ratio |
| --- | --- | ---: | ---: | ---: |
| `leunxckm` | local weak-init conditioned | 0.1485 | 0.2946 | 0.0247 |
| `vph578vo` | V-JEPA weak-init conditioned | 0.1601 | 0.2690 | 0.0261 |
| `fja3e512` | unconditioned strong-init | 0.1289 | 0.4009 | 0.0956 |

## Read

Strong init helped the conditioned setups, especially V-JEPA HF versus its weak-init cell, but it did not reverse the main conclusion for this plain 16f/128px comparison. The unconditioned strong-init token baseline is still better on source-view loss/SSIM than both conditioned strong-init cells.

This means the failure was not just missing RGB/uniform/token/head init. The stronger V-JEPA result still points to the combined recipe: static/dynamic split, deeper cross-attention, precomputed V-JEPA 2.1 features, and the same strong init.

Camera motion remains a diagnostic, not a disqualifier. The unconditioned run has the best image metrics here and also the largest adjacent camera deltas, so source-view metrics alone still allow camera/splat tradeoffs. The next fair test should keep the stronger recipe and add a camera-clamped/control lane, rather than clamping the default training path.
