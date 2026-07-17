# STAR UVT Birth/Split Opacity Initialization Sweep

Date: 2026-05-20 05:15 +07

## Why

The intermediate-radius sweep showed support radius is a smooth coverage/oracle tradeoff. This gate tested whether born-tube opacity could recover oracle/content while keeping the coverage bump.

## Harness Update

`research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py` now accepts:

```bash
--opacities 0.4,0.6,0.8
```

When set, it writes `support_birth_split.opacity` into the generated config and includes the opacity in row labels.

## Commands

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py \
  --target-sources uncovered_brightness,low_alpha \
  --reallocate-tubes 32 \
  --support-radii 80 \
  --opacities 0.4,0.6,0.8,0.9 \
  --tile-capacities 128 \
  --out-base outputs/benchmarks/2026-05-20_star_uvt_birthsplit_opacity_init_sweep_r80_cap128

PYTHONPATH=src/train:. rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py \
  --target-sources uncovered_brightness,low_alpha \
  --reallocate-tubes 32 \
  --support-radii 88 \
  --opacities 0.2,0.4,0.6,0.8 \
  --tile-capacities 128 \
  --out-base outputs/benchmarks/2026-05-20_star_uvt_birthsplit_opacity_init_sweep_r88_cap128
```

## Results

R80:

| row | pass | alpha >0.1 | alpha >0.5 | normal PSNR | forced PSNR | oracle PSNR |
|---|---:|---:|---:|---:|---:|---:|
| uncovered o0.4 | true | 0.414 | 0.124 | 5.735 | 14.583 | 25.177 |
| uncovered o0.6 | true | 0.414 | 0.127 | 5.748 | 14.588 | 25.083 |
| uncovered o0.8 | true | 0.415 | 0.129 | 5.757 | 14.592 | 25.015 |
| uncovered o0.9 | true | 0.415 | 0.130 | 5.760 | 14.594 | 24.987 |
| low-alpha o0.4 | true | 0.414 | 0.124 | 5.742 | 14.572 | 25.007 |
| low-alpha o0.6 | false | n/a | n/a | n/a | n/a | n/a |
| low-alpha o0.8 | false | n/a | n/a | n/a | n/a | n/a |
| low-alpha o0.9 | false | n/a | n/a | n/a | n/a | n/a |

R88:

| row | pass | alpha >0.1 | alpha >0.5 | normal PSNR | forced PSNR | oracle PSNR |
|---|---:|---:|---:|---:|---:|---:|
| uncovered o0.2 | true | 0.414 | 0.123 | 5.729 | 14.576 | 25.242 |
| uncovered o0.4 | true | 0.416 | 0.129 | 5.756 | 14.587 | 25.032 |
| uncovered o0.6 | true | 0.416 | 0.133 | 5.771 | 14.592 | 24.897 |
| uncovered o0.8 | true | 0.417 | 0.136 | 5.782 | 14.596 | 24.802 |
| low-alpha o0.2 | true | 0.415 | 0.123 | 5.736 | 14.567 | 25.112 |
| low-alpha o0.4 | true | 0.416 | 0.130 | 5.766 | 14.578 | 24.821 |
| low-alpha o0.6 | true | 0.417 | 0.134 | 5.783 | 14.586 | 24.658 |
| low-alpha o0.8 | false | n/a | n/a | n/a | n/a | n/a |

## Read

Born opacity is another smooth tradeoff, not the missing bridge. Lower opacity recovers target-background oracle and loss stability but reduces the coverage bump; higher opacity buys tiny coverage/normal-PSNR movement while losing oracle. It also exposes that low-alpha target sampling becomes optimizer-hostile at higher opacity: r80 low-alpha fails at `0.6+`, and r88 low-alpha fails at `0.8`.

The next useful gate should change support shape rather than scalar opacity. A practical candidate is anisotropic birth support: wide along the fitted trajectory but narrower across it, so we do not flood unrelated target-background regions while still covering missing motion tubes.
