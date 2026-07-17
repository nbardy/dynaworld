# STAR UVT Birth/Split Intermediate Radius Gate

Date: 2026-05-20 05:08 +07

## Why

The previous birth/split sweep showed `96px` support radius raises dense alpha coverage but drops target-background oracle. This gate tested intermediate radii at the safe `32`-birth, cap-128 setting before any longer continuation.

## Command

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py \
  --target-sources uncovered_brightness,low_alpha \
  --reallocate-tubes 32 \
  --support-radii 64,72,80,88 \
  --tile-capacities 128 \
  --out-base outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_intermediate_radius_cap128
```

## Outputs

- Summary: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_intermediate_radius_cap128.md`
- JSON: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_intermediate_radius_cap128.json`
- Dense diagnostic: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_intermediate_radius_cap128_dense_support.md`

## Results

| row | pass | alpha >0.1 | alpha >0.5 | normal PSNR | forced-alpha PSNR | oracle PSNR | max tile |
|---|---:|---:|---:|---:|---:|---:|---:|
| uncovered r64 | true | 0.411 | 0.119 | 5.713 | 14.579 | 25.319 | 100 |
| uncovered r72 | true | 0.413 | 0.124 | 5.734 | 14.587 | 25.187 | 100 |
| uncovered r80 | true | 0.415 | 0.129 | 5.757 | 14.592 | 25.015 | 100 |
| uncovered r88 | true | 0.417 | 0.136 | 5.782 | 14.596 | 24.802 | 100 |
| low-alpha r64 | true | 0.411 | 0.119 | 5.717 | 14.569 | 25.209 | 100 |
| low-alpha r72 | true | 0.413 | 0.124 | 5.740 | 14.579 | 25.034 | 100 |
| low-alpha r80 | false | n/a | n/a | n/a | n/a | n/a | 100 |
| low-alpha r88 | false | n/a | n/a | n/a | n/a | n/a | 100 |

The two failed low-alpha rows failed the trainer loss gate, not overflow:

- `low_alpha_n32_r80_cap128`: weighted loss `0.913757 -> 0.923922`, feature target `0.638905 -> 0.640059`, RGB-probe `0.006871 -> 0.007097`.
- `low_alpha_n32_r88_cap128`: weighted loss `0.925082 -> 0.934971`, feature target `0.642224 -> 0.643357`, RGB-probe `0.007071 -> 0.007290`.

## Read

Intermediate radius gives a smooth tradeoff, not a promotion. Larger radius monotonically raises alpha coverage and normal black-background PSNR, but it also monotonically lowers target-background oracle. The best passing cap-128 intermediate row is `uncovered_brightness_n32_r88_cap128` with alpha `>0.1` `0.417` and oracle `24.802`; this is weaker coverage than `r96` (`0.420`) and still below the `r64` oracle (`25.319`).

The immediate conclusion is that support radius is a blunt lever. A longer continuation is not justified yet. The next short gate should change how born tubes are initialized, such as lower initial opacity or a narrower anisotropic support shape, rather than only widening radius.
