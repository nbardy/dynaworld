# Alpha Background Ablation Refresh

Date: 2026-05-21 21:05:52 +07

## Goal

Refresh the matched alpha-background ablation for both renderer families on the
current workspace:

- dynamic gsplat feature splatting
- STAR UVT feature tubes

The two strategies are still:

- `random_rgb_after_colorizer`: colorize splatted features, then alpha-compose
  random RGB behind the output during training; use fixed black for eval.
- `random_feature_before_colorizer`: alpha-compose random feature background
  behind the splatted feature image before `FeatureToColor`; use fixed-zero
  feature background for eval.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  --steps 20 \
  --output-root outputs/benchmarks/2026-05-21_alpha_background_ablation_refresh_210512
```

## Artifacts

- `outputs/benchmarks/2026-05-21_alpha_background_ablation_refresh_210512/summary.md`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation_refresh_210512/summary.json`
- per-cell `config.json` and `result.json`
- STAR UVT contact sheets under the same output root

## Result

| renderer | strategy | train start | train end | eval loss | eval PSNR | warmed step ms | forward ms | backward ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic_gsplat | random RGB after colorizer | 0.50884 | 0.30373 | 0.47884 | 5.496 | 79.367 | 30.898 | 33.708 |
| dynamic_gsplat | random feature before colorizer | 0.26155 | 0.36929 | 0.25288 | 11.311 | 74.196 | 28.049 | 31.960 |
| star_uvt | random RGB after colorizer | 0.10114 | 0.04421 | 0.07775 | 11.093 | 31.235 | 6.979 | 17.534 |
| star_uvt | random feature before colorizer | 0.14059 | 0.07570 | 0.07024 | 11.534 | 32.152 | 6.834 | 17.930 |

## Interpretation

The refreshed current-code ordering matches the previous three runs. Random
feature background before `FeatureToColor` wins fixed-background eval PSNR in
both renderer families. The dynamic row is the clearest anti-cheat signal:
post-colorizer random RGB lowers the noisy train loss, but it collapses fixed
black eval PSNR, while feature-space background regularization keeps eval much
better.

This is still only an 8-frame, 64px, 20-step smoke. It is enough to prefer the
feature-space strategy for the next longer confirmation, but not enough to
declare the 256/512px or 300-video scale default.
