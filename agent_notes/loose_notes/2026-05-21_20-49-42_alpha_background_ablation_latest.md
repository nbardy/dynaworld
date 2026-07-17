# Alpha Background Ablation Latest Rerun

Date: 2026-05-21 20:49:42 Asia/Ho_Chi_Minh

## Goal

Rerun the matched alpha-background ablation on current code for both renderer
families:

- dynamic gsplat feature splatting
- STAR UVT feature tubes

The two strategies remain:

- `random_rgb_after_colorizer`: colorize features, then composite random RGB
  behind alpha during training.
- `random_feature_before_colorizer`: composite a random feature background
  behind alpha before `FeatureToColor`, then colorize.

## Command

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  --steps 20 \
  --output-root outputs/benchmarks/2026-05-21_alpha_background_ablation_latest
```

## Artifacts

- `outputs/benchmarks/2026-05-21_alpha_background_ablation_latest/summary.md`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation_latest/summary.json`
- per-cell `config.json`, `result.json`, and STAR contact sheets under the same
  output root

## Result

| renderer | strategy | train start | train end | eval loss | eval PSNR | warmed step ms | forward ms | backward ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic_gsplat | random RGB after colorizer | 0.50884 | 0.30385 | 0.47905 | 5.492 | 80.414 | 30.470 | 35.147 |
| dynamic_gsplat | random feature before colorizer | 0.26155 | 0.36928 | 0.25282 | 11.315 | 86.455 | 33.461 | 36.235 |
| star_uvt | random RGB after colorizer | 0.10114 | 0.04421 | 0.07775 | 11.093 | 28.327 | 6.505 | 16.114 |
| star_uvt | random feature before colorizer | 0.14059 | 0.07570 | 0.07024 | 11.534 | 29.899 | 6.807 | 16.535 |

## Interpretation

The current-code rerun preserves the earlier ordering. Feature-space random
background before the colorizer wins deterministic eval PSNR for both dynamic
gsplat and STAR UVT at this 8f/64px/20-step scale. The timing penalty is small
relative to the quality split: dynamic warmed step is about `1.08x` slower and
STAR warmed step is about `1.06x` slower in this noisy MPS run.

This is still not a final scale claim. The correct next experiment is the same
matrix at longer training and 128px/256px before changing scale defaults.
