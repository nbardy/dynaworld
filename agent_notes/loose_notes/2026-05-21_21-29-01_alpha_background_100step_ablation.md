# Alpha Background 100-Step Ablation

Date: 2026-05-21 21:29:01 +07

## Goal

Re-run the matched alpha-background ablation for the two renderer families:

- dynamic gsplat feature splatting
- STAR UVT feature tubes

The two compared strategies are:

- `random_rgb_after_colorizer`: colorize features first, then alpha-compose a
  random RGB background during training; evaluate on fixed black.
- `random_feature_before_colorizer`: inject random feature background before
  `FeatureToColor` during training; evaluate with fixed-zero feature
  background.

This extends the earlier 20-step 8f/64px smoke to 100 steps because the short
run looked like a possible early-step artifact.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  --steps 100 \
  --output-root outputs/benchmarks/2026-05-21_alpha_background_ablation_100step_212901
```

## Artifacts

- `outputs/benchmarks/2026-05-21_alpha_background_ablation_100step_212901/summary.md`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation_100step_212901/summary.json`
- per-cell `config.json` and `result.json`
- STAR UVT contact sheets under the same output root

## Result

| renderer | strategy | train start | train end | eval loss | eval PSNR | alpha mean | warmed step ms | forward ms | backward ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dynamic_gsplat | random RGB after colorizer | 0.50884 | 0.13941 | 0.15769 | 15.915 | 0.88957 | 81.078 | 31.133 | 34.731 |
| dynamic_gsplat | random feature before colorizer | 0.26155 | 0.33761 | 0.25449 | 10.917 | 0.33724 | 83.822 | 32.016 | 36.628 |
| star_uvt | random RGB after colorizer | 0.10114 | 0.006655 | 0.008134 | 20.897 | - | 31.741 | 6.835 | 18.547 |
| star_uvt | random feature before colorizer | 0.14059 | 0.009132 | 0.008586 | 20.662 | - | 34.831 | 7.219 | 19.815 |

## Interpretation

The longer run reverses the earlier 20-step ordering. Post-colorizer random RGB
is the better current default candidate: it wins fixed-black eval in both
renderer families, is slightly faster in both warmed timings, and in dynamic
gsplat it drives alpha much higher instead of leaving the model with a weak
coverage solution.

The 20-step feature-background win should be treated as early optimization
behavior, not a default-setting result.

## Remaining Work

- Confirm at 256px/512px and/or longer single-video runs before declaring this
  the scale default.
- If feature-background is revisited, test stronger scopes (`frame` or `pixel`)
  rather than only step-scope random features, because the current step-scope
  feature background did not force enough alpha coverage by 100 steps.
