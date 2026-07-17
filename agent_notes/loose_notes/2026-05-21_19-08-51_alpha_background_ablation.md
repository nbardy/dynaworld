# Alpha Background Ablation

Date: 2026-05-21 19:08:51 Asia/Ho_Chi_Minh

## Goal

Run a matched ablation for the two alpha-background strategies on both UVT STAR
feature tubes and dynamic gsplat feature splatting:

- `random_rgb_after_colorizer`: colorize splatted features, then composite
  random RGB behind alpha during training.
- `random_feature_before_colorizer`: add random feature background behind alpha
  before `FeatureToColor`, then colorize the combined feature image.

The concrete question was whether random feature background is a stronger
anti-cheat regularizer than random post-colorizer RGB background.

## Code Changes

- Added feature-space background modes to the shared RGB objective:
  - `BackgroundSpec.feature_train_mode`
  - `BackgroundSpec.feature_eval_mode`
  - `BackgroundSpec.feature_sample_scope`
  - `BackgroundSample.feature`
- Added `compose_feature_background(...)` so F-channel feature maps can receive
  `(1 - alpha) * random_feature` before colorization.
- Updated the dynamic trainer so train background sampling is delayed until
  after rasterization when feature-space background is active. This avoids
  sampling a 3-channel RGB background from `clip_frames` for an F32 feature map.
- Added STAR UVT trainer support for:
  - `fixed_black_after_colorizer`
  - `random_rgb_after_colorizer`
  - `random_feature_before_colorizer`
  - `fixed_zero_feature_before_colorizer`
- Added reproducible runner:
  `research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`

## Validation

Compile:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  src/train/objective/types.py \
  src/train/objective/background.py \
  src/train/objective/objective.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_star_uvt_feature_overfit.py
```

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_rgb_recon_objective.py \
  tests/test_star_uvt_background_cheat_diagnostic.py \
  -q
```

Result: `8 passed`.

Smoke ablation:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  --steps 2 \
  --output-root outputs/benchmarks/2026-05-21_alpha_background_ablation_smoke
```

Full short ablation:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  --steps 20 \
  --output-root outputs/benchmarks/2026-05-21_alpha_background_ablation
```

Current-code confirmation rerun:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  --steps 20 \
  --output-root outputs/benchmarks/2026-05-21_alpha_background_ablation_confirm
```

## Results

Generated summary:

- `outputs/benchmarks/2026-05-21_alpha_background_ablation/summary.md`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation/summary.json`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation_confirm/summary.md`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation_confirm/summary.json`

Short 8f/64px/20-step result table:

| renderer | strategy | train start | train end | eval loss | eval PSNR | warmed step ms | forward ms | backward ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic_gsplat | random RGB after colorizer | 0.50884 | 0.30386 | 0.47890 | 5.495 | 66.739 | 23.905 | 30.433 |
| dynamic_gsplat | random feature before colorizer | 0.26155 | 0.36928 | 0.25289 | 11.311 | 67.369 | 24.480 | 30.317 |
| star_uvt | random RGB after colorizer | 0.10114 | 0.04421 | 0.07775 | 11.093 | 24.940 | 5.545 | 14.343 |
| star_uvt | random feature before colorizer | 0.14059 | 0.07570 | 0.07024 | 11.534 | 24.328 | 5.302 | 14.003 |

Current-code confirmation rerun:

| renderer | strategy | train start | train end | eval loss | eval PSNR | warmed step ms | forward ms | backward ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dynamic_gsplat | random RGB after colorizer | 0.50884 | 0.30375 | 0.47880 | 5.496 | 67.718 | 24.127 | 31.122 |
| dynamic_gsplat | random feature before colorizer | 0.26155 | 0.36927 | 0.25286 | 11.312 | 67.378 | 24.369 | 30.492 |
| star_uvt | random RGB after colorizer | 0.10114 | 0.04421 | 0.07775 | 11.093 | 24.585 | 5.468 | 14.187 |
| star_uvt | random feature before colorizer | 0.14059 | 0.07570 | 0.07024 | 11.534 | 26.016 | 5.474 | 14.817 |

## Interpretation

The early signal favors `random_feature_before_colorizer`.

Dynamic gsplat showed the strongest split: the feature-background strategy had
much better deterministic eval PSNR (`11.31` vs `5.50`) at essentially the same
warmed step time. The post-colorizer RGB variant reduced train loss, but with
fixed-black eval it still relied on low alpha and scored poorly.

STAR UVT also favored feature-background eval (`11.53` vs `11.09`) and was
slightly faster in warmed timing. This is small but useful because it shows the
strategy is not only a dynamic-gsplat artifact.

This is not a final quality claim. It is a short single-video 8f/64px check.
The useful decision is: use feature-space random background as the candidate
default for the next longer/higher-res confirmation.

## Next

1. Run the same ablation at 128px/256px and at least 100-200 steps.
2. For dynamic gsplat, include the previous production eval mode if that was
   white background, so the comparison separates train regularization from eval
   background color.
3. Add the feature-background mode to the scale configs only after the longer
   confirmation, then rerun the chosen STAR UVT and dynamic gsplat overfit
   harnesses.
4. If feature-background remains better, make it the default for F-channel
   `FeatureToColor` training and keep post-colorizer random RGB as a legacy
   baseline.
