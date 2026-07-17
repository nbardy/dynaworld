# Alpha Background 128/256px Confirmation

Date: 2026-05-21 23:04:28 +07

## Goal

Extend the alpha-background ablation beyond the existing 8f/64px result for:

- dynamic gsplat feature splatting
- STAR UVT feature tubes

Strategies:

- `random_rgb_after_colorizer`: colorize features, then alpha-compose random RGB during train; fixed black eval.
- `random_feature_before_colorizer`: inject random feature background before `FeatureToColor` during train; fixed-zero feature eval.

## Code Change

`research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`
now accepts experiment-shape knobs instead of being hard-coded to the 8f/64px
smoke:

- `--render-size`
- `--frames`
- `--dynamic-video-path`
- `--star-video-path`
- `--dynamic-gaussians`
- `--star-tubes`
- `--star-frame-chunk-size`
- `--star-tile-capacity`

It also prints compact rows at the end. STAR UVT trainer stdout is redirected
to each cell's `trainer_stdout.log` for future runs so the terminal is not
flooded by per-step JSON.

Validation:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_background_cheat_diagnostic.py -q
```

Result: compile passed; pytest passed `2 passed in 1.25s`.

## Commands

128px confirmation:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  --steps 100 \
  --render-size 128 \
  --frames 8 \
  --dynamic-video-path test_data/test_video_small_128_4fps.mp4 \
  --star-video-path test_data/test_video_small_128_4fps.mp4 \
  --output-root outputs/benchmarks/2026-05-21_alpha_background_ablation_128px_100step_224500
```

256px confirmation:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  --steps 100 \
  --render-size 256 \
  --frames 8 \
  --dynamic-video-path test_data/test_video_384_3fps.mp4 \
  --star-video-path test_data/test_video_384_3fps.mp4 \
  --output-root outputs/benchmarks/2026-05-21_alpha_background_ablation_256px_100step_225800
```

## Results

| renderer | res | strategy | eval loss | eval PSNR | alpha mean | warm step ms | backward ms |
|---|---:|---|---:|---:|---:|---:|---:|
| dynamic_gsplat | 128 | random_rgb_after_colorizer | 0.11775 | 17.451 | 0.91298 | 163.53 | 89.35 |
| dynamic_gsplat | 128 | random_feature_before_colorizer | 0.12207 | 18.153 | 0.87880 | 162.77 | 85.84 |
| star_uvt | 128 | random_rgb_after_colorizer | 0.02037 | 16.909 | - | 84.97 | 52.10 |
| star_uvt | 128 | random_feature_before_colorizer | 0.02136 | 16.703 | - | 65.52 | 40.96 |
| dynamic_gsplat | 256 | random_rgb_after_colorizer | 0.13637 | 16.350 | 0.89354 | 246.15 | 156.72 |
| dynamic_gsplat | 256 | random_feature_before_colorizer | 0.15993 | 15.035 | 0.79603 | 217.59 | 140.38 |
| star_uvt | 256 | random_rgb_after_colorizer | 0.05301 | 12.757 | - | 145.55 | 108.30 |
| star_uvt | 256 | random_feature_before_colorizer | 0.02577 | 15.889 | - | 151.53 | 108.25 |

Artifacts:

- `outputs/benchmarks/2026-05-21_alpha_background_ablation_128px_100step_224500/summary.md`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation_128px_100step_224500/summary.json`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation_256px_100step_225800/summary.md`
- `outputs/benchmarks/2026-05-21_alpha_background_ablation_256px_100step_225800/summary.json`

## Interpretation

The background-policy answer is not universal.

- Dynamic gsplat: post-colorizer random RGB remains the safer anti-cheat choice
  by 256px. It wins eval loss, PSNR, and alpha coverage over random feature
  background.
- STAR UVT: the 64/128px result favored post-colorizer random RGB weakly, but
  the 256px result strongly favors random feature background for fixed-zero
  feature eval.
- 128px is mixed enough that it should not be used as the final policy gate:
  dynamic has lower eval loss and higher alpha under random RGB but higher PSNR
  under feature background; STAR has slightly better quality under random RGB
  but much faster timing under feature background.

Current recommendation: choose background strategy per renderer at the intended
scale. For the next matched 256px run, use post-colorizer random RGB for
dynamic gsplat and random feature background for STAR UVT. Do not promote one
global default to the 300-video scale lane yet.

## Remaining Work

- Repeat at the actual STAR UVT scale setting before a long/300-video run,
  especially if tube count or frame count changes.
- If 512px is the next target, run the same script at 512px with the selected
  STAR tube count and dynamic splat count rather than extrapolating from 256px.
- Consider testing `sample_scope=frame` or `pixel` only after the renderer-scale
  default is selected; the current result only covers step-scope randomization.
