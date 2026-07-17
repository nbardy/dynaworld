# STAR UVT Alpha Background Objective Boundary

## Goal

Continue the trainer/interface modularization goal without doing a broad
rewrite: move the alpha-background composition math behind one shared objective
boundary, keep the STAR UVT alpha-background strategies configurable, and leave
trainer-policy choices local.

## What Changed

- `src/train/objective/objective.py` now exposes tensor-level helpers:
  - `compose_rgb_background_tensor(...)`
  - `compose_feature_background_tensor(...)`
  - `colorize_and_compose_feature_rgb(...)`
- `RGBReconObjective.compose_rasterized(...)` already routes trainer RGB
  composition through the objective layer. This slice extends that boundary to
  standalone STAR UVT feature rendering and diagnostics.
- `src/train/star_uvt_feature_rendering.py` now maps legacy STAR alpha
  strategies into `BackgroundSpec`/`BackgroundPolicy`, then calls
  `colorize_and_compose_feature_rgb(...)`.
- `research_experiments/star_uvt_feature_tubes/background_cheat_diagnostic.py`
  now uses the same shared helper for both post-colorizer RGB background and
  pre-colorizer feature background rows.
- `research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py`
  keeps its old `colorize_and_compose(...)` name because older benchmark
  scripts import it, but the function is now only a compatibility shim around
  `colorize_and_compose_feature_rgb(...)`.
- `tests/test_rgb_recon_objective.py` now protects the two tensor-level
  composition contracts.

## Current State

The practical shared boundary is now:

```text
BackgroundSpec/BackgroundPolicy -> BackgroundSample
feature/RGB tensor + alpha + colorizer -> objective.colorize_and_compose_feature_rgb(...)
```

That gives STAR UVT feature rendering, diagnostic scripts, single-cam token-GS,
and multicam feature trainers one place for alpha/background composition
semantics. The configurable alpha-background ablation runner remains the place
to compare strategies across renderer families and scales:

```text
research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py
```

Important current result from the ablation docs: background strategy is not a
global default. At 256px dynamic gsplat favors post-colorizer random RGB, while
STAR UVT favors random feature background.

## Validation

- `py_compile` passed for the objective, STAR rendering helper, background cheat
  diagnostic, dense feature-tube prototype shim, old benchmark importers, and
  focused tests.
- Focused tests passed:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_objective_background_and_composition.py \
  tests/test_rgb_recon_objective.py \
  tests/test_star_uvt_background_cheat_diagnostic.py -q
```

Result: `15 passed in 0.76s`.

- Direct STAR composition smoke covered all four alpha-background strategies
  and returned finite `(2, 3, 3, 3)` RGB tensors.
- Dense prototype compatibility-shim smoke returned a finite `(1, 3, 2, 2)`
  RGB tensor through `dense_feature_tube_prototype.colorize_and_compose(...)`.
- Runtime STAR trainer smoke passed:

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc
```

Result: pass true, loss decreased `0.1860204972 -> 0.0416746489`, final full
RGB loss `0.0400530137`, final full RGB PSNR `13.9736485`, tile overflow sum
`0`, and gradients were seen for center, velocity, raw feature, opacity,
precision, and colorizer. This smoke uses the config's existing output paths,
so it refreshes those local benchmark artifacts.

## What Is Left

- Do not introduce a monolithic base trainer. Keep using narrow shared
  contracts: objective composition, logging cadence/W&B init, runtime payloads,
  validation media, metric helpers, STAR runtime/config/model/output helpers,
  and typed step/result helpers.
- The next useful cleanup slice is render-dispatch convergence or mixed
  same-view plus heldout scheduler evidence, not another RGB-composition
  rewrite.
- Old-code candidates to delete/unify after one more import audit:
  - older alpha-background one-off runners if the configurable ablation runner
    fully covers their parameter surface.
- Quality is still a separate problem. This cleanup makes the experiments less
  ambiguous; it does not solve STAR UVT visual quality or 300-video scale-up.
