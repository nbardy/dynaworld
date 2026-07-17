# STAR UVT Background Cheat Diagnostic

Date: 2026-05-21 14:55

## Question

The user remembered an older feature-splatting setup that avoided the
background-feature/colorizer cheat and asked whether it was recorded, whether it
used random background, why not use random feature background, and how modular
the code is for testing variants.

## Recovered Answer

The old fix is recorded. It was RGB-space random background after colorization,
not feature-space randomization:

```text
splat_rgb = colorize(rendered_features)
final_rgb = alpha * splat_rgb + (1 - alpha) * rgb_background
```

This appears in `TODO/alpha_mask_white_background_cheating.md` and
`agent_notes/loose_notes/2026-05-08_16-25-46_alpha_bg_bleed_features.md`.

The shared trainer path still has the same contract:

- `src/train/objective/background.py` supports `random_rgb` with `step`,
  `view`, `frame`, and `pixel` sample scopes.
- `src/train/objective/objective.py::compose_rgb` composites alpha after the
  feature colorizer.
- `src/train/train_video_token_implicit_dynamic.py` samples train background
  once per train step and requires alpha for F-channel training.
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` does the
  same for multicam feature training.
- Feature rasterizer `feature_background = 0.0` is only the raw feature fill,
  not the reconstruction background.

## Diagnostic Added

Added:

- `research_experiments/star_uvt_feature_tubes/background_cheat_diagnostic.py`
- `tests/test_star_uvt_background_cheat_diagnostic.py`

The diagnostic compares two toy objective paths:

1. `rgb_background_after_colorizer`:
   `rgb = alpha * colorizer(feature) + (1 - alpha) * rgb_background`
2. `feature_background_before_colorizer`:
   `rgb = colorizer(alpha * feature + (1 - alpha) * feature_background)`

Saved outputs:

- `outputs/benchmarks/2026-05-21_star_uvt_background_cheat_diagnostic.json`
- `outputs/benchmarks/2026-05-21_star_uvt_background_cheat_diagnostic.md`

Key measured row:

| mode | alpha | feature grad L2 | alpha grad | colorizer grad L2 |
| --- | ---: | ---: | ---: | ---: |
| rgb_background_after_colorizer | 0 | 0 | -0.73 | 0 |
| feature_background_before_colorizer | 0 | 0 | -1.782 | 1.64729 |

Interpretation: RGB background after colorizer gates colorizer gradients by
alpha. A fully empty pixel (`alpha = 0`) still gives an alpha/geometry signal
but gives the colorizer no reconstruction gradient. Feature background before
colorizer trains the colorizer even when `alpha = 0`, which is exactly the
background-feature cheat path we wanted to avoid. A random/noisy feature
background can be a useful negative-control ablation, but should not be the
default anti-cheat objective.

Partial alpha is not fully gated. At `alpha = 0.02`, the RGB-after-colorizer
path produced nonzero but tiny colorizer and feature gradients. This matches
the old note's "low-alpha edge bleed" warning and argues for alpha-bucket
diagnostics rather than feature-space randomization.

## STAR UVT Status

The shared dynamic-gsplat feature trainer is modular here. STAR UVT feature
projection is only partially modular:

- `dense_feature_tube_prototype.colorize_and_compose` already composes after
  colorization, but currently only accepts a fixed RGB tuple and defaults to
  black.
- The selected STAR UVT feature-tube runs mainly train target-grid feature loss
  plus a frozen RGB probe, so they do not yet use the shared
  `RGBReconObjective` random-background policy.
- `rgb_loss_weight > 0` in `train_star_uvt_feature_overfit.py` goes through
  `colorize_and_compose`, so that is the narrow place to wire a train-time RGB
  background policy when testing RGB reconstruction variants.

## Validation

Passed:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/background_cheat_diagnostic.py \
  tests/test_star_uvt_background_cheat_diagnostic.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_star_uvt_background_cheat_diagnostic.py \
  tests/test_objective_background_and_composition.py \
  tests/test_rgb_recon_objective.py -q
```

Result: `12 passed in 0.93s`.

## Next Experiments

1. Add alpha-bucket diagnostics to STAR UVT feature runs: colorizer-grad or RGB
   residual contribution for alpha buckets `[0,.01)`, `[.01,.05)`,
   `[.05,.2)`, `[.2,1]`.
2. Add a config-backed STAR UVT RGB background policy for the `rgb_loss_weight`
   path: `black`, `white`, `fixed_rgb`, `random_rgb_step`, and
   `random_rgb_pixel`.
3. Treat feature-background random/noise as a deliberate negative control:
   expected behavior is colorizer grad at alpha zero; if it "works" visually,
   verify alpha and support did not hollow out.
4. For the main feature-projection lane, prefer random RGB after any RGB
   colorizer/probe path and keep raw feature background fixed zero unless a
   specific parity test requires otherwise.
