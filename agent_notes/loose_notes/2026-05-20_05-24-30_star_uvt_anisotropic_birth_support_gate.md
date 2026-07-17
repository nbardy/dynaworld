# STAR UVT Anisotropic Birth Support Gate

## Question

The prior radius and born-opacity sweeps showed a smooth coverage/oracle tradeoff:
wider or higher-opacity support raises dense alpha a little, but gives back target
background oracle. The next hypothesis was that support shape, not scalar radius
or opacity, might move the frontier: put born tubes in an ellipse aligned with
the fitted target-point trajectory, wide along motion and narrow across it.

## Implementation

`src/train/train_star_uvt_feature_overfit.py` now supports:

- `support_birth_split.support_shape`: `isotropic` or `trajectory_ellipse`
- `support_birth_split.support_radius_along_px`
- `support_birth_split.support_radius_across_px`
- `support_birth_split.support_precision_radius_px`

Defaults preserve the old isotropic path by setting all three new radii to
`support_radius_px`. The `trajectory_ellipse` path uses the fitted
`velocity_uv` as the along axis, falls back to the x-axis if the fit has
near-zero speed, and still keeps the fixed tube budget by reallocating existing
tubes.

`research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py` now
accepts:

```bash
--support-shapes
--support-along-radii
--support-across-radii
--support-precision-radii
```

The generated row labels include the shape and anisotropic radii, for example
`trajectory_ellipse_a88_x24_p48`.

Focused validation:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  tests/test_star_uvt_feature_target_adapter.py \
  research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_visibility_support_bridge.py -q
```

Result: `36 passed`.

## Sweep

Command:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py \
  --target-sources uncovered_brightness,low_alpha \
  --reallocate-tubes 32 \
  --support-radii 64 \
  --support-shapes trajectory_ellipse \
  --support-along-radii 88 \
  --support-across-radii 24,32 \
  --support-precision-radii 48 \
  --opacities 0.4,0.6 \
  --tile-capacities 128 \
  --out-base outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128
```

Artifacts:

- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128.md`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128.json`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128_dense_support.md`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128_dense_support.json`

## Results

All eight rows passed with zero tile overflow and max/p95/cap `100/71/128`.

Dense alpha `>0.1` stayed in `0.408-0.409`, below the previous isotropic
uncovered baseline at `0.411` and below the intermediate-radius `r80/r88`
coverage rows at `0.415/0.417`. Forced-alpha PSNR stayed around
`14.554-14.566`; target-background oracle stayed high at `25.404-25.541`, but
that is because the narrow ellipse gives back coverage rather than solving
visibility.

Best dense coverage row:

- `low_alpha_n32_r64_trajectory_ellipse_a88_x32_p48_o0p6_cap128`
- alpha `>0.1`: `0.408975`
- normal PSNR: `5.685`
- forced-alpha PSNR: `14.556`
- oracle PSNR: `25.404`
- max tile count: `100/128`

## Read

This is a clean negative. The anisotropic ellipse is mechanically useful and
safe, but it does not move the coverage frontier. The likely reason is that
the current birth/split primitive fits one global line through target points;
for a broad 512px video target, that collapses many uncovered regions into one
average support stripe. Narrowing the stripe improves oracle by avoiding some
background contamination, but it gives up the actual alpha support we need.

Do not expand this exact grid. The next support-changing experiment should
split target points into multiple centers/trajectories before birth:

- multi-cluster birth/split with `K=4` or `K=8` target clusters and `32` total
  reallocated tubes
- stratified target-cell birth/split that reserves tubes per spatial target
  region rather than one fitted line
- only after that, repeat the radius/opacity/cap sweeps on the multi-center
  primitive

Current state: birth/split remains real progress as a fixed-budget
support-changing primitive, but the single-line ellipse variant is not the
quality bridge.
