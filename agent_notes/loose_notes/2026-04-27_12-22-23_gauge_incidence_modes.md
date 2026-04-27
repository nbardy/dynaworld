# Gauge Incidence Modes

Date: 2026-04-27

## Context

We needed to turn the incidence-kernel theory into the existing gauge-fields
harness so projected conics, ray-Gaussian peak-density line integrals, and
ray-Gaussian mass-normalized line integrals can be checked against each other
on the same data path.

## Implementation

Added `render.incidence_mode` to the gauge harness:

```text
projected_conic
ray_gaussian_line_peak
ray_gaussian_line_mass
```

The generic ray/incidence math lives in:

```text
research_experiments/gauge_fields/incidence.py
```

The trainer keeps model-specific rendering/compositing code; tests import the
incidence math directly instead of pulling the full trainer where possible.
The sweep and summary scripts use the shared `common.py` dynaworld path helper.

The new line modes share the existing `rank_adaptive_metric` event state:

```text
x_i(t), Sigma_i(t), alpha/mass strength, c_i
```

and swap only the ray-event optical-depth law.

The exact line helper computes finite-segment optical depth:

```text
tau_i(ray) = integral_{s0}^{s1} sigma_i(o + s d) ds
alpha_i = 1 - exp(-tau_i)
```

The initial mass-normalized implementation treated sigmoid opacity as literal
total 3D mass and saturated alpha. We changed the renderer-side conversion so
the learned alpha initializes as an approximate center-line optical strength:

```text
mass ~= alpha * 2*pi*det(Sigma)^(1/3)
```

For isotropic support this gives:

```text
tau_center ~= alpha
```

This keeps the mass-normalized law comparable to projected conics at init.

## Verification

Unit tests:

```bash
uv run --with pytest python -m pytest tests/test_gauge_incidence.py
```

Result:

```text
4 passed
```

The tests cover:

```text
mass-normalized isotropic whole-line formula
peak-density finite-segment formula vs numeric quadrature
rigid invariance
config validation/defaults
```

Python compile check:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/incidence.py \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/make_sweep_configs.py \
  research_experiments/gauge_fields/summarize_runs.py \
  tests/test_gauge_incidence.py
```

Result: passed.

Diff whitespace check:

```bash
git diff --check -- \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/make_sweep_configs.py \
  research_experiments/gauge_fields/summarize_runs.py \
  tests/test_gauge_incidence.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_128_16f_2048el.jsonc \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_peak_128_16f_2048el.jsonc \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_mass_128_16f_2048el.jsonc
```

Result: passed.

## Smoke Results

Tiny explicit-video smoke:

```text
config: local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc
steps: 1
```

| incidence_mode | eval_psnr | eval_l1 | alpha_coverage_050 | xmap_occ |
| --- | ---: | ---: | ---: | ---: |
| projected_conic | 6.4940 | 0.4362 | 0.0000 | 0.0078 |
| ray_gaussian_line_mass | 6.4025 | 0.4415 | 0.0000 | 0.0078 |
| ray_gaussian_line_peak | 6.3488 | 0.4447 | 0.0000 | 0.0000 |

Compact DeepView held-out-camera smoke:

```text
config: local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_incidence_smoke_32_2f_64el.jsonc
steps: 2
source: DeepView 03_Dog camera_0001
heldout: camera_0015
```

| incidence_mode | eval_psnr | heldout_eval_psnr | heldout_eval_l1 | heldout_alpha_coverage_050 |
| --- | ---: | ---: | ---: | ---: |
| projected_conic | 6.2087 | 5.3154 | 0.5048 | 0.0234 |
| ray_gaussian_line_mass | 5.8856 | 5.0739 | 0.5218 | 0.0000 |
| ray_gaussian_line_peak | 3.9617 | 3.7377 | 0.6199 | 0.0000 |

These are not representation conclusions. They are plumbing and calibration
signals. Projected conic remains the fast control. Mass-normalized line
incidence is close enough to run the real 128px/16f benchmark. Peak-density line
incidence is under-covered at the default initialization and should be treated
as a diagnostic unless retuned.

## New Configs

Full-size configs:

```text
src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_peak_128_16f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_mass_128_16f_2048el.jsonc
```

Compact held-out smoke config:

```text
src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_incidence_smoke_32_2f_64el.jsonc
```

## Next

Run the real matrix:

```text
free_dynamic_3dgs
screen_disk / projected_conic
rank_adaptive_metric / projected_conic
rank_adaptive_metric / ray_gaussian_line_mass
rank_adaptive_metric / ray_gaussian_line_peak, optional diagnostic
```

Selector:

```text
heldout_eval_psnr
heldout_eval_l1
heldout_xmap_occ
coverage and alpha health
runtime
```

Do not promote a line-integral law if it only wins source-view RGB.
