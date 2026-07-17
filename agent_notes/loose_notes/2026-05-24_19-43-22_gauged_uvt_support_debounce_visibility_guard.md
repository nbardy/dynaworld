# Gauged UVT Support Debounce Visibility Guard

## Context

The previous pass added image-level support-debounce tests:

- real support padding plus `0.05px` overshoot gives tiny tail-only error,
- center-only support plus the same overshoot gives large image error.

That still left one dangerous interpretation open: a support tolerance could
accidentally be treated as permission to reuse stale visibility order.

## Change

Added:

```text
test_projective_interval_support_debounce_still_repairs_visibility_order
```

The fixture combines:

- a tolerated `0.05px` support boundary overshoot,
- stale stored depth intervals/order,
- live depths with the opposite front-to-back order.

Expected behavior:

```text
support_margin_before.max_boundary_overshoot_px == 0.05
visibility_before.stale == True
refresh.rebinned == True
visibility_after.order_mismatch_samples == 0
refreshed cell order == (1, 0)
```

## Implication

`support_stale_overshoot_epsilon` only debounces support coverage. Visibility
staleness still forces the atlas refresh path. This keeps the chart validity
contract separated:

```text
support certificate != visibility certificate
```

That separation matters for the Gauged UVT goal because the renderer can share
projection/binning work only where both support and order remain certified.

## Verification

Focused debounce cluster:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_subpixel_support_debounce_has_bounded_tail_error \
  tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_subpixel_support_debounce_rejects_underspecified_support_assumption \
  tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_support_debounce_still_repairs_visibility_order -q

3 passed in 1.95s
```

Full focused projective plus interval-gated trainer gate:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py -q

119 passed in 26.64s
```
