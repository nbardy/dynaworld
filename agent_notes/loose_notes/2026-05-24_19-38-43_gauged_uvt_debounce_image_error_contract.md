# Gauged UVT Debounce Image-Error Contract

## Context

The cap128 support-guard path now has two ingredients:

- `slack_budgeted`, which spends crowded guarded-tile headroom on traces nearest
  to the support-event boundary.
- `support_stale_overshoot_epsilon`, which can debounce tiny support-boundary
  crossings instead of rebinning metadata on every live update.

The missing check was image-level: a tiny geometric overshoot is only safe if
the compiled support radius already encloses the meaningful footprint.

## Change

Added two trainer interval tests:

```text
test_projective_interval_subpixel_support_debounce_has_bounded_tail_error
test_projective_interval_subpixel_support_debounce_rejects_underspecified_support_assumption
```

The positive case uses real support padding. A `0.05px` tile-boundary overshoot
only omits a Gaussian tail, and strict rebin versus tolerant reuse has:

```text
max RGB error < 1e-4
mean RGB error < 1e-6
```

The negative control uses center-only support. The same nominal `0.05px`
boundary overshoot can drop the Gaussian core:

```text
max RGB error > 0.35
```

## Implication

Support debounce is not a universal safe default. It is valid only relative to a
certified footprint support and an explicit image/error budget. This aligns the
guard story with the theory: chart reuse requires a validity certificate, not
just a small number.

## Verification

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

117 passed in 18.20s
```

The follow-up visibility-guard stress added one more debounce invariant test.
The same focused gate now passes:

```text
119 passed in 26.64s
```
