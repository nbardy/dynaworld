# Projective Tail-Alpha Support Certificate

## Context

Goal memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The pixel-distance debounce was useful but not satisfying: a fixed
`support_stale_overshoot_epsilon` says how far a support boundary crossed a
tile, not how much radiance/alpha the renderer might omit. The user explicitly
pushed for richer math rather than "fit residual checks and fallback", so this
pass turns the debounce into an error-style certificate.

## Math

For the current compatible projective interval route, screen footprints are
isotropic Gaussians with `sigma_px`. If a trace was compiled with support radius
`r = uv_padding`, and after an optimizer update its padded support crosses a
tile boundary by `delta`, then the nearest omitted continuous point is at least:

```text
d = max(r - delta, 0)
```

from the Gaussian center along one screen axis. Therefore:

```text
alpha_omitted <= opacity_upper * exp(-0.5 * (d / sigma_px)^2)
```

This is conservative because it ignores the second screen axis. It is still
much richer than a bare pixel epsilon:

- real tail loss: `r` large, `delta` tiny, alpha bound small
- core loss: `r=0` or `delta>=r`, alpha bound becomes opacity

## Implementation

Added:

```text
support_stale_tail_alpha_epsilon
```

through:

- `ProjectiveCellIntervalBackendConfig`
- `ProjectiveCellIntervalTrainerState`
- `refresh_projective_cell_interval_atlas_if_stale(...)`
- feature-trainer metrics
- projective interval cache benchmark CLI and summaries

Refresh now skips support rebinning only if either the old pixel overshoot
epsilon allows reuse or the new omitted-tail alpha bound is below
`support_stale_tail_alpha_epsilon`. Invalid active samples still force refresh.
Visibility stale still forces refresh.

## Tests

New tests:

```text
test_projective_interval_support_tail_alpha_certificate_debounces_subpixel_sliver
test_projective_interval_support_tail_alpha_certificate_rejects_core_loss
```

The first uses a `0.05px` boundary overshoot with `uv_padding=4`,
`sigma_px=1`, and opacity `0.5`. The tail bound sits between `1e-4` and
`3e-4`, so `1e-4` rebins and `3e-4` reuses.

The second uses `uv_padding=0`, so the missing region can contain the core. The
bound is `0.5` and a `0.1` alpha budget still rebins.

## Verification

Targeted tests:

```text
2 passed in 1.71s
```

Focused projective/interval suite:

```text
125 passed in 14.55s
```

Benchmark dry run:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --steps 1 \
  --support-guard-policy slack_budgeted \
  --support-stale-tail-alpha-epsilon 0.0003 \
  --dry-run \
  --out-dir /tmp/star_uvt_tail_alpha_benchmark_dryrun
```

The generated configs and summaries include
`support_stale_tail_alpha_epsilon=0.0003`.

Follow-up telemetry pass:

```text
projective_interval_cache_last_support_tail_alpha_bound
projective_interval_cache_max_support_tail_alpha_bound
```

now appears in trainer reports and benchmark summaries. Focused suite after the
telemetry patch:

```text
126 passed in 16.39s
```
