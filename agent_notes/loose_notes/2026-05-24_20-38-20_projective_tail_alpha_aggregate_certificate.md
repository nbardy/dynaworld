# Projective Tail-Alpha Aggregate Certificate

## Context

The previous cap128 tail-alpha bracket treated the certificate as a max over
omitted primitive tails. That was useful for single-tail slivers, but it was not
the broader-scene bound we actually need. If many traces leak into the same
missing sample/tile, their omitted opacity can accumulate in the rendered image.

## Fix

`_support_tail_alpha_bound(...)` in
`third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py`
now aggregates omitted tail alpha by `(sample_index, tile_u, tile_v)` and returns
the maximum aggregate missing-tail value. It no longer returns the maximum
single primitive tail.

While rerunning the no-rebuild live-update path, the benchmark exposed a stale
metadata bug: `ProjectiveCellIntervalStaticAtlas` did not carry
`spatial_precision_uv`, yet `_materialize_projective_cell_atlas(...)` expected
it. The static atlas now preserves that field.

## Regression

Added:

```text
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_support_tail_alpha_certificate_aggregates_overlapping_tails
```

The fixture stacks 16 tiny omitted tails on the same tile. A `1e-3` budget now
rebins because the aggregate bound is about `0.00327`; a loose `4e-3` budget can
reuse, and the rendered residual remains below the aggregate bound.

## Corrected Artifacts

The old `0.00035` bracket is superseded.

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00045_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail0006_aggregate/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.md
```

Observed cap128 slack-budgeted behavior:

```text
epsilon 0.00035:
    max aggregate omitted-tail bound = 0.000404648
    measured stale refreshes / support rebins = 2 / 2
    identical final loss, zero overflow

epsilon 0.00045:
    max aggregate omitted-tail bound = 0.000526049
    measured stale refreshes / support rebins = 1 / 1
    identical final loss, zero overflow

epsilon 0.0006:
    max aggregate omitted-tail bound = 0.000656625
    measured stale refreshes / support rebins = 1 / 1
    identical final loss, zero overflow

epsilon 0.001:
    max aggregate omitted-tail bound = 0.000736007
    measured stale refreshes / support rebins = 0 / 0
    rebuilds = 1 vs cadence 4
    live updates = 7
    identical final loss, zero overflow
```

This bracket is path-dependent: raising the budget can skip earlier support
repairs, allowing later aggregate drift to grow. A single static threshold
should therefore be treated as a policy budget, not as a scene-independent
geometric constant.

## Image-Error Evidence

New verifier artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.md
```

At `tail_alpha_epsilon=0.00035`, the single-tail affine cases and rational
orbit case still reuse with max RGB error below their omitted-tail bound. The
new 64-trace overlap negative rejects reuse:

```text
aggregate omitted-tail bound = 0.01309515
forced loose reuse max RGB error = 0.00141417
```

This is the first broader-scene falsification loop for the certificate. It
proves the old max-per-trace rule was not strong enough and that the aggregate
rule catches a real multi-trace residual.

## Validation

Focused tail tests:

```text
3 passed in 7.52s
```

Focused projective suite:

```text
128 passed in 26.61s
```

## Next

The certificate is now mathematically better for overlap, but it is still local
and isotropic. Next falsification targets:

- anisotropic precision in the tail distance
- pixel-varying depth/support footprints
- real high-density scenes where many tails overlap across multiple tiles
- separating support-tail refreshes from visibility/order refreshes in summary
  metrics
