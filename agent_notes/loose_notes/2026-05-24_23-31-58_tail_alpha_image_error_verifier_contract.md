# Tail-Alpha Image-Error Verifier Contract

## Context

The Gauged UVT trace atlas uses a measured support-refresh path. Small support
drift can be reused when the omitted Gaussian tail is bounded, but missing core
support must still rebin. Earlier artifacts already measured this:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_metal_precision_rerun/summary.json
```

The weakness was that the saved JSONs were only checked by `all_passed`; future
runs could lose the aggregate-tail red-team or accidentally accept core support
reuse while still looking plausible.

## Change

Added a reusable report verifier to:

```text
research_experiments/star_uvt_feature_tubes/projective_tail_alpha_image_error_verifier.py
```

Exports:

```text
verify_tail_alpha_image_error_report(payload) -> list[str]
assert_tail_alpha_image_error_report(payload) -> None
```

The CLI also accepts:

```text
--verify-report outputs/.../summary.json
```

## Contract

Positive certified-tail cases must:

- pass,
- strict-rebin without certification,
- reuse stale support with certification,
- have `0 < support_tail_alpha_bound <= tail_alpha_epsilon`,
- have `max_abs_error <= 1.05 * support_tail_alpha_bound + 1e-7`,
- keep mean error no larger than max error.

Required positive cases include three affine boundary tails plus the tiny
rational orbit chart.

Negative controls must include:

- `core_loss_rejected`,
- `overlapping_tail_aggregate_rejected`.

Both must reject stale reuse, rebin, have aggregate tail bound above epsilon,
and show forced-reuse max RGB error above epsilon. This is the useful part:
the verifier guards against replacing the actual alpha-tail certificate with a
plain pixel-overshoot threshold.

## Verification

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_tail_alpha_image_error_verifier.py -q

7 passed in 9.09s
```

The three saved artifacts verified via CLI:

```text
verified outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.json
verified outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.json
verified outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_metal_precision_rerun/summary.json
```

Regenerated the base `tail_alpha_image_error` artifact because the local copy
predated the aggregate-tail case.

## Next

Use this verifier for every new tail-alpha image-error artifact. The next
meaningful expansion is broader real-scene support drift, not another local
toy: run the same contract on trained high-motion atlas updates and decide
whether remaining failures want tighter split/refit rules or a richer
fiber/halfspace cell.
