# Anisotropic Tail-Bound Verifier Contract

## Context

The isotropic tail-alpha image-error verifier now has a report contract. The
next mathematical gap was the gauged/rotated footprint case: if a trace atlas
uses per-trace SPD UV precision, stale support cannot be certified by the
scalar `sigma_px` radius alone. It needs the rectangle minimum:

```text
min_{x in omitted tile} (x - mu)^T P (x - mu)
```

where `P` is the 2x2 SPD UV precision block.

## Change

Added a reusable report verifier to:

```text
research_experiments/star_uvt_feature_tubes/projective_anisotropic_tail_bound_verifier.py
```

Exports:

```text
verify_anisotropic_tail_bound_report(payload) -> list[str]
assert_anisotropic_tail_bound_report(payload) -> None
```

The CLI now supports:

```text
--verify-report outputs/.../summary.json
```

## Contract

Positive cases must include:

- `diagonal_sigma_u1_v2_tail`,
- `rotated_precision_tail`,
- `two_trace_same_omitted_tile_sum`.

Each positive case must certify reuse, have non-empty omitted tiles, keep
`0 < omitted_alpha_bound <= tail_alpha_epsilon`, and keep observed max RGB
error below `1.01 * omitted_alpha_bound + 1e-7`.

The two-trace case must have a bound larger than each single-tail bound. This
guards the important aggregate rule: multiple omitted traces in one tile sum
their bounds before comparing to the budget.

The negative case `anisotropic_core_loss_rejected` must reject stale reuse,
have omitted bound above epsilon, and have forced-bad max error above `0.25`.

## Verification

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_anisotropic_tail_bound_verifier.py -q

6 passed in 8.81s
```

Both saved artifacts verified:

```text
verified outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound/summary.json
verified outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound_metal_precision_rerun/summary.json
```

## Next

Use this verifier whenever extending support-tail certificates to real
projective/gauged traces with non-isotropic UV precision. The next useful step
is not more toy cases; it is feeding the same contract with trained
high-motion/revolving-camera atlas updates and checking whether failures are
best solved by support guards, interval splitting, or richer fiber/halfspace
cells.
