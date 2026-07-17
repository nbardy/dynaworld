# Projective Exposure/Rolling Backward Artifact

## Context

The active objective demands clean derivatives and shared backward work for
fast 2D rasters across time from 4D spacetime primitives. The forward
exposure/rolling quadrature artifact already proved:

```text
frame = integral_tau Composite(TraceAtlas(u,v,tau)) d tau
```

This note records the matching backward pass.

## Current Model

For a final image adjoint `G(row, col, channel)`, the lowered sample-image
adjoint is:

```text
finite exposure:
    dL/d sample[q, :, :, :] = weight[q] * G

rolling shutter:
    dL/d sample[q, row, :, :] = row_weights[q,row] * G[row, :, :]
```

Then the existing interval-cell VJP can run once on the shared sample-indexed
atlas:

```text
direct_backward_projective_trace_cell_interval_atlas_metal(
    lowered_atlas,
    unique_times,
    sample_adjoint,
    sample_config,
)
```

This is the backward analogue of the forward sample lowering. Rolling does not
need one backward replay per row; it pushes row adjoints into the shared unique
time schedule.

## Work Done

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_backward_report.py
tests/test_star_uvt_projective_exposure_rolling_backward_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.md
```

The report compares Metal direct VJP against Torch autograd on the lowered
interval atlas for:

- finite-exposure scalar quadrature weights;
- rolling-shutter row weights over a deduplicated unique-time schedule.

## Evidence

Saved artifact summary:

```text
finite_has_metal_backward: true
rolling_has_metal_backward: true
rolling_unique_to_row_sample_ratio: 0.875
max_metal_grad_abs_error: 1.430511474609375e-06
max_metal_grad_rel_error: 6.377381396305282e-07
metal_backward_case_count: 2
```

Finite-exposure gradient comparison:

```text
coeff max abs/rel:   1.43e-6 / 5.47e-7
opacity max abs/rel: 1.19e-7 / 6.38e-7
color max abs/rel:   5.96e-8 / 5.88e-7
```

Rolling gradient comparison:

```text
coeff max abs/rel:   8.94e-8 / 2.71e-7
opacity max abs/rel: 1.19e-7 / 1.44e-7
color max abs/rel:   5.96e-8 / 1.02e-7
```

Tests/verifiers:

```text
.venv/bin/python -m py_compile ...: passed
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_exposure_rolling_backward_report.py -q
  8 passed in 5.82s
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_backward_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.json
  verified
```

## What This Does And Does Not Prove

It proves the adjoint weighting and shared interval-cell VJP contract for the
focused finite/rolling scenes. It does not yet prove:

- mixed fallback backward for fallback tile/sample regions;
- a row-weighted fused backward kernel that avoids materializing `[Q,H,W,C]`
  sample adjoints;
- full trainer integration for finite exposure or rolling shutter.

Those are the next derivative-side gates.
