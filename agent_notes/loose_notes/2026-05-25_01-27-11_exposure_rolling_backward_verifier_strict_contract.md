# Exposure/Rolling Backward Verifier Strict Contract

## Context

The active Gauged UVT Trace Atlas objective needs backward-pass reuse across
time, not only forward/evaluation reuse. The finite-exposure and rolling
backward report already checked the headline adjoint rule:

```text
dL/d sample_image[q,row] = weight[q,row] * dL/d final_image[row]
```

where global shutter uses scalar quadrature weights and rolling shutter uses a
`row_weights[Q,H]` matrix over deduplicated sensor-time samples. The verifier
still trusted some stale row evidence: the rolling reuse ratio and the aggregate
Metal compare errors.

## Work Done

Hardened:

```text
research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_backward_report.py
tests/test_star_uvt_projective_exposure_rolling_backward_report.py
```

The verifier now checks:

- the theory contract mentions sample-adjoint lowering;
- device fields are booleans;
- finite exposure has positive quadrature sample count and one lowered sample
  per quadrature sample;
- finite/rolling sample adjoints and rendered sample images are nonzero;
- rolling row count and row weight sums are valid;
- rolling `unique_to_row_sample_ratio` is recomputed from unique/total samples;
- coeff, opacity, and color reference gradients are nonzero;
- Metal aggregate `max_abs_error` and `max_rel_error` match the max over
  coeff/opacity/color subrows;
- the whole summary is recomputed from case rows.

Added negative tests for stale rolling ratio, stale Metal aggregate compare
fields, and non-boolean device metadata.

## Evidence

```text
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_backward_report.py \
  tests/test_star_uvt_projective_exposure_rolling_backward_report.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_exposure_rolling_backward_report.py -q
```

Result:

```text
11 passed in 25.19s
```

Saved artifact verification:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_backward_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.json

verified outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.json
```

Current saved summary:

```text
finite_has_metal_backward          = true
rolling_has_metal_backward         = true
rolling_unique_to_row_sample_ratio = 0.875
max_metal_grad_abs_error           = 1.430511474609375e-06
max_metal_grad_rel_error           = 6.377381396305282e-07
metal_backward_case_count          = 2
```

## Decision Implications

This strengthens the backward-pass half of the goal: final-frame adjoints can
be lowered to shared sample adjoints and accumulated through one interval-cell
VJP, with report rows now checked for internal consistency. It is still a
focused synthetic proof, not a full completion claim; the next useful step is
to carry the same strictness into a larger high-motion/trainer artifact or into
less synthetic rolling/fallback visibility.
