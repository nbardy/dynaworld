# Exposure/Rolling Verifier Strict Contract

## Context

The Gauged UVT Trace Atlas goal needs finite-exposure and rolling-shutter
evidence that the system integrates the rendered sensor-time field:

```text
I_frame(u,v) = integral_tau Composite(K, u, v, tau) d tau
```

not primitive opacity before visibility. A report already existed, but the
verifier trusted several stale row fields: complexity ratios, fallback sample
counts, and the Metal summary max/count.

## Work Done

Hardened:

```text
research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_quadrature_report.py
tests/test_star_uvt_projective_exposure_rolling_quadrature_report.py
```

The verifier now checks:

- device capability fields are booleans;
- finite exposure preserves quadrature sample count and source trace order;
- rolling shutter recomputes `unique / total_row_samples`;
- each complexity row recomputes interval/dense ratio and fallback fraction;
- non-fallback finite/rolling rows have zero fallback complexity;
- fallback rows have fallback cells, leave some cells on the fast path, and
  have strict tile/trace sample subsets;
- fallback complexity and fallback stats agree on cells/fraction;
- expected fallback images have positive L1 support;
- the whole summary, including `max_metal_abs_error` and `metal_case_count`, is
  recomputed from case rows.

Added focused negative tests for stale complexity ratios, bad fallback sample
counts, stale Metal summaries, and fallback/complexity mismatch.

## Evidence

```text
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_quadrature_report.py \
  tests/test_star_uvt_projective_exposure_rolling_quadrature_report.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_exposure_rolling_quadrature_report.py -q
```

Result:

```text
11 passed in 34.79s
```

The current artifact was regenerated and verified:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.json
verified outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.json
```

Current saved summary:

```text
finite_reference_lowered_max_abs_error = 0.0
rolling_rowwise_batched_max_abs_error  = 0.0
rolling_unique_to_row_sample_ratio     = 0.875
finite_fallback_fraction               = 0.5
rolling_fallback_fraction              = 0.5
max_metal_abs_error                    = 5.960464477539063e-08
metal_case_count                       = 4
```

## Decision Implications

This makes the exposure/rolling report stronger evidence for the camera-program
compiler: sample-time lowering, row-weight reuse, and fallback patching are now
verified as internally consistent report facts, not just as copied summary
numbers.

The broader goal is still active. Next work should connect this stricter
evaluation contract to larger high-motion artifacts or to the corresponding
backward verifier, then measure whether rolling/fallback work still grows
sublinearly when scene visibility is less synthetic.
