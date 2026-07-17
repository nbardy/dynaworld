# Projective Exposure/Rolling Quadrature Artifact

## Context

The active STAR UVT / gauged UVT thread needs the memory anchors to stay
concrete:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

After the bundle gauge value/gradient reports and shared-work audit, the next
gap was finite exposure and rolling shutter. The theory says a shutter image is
not made by integrating primitive opacity before visibility. It is:

```text
I_frame(u,v) = integral_tau Composite(TraceAtlas, u, v, tau) d tau
```

## Work Done

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_quadrature_report.py
tests/test_star_uvt_projective_exposure_rolling_quadrature_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.md
```

The report checks:

- finite-exposure quadrature samples lower into a shared interval atlas and
  match the direct sensor-time CPU oracle;
- rolling shutter row schedules lower into one unique-time schedule with a
  `row_weights[Q,H]` matrix and match rowwise CPU rendering;
- mixed finite/rolling fallback marks `visibility_ambiguous_depth` cells,
  renders non-fallback regions with interval Metal, patches fallback regions
  using live-depth reference ordering, then accumulates exposure/row weights;
- optional Metal parity runs when MPS and the relevant ops are available.

## Evidence

Saved artifact summary:

```text
finite_reference_lowered_max_abs_error: 0.0
rolling_rowwise_batched_max_abs_error: 0.0
rolling_unique_to_row_sample_ratio: 0.875
finite_fallback_fraction: 0.5
rolling_fallback_fraction: 0.5
max_metal_abs_error: 5.960464477539063e-08
metal_case_count: 4
```

Per-case Metal max errors on this machine:

```text
finite interval Metal:       5.96e-8
rolling row-weighted Metal:  2.98e-8
finite mixed fallback:       2.98e-8
rolling mixed fallback:      2.98e-8
```

Tests/verifiers:

```text
.venv/bin/python -m py_compile ...: passed
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_exposure_rolling_quadrature_report.py -q
  6 passed, 1 skipped in 20.17s
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_exposure_rolling_quadrature_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.json
  verified
```

## Backtrack / Important Nuance

The first artifact attempt failed on the non-fallback Metal interval paths by
about `0.044` when the generic direct fixture included `depth_affine_uv`. The
CPU oracle used the sidecar, while the Metal interval path being audited here
does not implement that optional sidecar path for this report. This report is
about quadrature lowering, row weights, and fallback patching, so the fixture
was narrowed to the sidecar-free cell atlas that current Metal implements.

That does not invalidate `depth_affine_uv`; it means depth sidecars remain a
separate visibility-depth acceptance path and should not be quietly folded into
an exposure/rolling artifact unless the corresponding Metal support is explicit.

## Current Model

Finite exposure and rolling shutter are now covered at the evaluation-contract
level:

```text
partition support/order events -> quadrature samples
quadrature samples -> sample-indexed interval atlas
rolling row schedules -> unique sample times + row weights
ambiguous visibility -> tile/sample fallback mask
render/patch samples -> accumulate weights
```

This supports the broader theory: the camera-ray bundle base is still
`B = Omega x T`; rolling shutter just makes the sensor-time weight/camera map
row-dependent. The trace atlas remains the reusable object.

## Next Questions

- Add backward coverage for finite/rolling quadrature once the interval
  backward path exposes a clean row-weighted accumulation contract.
- Decide whether `depth_affine_uv` belongs in the interval Metal renderer or
  stays as a CPU/compiler visibility sidecar only.
- Move from focused synthetic scenes to an orbit/rolling scene with nontrivial
  visibility event roots and measure fallback fraction under motion.
