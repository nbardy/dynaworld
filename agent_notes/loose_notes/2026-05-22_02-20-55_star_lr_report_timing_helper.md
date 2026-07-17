# STAR Feature1 LR Report Timing Helper

## Context

The feature1 LR reset and LR schedule reports still carried local
`_mean_no_first(...)` helpers after `report_artifacts.mean_timing_without_first`
became the shared STAR report contract for `step_timings_ms[1:]` timing means.

Both local helpers had the same semantics:

- read `step_timings_ms`
- skip the first timing entry
- average only entries that contain the requested key
- return `None` when there is no usable sample

## Changes

- Routed `star_uvt_feature1_lr_reset_report.py` through
  `mean_timing_without_first(...)`.
- Routed `star_uvt_feature1_lr_schedule_report.py` through
  `mean_timing_without_first(...)`.
- Removed both local `_mean_no_first(...)` copies.
- Updated `TODO/trainer_landscape_unification.md` and `CODE_ORGANIZATION.md`.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_lr_schedule_report.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_lr_reset_report.py \
  tests/test_star_uvt_report_artifacts.py
```

Passed.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_star_uvt_report_artifacts.py -q
```

Passed: `13 passed in 0.12s`.

`rg` no longer finds `_mean_no_first(...)` in the STAR UVT feature-tube scripts.
The known parent-project `uv` warning about `gsplats_browser/pyproject.toml`
lacking `[project]` remains harmless for these commands; exit codes were `0`.
