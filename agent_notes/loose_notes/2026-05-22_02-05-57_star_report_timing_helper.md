# STAR Report Timing Helper Cleanup

## Context

Several STAR report/matrix scripts repeated the same helper for summarizing
`step_timings_ms` while skipping the first warmup-ish step. The repeated helper
was small but live in four current scripts:

- `targetgrid_render_mode_trainer_matrix.py`
- `sparse_forward_timing_repeat.py`
- `sparse_forward_scale_matrix.py`
- `support_birth_split_sweep.py`

## Changes

- Added `mean_timing_without_first(row, key)` to
  `research_experiments/star_uvt_feature_tubes/report_artifacts.py`.
- Routed the four report/matrix scripts through the shared helper.
- Removed the duplicate local `_mean_without_first(...)` definitions.
- Added a focused test covering the first-step skip behavior.

## Validation

- `rtk uv run python -m py_compile ...` passed for the shared helper, all
  touched report scripts, and `tests/test_star_uvt_report_artifacts.py`.
- `rtk env PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py -q`
  passed: `11 passed`.
- Dry-runs passed:
  - `support_birth_split_sweep.py --dry-run`
  - `sparse_forward_timing_repeat.py --repeat 1 --dry-run`
  - `sparse_forward_scale_matrix.py --sizes 128 --dry-run`
  - `targetgrid_render_mode_trainer_matrix.py --modes feature_direct_atomic --dry-run`

## Notes

This did not change timing semantics. It only moves the repeated
`step_timings_ms[1:]` mean calculation into the shared report artifact module.
Each report still decides which timing keys become columns and how to aggregate
rows across runs.
