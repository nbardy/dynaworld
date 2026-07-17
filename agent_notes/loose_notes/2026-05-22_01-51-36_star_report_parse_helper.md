# STAR Report Parse Helper Cleanup

## Context

The trainer/interface cleanup goal is still active. This slice kept the cleanup
small and live-file driven: several STAR UVT feature-tube report/matrix scripts
had identical optional JSON readers and comma-separated argument parsers, and
one direct-feature matrix still carried a local subprocess launcher.

## Changes

- Added shared report helpers in
  `research_experiments/star_uvt_feature_tubes/report_artifacts.py`:
  - `load_optional_report_json(...)`
  - `split_csv_strings(...)`
  - `split_csv_ints(...)`
  - `split_csv_floats(...)`
- Routed these scripts through the shared helpers:
  - `direct_feature_mode_matrix.py`
  - `targetgrid_render_mode_trainer_matrix.py`
  - `sparse_forward_timing_repeat.py`
  - `sparse_forward_scale_matrix.py`
  - `support_birth_split_sweep.py`
- Routed `direct_feature_mode_matrix.py` through
  `run_logged_subprocess(...)` so it shares the same log/status/timeout/TMPDIR
  behavior as the other STAR report subprocess runners.
- Kept script-specific summary schemas and CSV writing local. Those are
  per-report tables, not a shared trainer contract.

## Validation

- `rtk uv run python -m py_compile ...` passed for the shared helper, all touched
  report scripts, and `tests/test_star_uvt_report_artifacts.py`.
- `rtk env PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py -q`
  passed: `9 passed`.
- Dry-runs passed:
  - `support_birth_split_sweep.py --dry-run`
  - `sparse_forward_timing_repeat.py --repeat 1 --dry-run`
  - `sparse_forward_scale_matrix.py --sizes 128 --dry-run`
  - `targetgrid_render_mode_trainer_matrix.py --modes feature_direct_atomic --dry-run`
  - `direct_feature_mode_matrix.py --modes direct_atomic --sizes 128 --dry-run`
- `rtk git diff --check` passed after the code patch.

## Remaining

The broader objective is not complete. Next useful cleanup remains live-file
driven: avoid new base-trainer abstraction, keep experiment-specific math local,
and continue only the duplicate helpers that prove up in current scripts. The
alpha-background ablation and benchmark evidence are still the higher-value
next functional gates after this report-plumbing cleanup.
