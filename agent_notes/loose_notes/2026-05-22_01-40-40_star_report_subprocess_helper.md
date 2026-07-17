# STAR Report Subprocess Helper

## Context

Continuation of the training-code/interface cleanup goal. The STAR feature-tube
report scripts already had shared JSON/text artifact helpers, but several
trainer-report scripts still repeated the same subprocess launcher:

- build `PYTHONPATH=src/train:third_party/.../star_uvt_v0`
- create a per-report `tmp` directory
- write the command line and stdout/stderr to a log file
- convert return code or timeout into `status` / `error`
- measure elapsed wall time

This is a real shared boundary because the report scripts intentionally launch a
fresh trainer process for isolation, but the process-launch/log/status contract
should not be copied at every call site.

## Change

- Added `LoggedSubprocessResult`, `run_logged_subprocess(...)`, and
  `run_star_uvt_feature_trainer_subprocess(...)` to
  `research_experiments/star_uvt_feature_tubes/report_artifacts.py`.
- Routed these reports through the shared wrapper:
  - `sparse_forward_timing_repeat.py`
  - `sparse_forward_scale_matrix.py`
  - `targetgrid_render_mode_trainer_matrix.py`
- Added focused tests for successful subprocess logging and nonzero return-code
  reporting in `tests/test_star_uvt_report_artifacts.py`.
- Left support-birth/split and dense-support launch wrappers local for now
  because they add distinct env/command behavior (`STAR_UVT_TILE_CAPACITY`,
  dense diagnostic command construction, baseline case fanout).

## Validation

- `uv run python -m py_compile research_experiments/star_uvt_feature_tubes/report_artifacts.py research_experiments/star_uvt_feature_tubes/sparse_forward_timing_repeat.py research_experiments/star_uvt_feature_tubes/sparse_forward_scale_matrix.py research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py tests/test_star_uvt_report_artifacts.py`
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py -q`
  - `6 passed`
- `PYTHONPATH=src/train:. uv run python research_experiments/star_uvt_feature_tubes/sparse_forward_timing_repeat.py --repeat 1 --dry-run --out-base /tmp/dynaworld_sparse_repeat_smoke`
- `PYTHONPATH=src/train:. uv run python research_experiments/star_uvt_feature_tubes/sparse_forward_scale_matrix.py --sizes 128 --dry-run --out-base /tmp/dynaworld_sparse_scale_smoke`
- `PYTHONPATH=src/train:. uv run python research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py --modes feature_direct_atomic --dry-run --out-base /tmp/dynaworld_render_mode_smoke`
- `git diff --check`

The `uv` commands printed the existing parent-workspace warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` missing a `[project]`
table, but completed successfully.
