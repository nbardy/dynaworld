# STAR Report CSV Helper Cleanup

## Context

The STAR report helper already owned JSON/text artifacts, optional JSON loads,
CSV-style CLI parsing, subprocess logging, and path bootstrap. One live artifact
write remained in the same report surface: `direct_feature_mode_matrix.py`
kept a local `csv.DictWriter` helper for its complete in-memory `summary.csv`.

## Changes

- Added `write_report_csv(...)` to
  `research_experiments/star_uvt_feature_tubes/report_artifacts.py`.
- The helper delegates to `train_artifacts.write_csv(...)` but preserves the
  direct-feature matrix's previous first-seen column order instead of using the
  generic sorted-field default.
- Routed `direct_feature_mode_matrix.py` through `write_report_csv(...)`.
- Removed the local `csv` import and `_write_csv(...)` helper from
  `direct_feature_mode_matrix.py`.
- Added a focused test for first-seen column-order preservation.

## Validation

- `rtk uv run python -m py_compile research_experiments/star_uvt_feature_tubes/report_artifacts.py research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py tests/test_star_uvt_report_artifacts.py`
  passed.
- `rtk env PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py -q`
  passed: `10 passed`.
- `rtk env PYTHONPATH=src/train:. uv run python research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py --modes direct_atomic --sizes 128 --dry-run --out-dir /tmp/dynaworld_direct_mode_csv_smoke`
  passed.

## Notes

This is intentionally an artifact-boundary cleanup only. The direct-feature
matrix still owns its benchmark command construction, row schema, markdown
summary, and sequential run policy.
