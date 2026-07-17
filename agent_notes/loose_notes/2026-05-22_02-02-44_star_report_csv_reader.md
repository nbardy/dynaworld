# STAR Report CSV Reader Cleanup

## Context

After adding `write_report_csv(...)`, two STAR report scripts still had local
CSV reader blocks using `csv.DictReader`:

- `logit_handoff_reduce_report.py`
- `gate4_quality_bracket_report.py`

Those readers were the same artifact-boundary concern as the writer cleanup:
root-relative CSV I/O belongs in `report_artifacts.py`, while row filtering,
type conversion, comparisons, and report-specific decisions should stay local.

## Changes

- Added `read_report_csv(...)` to
  `research_experiments/star_uvt_feature_tubes/report_artifacts.py`.
- Routed `logit_handoff_reduce_report.py` through `read_report_csv(...)` and
  removed its local `_read_rows(...)`/`csv.DictReader` helper.
- Routed `gate4_quality_bracket_report.py` through `read_report_csv(...)` and
  removed its direct `csv.DictReader` block.
- Extended `tests/test_star_uvt_report_artifacts.py` to cover readback from
  `write_report_csv(...)`.

## Validation

- `rtk uv run python -m py_compile research_experiments/star_uvt_feature_tubes/report_artifacts.py research_experiments/star_uvt_feature_tubes/logit_handoff_reduce_report.py research_experiments/star_uvt_feature_tubes/gate4_quality_bracket_report.py tests/test_star_uvt_report_artifacts.py`
  passed.
- `rtk env PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py -q`
  passed: `10 passed`.
- Fixture report invocation passed for
  `gate4_quality_bracket_report.py` using tiny temp JSON/CSV inputs.
- Fixture report invocation passed for
  `logit_handoff_reduce_report.py` using a root-relative temp matrix under
  `outputs/tmp_report_csv_reader_fixture`; that fixture was removed afterward.

## Notes

The first logit-handoff fixture attempt used an absolute `/tmp` matrix dir and
hit the report's existing `matrix_dir.relative_to(ROOT)` assumption. That is
pre-existing report behavior, not introduced by the CSV reader helper, so this
slice did not change it.
