# Dynamic Foam Report JSONL Loader Boundary

## Context

Continued the training-code modularization goal by tightening another small
artifact/report boundary. Dynamic Foam already had `report_artifacts.py` for
strict JSON object reads and parent-safe JSON writes, but some report/verifier
scripts still parsed JSONL histories locally.

## What Changed

- Added `load_report_jsonl(path, missing_ok=False)` to
  `research_experiments/dynamic_foam/report_artifacts.py`.
- The helper skips blank lines, requires every nonblank line to decode to a
  JSON object, reports line numbers for bad rows, and has explicit
  `missing_ok=True` for optional histories.
- `compare_powerfoam_cuda_metal_smoke.py` now loads
  `train_metrics_history.jsonl` through the helper.
- `verify_powerfoam_paper_acceptance.py` now loads
  `eval_metrics_history.jsonl` through the helper, while keeping
  paper-acceptance metric-schema checks local.
- Removed local `load_json` / `read_jsonl` wrapper code from those callers where
  it only forwarded to the report helper.

## Validation

- `py_compile` passed for the helper, both routed callers, and the focused test.
- Direct `--help` passed for:
  - `compare_powerfoam_cuda_metal_smoke.py`
  - `verify_powerfoam_paper_acceptance.py`
- Focused pytest passed:
  `tests/test_dynamic_foam_report_artifacts.py`,
  `tests/test_powerfoam_cuda_smoke.py::test_cuda_metal_comparison_contract_writes_json`,
  and `tests/test_powerfoam_paper_acceptance.py`.

## Notes

This is intentionally not a broad data-loader abstraction. JSONL row loading is
shared because it is a report-file contract. The callers still own their metric
schema, acceptance criteria, and comparison logic.
