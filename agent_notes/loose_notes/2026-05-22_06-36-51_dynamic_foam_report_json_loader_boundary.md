# Dynamic Foam Report JSON Loader Boundary

## Context

Continued the trainer/report cleanup goal after the Dynamic Foam report writer
and path-display helper had already landed. The remaining live duplication was
mostly strict JSON-object readers in Dynamic Foam report and verifier scripts.

## Changes

- Routed strict report-object reads through
  `research_experiments/dynamic_foam/report_artifacts.py::load_report_json`.
- Touched:
  - `research_experiments/dynamic_foam/compare_dynamic_powerfoam_motion_vs_repaint.py`
  - `research_experiments/dynamic_foam/diagnose_powerfoam_raytrace_support_gap.py`
  - `research_experiments/dynamic_foam/verify_dynamic_powerfoam_geometry_run.py`
  - `research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py`
  - `research_experiments/dynamic_foam/verify_powerfoam_4k_trainability.py`
  - `research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py`
  - `research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py`
- Kept wrapper functions where callers already used them, but moved the actual
  load/type-check contract to the helper.
- Added dual-mode imports for the helper so files still work as direct CLIs and
  also when imported as `research_experiments.dynamic_foam.*` from tests.

## Deliberately Not Routed

- `verify_powerfoam_4k_benchmarks.py` loads JSON lists, not JSON objects.
- `verify_powerfoam_paper_acceptance.py` still parses JSONL line-by-line.
- Modal smoke runner/settings, fixture builders, PLY merge helpers, and external
  blocker orchestration keep their local parsing because they are not simple
  report-object loads or intentionally have different error/streaming behavior.

## Validation

- `rtk .venv/bin/python -m py_compile` passed for all touched scripts plus
  `report_artifacts.py`.
- CLI help smokes passed for:
  - `compare_dynamic_powerfoam_motion_vs_repaint.py`
  - `diagnose_powerfoam_raytrace_support_gap.py`
  - `verify_dynamic_powerfoam_geometry_run.py`
  - `verify_powerfoam_cuda_smoke_results.py`
  - `verify_powerfoam_4k_trainability.py`
  - `verify_powerfoam_paper_acceptance.py`
  - `verify_powerfoam_completion_audit.py`
- Focused pytest passed:
  `tests/test_dynamic_foam_report_artifacts.py`,
  `tests/test_dynamic_powerfoam_metal.py::test_dynamic_powerfoam_geometry_summary_verifier_contract`,
  and `tests/test_powerfoam_paper_acceptance.py` (`11 passed`).
- Existing CUDA comparison helper coverage passed:
  `tests/test_dynamic_foam_report_artifacts.py` plus
  `tests/test_powerfoam_cuda_smoke.py::test_cuda_metal_comparison_contract_writes_json`
  (`4 passed`).
- `git diff --check` passed for the touched report/verifier scripts.
