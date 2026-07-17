# STAR report CSV parser follow-up

## Context

The current cleanup pass is unifying small duplicated boundaries without
changing trainer math or benchmark semantics. `star_uvt_feature_tubes` already
had typed CSV parser helpers in `report_artifacts.py`, but several adjacent
diagnostic and benchmark scripts still hand-parsed comma-separated values.

## Changed

- `direct_feature_kernel_benchmark.py`,
  `sparse_hidden_sigmoid_mse_kernel_benchmark.py`, and
  `sparse_hidden_target_area_kernel_benchmark.py` now parse `--feature-dims`
  through `split_csv_ints(...)`.
- `dense_alpha_failure_diagnostic.py` now parses `--raw-opacity-biases`
  through `split_csv_floats(...)` while preserving the empty-list default.
- `background_cheat_diagnostic.py` now parses `--alphas` through
  `split_csv_floats(...)`; its nonempty/range validation stays local because it
  is a diagnostic-specific contract.
- `firstclass_backward_breakdown.py` now parses `--modes` through
  `split_csv_strings(...)`.
- `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` now record
  this parser-boundary expansion.

## Validation

- `py_compile` passed for all touched STAR scripts plus
  `tests/test_star_uvt_report_artifacts.py`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py -q`
  passed: `13 passed in 0.38s`.
- `--help` smokes passed for:
  - `background_cheat_diagnostic.py`
  - `firstclass_backward_breakdown.py`
  - `direct_feature_kernel_benchmark.py`
  - `sparse_hidden_sigmoid_mse_kernel_benchmark.py`
  - `sparse_hidden_target_area_kernel_benchmark.py`
  - `dense_alpha_failure_diagnostic.py`
- `git diff --check` passed for the STAR script files before the docs update.

No MPS benchmark or train run was launched; this was a parser/interface cleanup
only.

## Handoff

Keep parsing and root-relative report artifact I/O in
`star_uvt_feature_tubes/report_artifacts.py`. Keep validation that depends on a
specific experiment's meaning local to that script. Do not pull kernel case
definitions, row schemas, or trainer-patch logic into the report helper unless
another exact duplicate appears.
