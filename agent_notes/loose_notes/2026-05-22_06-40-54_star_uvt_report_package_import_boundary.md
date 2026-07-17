# STAR UVT Report Package Import Boundary

## Context

While continuing the trainer/report modularization cleanup, package-import
checks showed that several STAR feature-tube reports and prototypes were
runnable as direct scripts but failed when imported as
`research_experiments.star_uvt_feature_tubes.*` because they used the
direct-script import form:

```python
from report_artifacts import ...
```

This mattered because focused tests import the background-cheat diagnostic and
visibility-support prototypes through the package path.

## Change

Added:

- `research_experiments/star_uvt_feature_tubes/__init__.py`

The package init imports the package-local `report_artifacts` module and
registers it under the legacy top-level module name. This preserves direct CLI
execution for the existing report/prototype scripts while making package
imports work without adding dual-mode import blocks to every STAR report file.

## Validation

- Package import smoke passed for:
  - `research_experiments.star_uvt_feature_tubes.background_cheat_diagnostic`
  - `research_experiments.star_uvt_feature_tubes.visibility_support_bridge_prototype`
  - `research_experiments.star_uvt_feature_tubes.visibility_support_birth_split_prototype`
  - `research_experiments.star_uvt_feature_tubes.dense_feature_tube_prototype`
- `py_compile` passed for the new package init, `report_artifacts.py`, and the
  package-imported STAR report/prototype modules above.
- Focused pytest passed:
  `tests/test_star_uvt_report_artifacts.py`,
  `tests/test_star_uvt_background_cheat_diagnostic.py`, and
  `tests/test_star_uvt_visibility_support_bridge.py` (`17 passed`).
- Direct CLI help still passed for:
  - `background_cheat_diagnostic.py`
  - `visibility_support_bridge_prototype.py`
  - `visibility_support_birth_split_prototype.py`

## Notes

This is import-boundary cleanup only. It does not change STAR UVT training math,
shader behavior, objective selection, or benchmark results.
