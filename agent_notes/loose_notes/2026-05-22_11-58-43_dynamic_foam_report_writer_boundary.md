# Dynamic Foam Report Writer Boundary

## Goal

Continue the modularization goal by routing a live report-output helper through
the shared artifact writer without changing Dynamic Foam script APIs.

## Change

- `research_experiments/dynamic_foam/report_artifacts.py` now calls
  `ensure_train_path()` on import and delegates `write_report_json(...)` to
  `train_artifacts.write_json(...)`.
- The Dynamic Foam report-facing API stays the same. Existing scripts still
  import `write_report_json(...)`, `load_report_json(...)`,
  `load_report_jsonl(...)`, and frame-index helpers from the Dynamic Foam report
  module.
- Strict Dynamic Foam read contracts stay local: JSON object validation, JSONL
  object row validation, missing-optional histories, and frame-index range
  checks are not pushed into the generic artifact module.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/report_artifacts.py src/train/train_artifacts.py

PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_dynamic_foam_report_artifacts.py -q

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python \
  research_experiments/dynamic_foam/compare_dynamic_powerfoam_motion_vs_repaint.py --help

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python \
  research_experiments/dynamic_foam/build_multiview_feature_triangulation_point_cloud.py --help
```

Results:

- `py_compile` passed.
- `tests/test_dynamic_foam_report_artifacts.py` passed: `6 passed in 0.06s`.
- Both Dynamic Foam help smokes passed, covering direct-script imports of
  `write_report_json(...)` and package-style imports of frame-index helpers.

## Handoff

This removes one more local `mkdir + json.dumps + write_text` writer from a
reusable report boundary. It is not a training-quality result; the active goal
still needs W&B/media/benchmark evidence before any trainer path is complete.
