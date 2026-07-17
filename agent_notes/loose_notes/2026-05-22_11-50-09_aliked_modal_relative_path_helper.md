# ALIKED Modal relative-path helper cleanup

## Context

The modularization goal is still active. `modal_powerfoam_aliked_geometry.py`
already used the Dynamic Foam report helper for JSON writes and frame-index
parsing, but it still carried a local `rel(path)` helper for project-relative
output display. `research_experiments/dynamic_foam/report_artifacts.py`
already exposes `relative_to_project(...)` for that boundary.

## Changed

- `modal_powerfoam_aliked_geometry.py` now imports
  `relative_to_project as rel` from `report_artifacts`.
- The local `rel(path)` copy was removed.
- The script keeps its local `repo_root()` and `/root/dynaworld` Modal fallback
  because that is Modal orchestration behavior, not a generic report helper.
- `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` record
  that ALIKED output display is now covered by the Dynamic Foam report helper.

## Validation

- `py_compile` passed for:
  - `research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py`
  - `research_experiments/dynamic_foam/report_artifacts.py`
  - `tests/test_dynamic_foam_report_artifacts.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_dynamic_foam_report_artifacts.py -q`
  passed: `6 passed in 0.01s`.

No Modal ALIKED job was launched.

## Handoff

Keep relative path display in the Dynamic Foam report helper. Keep Modal image
construction, remote path mapping, file mirroring, and repo-root fallback in
the ALIKED orchestrator.
