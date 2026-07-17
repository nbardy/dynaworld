# Dynamic Foam frame-index helper cleanup

## Context

The modularization goal is still active. A live scan found Dynamic Foam scripts
parsing frame-index lists locally in slightly different ways:

- `build_multiview_feature_triangulation_point_cloud.py`
- `build_pycolmap_known_pose_point_cloud.py`
- `modal_powerfoam_aliked_geometry.py`
- `diagnose_powerfoam_sections.py`

The duplicate part was not the experiment behavior; it was the
comma-separated frame-index parsing plus "at least one frame" and optional
range validation.

## Changed

- `research_experiments/dynamic_foam/report_artifacts.py` now exposes:
  - `parse_frame_indices(...)`
  - `validate_frame_indices(...)`
- Feature triangulation now calls `parse_frame_indices(..., allow_all=True,
  frame_count=...)`, preserving its `"all"` behavior.
- Known-pose pycolmap keeps its argparse shape (`--frame-index` or
  `--frame-indices ...`) and delegates only nonempty/range validation.
- ALIKED geometry orchestration now uses the shared parser for its
  comma-separated probe frame list.
- Section diagnostics now uses the shared parser for `--frames`.
- `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` record the
  new helper boundary.

## Validation

- `py_compile` passed for the Dynamic Foam report helper, the four touched
  scripts, and `tests/test_dynamic_foam_report_artifacts.py`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_dynamic_foam_report_artifacts.py -q`
  passed: `6 passed in 0.04s`.
- `--help` smokes passed for:
  - `build_multiview_feature_triangulation_point_cloud.py`
  - `build_pycolmap_known_pose_point_cloud.py`
  - `diagnose_powerfoam_sections.py`

I did not run the Modal ALIKED entrypoint; that script was covered by
`py_compile` and the shared parser unit tests only.

## Handoff

Keep frame defaults and CLI shapes in the owning script. Use the helper only
for common parsing and range validation. Do not move point-cloud construction,
Modal staging, or pycolmap command behavior into `report_artifacts.py`.
