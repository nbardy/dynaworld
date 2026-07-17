# Dynamic Foam Summary Write Boundary

## Context

Continued the modularization cleanup by targeting live Dynamic Foam scripts
that still hand-wrote adjacent JSON summaries with
`path.write_text(json.dumps(...))`. These were not trainer-math changes; the
goal was to keep summary/report serialization consistent with the existing
`report_artifacts.write_report_json(...)` helper.

## Changes

Routed JSON summary writes through `write_report_json(...)` in:

- `research_experiments/dynamic_foam/build_multiview_plane_sweep_point_cloud.py`
- `research_experiments/dynamic_foam/build_multiview_feature_triangulation_point_cloud.py`
- `research_experiments/dynamic_foam/prepare_ex4dgs_anchor_point_cloud.py`
- `research_experiments/dynamic_foam/verify_powerfoam_4k_trainability.py`

The point-cloud scripts still own their PLY generation and domain-specific
summary payloads. `prepare_ex4dgs_anchor_point_cloud.py` passes
`sort_keys=False` to preserve its previous unsorted JSON style as closely as
possible.

## Deliberately Not Routed

- PLY writers remain local.
- Dataset export payloads/manifests remain local for now because they are data
  contract files, not report summaries.
- Modal runner settings/results and fixture builders remain local because their
  file contracts differ from simple report summaries.

## Validation

- `py_compile` passed for all touched scripts plus `report_artifacts.py`.
- Direct CLI `--help` passed for:
  - `build_multiview_plane_sweep_point_cloud.py`
  - `build_multiview_feature_triangulation_point_cloud.py`
  - `prepare_ex4dgs_anchor_point_cloud.py`
  - `verify_powerfoam_4k_trainability.py`
- Package import smoke passed for the same four modules.
- Focused pytest passed:
  `tests/test_dynamic_foam_report_artifacts.py` plus
  `tests/test_powerfoam_direct.py::test_atomic_torch_save_preserves_existing_checkpoint_on_failure`
  (`4 passed`).
- The touched files no longer contain direct
  `write_text(json.dumps(...))` JSON summary writes.
- `git diff --check` passed for the touched files.
