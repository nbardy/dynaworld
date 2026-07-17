# Dynamic Foam Path Bootstrap Boundary

## Context

Continuation of the trainer/code-organization cleanup goal. The prior report
artifact pass left several Dynamic Foam scripts still carrying local
`ROOT = Path(__file__).resolve().parents[2]`, `TRAIN_SRC`, and
`sys.path.insert(...)` preambles. Those are not trainer math, but they are
repeated script plumbing and make direct CLI/package-import behavior drift.

## Change

- Added `research_experiments/dynamic_foam/experiment_paths.py` with shared
  Dynamic Foam path primitives:
  - `DYNAMIC_FOAM_ROOT`
  - `PROJECT_ROOT`
  - `TRAIN_SRC`
  - `POWERFOAM_METAL_ROOT`
  - `ensure_sys_path(...)`
  - `ensure_train_path()`
  - `relative_to_project(...)`
- Updated `report_artifacts.py` to re-export those helpers while preserving
  direct-script fallback imports.
- Routed these scripts through the shared path/bootstrap boundary:
  - `export_powerfoam_smoke_dataset.py`
  - `build_multiview_plane_sweep_point_cloud.py`
  - `build_multiview_feature_triangulation_point_cloud.py`
  - `prepare_ex4dgs_anchor_point_cloud.py`
  - `build_pycolmap_known_pose_point_cloud.py`
- Routed the smoke dataset transform and summary JSON writes through
  `write_report_json(...)`.

## Deliberate Non-Changes

- PLY writers stayed local. They are data-format writers, not report artifacts.
- Image/video/manifest semantics stayed local to the dataset and point-cloud
  scripts.
- External blocker config writes, PowerFoam fixture writes, Modal staging
  inputs, and embedded upstream CUDA smoke writes stayed local because they are
  execution inputs or upstream checkout outputs, not reusable Dynamic Foam
  report contracts.

## Validation

- `py_compile` passed for:
  - `experiment_paths.py`
  - `report_artifacts.py`
  - the smoke dataset exporter
  - the multiview plane-sweep builder
  - the multiview feature-triangulation builder
  - the EX4DGS anchor prep script
  - the known-pose pycolmap builder
- Direct `--help` passed for all five routed scripts.
- Smoke dataset export wrote `transforms_all.json` and
  `dynaworld_smoke_dataset.json` through the shared writer.
- Package import and direct-script import of `report_artifacts` both worked.
- `tests/test_dynamic_foam_report_artifacts.py -q` passed: `5 passed`.
- `git diff --check` passed for the touched files.

## Current State

This is another small interface cleanup, not a completion claim. The active
goal remains open. Remaining cleanup should keep using this standard: unify
only repeated boundaries with the same semantics, and leave experiment-specific
data formats, fixtures, configs, and trainer loops local.
