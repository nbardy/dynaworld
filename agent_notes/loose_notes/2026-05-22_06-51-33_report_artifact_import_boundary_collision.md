# Report Artifact Import Boundary Collision

## Context

This chunk continued the trainer/interface cleanup goal. The narrow target was
the report-artifact helper boundary for Dynamic Foam and STAR UVT
feature-tube report/prototype scripts.

Earlier cleanup added local `report_artifacts.py` helpers for both experiment
families. The intended shape is simple:

- shared parent-safe JSON/text/CSV helpers live near the scripts that use them
- experiment-specific schemas stay local
- direct script execution still works
- package imports used by tests and other tools do not depend on current
  working directory or top-level module name collisions

## What Changed

- Dynamic Foam report scripts use dual-mode imports:
  `from .report_artifacts import ...` for package import, with
  `from report_artifacts import ...` as a direct-script fallback.
- STAR UVT report/prototype scripts now use the same dual-mode import shape.
- Removed the STAR package-level `__init__.py` alias approach that registered
  package-local `report_artifacts` as the process-global top-level module.

## Why The Alias Was Rejected

The package alias looked convenient because many STAR scripts historically used
`from report_artifacts import ...`. Focused pytest showed it was not safe:

- a Dynamic Foam module/test can import its helper as top-level
  `report_artifacts`
- STAR package import then sees the Dynamic Foam helper under that global name
- STAR modules fail when they need APIs that only exist in the STAR helper, such
  as `write_report_text`

The durable rule is: experiment packages should not share a process-global
helper module name. Use relative imports for package use, and keep the
top-level fallback only for direct CLI execution.

## Validation

- `py_compile` passed for Dynamic Foam and STAR UVT report/prototype scripts.
- Mixed package import smoke passed in both orders:
  Dynamic Foam then STAR, and STAR then Dynamic Foam.
- Direct CLI `--help` passed for representative Dynamic Foam scripts:
  `compare_powerfoam_cuda_metal_smoke.py`,
  `compare_powerfoam_to_splats_nearest0040.py`,
  `diagnose_powerfoam_heldout_error.py`, `rank_video_motion.py`,
  `build_multiview_plane_sweep_point_cloud.py`,
  `build_multiview_feature_triangulation_point_cloud.py`,
  `prepare_ex4dgs_anchor_point_cloud.py`, and
  `verify_powerfoam_4k_trainability.py`.
- Direct CLI `--help` passed for representative STAR scripts:
  `background_cheat_diagnostic.py` and
  `visibility_support_bridge_prototype.py`.
- Focused pytest passed:
  `tests/test_dynamic_foam_report_artifacts.py`,
  `tests/test_powerfoam_cuda_smoke.py::test_cuda_metal_comparison_contract_writes_json`,
  `tests/test_dynamic_powerfoam_metal.py::test_dynamic_powerfoam_geometry_summary_verifier_contract`,
  `tests/test_powerfoam_paper_acceptance.py`,
  `tests/test_star_uvt_report_artifacts.py`,
  `tests/test_star_uvt_background_cheat_diagnostic.py`, and
  `tests/test_star_uvt_visibility_support_bridge.py`.

## Remaining Work

- This cleanup does not change training math or shader behavior.
- Keep unifying active repeated boundaries only when they have a real contract:
  config CLI, registry dispatch, device/sync, artifact I/O, W&B cadence, report
  helpers, and background/composition helpers.
- Do not centralize experiment-specific result schemas, optimizer contracts,
  checkpoint payloads, model internals, or benchmark math just to reduce file
  count.
- The next real research work remains the alpha-background ablation and the
  STAR UVT feature/visual quality bottleneck, not another base trainer class.
