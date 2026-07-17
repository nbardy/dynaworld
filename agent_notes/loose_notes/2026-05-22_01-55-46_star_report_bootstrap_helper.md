# STAR Report Bootstrap Helper Cleanup

## Context

The previous STAR report helper slice centralized optional JSON loading, CSV
argument parsing, and logged subprocess behavior. A second live duplication was
still present in the same report/matrix cluster: several scripts rebuilt
`ROOT`, `SRC_TRAIN`, and sometimes `STAR_UVT_ROOT`, then mutated `sys.path`
before importing `config_utils`.

`report_artifacts.py` already owns direct-script bootstrapping for the
Dynaworld root and `src/train`, so those scripts did not need their own
`sys.path` preambles.

## Changes

- Routed these scripts to import `ROOT` from `report_artifacts`:
  - `targetgrid_render_mode_trainer_matrix.py`
  - `sparse_forward_timing_repeat.py`
  - `sparse_forward_scale_matrix.py`
  - `support_birth_split_sweep.py`
- Routed `support_birth_split_sweep.py` to import `TRAIN_ROOT` and
  `STAR_UVT_ROOT` from `report_artifacts` for its dense diagnostic subprocess
  `pythonpath`.
- Removed the local `SRC_TRAIN` constants and `sys.path.insert(...)` preambles
  from that cluster.

## Validation

- `rtk uv run python -m py_compile ...` passed for the shared helper, all
  touched report scripts, and `tests/test_star_uvt_report_artifacts.py`.
- `rtk env PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py -q`
  passed: `9 passed`.
- Dry-runs passed after the import cleanup:
  - `support_birth_split_sweep.py --dry-run`
  - `sparse_forward_timing_repeat.py --repeat 1 --dry-run`
  - `sparse_forward_scale_matrix.py --sizes 128 --dry-run`
  - `targetgrid_render_mode_trainer_matrix.py --modes feature_direct_atomic --dry-run`
  - `direct_feature_mode_matrix.py --modes direct_atomic --sizes 128 --dry-run`

## Notes

This cleanup intentionally stayed at the script-interface layer. It did not
change trainer math, benchmark schemas, or report table formats. Those remain
owned by their individual reports.
