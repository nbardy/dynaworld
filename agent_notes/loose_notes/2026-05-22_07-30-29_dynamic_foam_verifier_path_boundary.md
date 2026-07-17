# Dynamic Foam Verifier Path Boundary

## Context

Follow-up to the Dynamic Foam path bootstrap cleanup. After moving point-cloud
and smoke-dataset scripts to `experiment_paths.py`, the remaining repeated
bootstrap was mostly in verifier and orchestration scripts:

- local `ROOT = Path(__file__).resolve().parents[2]`
- local `SRC_TRAIN` / `POWERFOAM_METAL`
- local `sys.path.insert(...)`
- local project-relative path display helpers

These were repeated script plumbing, not experiment-specific math.

## Change

Routed these Dynamic Foam scripts through the shared path boundary re-exported
by `report_artifacts.py`:

- `verify_powerfoam_4k_benchmarks.py`
- `verify_powerfoam_4k_trainability.py`
- `verify_powerfoam_clean_init_coverage.py`
- `diagnose_powerfoam_sections.py`
- `verify_powerfoam_completion_audit.py`
- `verify_powerfoam_paper_acceptance.py`
- `verify_powerfoam_cuda_smoke_results.py`
- `run_powerfoam_external_blockers.py`
- `powerfoam_cuda_smoke_runner.py`

They now use some combination of `PROJECT_ROOT`, `DYNAMIC_FOAM_ROOT`,
`POWERFOAM_METAL_ROOT`, `ensure_train_path()`, `ensure_sys_path(...)`, and
`relative_to_project(...)`.

## Deliberate Non-Changes

- `modal_powerfoam_aliked_geometry.py` keeps its custom `repo_root()` and
  `rel(...)` behavior because that script must run both from the local repo and
  inside Modal's `/root/dynaworld` staging layout.
- PowerFoam fixture writers remain local. Their JSON payloads are fixture/data
  artifacts, not generic Dynamic Foam reports.
- The CUDA smoke runner's embedded upstream `SMOKE_ENTRY` still writes its own
  JSON and calls `torch.cuda.synchronize()` because it executes inside the
  cloned upstream PowerFoam checkout.
- Modal settings/config/manifest input writes stay local because they are
  execution inputs, not reusable report artifacts.

## Validation

- `py_compile` passed for all nine routed scripts.
- Direct `--help` passed for:
  - `verify_powerfoam_4k_benchmarks.py`
  - `verify_powerfoam_4k_trainability.py`
  - `verify_powerfoam_clean_init_coverage.py`
  - `diagnose_powerfoam_sections.py`
  - `verify_powerfoam_completion_audit.py`
  - `verify_powerfoam_paper_acceptance.py`
  - `verify_powerfoam_cuda_smoke_results.py`
  - `run_powerfoam_external_blockers.py`
  - `powerfoam_cuda_smoke_runner.py`
- `tests/test_dynamic_foam_report_artifacts.py -q` passed: `5 passed`.
- Targeted `git diff --check` passed after the patch.

## Current State

The Dynamic Foam report/path boundary is now cleaner, but this is still a
slice toward the broader trainer modularization goal. The remaining cleanup
should keep separating shared plumbing from semantics-heavy trainer,
fixture, Modal, and upstream-runner code.
