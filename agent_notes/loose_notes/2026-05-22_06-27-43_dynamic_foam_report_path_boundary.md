# Dynamic Foam Report Path Boundary

## Context

Follow-up to the Dynamic Foam report artifact cleanup. The previous slice moved
simple report JSON reads/writes into `research_experiments/dynamic_foam/report_artifacts.py`,
but the routed report scripts still had local copies of:

```python
ROOT = Path(__file__).resolve().parents[2]

def rel(path: Path) -> str:
    ...
```

Those helpers only existed to serialize config/checkpoint/output/panel paths in
report JSON or stdout summaries.

## Changes

- `report_artifacts.py` now serves as the report path boundary too:
  - `PROJECT_ROOT`
  - `relative_to_project(path)`
- Routed these scripts to import `relative_to_project as rel`:
  - `diagnose_powerfoam_heldout_error.py`
  - `diagnose_powerfoam_topology_edges.py`
  - `diagnose_powerfoam_raytrace_support_gap.py`
  - `probe_powerfoam_camera_perturbations.py`
  - `diagnose_powerfoam_color_affine.py`
  - `verify_powerfoam_raytrace_start_support.py`
  - `verify_powerfoam_raytrace_real_view_alpha.py`
  - `compare_powerfoam_to_splats_nearest0040.py`
  - `compare_powerfoam_cuda_metal_smoke.py`
- `compare_powerfoam_to_splats_nearest0040.py` and
  `compare_powerfoam_cuda_metal_smoke.py` now use `PROJECT_ROOT` from the same
  helper for report default/output paths.
- Added a focused test for project-relative path shortening.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

Commands run:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/report_artifacts.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_topology_edges.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_raytrace_support_gap.py \
  research_experiments/dynamic_foam/probe_powerfoam_camera_perturbations.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_start_support.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py \
  research_experiments/dynamic_foam/compare_powerfoam_to_splats_nearest0040.py \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_dynamic_foam_report_artifacts.py \
  tests/test_powerfoam_cuda_smoke.py::test_cuda_metal_comparison_contract_writes_json \
  -q

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py --help
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_to_splats_nearest0040.py --help
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/diagnose_powerfoam_raytrace_support_gap.py --help
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_start_support.py --help

rg -n "ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]|def rel\\(" \
  <routed Dynamic Foam report scripts>
```

Results:

- `py_compile` passed.
- Focused pytest: `4 passed`.
- CLI import/help smokes passed for CUDA-vs-Metal, PowerFoam-vs-splats,
  support-gap, and raytrace-start-support scripts.
- The routed report scripts no longer contain local `ROOT` or `rel(...)`
  helper definitions.

## Remaining

This is report-path plumbing only. It does not change model construction,
renderer behavior, training math, or acceptance criteria.
