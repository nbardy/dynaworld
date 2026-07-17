# Dynamic Foam Report Artifact Boundary

## Context

Continued the trainer/code-organization cleanup by targeting a live report
duplication cluster in `research_experiments/dynamic_foam`. Several diagnostics
and comparison scripts wrote the same parent-safe, sorted, indented JSON reports
with hand-coded `mkdir + write_text(json.dumps(...))` blocks.

## Changes

- Added `research_experiments/dynamic_foam/report_artifacts.py` with:
  - `write_report_json(path, payload, sort_keys=True)`
  - `load_report_json(path)`
  - `relative_to_project(path)`
- Routed simple JSON report writes through the helper in:
  - `diagnose_powerfoam_heldout_error.py`
  - `diagnose_powerfoam_topology_edges.py`
  - `diagnose_powerfoam_raytrace_support_gap.py`
  - `probe_powerfoam_camera_perturbations.py`
  - `compare_dynamic_powerfoam_motion_vs_repaint.py`
  - `rank_video_motion.py`
  - `diagnose_powerfoam_color_affine.py`
  - `verify_powerfoam_raytrace_start_support.py`
  - `verify_powerfoam_raytrace_real_view_alpha.py`
  - `compare_powerfoam_to_splats_nearest0040.py`
  - `compare_powerfoam_cuda_metal_smoke.py`
- Routed two local JSON object loaders through `load_report_json(...)`.
- Left PLY writers, Modal staging/mirroring, config dumps, fixture writers,
  and streaming/log files local because those are different artifact contracts.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Validation

Commands run:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/report_artifacts.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_topology_edges.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_raytrace_support_gap.py \
  research_experiments/dynamic_foam/probe_powerfoam_camera_perturbations.py \
  research_experiments/dynamic_foam/compare_dynamic_powerfoam_motion_vs_repaint.py \
  research_experiments/dynamic_foam/rank_video_motion.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_start_support.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py \
  research_experiments/dynamic_foam/compare_powerfoam_to_splats_nearest0040.py \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py \
  tests/test_dynamic_foam_report_artifacts.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_dynamic_foam_report_artifacts.py \
  tests/test_powerfoam_cuda_smoke.py::test_cuda_metal_comparison_contract_writes_json \
  -q

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/compare_dynamic_powerfoam_motion_vs_repaint.py --help
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/rank_video_motion.py --help
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_to_splats_nearest0040.py --help
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/diagnose_powerfoam_topology_edges.py --help
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py --help
```

Results:

- `py_compile` passed for the helper, routed scripts, and helper tests.
- Focused pytest: `3 passed`.
- CLI import/help smokes passed for representative lightweight, comparison,
  and heavier PowerFoam diagnostic scripts.
- Direct sorted JSON write scan now leaves deliberate non-routed contracts:
  PLY/point-cloud summaries, Modal staging files, fixture writers, CUDA smoke
  runner internals, external-blocker config output, and other specialized
  artifact producers.

## Remaining

This is report plumbing only. It does not change trainer math, PowerFoam model
construction, or STAR/Dynamic quality evidence. Future cleanup should continue
from live scans and avoid forcing specialized artifact contracts into this
helper.
