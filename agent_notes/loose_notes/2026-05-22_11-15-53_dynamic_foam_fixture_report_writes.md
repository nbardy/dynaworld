# Dynamic Foam Fixture/Runner Report Writes

## Context

Continuation of the trainer/code organization cleanup goal. The mixed
same-view/heldout trainer validation was rechecked first because it looked like
a possible `__init__` validation smell, but the live code already normalizes
and validates mixed schedule keys in `resolve_config(...)`, so there was no P5
trainer edit to make there.

The live duplication target was Dynamic Foam JSON-object artifact writes. The
repo already has `research_experiments/dynamic_foam/report_artifacts.py` with
`write_report_json(...)` and `load_report_json(...)`, but a few runner/fixture
surfaces still hand-wrote sorted newline JSON objects.

## Changes

- `make_powerfoam_parity_fixture.py` now writes its generated fixture through
  `write_report_json(...)`.
- `make_powerfoam_official_parity_fixture.py` now writes local/official fixture
  outputs through `write_report_json(...)`.
- `powerfoam_cuda_smoke_runner.py` now writes per-lane settings JSON through
  `write_report_json(...)` and reads lane metrics through
  `load_report_json(...)`.
- `modal_powerfoam_aliked_geometry.py` now writes generated remote config JSONC
  through `write_report_json(...)`.
- Left JSONL manifest writing and the embedded upstream CUDA smoke entry alone:
  the former is line-oriented data, and the latter executes inside the cloned
  upstream PowerFoam checkout rather than as a reusable Dynaworld report helper.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/make_powerfoam_parity_fixture.py \
  research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py \
  research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_dynamic_foam_report_artifacts.py -q
rtk env PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/make_powerfoam_parity_fixture.py \
  --output /tmp/dynaworld_powerfoam_parity_fixture_smoke.json
rtk env PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py \
  --backend local \
  --output /tmp/dynaworld_powerfoam_official_local_fixture_smoke.json
rtk env PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  --output-dir /tmp/dynaworld_powerfoam_cuda_smoke_plan \
  --run-id helper-routing-plan-smoke
rtk git diff --check -- \
  research_experiments/dynamic_foam/make_powerfoam_parity_fixture.py \
  research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py \
  research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py
```

Report artifact tests passed (`5 passed`). Both fixture smoke commands wrote
JSON outputs under `/tmp`. The CUDA smoke runner plan wrote
`/tmp/dynaworld_powerfoam_cuda_smoke_plan/summary.json`. No CUDA/Modal execution
was run; this was an artifact-helper cleanup, not a PowerFoam quality or timing
gate.
