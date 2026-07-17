# ALIKED Modal Report Artifact Boundary

## Context

Continuation of the trainer-landscape modularization cleanup. The current
pattern is to share report/artifact primitives only where the output contract is
identical, while keeping execution inputs and copied remote artifacts local.

## Change

- Routed ALIKED/Colmap Modal report JSON writes through
  `research_experiments.dynamic_foam.report_artifacts.write_report_json(...)`:
  - `modal_powerfoam_aliked_geometry.py` writes `plan.json` and local
    probe/full result JSONs through the helper.
  - `modal_powerfoam_aliked_onnx_check.py` writes `onnx_check.json` through the
    helper.
  - `modal_colmap_cli_onnx_check.py` writes `colmap_cli_onnx_check.json`
    through the helper.
- Left remote `manifest.jsonl`, remote `config.jsonc`, copied returned JSON
  files, PLY artifacts, and canonical copied artifacts local because those are
  execution inputs, byte-preserving remote payloads, or domain-specific
  artifacts rather than generic Dynaworld report files.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  research_experiments/dynamic_foam/modal_powerfoam_aliked_onnx_check.py \
  research_experiments/dynamic_foam/modal_colmap_cli_onnx_check.py \
  research_experiments/dynamic_foam/report_artifacts.py
```

Passed.

```bash
rtk .venv/bin/python - <<'PY'
# Fake-modal import smoke for the three Modal helper modules.
PY
```

Passed: `modal_import_smoke=ok`.

```bash
rtk .venv/bin/python - <<'PY'
# Fake-modal write smoke for write_plan(...) and write_local_result(...).
PY
```

Passed: `modal_aliked_write_smoke=ok`.

## Handoff

This is another report-artifact boundary cleanup only. It does not change
remote Modal execution, COLMAP/ALIKED matching behavior, point-cloud generation,
or copied returned file contents.
