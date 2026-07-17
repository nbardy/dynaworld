# Renderer Scaling Report Artifact Boundary

## Goal

Continue the trainer/benchmark modularization pass by moving another live
benchmark-report surface onto shared report artifact helpers without changing
the renderer comparison table semantics.

## Change

- Added `research_experiments/report_artifacts.py` for generic top-level
  research report plumbing:
  - Dynaworld root resolution,
  - `src/train` path bootstrap,
  - JSON object reads,
  - JSONL reads,
  - CSV reads,
  - CSV writes,
  - JSON writes,
  - text writes.
- Routed `research_experiments/renderer_scaling_report.py` through that helper.
- Kept renderer-family row normalization, sorting, matched-row selection, and
  markdown table content local to the renderer scaling report.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/report_artifacts.py \
  research_experiments/renderer_scaling_report.py \
  tests/test_research_report_artifacts.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_research_report_artifacts.py -q
```

Passed: `1 passed in 0.01s`.

```bash
rtk .venv/bin/python research_experiments/renderer_scaling_report.py \
  --out-md /tmp/renderer_scaling_report_shared_artifacts.md \
  --out-csv /tmp/renderer_scaling_report_shared_artifacts.csv \
  --report-date 2026-05-22
```

Passed with `rows=145`. The smoke wrote:

- `/tmp/renderer_scaling_report_shared_artifacts.md`
- `/tmp/renderer_scaling_report_shared_artifacts.csv`

## Handoff

This is a report-plumbing cleanup, not a benchmark rerun. It keeps the renderer
scaling report ready for future STAR/dynamic/feature table refreshes using the
same artifact primitives as the trainer and benchmark stack.
