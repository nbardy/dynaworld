# Gauge Columns Legend Helper

## Context

Continuation of the modular trainer/code-organization goal. Gauge Fields media
helpers were already converging in `common.py`, but the sidecar legend files
were still hand-written in multiple places:

- `common.save_preview_strip(...)`
- `smiley_smoke.save_smiley_strip(...)`
- `cheat_probe_material_gauge.save_probe_strip(...)`
- `cheat_probe_material_gauge.save_diagnostic_strips(...)`

Each produced a `*_columns.txt` file with the same `columns: a | b | c` format.

## Change

- Added `write_columns_legend(path, columns)` to
  `research_experiments/gauge_fields/common.py`.
- Routed preview-strip, smiley, probe-preview, xmap/depth/alpha, and flow
  sidecar writes through the shared helper.
- Kept the actual image composition local to each script.

## Validation

- `rtk .venv/bin/python -m py_compile research_experiments/gauge_fields/common.py research_experiments/gauge_fields/smiley_smoke.py research_experiments/gauge_fields/cheat_probe_material_gauge.py`
- `rtk uv run python research_experiments/gauge_fields/smiley_smoke.py --help`
- `rtk uv run python research_experiments/gauge_fields/cheat_probe_material_gauge.py --help`
- `rtk rm -rf /tmp/dynaworld_smiley_columns_helper && rtk uv run python research_experiments/gauge_fields/smiley_smoke.py --output-dir /tmp/dynaworld_smiley_columns_helper --device cpu --frames 2 --size 16 --pixel-chunk 1024 && rtk cat /tmp/dynaworld_smiley_columns_helper/smiley_static_columns.txt`

The tiny CPU smiley smoke succeeded and the sidecar contained:

```text
columns: rgb | alpha | depth
```

## Handoff

This is a narrow artifact helper extraction. It does not change Gauge rendering
or probe semantics. The broader modularization goal remains active.
