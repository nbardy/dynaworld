# Gauge RGB MP4 Helper

## Context

Continuation of the modular trainer/code organization goal. The Gauge Fields
scripts already shared `save_side_by_side_mp4(...)` for target-vs-render
videos, but `smiley_smoke.py` and `cheat_probe_material_gauge.py` each carried
the same OpenCV single-RGB-video writer.

## Change

- Added `save_rgb_mp4(...)` to `research_experiments/gauge_fields/common.py`.
- Routed `smiley_smoke.py` through the shared helper.
- Routed `cheat_probe_material_gauge.py` through the shared helper directly
  from `common.py`, rather than relying on an accidental re-export through the
  Gauge-local trainer module.

## Validation

- `rtk .venv/bin/python -m py_compile research_experiments/gauge_fields/common.py research_experiments/gauge_fields/smiley_smoke.py research_experiments/gauge_fields/cheat_probe_material_gauge.py`
- `rtk uv run python research_experiments/gauge_fields/smiley_smoke.py --help`
- `rtk uv run python research_experiments/gauge_fields/cheat_probe_material_gauge.py --help`
- `rtk uv run python research_experiments/gauge_fields/smiley_smoke.py --output-dir /tmp/dynaworld_smiley_smoke_helper --device cpu --frames 2 --size 16 --pixel-chunk 1024`

The tiny CPU smiley run succeeded and wrote outputs to
`/tmp/dynaworld_smiley_smoke_helper`, exercising the shared MP4 writer.

## Handoff

This is another narrow artifact-boundary cleanup. It does not change Gauge
rendering or probe math. Future Gauge cleanup should continue to move repeated
artifact mechanics into `common.py` while keeping experiment semantics local.
