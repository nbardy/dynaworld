# V-JEPA Benchmark Config Helpers

## Goal

Continue the trainer/interface cleanup on a reusable benchmark surface without
changing benchmark math, trainer objectives, or renderer behavior.

## Change

Added shared helpers in
`research_experiments/vjepa_performance/vjepa_benchmark_common.py`:

- `effective_splat_count(cfg)`
- `set_total_splat_count(cfg, splat_count)`
- `apply_video_benchmark_shape(cfg, render_size=..., clip_length=..., steps=...)`
- `quiet_training_logging(cfg, log_every=...)`

Routed these scripts through the shared helpers:

- `benchmark_free_splats_throughput.py`
- `profile_fast_mac_render_phases.py`
- `compare_fast_mac_quality.py`
- `benchmark_multicam_vjepa.py`

Also renamed the remaining local parser in `benchmark_fast_mac_variants.py` from
generic `parse_csv_strings(...)` to `parse_variant_csv(...)`, because it now
does variant membership validation on top of the shared nonempty CSV parser.

## What Stayed Local

- benchmark-specific config patching
- run-name formats
- AMP mode handling
- source-frame cases
- trainer timing/profiling math
- output schemas

## Validation

Commands run from the Dynaworld root:

```bash
rtk .venv/bin/python -m py_compile research_experiments/vjepa_performance/vjepa_benchmark_common.py research_experiments/vjepa_performance/benchmark_free_splats_throughput.py research_experiments/vjepa_performance/profile_fast_mac_render_phases.py research_experiments/vjepa_performance/compare_fast_mac_quality.py research_experiments/vjepa_performance/benchmark_multicam_vjepa.py research_experiments/vjepa_performance/benchmark_fast_mac_variants.py
rtk uv run python research_experiments/vjepa_performance/benchmark_free_splats_throughput.py --help
rtk uv run python research_experiments/vjepa_performance/profile_fast_mac_render_phases.py --help
rtk uv run python research_experiments/vjepa_performance/compare_fast_mac_quality.py --help
rtk uv run python research_experiments/vjepa_performance/benchmark_multicam_vjepa.py --help
rtk uv run python research_experiments/vjepa_performance/benchmark_fast_mac_variants.py --help
```

All compile/help checks passed. The `uv run` help checks still emit the known
parent `pyproject.toml` warning from `/Users/nicholasbardy/git/gsplats_browser`;
the CLIs exit successfully.

## Notes

This is an interface cleanup only. It proves the benchmark entrypoints still
parse and import. It does not prove any V-JEPA quality or speed claim; those
still require actual benchmark rows and W&B/result artifacts.
