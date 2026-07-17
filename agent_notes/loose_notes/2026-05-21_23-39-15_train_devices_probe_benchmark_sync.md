# Train Devices Probe And Benchmark Sync

## Context

`src/train/train_devices.py` already owned the shared MPS/CUDA device primitive
for routed trainers and STAR runtime code. A follow-up scan found trainer-
adjacent probes and benchmark orchestrators still carrying local copies of the
same auto-device or sync helpers.

## Change

Routed these scripts through `train_devices`:

- `src/train/probe_colorize_init.py`
- `src/train/probe_colorize_matrix.py`
- `research_experiments/vjepa_performance/benchmark_free_splats_throughput.py`
- `research_experiments/vjepa_performance/benchmark_multicam_vjepa.py`
- `research_experiments/vjepa_performance/profile_fast_mac_render_phases.py`
- `research_experiments/vjepa_performance/compare_fast_mac_quality.py`
- `research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`

The colorize probes now use `resolve_torch_device(..., auto_cuda=True)`,
preserving the old auto order of MPS, then CUDA, then CPU. The benchmark/
ablation scripts now alias `sync_torch_device(...)` instead of defining local
`sync(...)` or `_sync_device(...)` functions.

Deep one-off WorldFoam and low-level kernel probes were intentionally left
alone. Those files are isolated research forks and often bake in local timing
rituals that should not be generalized without a promotion decision.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_devices.py \
  src/train/probe_colorize_init.py \
  src/train/probe_colorize_matrix.py \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  research_experiments/vjepa_performance/profile_fast_mac_render_phases.py \
  research_experiments/vjepa_performance/compare_fast_mac_quality.py \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_devices.py \
  tests/test_trainer_registry.py \
  tests/test_train_cli.py -q
```

The focused pytest slice passed with 17 tests.
