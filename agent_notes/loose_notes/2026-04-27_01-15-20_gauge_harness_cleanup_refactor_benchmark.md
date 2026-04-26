# Gauge Harness Cleanup And Refactor Benchmark

## Context

The gauge-field ablation lane had split into a material-field trainer and a
direct 3DGS control, but the control was importing shared data/media utilities
from `research_experiments/gauge_fields/train.py`. That made `train.py` both a
trainer and a library. The cleanup extracted the shared pieces into:

```text
research_experiments/gauge_fields/common.py
research_experiments/gauge_fields/data.py
```

The representation code remains separate:

```text
train.py                  material field: screen_disk / oriented_slab / rank_adaptive_metric
train_splat_baseline.py   free dynamic 3DGS control
```

This keeps the comparison harness shared while preserving the fact that the
representations expose different semantics and diagnostics.

## Commits

```text
0eead59 Add gauge held-out camera ablation lane
5944f8f Add gauge math prompt and shared helpers
```

The parent repo was advanced to these `dynaworld` commits in:

```text
eac0e5e Update dynaworld gauge harness cleanup
```

## Verification

Compile check:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/common.py \
  research_experiments/gauge_fields/data.py \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/train_splat_baseline.py \
  research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  research_experiments/gauge_fields/summarize_runs.py \
  src/train/multicam_val_data.py \
  src/dataset_pipeline/multicam_val.py
```

Smoke runs:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc \
  --device mps --no-wandb --output-dir /tmp/gauge_fields_screen_smoke_refactor

uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_oriented_slab_smoke_32_2f_32el.jsonc \
  --device mps --no-wandb --output-dir /tmp/gauge_fields_slab_smoke_refactor

uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc \
  --device mps --no-wandb --output-dir /tmp/gauge_fields_metric_smoke_refactor

uv run python research_experiments/gauge_fields/train_splat_baseline.py \
  src/train_configs/local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc \
  --device mps --steps 1 --no-wandb \
  --output-dir /tmp/gauge_fields_splat_baseline_smoke_refactor
```

All passed.

## Refactor Benchmark

Fresh no-W&B 80-step held-out-camera benchmark:

```text
outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step_refactor/
```

Summary:

```text
outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step_refactor/summary.md
outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step_refactor/summary.json
```

| representation | train PSNR | train L1 | held-out PSNR | held-out L1 | X-map occ | held-out X-map occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| free_dynamic_3dgs | 20.5017 | 0.0661 | 9.7392 | 0.2357 | n/a | n/a |
| screen_disk | 24.6535 | 0.0381 | 9.6479 | 0.2402 | 0.1409 | 0.1394 |
| rank_adaptive_metric | 24.3132 | 0.0389 | 9.5864 | 0.2406 | 0.1670 | 0.1526 |
| oriented_slab | 25.0256 | 0.0355 | 9.3320 | 0.2477 | 0.2002 | 0.1997 |

The ranking after refactor matches the previous held-out lane:

```text
source fit: oriented_slab > screen_disk > rank_adaptive_metric > free_dynamic_3dgs
held-out:   free_dynamic_3dgs > screen_disk > rank_adaptive_metric > oriented_slab
```

## Read

The shared harness cleanup did not perturb the benchmark. It made the code
cleaner while preserving the important result: source-camera PSNR is not a
representation-selection metric. Held-out camera evaluation must remain in every
serious ablation.

The pure Torch anisotropic support modes are still slow, especially
`rank_adaptive_metric`. That is now a measured systems problem, not just a code
organization issue.

