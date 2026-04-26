# Material Gauge Micro Sweep

## Context

After the 250-step material-gauge baseline reached 20.38 PSNR, the next useful
question was whether early fit is mostly controlled by element count or by
projected coverage/radius. Instead of running the full 18-cell matrix first, we
ran a compact 4-cell sweep:

```text
elements: 1024, 2048
radius: 0.07, 0.09
alpha_logit: 0.0
steps: 80
frames: 16
basis: 16
```

## Source Change

Added:

```text
research_experiments/gauge_fields/summarize_runs.py
```

Updated:

```text
research_experiments/gauge_fields/README.md
```

The summarizer reads `metrics.json`, `config.json`, and optional probe summaries
from material-gauge output directories and emits markdown/JSON tables sorted by
a selected metric.

## Commands

Generate configs:

```bash
rm -rf /tmp/gauge_micro_sweep_configs
uv run python research_experiments/gauge_fields/make_sweep_configs.py \
  --output-dir /tmp/gauge_micro_sweep_configs \
  --elements 1024,2048 \
  --radii 0.07,0.09 \
  --alpha-logits=0.0 \
  --steps 80 \
  --disable-wandb
```

Run sweep:

```bash
set -euo pipefail
for cfg in /tmp/gauge_micro_sweep_configs/*.jsonc; do
  uv run python research_experiments/gauge_fields/train.py "$cfg" --device mps --no-wandb
done
```

Summarize:

```bash
uv run python research_experiments/gauge_fields/summarize_runs.py \
  'outputs/gauge_fields/sweeps/gauge_fields_material_surfel_motion_128_16f_*el-r*p*-a0' \
  --out-md outputs/gauge_fields/sweeps/micro_sweep_80step_summary.md \
  --out-json outputs/gauge_fields/sweeps/micro_sweep_80step_summary.json
```

## Results

| Run | PSNR | L1 | Coverage | Radius p50 | Alpha > .90 | Xmap Occ |
|---|---:|---:|---:|---:|---:|---:|
| 2048 / r0.09 | 17.7753 | 0.0788 | 4.2986 | 3.2857 | 0.9799 | 0.2256 |
| 2048 / r0.07 | 17.6827 | 0.0793 | 2.8491 | 2.7026 | 0.8607 | 0.2400 |
| 1024 / r0.09 | 17.6021 | 0.0812 | 2.4955 | 3.5981 | 0.7627 | 0.2292 |
| 1024 / r0.07 | 17.0426 | 0.0884 | 1.7406 | 2.9833 | 0.2977 | 0.2581 |

## Interpretation

The early overfit is still very coverage-sensitive.

Increasing radius from 0.07 to 0.09 helps both capacities. In fact, 1024/r0.09
almost catches 2048/r0.07 by 80 steps:

```text
1024/r0.09: PSNR 17.60
2048/r0.07: PSNR 17.68
```

But radius is not a free win. The 2048/r0.09 cell has the best 80-step PSNR but
also the heaviest alpha carpet:

```text
coverage_budget: 4.30
alpha_coverage_090: 0.98
xmap_occ: 0.226
```

The 1024/r0.07 cell has the weakest RGB fit and weakest high-alpha coverage,
but the highest xmap occupancy:

```text
coverage_budget: 1.74
alpha_coverage_090: 0.30
xmap_occ: 0.258
```

So the practical tradeoff is now visible:

```text
larger radius -> easier RGB fit, heavier smear/coverage, lower xmap occupancy
more elements -> cleaner way to buy coverage, but slower
```

## Decision

Keep `2048/r0.07` as the stable baseline default for now. It is not the highest
80-step PSNR, but it is a better balance of fit, coverage, and xmap occupancy.

Run the full matrix next with:

```text
elements: 1024, 2048, 4096
radius: 0.05, 0.07, 0.09
alpha_logit: -1.2, 0.0
steps: 150
```

The full matrix should sort by PSNR, but report coverage and xmap occupancy in
the same table so radius-smear does not masquerade as representation quality.

