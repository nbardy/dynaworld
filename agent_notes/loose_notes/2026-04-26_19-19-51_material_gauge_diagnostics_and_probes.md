# Material Gauge Diagnostics And Probes

## Context

After the first material-surfel baseline commit, the chief scientist response
was clear: keep the baseline narrow, but make it measurable and red-teamable.

The next source drop therefore avoided new renderer work, learned cameras,
holonomy, or stronger losses. It added diagnostics and deterministic checkpoint
probes around the existing T-parameterized material gauge field.

## Changes

Updated:

```text
research_experiments/gauge_fields/train.py
research_experiments/gauge_fields/README.md
```

Added:

```text
research_experiments/gauge_fields/cheat_probe_material_gauge.py
research_experiments/gauge_fields/make_sweep_configs.py
```

## Diagnostic Metrics Added

The trainer now saves additional metrics into `metrics.json` and logs them to
W&B under `Diag/*` when W&B is enabled:

```text
projection_valid_fraction
projection_coverage_budget
projection_radius_px_p05/p50/p95/max
projection_radius_min_clamp_fraction
projection_radius_max_clamp_fraction
projection_depth_p05/p50/p95/max
alpha_coverage_090
alpha_hole_fraction
motion_delta_mean/p50/p95/max
motion_basis_norm_mean/p95
motion_coeff_norm_mean
motion_coeff_velocity_mean
motion_coeff_acceleration_mean
xmap_valid_fraction
xmap_occ
xmap_entropy
xmap_eff_bins
xmap_variance_x/y/z
xmap_local_smoothness
optional flow_magnitude stats
```

This makes coverage and gauge-collapse failures visible without changing the
training objective.

## Probe Harness

The new script:

```text
research_experiments/gauge_fields/cheat_probe_material_gauge.py
```

loads a checkpoint, renders the base model, applies deterministic
perturbations, re-renders, and writes:

```text
base_metrics.json
probe_summary.json
<probe>/probe_metrics.json
<probe>/preview.png
optional probe/base/absdiff mp4s
```

Current probes:

```text
depth_slide
radius_inflate
opacity_radius_trade
basis_scale_gauge
motion_phase_shift
```

## Sweep Generator

The new script:

```text
research_experiments/gauge_fields/make_sweep_configs.py
```

generates capacity/radius/alpha JSONC configs from the current 2048-element
baseline. It is intentionally only a config generator, not a launcher.

## Verification

Syntax:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/smiley_smoke.py \
  research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  research_experiments/gauge_fields/make_sweep_configs.py
```

Config parse:

```bash
uv run python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path('src/train').resolve()))
from config_utils import load_config_file
sys.path.insert(0, str(Path('research_experiments/gauge_fields').resolve()))
from train import gauge_config
for p in sorted(Path('src/train_configs').glob('local_mac_gauge_fields_material_surfel*.jsonc')):
    gauge_config(load_config_file(p))
    print(f'ok {p}')
PY
```

Diagnostic smoke:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc \
  --device cpu \
  --steps 2 \
  --no-wandb \
  --output-dir /tmp/gauge_fields_diag_smoke
```

Observed undercoverage signal from that smoke:

```text
projection_coverage_budget: 0.0552
projection_radius_px_p50: 0.75
alpha_coverage_050: 0.0
xmap_occ: 0.0078
```

This is the expected failure mode for the tiny smoke config.

Checkpoint probe smoke:

```bash
uv run python research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  --checkpoint outputs/gauge_fields/material_surfel_motion_128_16f_2048el_100step/checkpoint.pt \
  --output-dir /tmp/gauge_fields_cheat_probe_smoke \
  --device mps \
  --probe all \
  --no-video
```

First probe deltas on the saved 100-step baseline:

```text
basis_scale_gauge:
  delta_render_l1: 0.0
  delta_target_l1: 0.0

depth_slide:
  delta_render_l1: 0.00352
  delta_target_l1: 0.00027

motion_phase_shift:
  delta_render_l1: 0.01548
  delta_target_l1: 0.00398

opacity_radius_trade:
  delta_render_l1: 0.00441
  delta_target_l1: 0.00059

radius_inflate:
  delta_render_l1: 0.00522
  delta_target_l1: -0.00004
```

The basis-scale probe is exactly render-invariant, as expected from the
low-rank scale gauge. Radius inflation is nearly RGB-neutral and slightly
improves target L1 at this scale, which is a useful early warning that PSNR can
reward smear unless radius/coverage certificates are tracked.

## Takeaways

1. The scientist's requested renderer-health metrics are now first-class.
2. The red-team path is no longer only prose; there is a checkpoint probe
   harness with structured deltas.
3. The current material gauge still has real gauge freedoms. Some are harmless
   parameterization symmetries, like basis scale. Some are image-space cheats,
   like radius inflation.
4. Next useful run is still the 250/500-step baseline plus the generated
   capacity/radius/alpha sweep.

