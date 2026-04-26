# Material Gauge Support-Mode 250-Step Ablation

## Context

This follow-up tested the three material-gauge support modes on the same local
128px / 16-frame / 2048-element video overfit setup:

- `screen_disk`
- `oriented_slab`
- `rank_adaptive_metric`

The goal was not to prove novel-view geometry. The goal was to check whether
the richer support modes improve source-view fit and whether deterministic
cheat probes become less RGB-null than they are for the projected screen-disk
control.

## Commands

Training outputs:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc \
  --device mps --steps 250 \
  --output-dir outputs/gauge_fields/support_mode_250/screen_disk_2048el

uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_oriented_slab_motion_128_16f_2048el.jsonc \
  --device mps --steps 250 \
  --output-dir outputs/gauge_fields/support_mode_250/oriented_slab_2048el

uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_motion_128_16f_2048el.jsonc \
  --device mps --steps 250 \
  --output-dir outputs/gauge_fields/support_mode_250/rank_adaptive_metric_2048el
```

Probe outputs:

```bash
uv run python research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  --checkpoint outputs/gauge_fields/support_mode_250/<run>/checkpoint.pt \
  --output-dir outputs/gauge_fields/support_mode_250/<run>/probes \
  --device mps --probe all --no-video
```

Summary artifact:

```bash
uv run python research_experiments/gauge_fields/summarize_runs.py \
  'outputs/gauge_fields/support_mode_250/*_2048el' \
  --out-md outputs/gauge_fields/support_mode_250/support_mode_250_summary.md \
  --out-json outputs/gauge_fields/support_mode_250/support_mode_250_summary.json
```

## Reproducibility Fix

The first anisotropic probe attempt exposed a checkpoint-reload bug.
`support_knn_idx` was registered as `persistent=False`. Training used the KNN
graph built from initialization points, but probe reload rebuilt KNN from the
trained `x0`. That changed the transported support Jacobian and made anisotropic
checkpoints render differently after reload.

Fix:

- `support_knn_idx` is now a persistent buffer.
- `cheat_probe_material_gauge.py` tolerates missing `support_knn_idx` only for
  older checkpoints.

The oriented-slab and rank-adaptive-metric 250-step runs were rerun after this
fix. The current probe summaries reproduce their checkpoint baselines.

## Source-View Result

| support mode | eval PSNR | eval L1 | coverage budget | radius p50 | radius p95 | anisotropy p95 | motion mean | X-map occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oriented_slab | 20.5371 | 0.0562 | 2.9773 | 2.5830 | 3.9050 | 4.1821 | 0.0518 | 0.3057 |
| screen_disk | 20.3144 | 0.0582 | 2.8528 | 2.6705 | 3.1631 | 1.0000 | 0.0647 | 0.2544 |
| rank_adaptive_metric | 19.9614 | 0.0602 | 3.7441 | 2.8098 | 4.5002 | 3.7100 | 0.0396 | 0.1995 |

Observed source-view ranking:

```text
oriented_slab > screen_disk > rank_adaptive_metric
```

The oriented slab currently wins on both fit and X-map occupancy. The
rank-adaptive metric over-covers more heavily and has worse X-map occupancy,
which means the first SPECTRE-like metric support is not yet the better
practical primitive in this tiny source-view overfit.

## Cheat Probe Deltas

Target L1 deltas:

| run | depth slide | dormant insert | opacity split | radius inflate | motion shift | opacity/radius trade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screen_disk | 0.000151 | -0.000001 | -0.000009 | 0.000511 | 0.015821 | 0.000980 |
| oriented_slab | 0.000662 | 0.010121 | 0.013839 | 0.000382 | 0.014591 | 0.000671 |
| rank_adaptive_metric | 0.000930 | 0.006588 | 0.008726 | 0.000421 | 0.010664 | 0.000636 |

Render L1 deltas:

| run | depth slide | dormant insert | opacity split | radius inflate | motion shift | opacity/radius trade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screen_disk | 0.002576 | 0.000141 | 0.003109 | 0.006383 | 0.035620 | 0.005349 |
| oriented_slab | 0.006181 | 0.026091 | 0.032261 | 0.005973 | 0.033395 | 0.005197 |
| rank_adaptive_metric | 0.009482 | 0.022613 | 0.026153 | 0.004556 | 0.029252 | 0.004281 |

Coverage / identity deltas:

| run | dormant coverage | split coverage | radius coverage | base X-map occ | dormant X-map occ | split X-map occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screen_disk | 0.127607 | 1.437405 | 0.159118 | 0.254395 | -0.073242 | -0.005127 |
| oriented_slab | 0.048099 | 2.043853 | 0.162920 | 0.305664 | -0.111572 | -0.001465 |
| rank_adaptive_metric | 0.147541 | 2.927111 | 0.208796 | 0.199463 | -0.048096 | -0.010986 |

## Interpretation

The most useful signal is not just PSNR. The projected screen disk still has
near-null dormant insertion and opacity-split probes:

```text
screen dormant target delta ~= 0
screen opacity-split target delta ~= 0
```

The oriented slab makes those probes visible:

```text
oriented-slab dormant target delta = 0.010121
oriented-slab opacity-split target delta = 0.013839
```

That is the strongest positive evidence for richer support in this run. The
slab is less vulnerable to two classic RGB-fiber cheats while also fitting the
source video slightly better.

The rank-adaptive metric is not a win yet. It makes some cheats visible, but it
underfits source RGB, over-covers the frame, and collapses X-map occupancy
relative to slab. The likely issue is not the broad idea of transported metric
support; it is that this first implementation has too much coverage freedom and
not enough spectrum/rank pressure.

The `basis_scale_gauge` probe still has exactly zero render and target delta in
all modes. That remains an internal parameter gauge: rescaling the motion basis
and inverse-rescaling coefficients leaves the actual motion unchanged.

## Decision

For the next baseline, treat `oriented_slab` as the strongest current support
mode. Keep `screen_disk` as the required control. Treat `rank_adaptive_metric`
as experimental and do not promote it until it wins at least one of:

- omitted-frame PSNR / L1,
- camera/view stress certificates,
- X-map/flow consistency,
- cheat robustness at matched coverage.

## Next Tests

1. Add omitted-frame or held-out-camera evaluation before declaring geometric
   progress. Source-camera overfit is not enough.
2. Tune `rank_adaptive_metric` with lower coverage or explicit spectrum/rank
   regularization.
3. Add a matched primitive/parameter-count comparison against the splat
   baseline.
4. Add a view-stress report for small camera offsets, even if no RGB target is
   available.
5. Keep exact Gaussian line-integral alpha deferred until metric support shows
   a concrete advantage.

