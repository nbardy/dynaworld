# Fast-Mac v13c Temporal Active Masks

Status: diagnostics only. Do not enable approximate temporal mask pruning from
this note alone.

## What was inspected

- `src/train/renderers/fast_mac.py` already carries active-tile controls for
  v6+ RGB/features and the v11 fixed-bin feature variant:
  `use_active_tiles`, `active_policy`, sparse/dense thresholds, sorted active
  tiles, and stop-count controls.
- `src/benchmarks/trainer_phase_benchmark.py` is the current trainer-level
  split point for project/raster timing. It obtains batched projected state via
  `project_for_fast_mac_batch(...)`, then calls the selected fast-mac variant.
- `src/train/renderers/tiled.py` has the closest pure-Python bound/tile
  assignment reference. The new diagnostic mirrors that style at the active-set
  level instead of calling a variant-specific Metal bridge.
- Existing v11 benchmark scripts time variant kernels and parity, but they do
  not report temporal reuse of visible Gaussians, active tiles, or
  Gaussian-tile assignment pairs.

## New diagnostic harness

`src/benchmarks/temporal_raster_overlap_profile.py` profiles projected Gaussian
active sets. It is intentionally a synthetic projected-state approximation for
now:

- generates `B x G` projected centers with configurable per-frame drift/noise,
  radius, opacity, and feature dimension;
- converts each projected disk to screen bounds and tile assignments;
- reports visible-Gaussian overlap, active-tile overlap, and Gaussian-tile-pair
  overlap;
- supports small matrices by comma-separating `--gaussians`, `--radius-px`, and
  `--motion-px`;
- labels results as `input_mode=synthetic_projected_approximation`.

The most important metrics are:

- `active_tiles_fraction_mean`: how much of the frame's tile grid is active.
  If this is close to 1.0, a tile mask alone cannot save launch work.
- `gaussian_tile_pair_density_mean`: active Gaussian-tile pairs divided by
  dense `G * tiles_total` work. This is the practical sparse-work signal.
- `gaussian_tile_pair_adjacent_jaccard_mean`: frame-to-frame reuse of exact
  Gaussian-tile pairs. This is the minimum signal for any temporal cache.
- `gaussian_tile_pair_adjacent_retention_mean`: fraction of previous-frame
  pairs still active next frame. This tells whether reusing last-frame work
  would mostly be valid or mostly stale.
- `active_gaussians_per_tile` and `active_tiles_per_gaussian` quantiles:
  pressure points for per-tile capacity and fixed-bin sizing.

## Synthetic smoke results

Command:

```bash
uv run python src/benchmarks/temporal_raster_overlap_profile.py \
  --frames 4 \
  --gaussians 256,512 \
  --height 128 \
  --width 128 \
  --tile-size 16 \
  --feature-dim 32 \
  --radius-px 3.5 \
  --motion-px 0,2,8 \
  --noise-px 0.25 \
  --seed 7
```

Summary:

| G | motion px | active tile fraction | pair density | pair adjacent Jaccard | pair adjacent retention |
|---:|---:|---:|---:|---:|---:|
| 256 | 0 | 0.9844 | 0.037094 | 0.9508 | 0.9742 |
| 256 | 2 | 0.9844 | 0.037048 | 0.9145 | 0.9603 |
| 256 | 8 | 0.9844 | 0.036667 | 0.7633 | 0.8686 |
| 512 | 0 | 1.0000 | 0.035660 | 0.9462 | 0.9718 |
| 512 | 2 | 1.0000 | 0.035583 | 0.9137 | 0.9538 |
| 512 | 8 | 1.0000 | 0.035400 | 0.7504 | 0.8515 |

Readout: in this dense 128px synthetic regime, the tile grid is basically
saturated, so an active-tile-only mask would not remove much launch surface.
The Gaussian-tile pair set is much sparser than dense tile-by-G work, but its
temporal stability falls quickly with motion. That makes pair-level diagnostics
more promising than tile-presence diagnostics, but also raises correctness risk
for approximate temporal reuse.

## Thresholds before a real mask path

Only consider a real v13c temporal mask path if projected-state diagnostics on
trainer samples, not just synthetic inputs, satisfy all of these:

- `active_tiles_fraction_mean <= 0.70` if the proposed path masks only tiles.
  Above that, the saved launch surface is too small to justify new complexity.
- `gaussian_tile_pair_density_mean <= 0.25` if the proposed path masks
  Gaussian-tile pairs. Above that, dense/fixed-bin work is probably competitive.
- `gaussian_tile_pair_adjacent_jaccard_mean >= 0.90` and
  `gaussian_tile_pair_adjacent_retention_mean >= 0.95` for last-frame reuse
  without a conservative refresh every frame.
- No visible-Gaussian churn spikes on real multicam clips: adjacent visible
  Jaccard should stay high per camera/view. If visibility churn is caused by
  camera motion or clipping, a temporal mask cache must be invalidated.
- Pair-count and per-tile occupancy quantiles must stay inside the existing
  fixed-bin capacity envelope. A sparse mask that triggers overflow fallback is
  not a win.

These are go/no-go gates, not quality claims. They must be paired with raster
parity checks and wall-clock timing before v13c becomes a benchmark candidate.

## Why pruning stays disabled

Approximate mask pruning can silently drop small but still visible
contributors. That is especially dangerous in the F32 feature path because
downstream colorization can amplify feature errors that look negligible in
alpha space. Before pruning exists, we need:

1. real projected-state profiles from `project_for_fast_mac_batch(...)`;
2. parity checks against unpruned v11/v12 outputs for features, alpha, RGB, and
   gradients;
3. timing evidence that the mask path beats current fixed-bin/hostmeta behavior
   after accounting for mask construction and invalidation.

Until those are all true, temporal active sets are a measurement surface only.
