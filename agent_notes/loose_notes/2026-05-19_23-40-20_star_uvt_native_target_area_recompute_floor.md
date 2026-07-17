# STAR UVT Native Target-Area Recompute Floor

Date: 2026-05-19 23:40:20 +07

## Why

The feature/geometry split showed that raw feature atomics and geometry/opacity
atomics were not independent big halves of native hidden64 target-area backward.
The next question was whether a "no output gradients" mode would expose a much
lower shared floor.

## What Changed

Exposed the already-supported mode-bit combination as a named benchmark mode:

- `target_area_recompute_only = 3`

This sets both skip bits in the native target-area hidden VJP:

- skip feature-gradient atomics,
- skip ma/q/opacity geometry-gradient atomics.

It still replays target-area samples, accumulates the pixel feature, runs the
hidden64 colorizer forward, computes sigmoid RGB, runs hidden64 VJP, and enters
the reverse loop envelope. It is intentionally not trainable.

Files changed:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`
- `research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py`

No Metal rebuild was needed for this mode because the native kernel already
accepted mode bits `0..3`.

## Commands

Tiny parity:

```bash
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode target_area_recompute_only \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_recomputeonly_tiny_parity_h64.json
```

256 timing:

```bash
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode target_area_recompute_only \
  --timing-frames 64 --timing-size 256 --timing-tubes 8192 \
  --timing-feature-dim 32 --tile-capacity 256 --grid-side 64 --patch-size 4 \
  --timing-repeat 5 --skip-baseline \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_recomputeonly_timing_64f256_8192t_h64_cap256_nativeonly.json
```

512 timing:

```bash
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode target_area_recompute_only \
  --timing-frames 64 --timing-size 512 --timing-tubes 8192 \
  --timing-feature-dim 32 --tile-capacity 256 --grid-side 64 --patch-size 8 \
  --timing-repeat 3 --skip-baseline \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_recomputeonly_timing_64f512_8192t_h64_cap256_nativeonly.json
```

## Results

| Size | Full backward | Recompute-only backward |
|---:|---:|---:|
| 256 | `581.33ms` | `571.30ms` |
| 512 | `1919.75ms` | `2101.73ms` |

Recompute-only passed loss parity and returned zero gradients by design.

## Conclusion

The shared replay/hidden64 VJP envelope is the native target-area bottleneck.
Removing all output-gradient atomics does not create a useful speed floor. The
next native target-area work should not be another gradient-masking reducer; it
needs to reduce shared hidden64 recompute, use a compact scalar/prefix tape, or
change the support/objective.

Report:

`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_recompute_floor_gate.md`
