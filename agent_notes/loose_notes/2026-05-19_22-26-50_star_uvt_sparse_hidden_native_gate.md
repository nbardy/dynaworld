# STAR UVT Sparse Hidden Native Gate

Date: 2026-05-19

## What Changed

Added a benchmark-only sparse hidden sigmoid-MSE native STAR UVT backward:

- Python wrapper:
  `direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins(...)`
- Torch schema:
  `direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins`
- Metal bridge and kernel:
  `direct_atomic_feature_sparse_hidden_sigmoid_mse_backward`
- Benchmark harness:
  `research_experiments/star_uvt_feature_tubes/sparse_hidden_sigmoid_mse_kernel_benchmark.py`

The op reuses cached sparse tile bins, renders only selected pixels, computes a
hidden no-pre-norm `FeatureToColor` RGB/loss VJP inside Metal, returns STAR
gradients and a scalar sparse loss, and skips colorizer parameter gradients.

## Results

Tiny parity passes for F4/F32 and H32/H64. Max errors are under `9e-07` for
loss and under `1.5e-08` for STAR gradients.

Main timing at `64f/8192t/F32`, warm3/repeat5:

| Hidden | Res | Sparse side | Baseline total | Fused total | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 128 | 64 | `100.14ms` | `105.27ms` | `0.95x` |
| 32 | 256 | 64 | `35.77ms` | `28.68ms` | `1.25x` |
| 32 | 512 | 32 | `9.73ms` | `7.06ms` | `1.38x` |
| 32 | 512 | 64 | `29.66ms` | `18.47ms` | `1.61x` |
| 32 | 512 | 128 | `111.17ms` | `64.17ms` | `1.73x` |
| 64 | 256 | 64 | `51.27ms` | `40.21ms` | `1.28x` |
| 64 | 512 | 64 | `45.09ms` | `28.40ms` | `1.59x` |

All timing rows have zero overflow.

## Interpretation

This is a positive native gate, but only at the sparse visual boundary. Dense
hidden fusion was a correctness scaffold and not a speed route. Sparse hidden
fusion works because it avoids dense `[T,H,W,F]` support and removes the
Torch-side hidden-VJP prep over selected pixels.

The 128px side64 row loses because the same selected-pixel count and 8192 tubes
are packed into fewer tiles. This is a tile-occupancy warning: sparse visual
cost scales with selected pixels and per-selected-pixel tile occupancy, not
with dense image area alone.

## Next

Use this as a candidate for native sparse visual/probe trainer integration, but
compare against the selected sparse-forward batched target/probe VJP route before
promotion. Do not spend more time on dense hidden native fusion unless the
objective changes.
