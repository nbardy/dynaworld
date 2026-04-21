# 4K 64K Renderer Stack Benchmark

## Setup

User asked for a key-use-case comparison across direct Torch, Taichi, and
fast-mac at 4K and 64K total splats for batch sizes 1 and 4.

Assumption used:

- Resolution: `4096x4096`.
- Total splats per batch fixed at `65,536`.
- B=1 uses `G=65,536` per image.
- B=4 uses `G=16,384` per image.
- Cases: `sparse_sigma_1_5` and `medium_sigma_3_8`.
- Timing: warmup `1`, timed iterations `3`.
- Direct Torch was requested but skipped by the benchmark because the work size
  is `1,099,511,627,776` pixel-splat evaluations, far above the configured
  direct-Torch cap of `64,000,000`.
- Taichi had to run with `tile_size=32`; at 4K its default 16x16 tile mode
  asserts because `256*256` tiles hits the 16-bit tile-id cap.

Benchmark harness update:

- `src/benchmarks/mac_renderer_stack_compare.py` now includes v5 as
  `metal_v5_native`.
- Added `--taichi-tile-size`.

Output CSVs:

- `benchmark_outputs/mac_renderer_stack_4096_b1_total65536_forward.csv`
- `benchmark_outputs/mac_renderer_stack_4096_b4_total65536_forward.csv`
- `benchmark_outputs/mac_renderer_stack_4096_b1_total65536_backward.csv`
- `benchmark_outputs/mac_renderer_stack_4096_b4_total65536_backward.csv`

## Forward Mean Timings

| Batch / Case | Taichi | v2 | v3 | v5 | Best fast | Best vs Taichi |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| B=1, sparse 1-5 | `429.057 ms` | `16.902 ms` | `15.013 ms` | `9.660 ms` | v5 | `44.41x` / `+4341%` |
| B=1, medium 3-8 | `234.464 ms` | `32.072 ms` | `20.276 ms` | `15.012 ms` | v5 | `15.62x` / `+1462%` |
| B=4, sparse 1-5 | `1959.834 ms` | `51.446 ms` | `63.891 ms` | `26.336 ms` | v5 | `74.42x` / `+7342%` |
| B=4, medium 3-8 | `2053.581 ms` | `64.992 ms` | `53.015 ms` | `22.290 ms` | v5 | `92.13x` / `+9113%` |

## Forward+Backward Mean Timings

| Batch / Case | Taichi | v2 | v3 | v5 | Best fast | Best vs Taichi |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| B=1, sparse 1-5 | `1155.436 ms` | `105.146 ms` | `68.963 ms` | `67.432 ms` | v5 | `17.13x` / `+1613%` |
| B=1, medium 3-8 | `2585.650 ms` | `193.886 ms` | `89.777 ms` | `95.694 ms` | v3 | `28.80x` / `+2780%` |
| B=4, sparse 1-5 | `4623.423 ms` | `239.023 ms` | `338.144 ms` | `296.362 ms` | v2 | `19.34x` / `+1834%` |
| B=4, medium 3-8 | `11141.139 ms` | `489.933 ms` | `627.433 ms` | `329.199 ms` | v5 | `33.84x` / `+3284%` |

## Current Read

Direct Torch is not a practical measured baseline at this size. The measured
baseline is Taichi/Metal with 32x32 tiles.

For forward-only rendering, v5 won every fixed-total 64K 4K case. Native batch
is especially strong at B=4.

For forward+backward, the winner depends on footprint and batch:

- v5 narrowly wins B=1 sparse.
- v3 wins B=1 medium.
- v2 wins B=4 sparse.
- v5 wins B=4 medium by a wide margin.

The biggest practical result is that fast-mac beats Taichi by one to two orders
of magnitude in the 4K/64K key case, while v5 is now clearly the forward and
B=4 medium training path to keep developing.
