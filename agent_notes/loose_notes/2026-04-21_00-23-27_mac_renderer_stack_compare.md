# Mac Renderer Stack Compare

## Context

The Taichi engineer reported large native-batch speedups against direct Torch:

- `B=4`, 64x64, 128 splats: `7.6x` forward, `16.9x` forward+backward.
- `B=4`, 128x128, 128 splats: `30.8x` forward, `36.2x` forward+backward.

We needed the same table with four paths side by side:

- direct Torch projected 2D reference,
- Taichi native Metal batch (`rasterize_batch` when `B > 1`),
- `fast-mac-gsplat` v2 pure Metal looped over frames,
- `fast-mac-gsplat` v3 pure Metal looped over frames.

The new runner is:

```bash
uv run python src/benchmarks/mac_renderer_stack_compare.py ...
```

It generates one circular packed-2D projected Gaussian workload and feeds that
same workload to all renderers. When direct Torch is enabled, output diffs were
~`1e-7` to `5e-7`, so these are comparing the same math to float noise.

## Caveats

- The machine had unrelated CPU-heavy Python jobs active, so tiny launch-heavy
  timings have host-side noise.
- v2/v3 do not have native batch APIs yet. The runner loops over `B` frames for
  v2/v3 and reports both total and per-frame time. Taichi uses native batch for
  `B > 1`.
- The direct Torch reference here is a looped MPS reference over images/splats,
  not necessarily the same direct Torch baseline used in the Taichi README. This
  matters a lot for speedup factors.
- Large `1024x1024/G=65536` direct Torch was intentionally skipped.

## Taichi README Batch Scale

Commands:

```bash
uv run python src/benchmarks/mac_renderer_stack_compare.py --height 64 --width 64 --gaussians 128 --batch-size 4 --warmup 2 --iters 5
uv run python src/benchmarks/mac_renderer_stack_compare.py --height 64 --width 64 --gaussians 128 --batch-size 4 --warmup 2 --iters 5 --backward
uv run python src/benchmarks/mac_renderer_stack_compare.py --height 128 --width 128 --gaussians 128 --batch-size 4 --warmup 2 --iters 5
uv run python src/benchmarks/mac_renderer_stack_compare.py --height 128 --width 128 --gaussians 128 --batch-size 4 --warmup 2 --iters 5 --backward
```

CSV outputs:

- `benchmark_outputs/mac_renderer_stack_64_b4_g128_forward.csv`
- `benchmark_outputs/mac_renderer_stack_64_b4_g128_backward.csv`
- `benchmark_outputs/mac_renderer_stack_128_b4_g128_forward.csv`
- `benchmark_outputs/mac_renderer_stack_128_b4_g128_backward.csv`

### 64x64, B=4, G=128

| Case | Mode | Torch | Taichi | v2 | v3 |
| --- | --- | ---: | ---: | ---: | ---: |
| sigma 1-5 | forward | 126.618 ms | 14.759 ms | 12.861 ms | 20.065 ms |
| sigma 3-8 | forward | 145.510 ms | 17.993 ms | 15.916 ms | 27.048 ms |
| sigma 1-5 | fwd+bwd | 685.026 ms | 35.858 ms | 21.535 ms | 26.991 ms |
| sigma 3-8 | fwd+bwd | 593.915 ms | 36.881 ms | 33.096 ms | 51.422 ms |

Interpretation:

- Taichi speedups vs this direct Torch reference are `8.1-8.6x` forward and
  `16.1-19.1x` forward+backward, which is close to the README claim.
- v2 is fastest at this low-res, low-splat batch scale. v3 has more overhead
  than v2 and does not pay it back yet.

### 128x128, B=4, G=128

Initial run:

| Case | Mode | Torch | Taichi | v2 | v3 |
| --- | --- | ---: | ---: | ---: | ---: |
| sigma 1-5 | forward | 158.351 ms | 24.516 ms | 20.270 ms | 36.331 ms |
| sigma 3-8 | forward | 156.158 ms | 19.900 ms | 17.110 ms | 26.920 ms |
| sigma 1-5 | fwd+bwd | 702.222 ms | 47.335 ms | 24.254 ms | 28.558 ms |
| sigma 3-8 | fwd+bwd | 717.170 ms | 66.641 ms | 28.428 ms | 39.775 ms |

Forward rerun with more warmup/iters:

| Case | Mode | Torch | Taichi | v2 | v3 |
| --- | --- | ---: | ---: | ---: | ---: |
| sigma 1-5 | forward | 83.627 ms | 11.961 ms | 11.050 ms | 16.823 ms |
| sigma 3-8 | forward | 97.898 ms | 16.517 ms | 11.517 ms | 16.688 ms |

Interpretation:

- The Taichi absolute times are in the same broad range as the README
  (`21.665 ms` forward, `37.359 ms` fwd+bwd), but the direct Torch baseline here
  is much faster than the README baseline (`667.194 ms` forward,
  `1350.545 ms` fwd+bwd).
- The discrepancy is mostly the Torch baseline implementation, not a failure of
  Taichi. Absolute time is the safer comparison.

## Fast-Mac Small Single-Frame Scale

Commands:

```bash
uv run python src/benchmarks/mac_renderer_stack_compare.py --height 128 --width 128 --gaussians 512 --batch-size 1 --warmup 2 --iters 5
uv run python src/benchmarks/mac_renderer_stack_compare.py --height 128 --width 128 --gaussians 512 --batch-size 1 --warmup 2 --iters 5 --backward
```

CSV outputs:

- `benchmark_outputs/mac_renderer_stack_128_b1_g512_forward.csv`
- `benchmark_outputs/mac_renderer_stack_128_b1_g512_backward.csv`

| Case | Mode | Torch | Taichi | v2 | v3 |
| --- | --- | ---: | ---: | ---: | ---: |
| sigma 1-5 | forward | 107.720 ms | 18.254 ms | 4.007 ms | 5.099 ms |
| sigma 3-8 | forward | 113.577 ms | 14.365 ms | 3.680 ms | 5.209 ms |
| sigma 1-5 | fwd+bwd | 612.363 ms | 44.268 ms | 8.769 ms | 8.578 ms |
| sigma 3-8 | fwd+bwd | 636.652 ms | 43.138 ms | 6.236 ms | 6.858 ms |

Interpretation:

- Taichi is much better than direct Torch.
- v2/v3 are several times faster than Taichi at this single-frame 128px scale.
- v2 is still the better forward-only low-overhead path. v3 is roughly tied or
  slightly better only once backward is included.

## Large No-Torch Stress

Commands:

```bash
uv run python src/benchmarks/mac_renderer_stack_compare.py --height 1024 --width 1024 --gaussians 65536 --batch-size 1 --warmup 3 --iters 5 --no-include-torch --no-check-outputs
uv run python src/benchmarks/mac_renderer_stack_compare.py --height 1024 --width 1024 --gaussians 65536 --batch-size 1 --warmup 2 --iters 3 --backward --no-include-torch --no-check-outputs
```

CSV outputs:

- `benchmark_outputs/mac_renderer_stack_1024_b1_g65536_forward_rerun.csv`
- `benchmark_outputs/mac_renderer_stack_1024_b1_g65536_backward.csv`

| Case | Mode | Taichi | v2 | v3 |
| --- | --- | ---: | ---: | ---: |
| sigma 1-5 | forward | 54.323 ms | 14.410 ms | 9.717 ms |
| sigma 3-8 | forward | 63.798 ms | 16.919 ms | 18.117 ms |
| sigma 1-5 | fwd+bwd | 352.500 ms | 41.148 ms | 20.238 ms |
| sigma 3-8 | fwd+bwd | 1081.152 ms | 119.657 ms | 39.510 ms |

Interpretation:

- v3 is the real large-scene training path. The backward recompute/tile-order
  changes become decisive at 65k splats.
- Taichi remains valuable for compatibility and batch integration, but its
  reference Metal backward is much slower at high overlap.
- The v2/v3 crossover depends on splat footprint: v3 wins sparse 1-5px forward,
  ties/loses medium 3-8px forward here, and clearly wins backward in both.

## Current Model

Renderer choice should be staged:

- Use Taichi when Taichi compatibility or native batched autograd integration is
  the goal.
- Use v2 for early low-res/low-splat bootstrap and forward-heavy smoke tests.
- Use v3 for large splat counts, larger footprints, and training workloads where
  backward dominates.

The biggest benchmarking lesson is to stop reporting only speedup factors. The
same Taichi absolute timing can look like `7x` or `30x` depending on which direct
Torch reference is used.
