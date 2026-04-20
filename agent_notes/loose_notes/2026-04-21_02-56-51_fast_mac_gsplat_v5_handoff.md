# fast-mac-gsplat v5 Handoff

## Context

The scientist handed off `torch_metal_gsplat_v5.tar.gz` as a source bundle for
the next Torch+Metal rasterizer attempt. The major claimed change was not just a
new single-image kernel, but the API shape needed by the trainer:

- `[B,G,2/3] -> [B,H,W,3]` batchwise rendering.
- eval forward can skip sorted-ID writeback.
- train forward keeps saved sorted IDs and stop counts for backward.
- `batch_strategy=auto|flatten|serial`.
- benchmark matrix for batch sizes, strategies, shuffling, seeds, and overflow.

The goal of this work chunk was to unpack it into `fast-mac-gsplat`, build it
on the real Mac stack, add it to the existing v2/v3 benchmark comparison, and
collect enough numbers to know whether v5 is an immediate default or a branch
to keep developing.

## What Changed

In `third_party/fast-mac-gsplat`:

- Extracted the handoff into `variants/v5/`.
- Preserved the original tarball at
  `source_artifacts/torch_metal_gsplat_v5.tar.gz`.
- Added v5 to `benchmarks/compare_v2_v3.py`.
- Added median timing to the comparison script because MPS timings had visible
  outliers even after warmup.
- Added v5 tile/profile stats to the comparison script.
- Fixed v5 `tests/reference_check.py` and `benchmarks/benchmark_mps.py` import
  paths so they run directly from the v5 directory.
- Fixed v5 standalone benchmark conic generation from `1 / sigma` to
  `1 / sigma^2`.
- Wrote `docs/v5_field_report.md`.
- Updated `README.md` with v5 build commands and measured numbers.

## Build And Correctness

Command:

```bash
cd third_party/fast-mac-gsplat/variants/v5
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python setup.py build_ext --inplace
```

Result:

- Build succeeded.
- Initial compiler warning: unused local `meta` in `gsplat_metal.mm`.
- No link/import failure.

Reference:

```bash
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python tests/reference_check.py
```

Result:

- B=1 and B=2 both passed.
- Image max error was about `5.96e-08`.
- Grad max errors were around `1e-10` to `2e-09` for means, conics, colors, and
  opacities.

This means the v5 handoff was not just theoretical source. It built on the Mac
and matched the CPU reference on small scalar and batched cases.

## Benchmark Findings

B=1, 4096x4096, 65,536 splats:

| Case | v2 forward | v3 forward | v5 forward | Winner |
| --- | ---: | ---: | ---: | --- |
| sigma 1-5 | `15.643 ms` | `13.675 ms` | `20.110 ms` | v3 |
| sigma 3-8 | `23.758 ms` | `13.350 ms` | `11.322 ms` | v5 |

| Case | v2 fwd+bwd | v3 fwd+bwd | v5 fwd+bwd | Winner |
| --- | ---: | ---: | ---: | --- |
| sigma 1-5 | `72.585 ms` | `48.684 ms` | `61.780 ms` | v3 |
| sigma 3-8 | `135.596 ms` | `60.360 ms` | `73.231 ms` | v3 |

v5 is therefore not a blanket replacement for v3. Its eval-forward path can win
in the 4K medium-sigma case, likely because it avoids training writeback, but
the training/backward path still loses to v3 for B=1.

Native v5 batch probe, medium sigma 3-8:

- 4096x4096 B=1 forward: `16.691 ms`.
- 4096x4096 B=4 forward: `32.576 ms` total, or `8.144 ms/frame`.
- 4096x4096 B=1 fwd+bwd: `64.997 ms`.
- 4096x4096 B=4 fwd+bwd flatten: `274.373 ms` total.
- 1024x1024 B=1 forward: `13.461 ms`.
- 1024x1024 B=4 forward: `21.740 ms` total, or `5.435 ms/frame`.
- 1024x1024 B=1 fwd+bwd: `28.111 ms`.
- 1024x1024 B=4 fwd+bwd: `99.180 ms` total, or `24.795 ms/frame`.

The batch feature is already useful for forward throughput. It is much less
decisive for backward, where global gradient writes and replay bandwidth still
dominate.

Tile ablation at 1024x1024, 65,536 splats, B=1, medium sigma, fwd+bwd:

- tile 8 / chunk 32 / cap 1024: `31.853 ms`.
- tile 16 / chunk 64 / cap 2048: `28.259 ms`.
- tile 32 / chunk 128 / cap 2048: `52.407 ms`.

The 16x16 default is still the best of the tried options.

## Things That Changed Our Model

The scientist's direction was correct at the API level: native batch is the
right shape for training-loop integration. But the performance outcome is more
specific than "v5 is faster." The measured split is:

- v2 remains the low-overhead tiny-scene path.
- v3 remains the best measured B=1 high-res training/backward path.
- v5 is the batched/eval branch and already shows strong batched forward
  throughput.

The v5 benchmark conic bug was also important. A benchmark can use the same
sigma labels and still describe a different workload if it passes `1 / sigma`
where the kernel expects `1 / sigma^2`.

## Open Questions

- Is v5's B=1 backward slower because of batch-general indexing, extra
  stop-count bookkeeping, or compiler specialization loss?
- Would a B=1-specialized v5 compile path recover v3 backward speed while
  keeping the batched API for B>1?
- Is B=4 backward variance due to other GPU load, MPS allocator behavior,
  global atomics, or command scheduling?
- How do the synthetic uniform-splat results transfer to real projected
  Dynaworld Gaussian distributions?

## Next Useful Tests

1. Profile v3 vs v5 B=1 backward in Xcode GPU tools.
2. Add a v5 single-image specialization and ablate it against the current
   batch-general kernel.
3. Re-run B=4 backward with a quiet GPU and more iterations.
4. Feed v5 the actual trainer projected splat distributions, not only random
   uniform synthetic scenes.
