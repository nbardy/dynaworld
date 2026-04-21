# fast-mac v7 hardware benchmark integration

## Context

The scientist handed off `/Users/nicholasbardy/Downloads/torch_metal_gsplat_v7_hardware.tar.gz`.
The request was to add it to `third_party/fast-mac-gsplat`, wire it into the
benchmark stack, and add a "full" benchmark runner that can sweep renderers,
resolutions, splat counts, batch sizes, and projected-splat distributions while
writing a Markdown report.

## What changed

- Extracted v7 under `third_party/fast-mac-gsplat/variants/v7`.
- Preserved the original handoff tar under
  `third_party/fast-mac-gsplat/source_artifacts/torch_metal_gsplat_v7_hardware.tar.gz`.
- Added v7 as an optional row to `benchmarks/compare_v2_v3.py` via
  `--include-v7`.
- Added `benchmarks/benchmark_full_matrix.py`, which:
  - runs each benchmark cell in a subprocess,
  - supports `torch_direct`, `v2_fastpath`, `v3_candidate`, `v5_batched`,
    `v6_direct`, `v6_auto`, and `v7_hardware`,
  - supports `microbench_uniform_random`, `sparse_screen`,
    `clustered_hot_tiles`, `layered_depth`, and `overflow_adversarial`,
  - writes a Markdown report plus optional JSONL,
  - records skips/errors/timeouts rather than letting one experimental variant
    poison the full run.
- Documented the full sweep command in the fast-mac README.
- Added `docs/v7_field_report.md` in fast-mac.

## Fixes needed before v7 ran

The source handoff was not directly runnable. Local fixes:

- Added direct-script import path fixes for `tests/reference_check.py` and
  `benchmarks/benchmark_v7.py`.
- Added the missing `if __name__ == "__main__": main()` to the reference check.
- Fixed Metal shader compilation by removing a duplicate `[[position]]`
  fragment declaration.
- Corrected hardware blend order: the Torch wrapper depth-sorts front-to-back,
  but standard hardware source-over blending needs far-to-near draw order to
  reproduce the same equation.
- Expanded quad bounds from the handoff's shrunken/clamped edge behavior to
  `[0, W] x [0, H]` support. The first multi-splat reference mismatch was on
  edge pixels that the dense reference still saw but the hardware quad missed.
- Fixed backward math:
  - `dL/dcolor` is `grad * weight`, not `grad * weight / alpha`.
  - mean gradients had the wrong sign.

## Correctness result

Command:

```bash
cd third_party/fast-mac-gsplat/variants/v7
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python setup.py build_ext --inplace
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python tests/reference_check.py
```

Result:

```text
image max error: 5.960464477539063e-08
means grad max error: 2.473825588822365e-10
conics grad max error: 9.313225746154785e-10
colors grad max error: 9.313225746154785e-10
opacities grad max error: 1.862645149230957e-09
```

This validates the small projected-2D path (`B=1`, `G=4`, `16x16`) against the
dense reference. It does not validate large-scene performance.

## Benchmark smoke

Command:

```bash
cd third_party/fast-mac-gsplat
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python benchmarks/benchmark_full_matrix.py \
  --resolutions 128x128 \
  --splats 64 \
  --batch-sizes 1 \
  --distributions microbench_uniform_random,clustered_hot_tiles \
  --renderers torch_direct,v6_direct,v7_hardware \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 60 \
  --output-md benchmarks/full_rasterizer_benchmark_smoke.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_smoke.jsonl
```

Observed smoke rows:

| Distribution | Mode | Torch | v6 direct | v7 hardware |
|---|---|---:|---:|---:|
| microbench_uniform_random | forward | 207.970 ms | 10.436 ms | 25.654 ms |
| microbench_uniform_random | forward+backward | 712.614 ms | 10.941 ms | 13.248 ms |
| clustered_hot_tiles | forward | 271.543 ms | 71.682 ms | 36.989 ms |
| clustered_hot_tiles | forward+backward | 358.383 ms | 13.973 ms | 15.422 ms |

The smoke report files were deleted after inspection so the commit contains the
reusable runner, not a stale partial result.

## Current model

v7 proves the hardware render-pipeline idea can be made numerically consistent
for a small case and gives a useful alternative benchmark branch. It is not yet
the default fast training renderer.

The biggest architectural problem is not the shader math. The ObjC++ bridge
still copies MPS tensors to CPU/shared buffers, reads the render texture back to
Torch, and does backward through a compute replay with global atomics. That
means v7 may look promising in tiny clustered forward-only cases but still has
large fixed overhead and a backward path that is structurally different from the
v6 direct compute renderer.

The next meaningful v7 experiment is resource plumbing: eliminate CPU round
trips and benchmark again on larger cases. Shader tweaks before that are likely
to be second-order.

## Commit state

fast-mac commit: `8c4679d Add v7 hardware rasterizer benchmarks`

Pushed to: `git@github.com:nbardy/fast-mac-gsplat.git` `main`
