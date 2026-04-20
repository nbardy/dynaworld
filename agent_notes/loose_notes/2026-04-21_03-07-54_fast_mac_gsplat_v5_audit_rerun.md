# fast-mac-gsplat v5 Audit Rerun

## Context

The user asked to continue, make sure benchmarks were run, and audit the v5
code rather than just accepting the source handoff. This was a second pass after
v5 had already been extracted, built, benchmarked once, documented, committed,
and pushed.

## Code Audit Findings

The v5 wrapper/bridge/kernel shape looked broadly consistent with the earlier
reference results:

- Torch `[G,3]` arrays are treated as packed flat `float*`, not Metal `float3*`.
- v5 sorts depths per batch in Torch, then flattens `[B,G,...]`.
- tile IDs include the batch offset, so per-tile bins are isolated by batch.
- overflow tiles are rendered by a separate slow path and scattered back into
  only the valid tile region.
- edge-tile invalid pixels can exist in the temporary overflow tile image, but
  Python only scatters/gathers the valid sub-rectangle.

Two concrete hardening issues were fixed:

1. The training path raised on tile overflow when `enable_overflow_fallback` was
   disabled, but the eval path did not. The eval path now raises the same
   `Tile overflow detected...` error instead of returning a fast-cap render.
2. Python input validation now requires all five tensors
   `means2d/conics/colors/opacities/depths` to be same-device MPS float32. This
   makes mixed CPU/MPS or dtype mistakes explicit before the wrapper reaches
   Torch gather or the C++ backend.

`tests/reference_check.py` now has a regression check for the eval overflow
disabled behavior.

## Validation Rerun

Command:

```bash
cd third_party/fast-mac-gsplat/variants/v5
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python tests/reference_check.py
```

Result:

- B=1 image max error: `5.960464477539063e-08`
- B=1 means grad max error: `2.4010660126805305e-10`
- B=1 conics grad max error: `9.313225746154785e-10`
- B=1 colors grad max error: `9.313225746154785e-10`
- B=1 opacities grad max error: `1.862645149230957e-09`
- B=2 image max error: `5.960464477539063e-08`
- B=2 means grad max error: `3.710738383233547e-10`
- B=2 conics grad max error: `1.862645149230957e-09`
- B=2 colors grad max error: `9.313225746154785e-10`
- B=2 opacities grad max error: `1.862645149230957e-09`
- eval overflow disabled raises: `ok`

## Benchmark Rerun And Contamination

The benchmark scripts were rerun, but the machine was not clean:

- A Taichi training run was active:
  `src/train/dynamicTokenGS.py ... taichi_8192splats.jsonc`
- Several Python 3.14 geometry/codec benchmark jobs were running at around
  `80%` CPU each.
- Load average was around `21`.
- Xcode/GPU tooling services were also resident.

Because of that, the later timing reruns showed large outliers. I did not treat
them as replacement headline numbers.

Contaminated rerun examples:

4096x4096, 65,536 splats, forward-only, warmup 4, iters 7:

- sparse sigma 1-5:
  - v2 mean `54.965 ms`, median `45.838 ms`, min `24.400 ms`, max `117.852 ms`
  - v3 mean `36.474 ms`, median `18.749 ms`, min `17.511 ms`, max `79.491 ms`
  - v5 mean `125.387 ms`, median `58.025 ms`, min `20.562 ms`, max `402.143 ms`
- medium sigma 3-8:
  - v2 mean `165.940 ms`, median `47.131 ms`, min `27.355 ms`, max `890.695 ms`
  - v3 mean `165.513 ms`, median `82.820 ms`, min `20.399 ms`, max `530.583 ms`
  - v5 mean `128.552 ms`, median `26.903 ms`, min `22.903 ms`, max `389.986 ms`

1024x1024, 65,536 splats, forward+backward, warmup 2, iters 5:

- sparse sigma 1-5:
  - v2 mean `133.554 ms`, median `72.692 ms`, min `54.164 ms`, max `397.357 ms`
  - v3 mean `66.879 ms`, median `65.993 ms`, min `51.782 ms`, max `83.843 ms`
  - v5 mean `121.720 ms`, median `66.145 ms`, min `37.741 ms`, max `372.548 ms`
- medium sigma 3-8:
  - v2 mean `350.646 ms`, median `206.459 ms`, min `155.063 ms`, max `687.570 ms`
  - v3 mean `222.882 ms`, median `78.898 ms`, min `47.953 ms`, max `493.457 ms`
  - v5 mean `277.677 ms`, median `271.714 ms`, min `69.024 ms`, max `572.134 ms`

Native v5 B=4, 1024x1024, 65,536 splats, medium sigma 3-8,
`batch_strategy=flatten`, forward+backward:

- mean `464.443 ms`
- median `278.107 ms`
- forward mean `208.182 ms`
- backward mean `256.260 ms`
- profile total pairs `2,107,600`
- max pairs per tile `177`
- overflow tiles `0`

These numbers are useful as a smoke that the benchmarks exercise the path under
load, but not as fair speed data. The earlier quieter run remains the better
headline for README/report tables.

## Updated Belief

The code is more robust after the audit, and the benchmark harnesses do run.
The fair benchmark question is not fully closed because the current machine is
actively contended. The next clean timing pass should be run after stopping the
Taichi training job and the CPU-heavy codec probes, then using median/min/max
rather than mean alone.
