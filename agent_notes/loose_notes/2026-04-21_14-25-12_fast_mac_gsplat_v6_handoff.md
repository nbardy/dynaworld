# fast-mac-gsplat v6 Handoff

## Context

The scientist handed off `torch_metal_gsplat_v6.tar.gz` with batch-first active
tile scheduling, optional count-sorted active tiles, no unconditional grad clone,
adaptive stop counts, overflow fallback, and Torch-side stable depth sort. The
task was to add it to the repo, build it on the Mac, wire it into benchmarks,
and collect first evidence.

## What Changed

- extracted the bundle to `third_party/fast-mac-gsplat/variants/v6/`
- saved the original tarball at
  `third_party/fast-mac-gsplat/source_artifacts/torch_metal_gsplat_v6.tar.gz`
- patched v6 direct script imports so `tests/reference_check.py` and
  `benchmarks/benchmark_mps.py` work when run by path
- built the extension successfully with the Dynaworld `.venv`
- added v6 to fast-mac's `benchmarks/compare_v2_v3.py`
- added v6 to Dynaworld's `src/benchmarks/mac_renderer_stack_compare.py`
- added benchmark flags for v6 active-tile ablations:
  `--active-tiles`, `--no-active-tiles`, `--sort-active-tiles`,
  `--no-sort-active-tiles`
- extended v6 matrix benchmark to sweep active-tile and active-sort modes
- wrote `third_party/fast-mac-gsplat/docs/v6_field_report.md`

## Build And Correctness

Build command:

```bash
cd third_party/fast-mac-gsplat/variants/v6
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python setup.py build_ext --inplace
```

Build passed. The ObjC++ bridge emitted two unused-variable warnings for local
`meta` variables, but no compile failure.

Reference command:

```bash
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python tests/reference_check.py
```

Reference passed:

```text
B=1 image max error: 5.960464477539063e-08
B=1 means grad max error: 2.4010660126805305e-10
B=1 conics grad max error: 9.313225746154785e-10
B=1 colors grad max error: 9.313225746154785e-10
B=1 opacities grad max error: 1.862645149230957e-09
B=2 image max error: 5.960464477539063e-08
B=2 means grad max error: 3.710738383233547e-10
B=2 conics grad max error: 1.862645149230957e-09
B=2 colors grad max error: 9.313225746154785e-10
B=2 opacities grad max error: 1.862645149230957e-09
```

## Benchmark Evidence

At `1024x512`, `G=2048`, `B=1`, `warmup=1`, `iters=5`, v6 is not a forward
winner:

```text
Forward sparse:  v2 4.012 ms, v3 4.669 ms, v5 4.066 ms, v6 5.579 ms
Forward medium:  v2 4.431 ms, v3 5.021 ms, v5 4.685 ms, v6 7.737 ms
```

Forward+backward:

```text
Sparse: v2 12.905 ms, v3 7.728 ms, v5 11.355 ms, v6 8.562 ms
Medium: v2 6.946 ms, v3 5.634 ms, v5 6.339 ms, v6 7.979 ms
```

At `4096x4096`, `G=65536` per image, `B=4`, medium sigma, forward+backward:

```text
v5: mean 248.405 ms, median 246.670 ms, fwd 38.699 ms, bwd 209.707 ms
v6 default: mean 354.122 ms, median 309.023 ms, fwd 116.762 ms, bwd 237.361 ms
v6 best ablation: mean 301.263 ms, median 301.098 ms, fwd 111.459 ms, bwd 189.804 ms
```

The best v6 ablation disabled active compaction and active sorting. That is an
important negative result: the new scheduling layer does not pay for itself on
the current uniform synthetic case.

At `4096x4096`, fixed total `65536` splats across `B=4` (`G=16384` per image):

```text
Sparse forward+backward: v5 168.412 ms, v6 223.853 ms
Medium forward+backward: v5 194.507 ms, v6 238.707 ms
```

## Current Belief

v6 is correct and useful as a research branch, but v5 should remain the
recommended renderer for current workloads. v6's backward can be modestly faster
in the best ablation, but forward is about 3x slower at the big B=4 4K shape,
and the total loses.

The likely reason is workload mismatch. The tested scenes have nearly all tiles
active, no overflow, no dense active tiles, and stop ratios near 1.0. Active tile
compaction, count sorting, and adaptive stop counts need skewed scenes or
pathological occupancy to win; uniform splat distributions mostly pay overhead.

## Follow-Up Tests

- clustered scene where many screen tiles are empty
- overflow stress scene where count scheduling might reduce tail latency
- v6 forward kernel profile against v5 to find the 3x forward regression
- possible v6.1 direction: keep the faster backward idea but restore a v5-like
  forward path for normal tiles
