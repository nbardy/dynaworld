# fast-mac-gsplat v3 handoff

## Context

The chief scientist provided a v3 Torch+Metal gsplat bundle intended to improve
the previous validated fastpath. The requested work was to extract it into a new
subfolder under `third_party/fast-mac-gsplat`, write a field note explaining
what we fixed/learned from the previous shader, add benchmark coverage, and
compare speed.

## What changed

- Extracted `/Users/nicholasbardy/Downloads/torch_metal_gsplat_v3.tar.gz` into
  `third_party/fast-mac-gsplat/variants/v3/`.
- Preserved the source tarball at
  `third_party/fast-mac-gsplat/source_artifacts/torch_metal_gsplat_v3.tar.gz`.
- Patched `variants/v3/setup.py` so PyTorch extension sources are relative to
  the v3 package root; the original absolute source paths failed
  `build_ext --inplace`.
- Added `benchmarks/compare_v2_v3.py` to run the validated v2 fastpath and v3
  candidate side by side on the same projected MPS tensors.
- Added `docs/chief_scientist_field_report.md` inside the submodule.
- Updated the submodule README and v3 engineering notes with local validation
  and benchmark results.

## Validation

v3 build and reference:

```text
cd third_party/fast-mac-gsplat/variants/v3
uv run python setup.py build_ext --inplace
uv run python tests/reference_check.py
```

Reference result:

```text
image max error: 5.960464477539063e-08
means grad max error: 2.4010660126805305e-10
conics grad max error: 9.313225746154785e-10
colors grad max error: 9.313225746154785e-10
opacities grad max error: 1.862645149230957e-09
```

Small v2/v3 image parity check at 128x128 / 512 splats returned max diff `0.0`.

4K / 64K forward comparison:

```text
height=4096 width=4096 gaussians=65536 warmup=1 iters=3 backward=False

case=sparse_sigma_1_5 sigma=[1.0, 5.0]
v2_fastpath    forward          mean_ms=   13.563 min_ms=   12.342 max_ms=   15.024 vs_v2= 1.000x
v3_candidate   forward          mean_ms=    8.805 min_ms=    7.300 max_ms=   10.252 vs_v2= 0.649x

case=medium_sigma_3_8 sigma=[3.0, 8.0]
v2_fastpath    forward          mean_ms=   19.964 min_ms=   17.032 max_ms=   22.548 vs_v2= 1.000x
v3_candidate   forward          mean_ms=   13.876 min_ms=   13.047 max_ms=   14.946 vs_v2= 0.695x
```

4K / 64K forward+backward comparison:

```text
height=4096 width=4096 gaussians=65536 warmup=1 iters=3 backward=True

case=sparse_sigma_1_5 sigma=[1.0, 5.0]
v2_fastpath    forward_backward mean_ms=   74.045 min_ms=   71.874 max_ms=   75.522 vs_v2= 1.000x
v3_candidate   forward_backward mean_ms=   57.253 min_ms=   57.180 max_ms=   57.306 vs_v2= 0.773x

case=medium_sigma_3_8 sigma=[3.0, 8.0]
v2_fastpath    forward_backward mean_ms=  137.559 min_ms=  136.339 max_ms=  139.467 vs_v2= 1.000x
v3_candidate   forward_backward mean_ms=   68.198 min_ms=   67.509 max_ms=   69.444 vs_v2= 0.496x
```

## Current model

The previous shader's biggest correctness issue was `float3*` indexing over
tightly packed Torch `[G,3]` tensors. The previous performance issue that was
easy to fix was duplicate tile-local sorting in backward; saving the forward
sorted order helped a lot.

The remaining backward gap is probably dominated by recomputing alpha chains,
reverse scanning, repeated parameter loads, and global atomic gradient writes.
v3 supports that model: it is faster despite still sorting tile IDs again in
fast backward, because it stages parameters in threadgroup memory and reduces
per-Gaussian gradients inside the tile before global atomics.

## Open questions

- Port v2's saved-sorted-order backward optimization into v3 and measure how
  much the remaining local sort matters.
- Add benchmark counters for max tile count, mean tile count, total tile pairs,
  and overflow tile count.
- Stress-test v3 overflow fallback with deliberately clustered splats.
- Benchmark compiled 8x8 / 16x16 / 32x32 variants only after the occupancy
  counters justify changing tile shape.
