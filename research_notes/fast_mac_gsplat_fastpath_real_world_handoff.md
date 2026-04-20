# Fast Mac gsplat fastpath real-world handoff

Date: 2026-04-20

## Summary

The second Torch/Metal handoff in `submodules/fast-mac-gsplat-fastpath/` is a real fastpath, not a scaffold. After small packaging/schema fixes and one important Metal layout correctness fix, it builds locally, registers a Torch custom op, renders on MPS, and produces backward gradients.

This is best understood as a projected 2D differentiable Gaussian rasterizer for Apple Silicon:

- Torch owns the training-loop interface.
- Torch MPS handles global depth sort via `argsort`.
- Torch MPS handles tile prefix offsets via `cumsum`.
- Metal handles tile counting, bin emission, local per-tile sort, forward rasterization, and backward rasterization.

That split is pragmatic. It is not pure Metal end-to-end, but it keeps the fragile global primitives in Torch and puts the hot per-tile raster work in Metal.

## What was fixed

The original C++ op schemas declared fixed tuple returns. PyTorch requires fixed C++ tuple types for those schemas; returning `std::vector<torch::Tensor>` registers as `Tensor[]` and aborts at extension import. The binding now returns fixed `std::tuple<...>` values.

The biggest correctness bug was Metal `float3` layout. Torch `[G, 3]` float tensors are tightly packed with a 12-byte row stride. Metal `float3` has 16-byte alignment/stride. Reading conics/colors as `device float3*` made splat 0 work and splats 1+ read corrupted values. The kernels now read conics/colors as flat `device float*` and manually load `index * 3 + channel`.

The lower x/y SnugBox bounds also had a suspicious `+1` offset. Removing it was not the main multi-splat fix, but made the tile bounds match the direct CPU reference more naturally.

## Validation

Local validation after fixes:

- `python setup.py build_ext --inplace` succeeded.
- `import torch_gsplat_bridge_fast` registered `torch.ops.gsplat_metal_fast`.
- Tiny MPS forward/backward smoke worked.
- 16x16 / 4-splat CPU reference check matched forward/backward at about `1e-8` absolute error.

Small correctness check max errors:

- image: `5.74e-08`
- means grad: `4.29e-10`
- conics grad: `1.05e-08`
- colors grad: `3.01e-09`
- opacities grad: `1.02e-08`

Large synthetic projected smoke:

- 4096x4096, 65,536 projected splats
- forward-only warmed mean: about `13.3ms`
- one forward+backward smoke: about `9.9ms` forward, `99.6ms` backward, `109.5ms` total

Those large timings used already-projected 2D splats with small random radii. They should not be sold as full 3D scene-quality benchmarks.

## Why backward is much slower than forward

The backward kernel is doing substantially more work than the forward kernel.

Forward locally sorts a tile, scans splats front-to-back for each pixel, updates transmittance/color, and writes one RGB output per pixel.

Backward locally sorts the same tile again, then recomputes the forward alpha/transmittance chain. After that it reverse-scans the contributing splats to propagate gradients. It also emits many atomic adds into shared gradient buffers:

- 3 atomics for colors
- 3 atomics for conics
- 2 atomics for means
- 1 atomic for opacity

That can be up to 9 atomic gradient writes per contributing pixel-splat pair. Forward has no comparable cross-pixel atomic accumulation; it mostly writes one pixel once. In dense or overlapping scenes, atomics and the extra reverse scan are the expected bottlenecks.

The current backward is therefore correctness-first and memory-light, not maximally optimized. It saves compact tile bins rather than huge per-pixel activation volumes, then pays for recomputation and atomics during backward.

Likely backward optimization directions:

- Accumulate per-splat gradients inside threadgroup memory where possible, then issue fewer global atomics.
- Split gradients into separate kernels or reductions if contention dominates.
- Consider storing a small amount of forward state, such as final transmittance or contribution counts, if memory allows.
- Add lower-precision or packed gradient variants only after the f32 path is fully validated.
- Benchmark by overlap regime, not just `G`, because atomics scale with contributing pixel-splat pairs.

## Is 16x16 tile specialization a problem?

Not immediately. A 16x16 tile is a reasonable first target:

- 256 pixels maps cleanly to one threadgroup's pixel work.
- Tile-local splat lists stay smaller than a 32x32 tile.
- Threadgroup memory for the local ID list is bounded and simple.
- This shape is common for tiled rasterizers because it balances binning overhead against per-tile overlap.

It is currently a compile-time specialization, not a normal runtime hyperparameter. The shader defines `GSP_TILE_SIZE 16u` and the Python wrapper rejects non-16 tile sizes. Making tile size configurable means compiling more kernel variants or making the kernels more dynamic.

Potential variants:

- 8x8: more tiles and more binning overhead, but shorter per-tile splat lists. Could help very dense overlap or small-radius splats.
- 16x16: current balanced default.
- 32x32: fewer tiles, but much longer splat lists and too many pixels for the current threadgroup shape. Likely worse unless the kernel is redesigned.

The algorithm is not too narrow, but this implementation is intentionally focused. Treat tile size as a kernel-family design choice first, not a sweepable training hyperparameter yet.

## Current caveats

- Requires MPS tensors and float32.
- CPU tensors are not the intended path.
- Depth ordering is piecewise constant; depth gradients are zero.
- Uses PyTorch internal MPS headers, so PyTorch version pinning matters.
- Specialized to 16x16 tiles.
- Has a hard tile-list cap, currently 4096 splats per tile.
- Does not include a full 3D camera projection frontend.
- Not yet integrated into the Dynaworld benchmark matrix or training loop.

## Real-world assessment

This is promising. Forward speed on synthetic projected 4K/64K splats is excellent for a local Mac path, and the small reference checks show the core math can be correct after the layout fix. The backward path is much slower, but the reason is understandable: recompute plus many global atomics.

The implementation is good enough to keep and develop. It is not yet a finished production renderer, but it is a credible Torch-compatible differentiable Metal rasterizer base.
