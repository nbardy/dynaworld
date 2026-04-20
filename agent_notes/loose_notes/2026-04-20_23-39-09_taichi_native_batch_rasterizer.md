# Taichi Native Batch Rasterizer

Implemented a side-by-side native batch path in the Taichi fork instead of using
an atlas.

Submodule changes:

- Added `map_to_tiles_batch(...)` with keys sorted by flattened
  `(batch_id, tile_id, depth)` and tile ranges shaped as
  `[B, tiles_y, tiles_x, 2]`.
- Added `forward_batch_kernel(...)` and `backward_batch_kernel(...)` for the
  `metal_reference` / `pixel_reference` path. These kernels write/read
  `[B, H, W, C]` images and `[B, H, W]` alpha directly.
- Added `rasterize_batch(...)` and `rasterize_with_tiles_batch(...)` to the
  public rasterizer API.
- Added docs in the Taichi README and NOTES.
- Added a CPU regression test comparing `rasterize_batch` against a loop over
  single-image `rasterize`.

Dynaworld adapter changes:

- Added batched Taichi projection/packing in `src/train/renderers/taichi.py`.
- Updated `render_gaussian_frames(..., mode="taichi")` to call the native
  batched Taichi renderer instead of looping through single-frame slices.

Checks run:

```text
compileall: passed
git diff --check inside third_party/taichi-splatting: passed
CPU batch-vs-loop forward image max: 0.0
CPU batch-vs-loop alpha max: 0.0
CPU packed grad max diff: 3.725290298461914e-09
CPU feature grad max diff: 0.0
MPS/Metal batch image shape: (3, 32, 32, 3)
MPS/Metal batch alpha shape: (3, 32, 32)
MPS/Metal packed/features grads finite: true
Dynaworld render_taichi_3dgs_batch smoke output: (2, 3, 32, 32), finite output and finite xyz/rgb grads
```

`pytest` is not installed in this `uv` env, so the test file was not executed
through pytest here. The equivalent direct Python check was run.

Surprise:

The old single-image Taichi loop comparison can trip an MPS autograd
version-counter error when it passes `packed[b]` slice views through the Taichi
autograd function. The native batch path completed backward cleanly and avoids
that view-slicing call pattern in the training loop.
