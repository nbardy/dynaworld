# PowerFoam Tiled Kernel Plan

## Context

The user asked for a concrete new-kernel plan, specifically avoiding
`instance x W x H` backward memory blowups. They also clarified that we can
write a new shader instead of fixing the old dense baseline.

## What Changed

- Added `research_notes/foam_papers/powerfoam_metal_tiled_kernel_plan.md`.
- Added `third_party/powerfoam-metal/csrc/metal/powerfoam_tiled_kernels.metal`.
- Added `third_party/powerfoam-metal/csrc/metal/powerfoam_streaming_kernels.metal`
  after the user pushed back that tiled Metal has often been slower locally.
- Kept the existing dense PowerFoam shader as a reference path rather than the
  target architecture.

## Technical Direction

The official-style tiled path mirrors PowerFoam:

1. Count visible `(tile, cell)` intersections.
2. Prefix-sum to tile offsets.
3. Write tile-cell ids plus sort keys.
4. Sort by packed tile id and power distance.
5. Render one 8x8 tile per threadgroup.
6. Save `tile_offsets`, `tile_cell_ids`, `tile_stop_counts`, and per-pixel
   `log_t`.
7. Backward replays the same tile list in reverse and atomically scatters
   gradients.

This is memory-safe because it stores visible tile-cell intersections, not
visible pixel-cell intersections.

But the first implementation should probably be the non-tiled streaming path:
one thread per pixel, global sorted cell order, optional per-cell screen bounds,
save `log_t` plus `pixel_stop`, and reverse replay from `pixel_stop - 1`. This
also avoids `N*H*W` memory and skips the tile count/write/sort/barrier overhead
that hurt earlier Metal rasterizers.

## Important Detail

Power-face endpoint gradients are not cleanly owned by one cell: when a face
between cell `i` and neighbor `j` clips the interval, the endpoint derivative
scatters to both cells. That makes a literal cell-owned "per-splat backward"
less natural than GS. The first exact path should be pixel-owned reverse replay
with global atomics; if atomics dominate, the next optimization is tile-cell
partial gradients reduced from `M -> N`.

## Validation Pending

The new shader file compiles independently:

```bash
xcrun -sdk macosx metal -c third_party/powerfoam-metal/csrc/metal/powerfoam_tiled_kernels.metal -o /tmp/powerfoam_tiled_kernels.air
```

The existing dense reference check still passes after the earlier power-face
sign fix:

```text
features max error: 5.8710575103759766e-06
alpha max error: 7.867813110351562e-06
powerfoam Metal reference check passed
```

Next step: add a C++ launcher under a new op name.
