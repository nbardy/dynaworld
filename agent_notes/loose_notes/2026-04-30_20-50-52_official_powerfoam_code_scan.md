# Official PowerFoam Code Scan

User provided `https://github.com/theialab/powerfoam` and explicitly said to
clone it. I refreshed a scratch clone at:

```text
/tmp/powerfoam_official
commit 25d6f7b Fix clone command in README
```

I did not vendor the official repo into Dynaworld.

## Key Read

The official code is CUDA/Warp, not Metal. README reference stack is Linux,
CUDA 12.x, `torch==2.9.1+cu128`, and `warp-lang==1.10.0`.

The rasterizer is all in `powerfoam/rasterize.py`. It is not the dense
pixel-times-cell loop that our first Metal prototype implements. Official
forward uses:

- sphere/tile visibility counting
- prefix sum into tile offsets
- per-tile primitive emission
- packed sort key: tile id plus power distance
- per-tile sorted primitive lists
- one thread per pixel in 8x8 tiles
- neighbor power-face clipping inside the pixel loop
- oriented dipole/detail-plane clipping and soft Voronoi texture lookup

For generic/non-pinhole cameras, the official rasterizer builds a cone hierarchy
over rays before visibility counting. For pinhole cameras, it projects each
sphere to an oriented screen-space bounding box.

## Backward

The paper does not isolate backward performance. The official code does contain
custom backward:

- `RasterGradFn` wraps `Rasterizer._forward/_backward`.
- `_forward` saves spheres, normals/densities, texel buffers, adjacency,
  `adjacency_diff`, sorted tile primitive lists, offsets, early-stop counters,
  `color_out`, and `log_t_out`.
- `_backward` launches `backward_kernel`, one thread per pixel, traversing the
  saved per-tile primitive list in reverse.
- It replays clipping to identify whether each interval endpoint came from the
  sphere, a neighboring power face, or the dipole plane.
- It propagates alpha/transmittance gradients and routes endpoint gradients to
  sphere parameters, neighbor sphere parameters, normal/density, and texture.

The helper derivative functions are in `powerfoam/rendering_math.py`:

- `ray_sphere_intersect_bwd`
- `ray_pface_intersect_bwd`
- `ray_plane_intersect_bwd`

The detail-site texture backward lives in `powerfoam/texture.py`.

## Benchmark Caveat

Official `benchmark.py` measures forward FPS for rasterize/raytrace. It does
not expose backward timing. Training uses NVTX ranges for `Forward`, `Losses`,
`Backward`, and `Optimizer Step`, so a backward timing harness should wrap
`model.forward(...); loss.backward()` or use profiler/NVTX around the existing
training loop.

## Implication For Our Metal Port

The relevant next step is not FasterGS per-Gaussian backward first. It is:

1. Port PowerFoam tile count/write/sort and per-tile sorted lists to Metal.
2. Add a feature-foam backward matching our current simpler output contract:
   `(features, alpha)`.
3. Add density and feature gradients first.
4. Add position/radius endpoint gradients through sphere and power-face
   derivatives.
5. Only then add official dipole/detail-site texture and spherical-Voronoi
   appearance.

FasterGS remains a useful later optimization template, but foam endpoint
ownership scatters gradients to adjacent cells, so the official reverse tile
replay is the safer first implementation target.
