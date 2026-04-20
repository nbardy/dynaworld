# Metal-first projected Gaussian rasterizer (MLX)

This is the first serious Mac backend for the bottlenecked part of your pipeline: the differentiable rasterizer after 3D Gaussians are already projected to 2D conics.

## What is in here

`mlx_projected_gaussian_rasterizer.py`

It implements a Metal-backed MLX custom function with:

1. global depth sort
2. tight opacity-aware `SnugBox` bounds
3. exact tile / ellipse intersection counting and emitting
4. pair sorting by tile id after depth sort
5. per-tile exact front-to-back alpha compositing
6. low-memory backward by recomputing and reverse-scanning alpha
7. atomic gradient accumulation to Gaussian parameters

## Shapes

Inputs:

- `means2d`: `[G, 2]`
- `conics`: `[G, 3]` where each row is `[a, b, c]`
- `colors`: `[G, 3]`
- `opacities`: `[G]`
- `depths`: `[G]`

Outputs:

- image: `[H, W, 3]`

Gradients are returned for:

- `means2d`
- `conics`
- `colors`
- `opacities`

Depth gradients are intentionally zero in this version because the sort order is treated as piecewise constant.

## What is fast about it

The important change is not a tiny math tweak. It is the work representation:

- do **not** materialize dense `[G, H, W]` tensors
- do **not** evaluate Gaussians on pixels they cannot touch
- do **not** save full alpha/transmittance stacks for backward

Instead:

- compute a tight screen-space bbox
- count exact tile intersections
- build a compact tile pair list
- rasterize only those tile-local pairs
- recompute local alpha in backward

This is exactly the direction you want if the current wall is activation memory and full-frame render cost.

## What is exact vs conservative

This version is **exact** in these senses:

- exact front-to-back alpha blending
- exact backward for the rasterizer recurrence given fixed visibility / sort / support
- exact ellipse / tile intersection test using the minimum of the quadratic form over a tile rectangle

This version is **not** differentiating through:

- sort order
- bbox clipping
- support thresholding / pair emission

That is intentional.

## One nuance about AccuTile

Speedy-Splat's AccuTile is a very specialized way to compute exact Gaussian/tile intersections cheaply by exploiting the ellipse geometry row-by-row.

The current implementation here uses an exact convex quadratic box-intersection test instead:

- it is still exact
- it is easier to port across backends
- it is probably a bit slower than the paper's most optimized row-sweep formulation

So think of this as:

- **correct and backend-friendly now**
- **replaceable with the paper's tighter row/column sweep later**

## Why MLX here

MLX gives a much cleaner Metal path on Apple silicon than trying to force everything through eager PyTorch + MPS. The kernels here are written as custom Metal kernels under MLX and wrapped as an autodiff-capable custom function.

## Next port

Once this is stable on Mac, the CUDA port should keep the same ABI and stage layout:

1. depth sort
2. snugbox / count
3. emit tile pairs
4. sort by tile
5. tile histogram / ranges
6. forward tile raster
7. backward tile raster

That way Metal and CUDA share the same mental model and tests.
