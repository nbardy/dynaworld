# DynamicPowerFoam Metal Prototype

This package is a fork of `third_party/powerfoam-metal` for Dynamic PowerFoam
experiments. The kernels are intentionally still the same bounded power-cell
rasterization core; dynamic behavior is decoded in Python before the selected
frame's cells are passed to Metal.

It intentionally implements the smallest useful raster contract for Dynaworld:

- per-cell positions, radii, densities, and arbitrary `F`-channel features
- a supplied neighbor graph, currently expected to be the Cech/overlap graph or
  a conservative superset
- front-to-back painter ordering by power distance from each batch camera origin
- outputs `(features, alpha)` with shapes `[B,H,W,F]` and `[B,H,W]`
- optional local-linear per-cell features, where `[N,C,4]` stores base/x/y/z
  coefficients evaluated at the ray midpoint in radius-normalized cell space
- optional surface-linear per-cell features, using the same `[N,C,4]` layout but
  clipping by a fixed camera-facing plane through each cell center and sampling
  the linear feature at that surface point
- optional oriented-surface-linear features, which add a learned `[N,3]`
  surface normal while keeping the same streaming replay backward contract
- optional oriented texel-surface features, which add learned local `[N,S,2]`
  detail sites, per-site `[N,S,C]` features, and a differentiable
  normal/tangent/bitangent material frame over the learned surface plane

It implements a streaming replay backward for centers, radii, densities, and
features, including texel-site and material-frame gradients in the
oriented-texel mode. It does not yet implement a fused time-decoding kernel,
detail-site heights, spherical-Voronoi view-dependent color, foam densification,
or the final tiled high-throughput kernel.

Build from the Dynaworld root:

```bash
( cd third_party/dynamic-powerfoam-metal && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Run the reference check:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/dynamic-powerfoam-metal/tests/reference_check.py
```

Run the backward parity checks:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/dynamic-powerfoam-metal/tests/backward_check.py
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/dynamic-powerfoam-metal/tests/linear_texture_check.py
```

Write a random foam PNG:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/dynamic-powerfoam-metal/tests/render_random_png.py --cells 512 --height 256 --width 256
```

Run a forward timing sweep, optionally compared with the current
`v5_features` Gaussian renderer:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/dynamic-powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py --cells 256,1024 --resolutions 128x128,256x256 --compare-gs
```

Add `--foam-backward` to include the Metal backward pass in the timing.
Add `--foam-linear` to benchmark the local-linear feature mode.
Add `--foam-surface` to benchmark the fixed surface-linear feature mode.
Add `--foam-oriented-surface` to benchmark the learned-normal surface-linear
feature mode.
Add `--foam-texel-surface` to benchmark the learned-normal detail-site mode.
