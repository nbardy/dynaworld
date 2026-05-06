# PowerFoam Metal Prototype

This package is a trainable Metal prototype for the bounded power-cell
rasterization core described in the PowerFoam paper.

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

It implements a streaming/tiled replay backward for centers, radii, densities,
features, quaternion frames, texel sites, height displacement, and
spherical-Voronoi color. The Dynaworld trainer exposes `adjacency_mode:
"cech_aabb"` as the correctness path; `knn` is only an approximate speed
ablation. A non-gradient tiled aux pass exposes normal distance, accumulated
normal, arbitrary depth-quantile maps, contribution, point error, and visibility mask
for trainer statistics; the Dynaworld trainer also has EMA-driven replacement,
grow, and prune plumbing from those stats. The current high-throughput tiled
path uses 16x16 Metal tiles. It does not yet implement differentiable
depth aux losses or external normal-supervision gradients; normal-distance,
contribution/sparsity, and interpenetration losses are wired through the
Dynaworld trainer. Paper-scale densification/pruning schedules and acceptance
runs, a full static SfM benchmark, and a ray-tracing adjacency walk are still
missing.

Build from the Dynaworld root:

```bash
( cd third_party/powerfoam-metal && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Run the reference check:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/reference_check.py
```

Run the backward parity checks:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/backward_check.py
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/linear_texture_check.py
```

Write a random foam PNG:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/render_random_png.py --cells 512 --height 256 --width 256
```

Run a forward timing sweep, optionally compared with the current
`v5_features` Gaussian renderer:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py --cells 256,1024 --resolutions 128x128,256x256 --compare-gs
```

Add `--foam-backward` to include the Metal backward pass in the timing.
Add `--foam-linear` to benchmark the local-linear feature mode.
Add `--foam-surface` to benchmark the fixed surface-linear feature mode.
Add `--foam-oriented-surface` to benchmark the learned-normal surface-linear
feature mode.
Add `--foam-texel-surface` to benchmark the learned-normal detail-site mode.
