# RADFoam / PowerFoam integration scan

## What I checked

- Project pages:
  - https://radfoam.github.io/
  - https://powerfoam.github.io/
- Code snapshots cloned to `/tmp/dynaworld_foam_scan/`:
  - `theialab/radfoam` at `3e7b52cf74e37ab2ab5e695f53570f515f537e3d`
  - `theialab/powerfoam` at `25d6f7b42aa1d42d30d4eead10e9e6b2101731db`
- Local Dynaworld seams:
  - `src/train/runtime_types.py`
  - `src/train/rendering.py`
  - `src/train/renderers/fast_mac.py`
  - `src/train/objective/types.py`
  - `src/train/objective/objective.py`
  - `src/train/gs_models/blocks.py`
  - `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`

## Read of the two foam papers/codebases

PowerFoam is the direct rasterization candidate. Its project page frames the key change as bounded power-diagram cells with controllable extents, so the cells are spatially bounded and amenable to tile culling. The released code is Python + `warp-lang`; `powerfoam/rasterize.py` exposes an autograd `Rasterizer` that returns `(color_out, opacity_out, normal_distance_out, normal_out, quantile_depths_out, err_out, contrib_out, point_err_out, prim_visible_mask)`.

RADFoam is a ray-tracing candidate, not a rasterization backend. It uses learnable Voronoi sites, CUDA/C++ bindings, Delaunay adjacency, and a `TraceRays` autograd function. The reference repo requires CUDA and the README calls out Delaunay robustness stalls. It can still be a useful representation target, but it should come after PowerFoam because its rendering API is ray batches and start-point traversal rather than image/tile raster output.

## Local Dynaworld fit

The current Dynaworld model output type is `GaussianSequence`, with tensors:

- `xyz [K, G, 3]`
- `scales [K, G, 3]`
- `quats [K, G, 4]`
- `opacities [K, G, 1]`
- `rgbs [K, G, F]`

The objective stack is already closer to representation-agnostic than the model side. `RGBReconObjective` only needs a `RasterizedView(features=[K,F,H,W], alpha=[K,H,W] | None, ...)` and then colorizes/composites. That means the right architectural move is not "make foam look like GaussianSequence forever"; it is:

1. introduce a renderable primitive payload abstraction,
2. keep the objective/colorize/background path shared,
3. add renderer adapters that turn each decoded primitive family into `RasterizedView`.

## Suggested path

Start with PowerFoam in a small CUDA-only lane:

1. Vendor or submodule `theialab/powerfoam` under `third_party/powerfoam`.
2. Add a `FoamSequence`/`PowerFoamSequence` dataclass beside `GaussianSequence`, with per-frame:
   - `points/sites [K, N, 3]`
   - `radii [K, N]`
   - `density [K, N, 1]`
   - `quats/normals [K, N, 4]` or direct `normals [K, N, 3]`
   - `texel_sites [K, N, S, 2 or 3]`
   - `texel_rgb/features [K, N, S, F]`
   - `texel_height [K, N, S]`
   - `adjacency`, `adjacency_offsets`
3. Add `renderers/powerfoam.py` that converts `CameraSpec` to PowerFoam's camera convention and calls the Warp rasterizer one frame at a time, returning `RasterizedView`.
4. Add a model head parallel to `GaussianParameterHeads` that decodes world tokens into the PowerFoam fields.
5. First smoke target: overfit 1-2 frames, known-camera, RGB only, no feature splatting, no resampling/densification, fixed adjacency rebuild cadence.
6. Only after RGB works, extend `texel_rgb` to F-channel features and reuse the existing `FeatureToColor`/alpha-aware RGB composition path.

Treat RADFoam as second stage:

1. Add a `renderers/radfoam.py` ray-batch adapter that renders full images by generating rays from `CameraSpec` and reshaping `rgba`.
2. Use it for offline CUDA experiments first.
3. Do not try to call it a rasterizer option until there is either a PowerFoam-style bounded raster path or a local raster approximation. RADFoam's core contract is ray tracing through Delaunay/Voronoi adjacency.

## Main blockers

- Local Mac/MPS training cannot run either reference backend as-is. PowerFoam uses Warp with `torch.cuda.current_stream`; RADFoam validates CUDA tensors in its bindings.
- PowerFoam's released rasterizer is RGB/normal/depth-oriented, not F-channel feature-native. The first integration should keep RGB before expanding to feature splats.
- Both foam methods rely on adjacency/topology. Tokens can decode sites and attributes, but adjacency must be built/rebuilt from geometry, not emitted as unconstrained token output.
- The current config/model factory assumes Gaussian primitive names. A clean path needs a representation key such as `model.primitive_type: "gaussian" | "powerfoam" | "radfoam"` and renderer validation that rejects incompatible backend/device combinations early.

## Current recommendation

PowerFoam is the practical first "foam option for rasterization." RADFoam should be kept as a ray-traced comparison/backend, useful for the long-term ray effects story but not the first token-to-raster target.
