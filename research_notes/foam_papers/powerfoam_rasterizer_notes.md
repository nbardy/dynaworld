# PowerFoam Rasterizer Notes

Artifacts:

- PDF: `research_notes/foam_papers/pdfs/powerfoam.pdf`
- Extracted text: `research_notes/foam_papers/text/powerfoam.txt`
- Current implementation status: `research_notes/foam_papers/foam_implementation_status_2026-05-02.md`
- Local prototype: `third_party/powerfoam-metal/`
- Tiled Metal kernel plan: `research_notes/foam_papers/powerfoam_metal_tiled_kernel_plan.md`
- Draft tiled shader: `third_party/powerfoam-metal/csrc/metal/powerfoam_tiled_kernels.metal`
- Direct-fit trainer: `src/train/train_powerfoam_direct.py`
- Direct-fit config: `src/train_configs/local_mac_powerfoam_direct_128_smoke.jsonc`

Status note: this is a historical mechanism note. As of 2026-05-05, the current
truth table is `TODO/powerfoam_full_reproduction_todo.md` plus `BASELINES.md`.
The Metal path has moved past the original forward-only prototype: it now has
streaming, tiled, and raytrace execution paths; replay backward; `cech_aabb`
trainer adjacency; strict quaternion height+SV primitive math; normal-distance
gradients; posed-camera trainer plumbing; and saved synthetic 4K
forward+backward verifier artifacts. It is still not the full official
CUDA/Warp PowerFoam training system.

## Mechanism To Preserve

PowerFoam represents space with bounded power cells. A cell site has position
`p_i`, radius/weight `r_i`, density, and appearance. Its region is the subset
of its bounding sphere where its power distance is no larger than every
neighbor:

```text
||x - p_i||^2 - r_i^2 <= ||x - p_j||^2 - r_j^2
```

For a ray `x = o + t d`, each neighbor becomes a half-space clip:

```text
2 (o + t d) dot (p_j - p_i)
  <= ||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2
```

The rasterization trick is to process cells in front-to-back painter order by
power distance from the camera origin:

```text
||c - p_i||^2 - r_i^2
```

The paper relies on a neighbor graph such as the Cech complex, or a conservative
superset, so each candidate cell only clips against overlapping nearby cells.
The paper reports that false-positive edges preserve correctness but add some
render cost.

## Implemented Local Core

`third_party/powerfoam-metal` implements the bounded-cell Metal core:

- MPS Torch custom op plus runtime Metal shader loader.
- Inputs: `points [N,3]`, `radii [N]`, `densities [N]`,
  `features [N,F]`, `adjacency [E]`, `offsets [N+1]`, `rays [B,H,W,6]`.
- Optional `sorted_ids [B,N]`; otherwise the Python wrapper sorts by camera
  power distance from `rays[:,0,0,:3]`.
- Output: rendered color/features, accumulated `alpha`, and selected auxiliary
  outputs such as normal distance for the full height+SV training path.
- The streaming/tiled raster paths clip ray-sphere intervals against neighbor
  radical planes and alpha-composite front-to-back.
- The raytrace path walks the current adjacency graph and replays a capped event
  list for backward.
- Full primitive mode supports strict quaternion frames, detail-site
  height/displacement, spherical-Voronoi view-dependent color, and gradients for
  geometry/material state.

This gives a real "feature foam" path in the Dynaworld sense because the
feature channel count is arbitrary. The current full height+SV mode also covers
the paper texture mechanism locally, but not the full official training system
or paper-scale acceptance.

## Still Not A Full Reproduction

- The official CUDA/Warp backend is not ported to Metal and cannot be exercised
  locally without a CUDA/Warp host.
- Official fixture generation is wired but still needs a CUDA host to produce
  the upstream-output JSON fixture.
- Arbitrary depth-quantile loss gradients and external normal-supervision
  gradients are not wired unless selected work adds them.
- The current clean multiview geometry is weak. DeepView pycolmap artifacts are
  mostly two-view tracks, and higher SIFT resolution did not close the heldout
  quality gap.
- Paper-scale grow/prune/resampling schedules and acceptance runs remain open.

## Validation

Build:

```bash
( cd third_party/powerfoam-metal && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Reference check:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/reference_check.py
```

Observed on 2026-04-30 for the first constant-feature prototype:

```text
features max error: 5.8710575103759766e-06
alpha max error: 7.867813110351562e-06
powerfoam Metal reference check passed
```

Visual random-foam smoke:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/render_random_png.py --cells 384 --height 192 --width 192 --seed 3 --neighbors 48 --adjacency overlap --out third_party/powerfoam-metal/outputs/random_foam_384_192.png
```

Observed on 2026-04-30:

```text
wrote third_party/powerfoam-metal/outputs/random_foam_384_192.png
wrote third_party/powerfoam-metal/outputs/random_foam_384_192_alpha.png
cells=384 resolution=192x192 adjacency=overlap avg_degree=7.45
```

Forward speed smoke versus the current `v5_features` Gaussian renderer:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py --cells 256,1024 --resolutions 128x128,256x256 --neighbors 32 --adjacency knn --warmup 2 --iters 5 --compare-gs
```

Observed on 2026-04-30:

```text
powerfoam_metal    128x128 N=256  fwd_med=4.378ms
gsplat_v5_features 128x128 N=256  fwd_med=4.077ms
powerfoam_metal    128x128 N=1024 fwd_med=3.846ms
gsplat_v5_features 128x128 N=1024 fwd_med=3.138ms
powerfoam_metal    256x256 N=256  fwd_med=4.100ms
gsplat_v5_features 256x256 N=256  fwd_med=2.699ms
powerfoam_metal    256x256 N=1024 fwd_med=11.404ms
gsplat_v5_features 256x256 N=1024 fwd_med=3.506ms
```

This is only a historical smoke matrix, not the current renderer decision. Later
work added tiled and raytrace paths; current selected synthetic 4K train evidence
is tracked in `TODO/powerfoam_full_reproduction_todo.md`.

## Backward Pass

This section records the original porting target. The local Metal path now has
custom autograd replay backward for the active streaming/tiled/raytrace modes,
including full height+SV raytrace backward and normal-distance gradients.

The official PowerFoam code does have a custom backward path in
`powerfoam/rasterize.py`: a `RasterGradFn` calls `Rasterizer._backward`, which
launches a Warp `backward_kernel`. That kernel walks the per-tile primitive list
in reverse, replays interval clipping, and accumulates gradients into sphere,
density/normal, and texture buffers. The helper derivatives live in
`powerfoam/rendering_math.py` and `powerfoam/texture.py`.

The paper establishes differentiability and gives the representation/rendering
structure, but the fast backward recipe is mostly in the implementation:

- save/reuse tile primitive lists, offsets, early-stop counters, color, and log
  transmittance from forward
- traverse front-to-back in forward and reverse order in backward
- propagate through alpha/transmittance recurrence
- route interval endpoint gradients to whichever boundary won the max/min:
  sphere hit, neighboring power face, or oriented dipole plane
- use atomics/scattered adds for cell and neighbor gradients

The remaining backward caveats are narrower: arbitrary depth-quantile gradients
and external normal-supervision gradients are not wired as differentiable losses
unless future work selects them.

## Official Code Scan

Scratch clone inspected on 2026-04-30:

```text
/tmp/powerfoam_official
commit 25d6f7b Fix clone command in README
```

The official implementation is CUDA/Warp, not Metal. The README requires a
Linux CUDA environment and lists `torch==2.9.1+cu128` plus
`warp-lang==1.10.0` as the reference stack.

Important files:

- `powerfoam/rasterize.py`: raster forward, custom autograd wrapper, backward,
  visualization, and forward-only benchmark path.
- `powerfoam/rendering_math.py`: analytic derivatives for ray-sphere,
  power-face, and plane intersections.
- `powerfoam/texture.py`: detail-site soft Voronoi texture forward/backward.
- `benchmark.py`: forward FPS benchmark for rasterize/raytrace; it does not
  expose backward timing.

The official raster forward pipeline is:

1. `count_visible_*_kernel`: count sphere/tile intersections. Pinhole cameras
   project each sphere to an oriented bounding box; generic cameras precompute a
   cone hierarchy over rays.
2. Prefix sum counts to per-tile offsets.
3. `write_visible_*_kernel`: emit `tile_prim_indices` and `sort_keys`.
4. `torch.argsort(sort_keys)`: sort by packed tile id plus power distance.
5. `prefetch_adjacency_kernel`: pack neighbor relative offsets into
   `adjacency_diff` as fp16.
6. `forward_kernel`: one thread per pixel in an 8x8 tile, iterating only that
   tile's sorted primitive list. It clips the sphere interval by neighbor power
   faces, then by the oriented dipole/detail plane, accumulates `rgb`, `log_t`,
   normal terms, contribution, quantile depths, and `tile_early_stop_counter`.

The official backward pipeline is:

1. `RasterGradFn.forward` saves `all_spheres`, `all_nsigmas`, texel buffers,
   adjacency, `adjacency_diff`, `tile_prim_indices`, tile offsets,
   `tile_early_stop_counter`, `color_out`, and `log_t_out`.
2. `Rasterizer._backward` converts alpha gradient to log-transmittance gradient
   with `grad_log_t_in = -grad_opacity_in * exp(log_t_out)`.
3. `backward_kernel` launches one thread per pixel, walks the saved tile list in
   reverse from `early_stop - 1` to zero, replays ray-sphere and power-face
   interval clipping, remembers which boundary won each endpoint, and propagates
   gradients through alpha/transmittance.
4. Endpoint gradients route to sphere derivatives or power-face derivatives.
   Power-face endpoints scatter gradients into both the current primitive and
   its adjacent primitive. The oriented dipole plane and detail texture are
   handled through `plane_intersection_bwd`.

This changes our local porting order:

- First wire and benchmark the non-tiled streaming Metal path. It keeps the
  memory-safe reverse replay contract without paying tile count/write/sort
  overhead up front.
- Keep the official-style tile count/write/sort path as the candidate-list
  acceleration if streaming becomes compute-bound.
- Then port the simpler volume backward for `(features, alpha)` only:
  feature gradients, density gradients, and endpoint gradients into
  positions/radii through sphere and power-face boundaries.
- Only after that port official dipole normals, detail-site displacement, and
  spherical-Voronoi color. These are paper-faithful but not needed for the
  first Dynaworld feature-foam contract.

The official `benchmark.py` only measures forward FPS. To measure backward
speed in the official code, add a new harness around `model.forward(...); loss.backward()`
with CUDA events or NVTX ranges. The training loop already labels `Forward`,
`Losses`, `Backward`, and `Optimizer Step` with NVTX, but does not print a
standalone backward table.

## Integration Shape

The likely trainer-facing boundary should mirror the current feature splat
renderer:

```python
out_features, out_alpha = rasterize_power_foam(
    points, radii, densities, features, adjacency, offsets, rays, config
)
```

Then the existing `FeatureToColor` path can own feature-to-RGB decoding for
`F != 3`, while `F == 3` can be treated as direct RGB if we want a simple visual
smoke. The baseline Gaussian rasterizers should stay untouched; PowerFoam now
has separate runtime and held-out-camera gates tracked in `BASELINES.md`.

## Direct-Fit Scaffold

`arch=powerfoam_direct` is the slow correctness/reference scaffold. It uses a
vectorized Torch renderer, including posed-camera smoke coverage, to check the
representation and loss path independently from Metal. The trainable Metal
trainer is `src/train/train_powerfoam_metal.py`.

Config:

```bash
PYTHONPATH=src/train uv run python src/train/train.py src/train_configs/local_mac_powerfoam_direct_128_smoke.jsonc
```

The checked-in config fits `test_data/test_video_small_128_4fps.mp4` with:

- 16 loaded frames at 128x128
- 32 independent RGB foam cells per frame
- 8 fixed KNN power-face neighbors per cell
- CPU execution, because this temporary Python/Torch renderer is launch-overhead
  dominated on MPS
- 100 training steps

Observed 2026-04-30:

```text
step 100 sampled-frame L1: 0.097700
step 100 full-video eval L1: 0.130358
step 100 full-video eval MSE: 0.032068
```

Artifacts are written to
`outputs/powerfoam_direct/local_mac_powerfoam_direct_128_smoke/`.
