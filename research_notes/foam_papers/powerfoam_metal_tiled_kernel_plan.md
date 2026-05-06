# PowerFoam Metal Kernel Plan

Status note, 2026-05-05: this file is a historical implementation plan. The
local Metal path has moved beyond the first runnable feature-foam target below:
streaming, tiled, and raytrace paths exist, with replay backward for the full
quaternion height+SV primitive and synthetic 4K verifier artifacts. Keep this
file for design rationale; use `TODO/powerfoam_full_reproduction_todo.md` and
`BASELINES.md` for current implementation status.

## Goal

Build the PowerFoam rasterizer as a first-class Metal path, not as an extension
of the dense prototype. The memory contract is the important part: never
materialize `cells x height x width` intersections, alpha, or gradients.

Tiles are optional. The required mechanism is a memory-safe schedule plus exact
reverse replay. Prior Metal renderer work showed that tile count/write/sort and
threadgroup barrier overhead can beat the culling win in small or mostly-active
cases. Therefore the first serious implementation should include a non-tiled
streaming path and benchmark it before committing to a tiled path.

The first runnable target is feature foam:

```python
features_out, alpha_out = rasterize_powerfoam_tiled(
    points, radii, densities, features, adjacency, offsets, rays, cameras
)
```

`features` may be RGB (`F=3`) or latent features (`F=32` or larger). The
trainer can keep the existing `FeatureToColor` decoder downstream.

## Source Mapping

Official PowerFoam uses a Warp/CUDA-style tiled rasterizer:

- count visible sphere/tile intersections
- prefix-sum counts into tile offsets
- write `(tile, primitive)` pairs plus a tile-local sort key
- sort by packed tile id and power distance
- render one 8x8 tile per threadgroup
- save only tile lists, offsets, early-stop counters, `color_out`, and `log_t`
- backward replays each tile list in reverse and atomically scatters gradients

The Metal path should mirror that structure.

| Official PowerFoam | Metal feature-foam v2 |
| --- | --- |
| `all_spheres [N,4]` | `points [N,3]`, `radii [N]` |
| `all_nsigmas [N,4]` | `densities [N]`, no normals in phase 1 |
| texture sites/rgbh | omitted; constant per-cell feature vector |
| `adjacency`, `adjacency_offsets` | same CSR graph |
| `adjacency_diff [E,4]` | optional phase-2 cache of face inputs |
| `tile_prim_indices_offsets` | `tile_offsets [B*T+1]` |
| `tile_prim_indices` | `tile_cell_ids [M]` |
| `tile_early_stop_counter` | `tile_stop_counts [B*T]` |
| `log_t_out` | `log_t [B,H,W]` |
| `color_out` | `feature_out [B,H,W,F]` |

The official tiled path scales as:

```text
O(B*H*W*F) outputs
+ O(B*H*W) logT/alpha
+ O(M) tile-cell intersections
+ O(E) adjacency
+ O(N*F) cell features
```

There is no `O(N*H*W)` tensor.

The non-tiled streaming path scales even smaller in auxiliary memory:

```text
O(B*H*W*F) outputs
+ O(B*H*W) logT/alpha/pixel_stop
+ O(B*N) sorted ids and optional screen bounds
+ O(E) adjacency
+ O(N*F) cell features
```

Its compute is `O(B*H*W*N)` with cheap screen-bound skips before expensive
ray-cell clipping. That may still be faster than tiled machinery for the local
Mac regimes where our previous tiled Metal shaders lost to simpler paths.

## Geometry Contract

Cell `i` owns points whose power distance is no larger than every neighbor:

```text
||x - p_i||^2 - r_i^2 <= ||x - p_j||^2 - r_j^2
```

For a ray `x = o + t d`, neighbor `j` clips cell `i` by a plane:

```text
n = p_j - p_i
h = 0.5 * (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)
t_face = (h - dot(o, n)) / dot(d, n)
```

If `dot(d, n) > 0`, the face clips `t_far`; otherwise it clips `t_near`.
The sign of the radius terms is easy to get wrong: it is `+ r_i^2 - r_j^2`.

Cells are sorted within each tile by camera-origin power distance:

```text
sort_power_i = ||camera_origin - p_i||^2 - r_i^2
```

## Path A: Non-Tiled Streaming Pipeline

This should be implemented and measured first. It is memory-safe, simple, and a
better answer to the concern that tiled Metal overhead can dominate.

### A1. Precompute Sort And Screen Bounds

Host or a small kernel computes one global order per batch by camera-origin
power distance:

```text
sort_power_i = ||camera_origin - p_i||^2 - r_i^2
sorted_ids[b] = argsort(sort_power)
```

A companion kernel or host helper computes conservative screen bounds per
`(batch, cell)`:

```text
screen_bounds[b, cell] = [x0, y0, x1, y1]
```

The bounds are not a correctness source; they are only an early skip. If unsure,
set the bounds to the full image and the renderer remains correct.

### A2. Streaming Forward

One thread owns one pixel. It walks the global sorted cell order, skips cells
whose screen bounds do not contain the pixel, then runs exact sphere and power
face clipping.

```text
kernel forward_streaming(...):
  pixel = thread_position_in_grid
  b, y, x = decode pixel
  logT = 0
  out[pixel, :] = 0
  stop = N

  for order in [0, N):
    if exp(logT) < threshold:
      stop = order
      break

    cell = sorted_ids[b, order]
    if pixel outside screen_bounds[b, cell]:
      continue

    hit, t0, t1 = clipped_interval(cell, ray[pixel])
    if hit and t1 > t0:
      delta = -density[cell] * (t1 - t0)
      alpha = 1 - exp(delta)
      out[pixel, :] += exp(logT) * alpha * features[cell, :]
      logT += delta

  log_t[pixel] = logT
  alpha[pixel] = 1 - exp(logT)
  pixel_stop[pixel] = stop
```

Saved state is only `log_t` and `pixel_stop`, plus the global inputs and
`sorted_ids/screen_bounds`.

### A3. Streaming Backward

Backward loops `order = pixel_stop[pixel] - 1 ... 0`, recomputes interval
boundaries, and scatters gradients with atomics. This is the same math as the
tiled reverse replay, but avoids tile list construction entirely.

This is the best first "no `N*W*H`" implementation:

- memory-safe
- no tile sort/cumsum/list buffers
- no threadgroup barrier hazards
- exact, because `pixel_stop` records the early-stop boundary
- easy to compare against the dense reference

The draft shader for this path is
`third_party/powerfoam-metal/csrc/metal/powerfoam_streaming_kernels.metal`.

## Path B: Tiled Candidate-List Pipeline

### 1. Count Visible Tiles

One thread handles one `(batch, cell)` pair. The first implementation can use a
conservative pinhole projected sphere radius; the paper's OBB test can replace
it after correctness and backward are stable.

```text
kernel count_visible(points, radii, cameras, tile_counts):
  b, cell = decode thread id
  center_cam = world_to_camera(points[cell], cameras[b])
  if center_cam.z <= near + radius: return

  pixel_center = project(center_cam)
  pixel_radius = max(fx, fy) * radius / max(center_cam.z - radius, eps)
  tile_box = clamp(pixel_center +/- pixel_radius)

  for tile in tile_box:
    atomic_add(tile_counts[batch_tile(tile) + 1], 1)
```

Host-side first pass:

```python
tile_offsets = cumsum(tile_counts.to(int64), dim=0)
M = int(tile_offsets[-1])
tile_cell_ids = empty([M], int32)
sort_keys = empty([M], uint64)
tile_cursors = zeros([B * T], uint32)
```

This uses Torch/MPS `cumsum` and `argsort` initially. A Metal scan/radix sorter
is a later optimization, not the first correctness blocker.

### 2. Write Visible Pairs

```text
kernel write_visible(points, radii, cameras, tile_offsets, cursors):
  recompute same tile_box
  for tile in tile_box:
    local = atomic_add(cursors[global_tile], 1)
    dst = tile_offsets[global_tile] + local
    tile_cell_ids[dst] = cell
    sort_keys[dst] = pack(global_tile, sortable_float(sort_power))
```

Host sort:

```python
perm = torch.argsort(sort_keys)
tile_cell_ids = tile_cell_ids[perm]
```

Because `tile_offsets` were produced from per-tile counts, sorting by packed tile
id preserves contiguous per-tile ranges.

### 3. Tiled Forward

Dispatch one 8x8 threadgroup per `(batch, tile)`. Each thread owns one pixel.
The threadgroup reduction only decides when the whole tile can stop; all
barriers are in tile-uniform control flow.

```text
kernel forward_tiles(...):
  tile_group = threadgroup_position_in_grid
  tid = thread_position_in_threadgroup  # 0..63
  pixel = tile_pixel(tile_group, tid)

  zero feature_out[pixel, :]
  logT = 0
  start, end = tile_offsets[tile], tile_offsets[tile + 1]

  for k in [start, end):
    threadgroup_max_trans = max(valid_pixel ? exp(logT) : 0)
    if threadgroup_max_trans < threshold:
      if tid == 0: tile_stop_counts[tile] = k - start
      break

    cell = tile_cell_ids[k]
    hit, t0, t1, near_id, far_id = clipped_interval(cell, ray)
    if hit and t1 > t0:
      delta = -density[cell] * (t1 - t0)
      alpha = 1 - exp(delta)
      weight = exp(logT) * alpha
      feature_out[pixel, :] += weight * features[cell, :]
      logT += delta

  if no early stop:
    if tid == 0: tile_stop_counts[tile] = end - start
  log_t[pixel] = logT
  alpha_out[pixel] = 1 - exp(logT)
```

Forward save set for backward:

```text
tile_offsets
tile_cell_ids
tile_stop_counts
log_t
points/radii/densities/features
adjacency/offsets
rays/cameras
```

No per-cell per-pixel intermediates are saved.

## Backward: Exact Reverse Replay

The first backward should be pixel-owned reverse replay with global float
atomics. This is the official-code shape and it avoids the `N*H*W` blowup.

For one contributing segment:

```text
delta = -sigma * dt
alpha = 1 - exp(delta)
T_before = exp(logT_before)
out += T_before * alpha * feature
logT_after = logT_before + delta
```

Start the reverse pass from the final alpha gradient:

```text
g_logT_after = -g_alpha[pixel] * exp(logT_final)
```

Then for each cell in reverse tile order:

```text
recompute clipped interval and boundary ids
if no contribution: continue

delta = -sigma * dt
logT_before = logT_after - delta
T = exp(logT_before)
e = exp(delta)
alpha = 1 - e
s = dot(g_feature_out[pixel, :], feature[cell, :])

atomic_add(g_features[cell, f], g_feature_out[pixel, f] * T * alpha)

g_delta = g_logT_after - s * T * e
atomic_add(g_densities[cell], g_delta * -dt)

g_t_near = g_delta * sigma
g_t_far = g_delta * -sigma
route_endpoint_grad(near_id, g_t_near)
route_endpoint_grad(far_id, g_t_far)

g_logT_before = g_logT_after + s * T * alpha
logT_after = logT_before
g_logT_after = g_logT_before
```

Endpoint routing:

```text
near_id/far_id == -2:
  near-plane clamp; no cell gradient

near_id/far_id == -1:
  sphere boundary; add ray_sphere_intersect_bwd gradient to current cell

near_id/far_id >= 0:
  power face boundary against adjacency[edge]
  add ray_pface_intersect_bwd gradient to current cell and neighbor cell
```

This is the key foam difference versus a clean Gaussian per-splat backward:
power-face endpoints scatter gradients into both the current cell and the
neighbor cell. A true "cell owns its whole gradient" kernel is therefore not the
first exact implementation.

## Faster Backward Variants

### Phase 2A: Pixel-Owned Atomics

This is the first implementation.

Pros:

- exact
- no `N*H*W` tensors
- minimal saved state
- easiest to finite-difference against CPU and compare to official code

Cons:

- global atomic pressure on `points`, `radii`, `densities`, and `features`
- feature grads cost `F` atomics per hit

### Phase 2B: Tile-Cell Partial Gradients

If global atomics dominate, keep the same reverse replay but write partials to
the tile-cell intersection list:

```text
partial_geom[M, 4]
partial_density[M]
partial_feature[M, F]   # or chunked F slices
```

Each tile owns a contiguous range of `M`, so pixel threads can reduce inside the
threadgroup for the current tile-cell intersection before one write. A separate
reduce-by-cell pass accumulates `M -> N`.

This is the closest foam analog to "per-splat backward" without storing
`N*H*W`. It scales with visible tile-cell intersections, not visible
pixel-cell intersections.

### Phase 2C: Cell-Owned Backward

Do not start here. A cell-owned kernel would loop over its support pixels and
needs the ordered prefix/suffix transmittance for every pixel. Without storing
checkpoints this becomes repeated tile-list replay, often `O(K^2)` per tile.
With checkpoints it starts reintroducing per-pixel saved state. Power-face
gradient ownership is also shared with neighbor cells.

## Metal Design Notes

- Use 8x8 tiles first. PowerFoam uses this size, and 64 threads keeps one pixel
  per thread with simple threadgroup reductions.
- Keep tile-stop reductions in uniform threadgroup control flow. Prior GS work
  showed that barriers inside per-pixel-divergent control paths can pass tiny
  tests and fail saturated train cases.
- Do not put `float3` in host-visible packed arrays. Torch `[N,3]` rows are
  12-byte packed; Metal `float3` has 16-byte alignment in structs. Load from
  flat `device float*` using `idx * 3`.
- Use `device atomic_float*` for the first exact backward, matching the fast-mac
  Gaussian kernels.
- Size dispatch threadgroups from the pipeline on the host. Apple documents
  `threadExecutionWidth` as the SIMD width for a specific pipeline; do not bake
  in a hardware-wide SIMD assumption.
- Threadgroup memory is useful for tile-wide reductions and cooperative chunks,
  but it must stay under device and pipeline limits.

## Implementation Order

0. Use `arch=powerfoam_direct` as the lean trainer scaffold while the Metal
   backward is pending. It fits random per-frame foam directly to the 128px test
   video through a vectorized Torch renderer.
1. Keep the dense shader only as `powerfoam_reference_forward`.
2. Add `powerfoam_streaming_kernels.metal` and wire that first; benchmark before
   paying tiled count/write/sort overhead.
3. Keep `powerfoam_tiled_kernels.metal` as the official-style candidate-list
   design if streaming is compute-bound.
4. Add a C++ launcher that exposes a new op name, not the old dense op.
5. Validate streaming/tiled forward against the dense Metal reference on small random
   scenes with the same rays and adjacency.
6. Add backward for `features` and `densities`; finite-difference CPU reference.
7. Add point/radius endpoint gradients; finite-difference tiny scenes with one
   sphere boundary and one power-face boundary isolated.
8. Benchmark:
   - resolutions: `128, 256, 512`
   - cells: `256, 1024, 4096, 16384`
   - feature dims: `3, 32`
   - forward, backward, forward+backward
   - compare against current fast-mac GS with the same batch/resolution/F
9. Only after that, port dipole normals and detail-site texture.

## Open Risks

- Conservative sphere projection can over-bin large or near-camera cells. That
  is acceptable for first correctness but not the final performance story.
- Sorting with `torch.argsort` may dominate at high `M`; we should measure
  before writing a Metal radix path.
- Atomic feature gradients can dominate for `F=32`. The tile-cell partial
  gradient path is the planned escape hatch.
- A learned adjacency graph must be conservative. Missing a power neighbor is a
  correctness bug; extra neighbors are only a speed cost.

## References

- Official code scan: `powerfoam/rasterize.py` in `/tmp/powerfoam_official`
- Local draft shader: `third_party/powerfoam-metal/csrc/metal/powerfoam_tiled_kernels.metal`
- Apple Metal docs checked for this plan:
  - `https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth`
  - `https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)`
  - `https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon`
