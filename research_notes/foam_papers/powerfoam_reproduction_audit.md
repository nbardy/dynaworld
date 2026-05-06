# PowerFoam reproduction audit

Date: 2026-04-30
Updated: 2026-05-01 after the Metal oriented-texel material-frame pass.

Supersession note, 2026-05-05: this audit is historical. The implementation has
since added local Metal `cech_aabb` adjacency, strict quaternion height+SV
primitive modes, tiled and raytrace replay backward, normal-distance gradients,
posed-camera trainer plumbing, grow/prune/resample plumbing, and synthetic 4K
forward+backward verifier artifacts. Do not use the "current verdict" and
"current closeness table" below as current status. The live acceptance matrix is
`TODO/powerfoam_full_reproduction_todo.md`; measured rows are in `BASELINES.md`.

## Current verdict

There are now two local PowerFoam-family paths, and they should not be judged
as the same artifact:

1. `src/train/powerfoam_direct.py` is a slow Torch/math reference. It is the
   closest local path to the paper primitive: bounded power cells, quaternion
   frame, local detail sites, per-site height/displacement, spherical-Voronoi
   view color, and the regularizer hooks are present. It is still not a full
   paper reproduction because the training contract and systems layer are not
   official/scalable.
2. `third_party/powerfoam-metal/` plus `src/train/train_powerfoam_metal.py` is
   the trainable Metal path. It now has replay backward for bounded power cells,
   learned oriented planes, and learned local texel/detail sites. It is not a
   full PowerFoam renderer. It is a partial fast core with an appearance-detail
   layer.

Bottom line:

- **Math-reference closeness:** `powerfoam_direct` is roughly 65-75% of the
  paper primitive math, but much less of the paper training system.
- **Fast Metal closeness:** the Metal path is roughly 35-45% of the paper
  primitive math, but it proves the key memory-safe replay-backward contract.
- **End-to-end reproduction closeness:** neither path is close to a full
  PowerFoam reproduction yet because neither has official Cech/AABB adjacency,
  densification/resampling/pruning, static multi-view SfM training, or the full
  tiled raster/ray-tracing system.

The remaining paper-critical systems are:

- official Cech/AABB adjacency construction
- tiled rasterization candidate lists and replay backward for the full primitive
- densification, pruning, resampling, and contribution/error EMAs
- the full official loss stack in the fast path
- official SfM/static-scene training contract
- ray-tracing backend and Steiner-point acceleration
- non-pinhole rasterization support

The biggest local math mismatches to fix first are texel-site scale,
gradient flow through spherical-Voronoi color lookup, and softplus beta.

Follow-up implementation status: the Torch reference now fixes those three
local math mismatches, fixes the camera-facing quaternion sign, and exposes
official-style normal/contribution/point-error/visibility stats plus
SSIM/normal/contribution/interpenetration loss hooks with official-style
exponential regularizer decay. The Metal path now adds a learned
oriented-texel-surface mode with a differentiable normal/tangent/bitangent
material frame and measured a real quality jump on the tiny video fit
(`t4enmpcc`, final `Eval/L1 = 0.01899`). It is still not the final PowerFoam
reproduction until the Cech/AABB adjacency, densification/resampling, static
SfM scene trainer, full SV/height primitive math in Metal, and tiled replay
rasterizer/backward are implemented.

## Current closeness table

| Component | Paper/official PowerFoam | Torch direct path | Metal path | Audit |
|---|---|---|---|---|
| Bounded power cells | `p_i`, `r_i`, sphere bound, radical-plane cells | Present | Present | Core cell math is in both paths. |
| Power-face sign | `+ r_i^2 - r_j^2` in cell `i` face offset | Present | Present | Previously a footgun; currently correct in checked code paths. |
| Power-distance sort | `||Q-p_i||^2-r_i^2` | Fixed-origin only | Camera-origin ray bundle, but trainer uses origin camera | Correct for current fixed-origin video trainer; full camera support needs explicit `Q`. |
| Cech adjacency | AABB/Cech overlap graph, false edges safe | Dense/KNN/overlap toy builder | CPU dense KNN/overlap CSR | Not official; KNN can be mathematically wrong if it misses true overlapping neighbors. |
| Quaternion frame | Learned quaternion gives normal/tangent/bitangent | Present | Partial; explicit material frame | Metal now has differentiable tangent/bitangent texel coordinates, but not a quaternion parameterization. |
| Oriented dipole | Plane splits density/empty side | Present | Partial | Metal clips by learned plane, but lacks full local frame and height. |
| Detail sites | `s_ij` local radius-normalized plane coords | Present | Partial | Metal has local 2D texel sites in a learned material frame, but still lacks height/SV color. |
| Detail height | `h_ij * r_i` displaces surface | Present | Missing | Major Metal missing piece; detail sites are appearance-only in Metal. |
| Spherical Voronoi color | `sv_dof=8` axes/RGB per detail site | Present | Missing | Metal texel mode uses direct per-site RGB/features, not SV view color. |
| SV geometry detach | Detach texel point for SV color query | Present | N/A | Correct in Torch; Metal has no SV yet. |
| Loss stack | RGB + SSIM + normal + sparse/contrib + connect | Mostly present in direct trainer | Minimal L1/MSE/radius | Fast path lacks official regularizers. |
| Contribution/error EMAs | Used for pruning/resampling | Missing | Missing | Required for adaptive capacity. |
| Densification/pruning | Grow 100k->500k/1.2M with error-driven resampling | Missing | Missing | Required for paper-scale quality. |
| Tiled rasterizer | tile visible lists, sort, replay backward | Missing | Missing | Metal streaming replay is correct-ish but not official high-throughput path. |
| Ray tracing | adjacency walk + Steiner points | Missing | Missing | Paper's unification claim not reproduced locally. |
| Static multi-view scene contract | SfM init, posed cameras, held-out views | Missing; per-frame video adapter | Missing; per-frame video adapter | Our tiny video fit is a useful local test, not paper reproduction. |

## Mathematically flawed or paper-incomplete local choices

### 1. Metal texel coordinates now have a material frame, but not a quaternion frame

Full PowerFoam stores a quaternion and derives normal/tangent/bitangent. Detail
sites live in the tangent/bitangent plane:

```text
world_site = p_i + r_i * (u_ij * tangent_i + v_ij * bitangent_i)
```

The current Metal `oriented_texel_surface` mode now packs normal, tangent, and
bitangent into the shader feature layout, projects the surface hit with

```text
u = dot((x_hit - p_i) / r_i, tangent_i)
v = dot((x_hit - p_i) / r_i, bitangent_i)
```

and propagates gradients into the tangent/bitangent frame. The trainer keeps a
raw tangent parameter and orthonormalizes it against the learned normal. This
fixes the earlier `local_coord.xy` front-facing approximation.

Remaining mismatch: official PowerFoam uses a quaternion, so frame updates live
on a compact rotation parameterization. The Metal trainer uses an explicit raw
tangent plus normal. That is a valid differentiable material-frame parameter,
but not the paper parameterization.

Validation on 2026-05-01:

```text
third_party/powerfoam-metal/tests/backward_check.py
third_party/powerfoam-metal/tests/linear_texture_check.py
```

Both passed after the frame-layout change. A 1-step trainer smoke with 64 cells,
2 frames, and 32px render also passed and showed nonzero normal/tangent/texel
movement.

### 2. Metal detail sites have color but no height/displacement

Full PowerFoam detail sites carry both displacement and radiance:

```text
h(x) = sum_j w_j(x) h_ij / sum_j w_j(x)
x_final = x_base + h(x_base) n_i
```

The current Metal texel mode only interpolates per-site features/colors. It does
not modify the surface intersection. This means it can paint detail onto a
plane, but cannot model local geometric relief.

Fix: add per-site height, two-stage plane query like the official
`texture.py`, and backward through height, texel sites, center/radius, and
normal/frame.

### 3. Metal color is not Spherical Voronoi

Full PowerFoam per-site color is view-dependent:

```text
c_ij(v) = 0.5 + sum_m exp(-tau_ijm ||v-u_ijm||) c_ijm / sum_m exp(...)
```

The current Metal path uses learned RGB/features per texel site. This is a
good Dynaworld feature-foam primitive, but not the paper appearance model.

Fix: add SV axes/temps/RGB per texel, detach geometry for the SV color query to
match official code, and test view-dependent color against the Torch reference.

### 4. KNN adjacency can change cell geometry

A Cech superset can include false edges safely. KNN is not a Cech superset and
can miss true overlapping cells. Missing a true overlapping neighbor means a
necessary radical plane is absent; the rendered cell can be too large.

Fix: for correctness modes, build overlap/Cech adjacency, even if CPU-side and
slow. Use KNN only as a speed/ablation mode and label it mathematically
approximate.

### 5. The direct path assumes camera origin zero in several places

The direct reference computes view direction as `normalize(texel_site_world)` and
sorts by fixed-origin power distance. That is fine for the current small
canonical camera test, but not a full posed-camera PowerFoam reproduction.

Fix: pass camera origin through the direct renderer and SV color query before
using it as a multi-view static-scene oracle.

### 6. The fast trainer is per-frame foam, not a static scene

Official PowerFoam learns one static 3D scene from posed views. Our video
trainer learns independent per-frame foam states. That can validate renderer
math and direct-fit capacity, but it does not validate PowerFoam's novel-view
contract.

Fix: add a static-scene PowerFoam trainer with posed cameras and held-out-view
metrics.

## What we did reproduce

Local implementation: `src/train/powerfoam_direct.py`.

Covered concepts:

- power cells with center/radius/density
- quaternion-derived normal/tangent/bitangent frames
- oriented internal plane/dipole behavior
- local detail sites with per-site height
- spherical-Voronoi directional color basis
- power-face clipping against neighboring cells
- density compositing along rays
- per-attribute optimizer parameter groups
- step-0 visual logging through the existing W&B harness

This is enough to test whether the representation can fit small images/videos,
and it gives us a CPU/Torch-readable reference for a future Metal kernel.

## Paper and official-code free parameters

Per primitive:

- `p_i`: cell center, 3 floats
- `r_i`: cell radius, positive scalar
- `sigma_i`: density, positive scalar
- `q_i`: orientation quaternion
- `s_ij`: local detail-site offset for each texel site
- `h_ij`: detail-site height for each texel site
- `a_ijk`: spherical-Voronoi axis for each site and SV basis component
- `c_ijk`: spherical-Voronoi color coefficient for each site and SV basis component

Paper/default counts:

- `num_texel_sites = 8`
- `sv_dof = 8`
- indoor/DL3DV: `init_points = 100k`, `final_points = 500k`
- outdoor: `final_points = 1.2M`
- DTU-style smaller config: `init_points = 20k`, `final_points = 50k`

Official optimizer groups are separate for points, radii, quats, density,
texel sites, SV axes, SV RGB, and texel height, with scheduled learning rates.
Our implementation only uses static group multipliers.

## Math mismatches fixed in the Torch reference

### 1. Texel-site scale

Official PowerFoam stores local texel-site offsets in a radius-normalized local
frame and multiplies by radius in forward:

```text
world_site = point + radius * (u * tangent + v * bitangent)
```

The local path originally stored world-scale offsets. It now stores normalized
local offsets and multiplies by decoded radius in forward.

### 2. Spherical-Voronoi geometry gradients

Official code detaches texel-site positions before SV color lookup. Color
coefficients/axes learn color; geometry does not receive gradient through the
view-dependent color query.

The local path now detaches texel-site world positions before the SV color
query, matching the official code. Geometry still receives gradients through
the texel/plane intersection itself.

### 3. Positive-parameter decode

Official code uses `softplus(..., beta=100)` for radii and density.

The local path now uses beta-100 softplus/inverse-softplus for radii and
density.

### 4. Detail-site interpolation

Paper equations write an exponential in distance. Official code uses
radius-normalized squared distance for detail-site interpolation:

```text
w_j = exp(-temperature * ||x - s_j||^2 / r_i^2)
```

Our implementation matches the official code here, not the paper text. That is
probably the right reproduction target.

### 5. Depth/order

The supplement proves rasterization order using power distance:

```text
||camera - p_i||^2 - r_i^2
```

Our toy camera path uses the fixed-origin version. That is acceptable only for
the current canonical camera-at-origin trainer. Full camera support needs the
camera-origin generalization.

## Missing paper systems

### Adjacency

Official PowerFoam uses an AABB tree to build a Cech-complex superset of the
power-cell adjacency graph. It accepts false edges for exactness and rebuilds
on a schedule.

Our path uses dense pairwise distances plus overlap/KNN fallback. It is useful
for tiny tests, but it is neither the official algorithm nor scalable.

### Rasterizer

Official rasterization builds tile primitive lists, sorts by power order,
tracks tile offsets/early-stop counters, and stores compact tensors for
backward replay.

Our Torch reference loops over all cells for each image. It is intentionally
simple and slow. It does not represent the final memory/performance contract.

### Backward pass

Official backward recomputes intersections and accumulates gradients per
primitive. It does not materialize an `instances x H x W` gradient tensor.

Our Torch autograd path materializes broad per-cell/per-pixel intermediates.
It is fine as an oracle for small shapes, but it is not the fast backward pass.

### Training and losses

Official training uses:

- RGB reconstruction
- SSIM term
- normal loss
- sparsity/contribution loss
- connectivity/interpenetration loss
- contribution and point-error EMAs
- scheduled densification/resampling/pruning
- downsampled first training phase
- random background/alpha handling

Our direct trainer uses only a small reconstruction-oriented loss stack and
does not yet emit the renderer statistics required by the official regularizers
or resampler.

### Scene contract

Official PowerFoam is a static 3D scene representation initialized from SfM
points and trained against multiple posed views. Our local `DirectPowerFoamVideo`
uses per-frame foam parameters for the small video test. That is a Dynaworld
adaptation, not a literal PowerFoam reproduction.

### Ray tracing and non-pinhole support

The paper includes a ray-tracing path over the dual graph and Steiner-point
acceleration for faster cell traversal. It also calls out rasterization over
generic ray sets for non-pinhole cameras.

Our local implementation has neither. It only renders a simple pinhole-like
fixed camera ray bundle.

## What should be fixed next

1. Add a static-scene PowerFoam trainer alongside the per-frame video adapter.
2. Add official learning-rate schedules.
3. Implement densification/resampling/pruning with contribution and point-error
   EMAs.
4. Replace dense adjacency with a Cech/AABB-style builder or a close Metal-side
   equivalent with CSR adjacency buffers.
5. Write the Metal tiled/replay rasterizer and backward pass; use the Torch path
   only as a tiny-shape numerical oracle.
6. Add parity tests against official CUDA on tiny random scenes before trusting
   training curves.
