# PowerFoam Mathematical Aspects Deep Dive

Date: 2026-05-01

Supersession note, 2026-05-05: this is a mathematical and planning deep dive,
not the current implementation status. Several "current Metal path" and
"missing" statements below were overtaken by later local work: `cech_aabb`
adjacency, strict quaternion height+SV modes, tiled and raytrace replay
backward, normal-distance gradients, posed-camera trainer plumbing, and
synthetic 4K verifier artifacts now exist. Use
`TODO/powerfoam_full_reproduction_todo.md` and `BASELINES.md` for current
status.

This note expands the mathematical model behind PowerFoam and maps it to the
local Dynaworld implementation work. It is meant to answer:

- What are the core mathematical objects?
- What are the free parameters?
- How are cells, power diagrams, dipoles, detail sites, color, and orientation
  defined?
- What is "Spherical Voronoi" in this paper, and how is it different from
  reflection?
- How do parameters move during optimization?
- What does our Metal path implement, and what is still missing?

Source anchors:

- Paper PDF: `research_notes/foam_papers/pdfs/powerfoam.pdf`
- Extracted paper text: `research_notes/foam_papers/text/powerfoam.txt`
- Project page: `https://powerfoam.github.io/`
- Official code clone inspected at `/tmp/powerfoam_official`
- Local implementation notes:
  - `research_notes/foam_papers/powerfoam_reproduction_audit.md`
  - `research_notes/foam_papers/powerfoam_rasterizer_notes.md`
  - `third_party/powerfoam-metal/`
  - `src/train/powerfoam_direct.py`
  - `src/train/train_powerfoam_metal.py`

## Executive Model

PowerFoam is best understood as a nested Voronoi representation:

```text
3D bounded power diagram
    -> gives finite volumetric cells and exact non-popping front-to-back order

oriented dipole surface inside each cell
    -> turns each volumetric cell into a local surface/empty-space primitive

2D soft Voronoi field on the dipole plane
    -> gives high-frequency displacement and texture without increasing cell count

spherical Voronoi function at each 2D detail site
    -> gives view-dependent color/directional radiance
```

The most important conceptual shift from 3DGS is that PowerFoam is not a
collection of overlapping translucent blobs. It is a partition-like geometry:
cells meet along power faces, and a ray can be integrated by visiting cells in a
provably valid order. That order is defined by power distance from the camera.

The most important conceptual shift from Radiant Foam is boundedness. Radiant
Foam uses unbounded Voronoi cells, which are natural for ray traversal but bad
for tile rasterization. PowerFoam replaces Voronoi cells with bounded power
cells, where the same radius both limits the support sphere and changes the
radical planes between neighboring cells. That gives useful gradients for both
cell extent and cell boundary.

The most important conceptual shift from "solid colored foam cells" is the
dipole/detail-site layer. Full PowerFoam does not simply color a whole cell. A
cell contains an oriented surface. That surface has local texture sites, local
height displacement, and per-site view-dependent color.

## Notation

Per primitive/cell index:

```text
i                   primitive/cell index
j                   neighboring primitive/cell index
k                   number of detail sites per primitive
m                   spherical-Voronoi basis index

p_i in R^3          cell center / power site / face center
r_i > 0             power radius and bounding sphere radius
sigma_i > 0         density on the high-density side of the dipole
q_i in R^4          orientation quaternion
n_i in R^3          normal derived from q_i
t_i, b_i in R^3     tangent and bitangent derived from q_i
s_ij in R^2         local detail-site coordinate on the dipole plane
h_ij in R           detail-site height/displacement scalar
a_ijm in R^3        raw spherical-Voronoi axis vector
u_ijm in S^2        normalized direction axis from a_ijm
tau_ijm >= 0        spherical-Voronoi temperature from ||a_ijm||
c_ijm in R^3        RGB color coefficient/offset for spherical axis m
```

Ray notation:

```text
o in R^3            ray origin
d in R^3            ray direction, normally unit length
x(t) = o + t d      ray point
t_near, t_far       current ray interval inside a cell
T                   accumulated transmittance before current segment
alpha               opacity contribution of current segment
```

The paper uses continuous geometry. The official implementation uses raw
trainable tensors decoded into constrained quantities:

```text
r_i      = softplus(raw_r_i, beta=100)
sigma_i  = softplus(raw_sigma_i, beta=100)
q_i      = raw_q_i / ||raw_q_i||
u_ijm    = a_ijm / ||a_ijm||
tau_ijm  = ||a_ijm||
```

## 1. From Voronoi To Power Cells

### 1.1 Ordinary Voronoi Foam

Radiant Foam starts with Voronoi cells:

```text
V_i = { x in R^3 : ||x - p_i|| <= ||x - p_j|| for all j }
```

The boundary between cells `i` and `j` is the plane equidistant from their
sites. A ray can walk cell-to-cell through adjacent faces. That is why foam
representations are attractive for ray tracing: the representation itself gives
a mesh-like adjacency graph.

But ordinary Voronoi cells are unbounded. For rasterization, that is a problem:

- a cell can project over a huge screen region
- tile culling needs a projected convex hull, not a simple radius
- empty/unbounded regions can still affect traversal bookkeeping
- training-time adjacency wants Delaunay-like updates

### 1.2 Power Distance

PowerFoam uses weighted sites, where each site has a center and a radius. The
power distance from a point `x` to primitive `i` is:

```text
pow_i(x) = ||x - p_i||^2 - r_i^2
```

The unbounded power cell is:

```text
Pi_i = { x in R^3 : pow_i(x) <= pow_j(x) for all j }
```

When all radii are equal, subtracting `r_i^2` changes nothing, so the power
diagram reduces to the ordinary Voronoi diagram.

### 1.3 Bounded Power Cell

PowerFoam uses the intersection of the power cell with the cell sphere:

```text
B_i = { x : ||x - p_i|| <= r_i and pow_i(x) <= pow_j(x) for neighbors j }
```

This is the "bounded power diagram" representation in the paper.

The same radius has two roles:

1. It is the support radius, so screen-space culling can use a sphere.
2. It is the power weight, so changing it moves radical planes too.

This is crucial. If the sphere bound were independent from the cell partition,
it could fail to get gradients in common cases. With power cells, radius changes
affect both the spherical cap boundary and the neighbor faces.

## 2. Radical Planes And Ray Clipping

The cell ownership inequality against neighbor `j` is:

```text
||x - p_i||^2 - r_i^2 <= ||x - p_j||^2 - r_j^2
```

Expand both sides:

```text
x dot x - 2 x dot p_i + ||p_i||^2 - r_i^2
    <=
x dot x - 2 x dot p_j + ||p_j||^2 - r_j^2
```

Cancel `x dot x`:

```text
2 x dot (p_j - p_i)
    <=
||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2
```

Define:

```text
n_ij = p_j - p_i
h_ij = 0.5 * (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)
```

Then the half-space test is:

```text
x dot n_ij <= h_ij
```

For a ray `x(t) = o + t d`, the plane crossing is:

```text
t_face = (h_ij - o dot n_ij) / (d dot n_ij)
```

If `d dot n_ij > 0`, the ray exits cell `i` at that plane, so it tightens
`t_far`.

If `d dot n_ij < 0`, the ray enters cell `i` at that plane, so it tightens
`t_near`.

If `d dot n_ij = 0`, the ray is parallel; either it is inside the half-space for
all t or outside for all t.

### 2.1 Sphere Interval

Before neighbor clipping, the ray interval inside the support sphere is found
from:

```text
||o + t d - p_i||^2 = r_i^2
```

Let:

```text
oc = o - p_i
A = d dot d
B = 2 oc dot d
C = oc dot oc - r_i^2
disc = B^2 - 4 A C
```

If `disc < 0`, no sphere hit. Otherwise:

```text
t0 = (-B - sqrt(disc)) / (2 A)
t1 = (-B + sqrt(disc)) / (2 A)
```

The sphere gives initial interval:

```text
t_near = max(t0, near_plane)
t_far  = t1
```

Then each neighbor radical plane clips that interval.

### 2.2 Endpoint Identity Matters For Backward

For differentiable replay, each final endpoint came from one of:

```text
sphere boundary
neighbor power face
oriented dipole plane
near plane / previous clamp
```

Backward must route gradients to the parameter that created the active
endpoint.

This is why PowerFoam backward is naturally a replay backward:

1. Recompute the clipped interval.
2. Remember which boundary won `t_near` and `t_far`.
3. Propagate through alpha/transmittance.
4. Scatter endpoint gradients to centers/radii/normals/texels.

It should not materialize a dense `N * H * W` gradient tensor.

## 3. Adjacency: Alpha Complex Versus Cech Superset

The true minimal graph for bounded power-cell face testing is the alpha-complex
of the support spheres. The paper argues that the cheaper Cech graph is enough:

```text
edge(i,j) exists if ||p_i - p_j|| <= r_i + r_j
```

This graph includes every overlapping sphere pair. It is a superset of the true
face graph. Extra neighbors induce extra radical planes, but those planes do
not change the actual bounded cell when the edge is false-positive. They only
cost extra clipping work.

This is a key engineering point:

- Delaunay/regular triangulation is expensive to rebuild during training.
- Cech can be built from sphere overlap / AABB collision.
- False edges preserve correctness.
- Connectivity loss encourages fewer overlaps, which helps keep this graph
  sparse.

Official code builds an AABB tree and Cech complex in:

```text
/tmp/powerfoam_official/powerfoam/scene.py
/tmp/powerfoam_official/powerfoam/bvh.py
```

Our current tiny trainer uses KNN/overlap approximations, not the official
scalable AABB/Cech builder.

## 4. Pop-Free Rasterization Order

Gaussian splatting uses a heuristic depth ordering. For distorted cameras,
large splats, or overlapping geometry, the correct per-ray order can differ
from a global primitive order, creating popping or view-dependent artifacts.

PowerFoam has a stronger ordering property.

For a camera origin `Q`, sort by:

```text
sort_i(Q) = pow_i(Q) = ||Q - p_i||^2 - r_i^2
```

The supplement proves that this is a valid painter order for power cells:
if any part of cell `i` occludes cell `j` from `Q`, then:

```text
pow_i(Q) < pow_j(Q)
```

The proof relies on the fact that every power-cell boundary is still a
half-space. Weights shift the plane offset, but do not make the boundary curved
or non-convex.

Practical consequence:

- one global sort per camera is enough for rasterization from that camera origin
- the result is exact for any ray direction through the same camera origin
- this supports fisheye/non-pinhole ray bundles better than splat depth sorting

This does not mean sorting is free. It means the sorted order is mathematically
valid for the cell complex.

## 5. Volume Rendering In A Cell

Once a ray segment inside a cell is known, PowerFoam uses standard front-to-back
alpha compositing.

For a segment length:

```text
Delta_i = t_far - t_near
```

and density `sigma_i`, the optical thickness is:

```text
m_i = sigma_i * Delta_i
```

Opacity:

```text
alpha_i = 1 - exp(-m_i)
```

Color accumulation:

```text
C_out += T_i * alpha_i * color_i
T_{i+1} = T_i * exp(-m_i)
```

Accumulated alpha:

```text
A_out = 1 - T_final
```

Backward through the recurrence needs both:

- gradients from direct color contribution
- gradients through later transmittance, because changing an earlier segment
  changes how much later cells are visible

Our Metal replay backward already follows this recurrence for the simplified
feature modes.

## 6. Oriented Points / Dipoles

Plain volumetric cells have an inefficiency: surfaces appear as interfaces
between high-density cells and low-density/empty cells. That means many
primitives can be spent just to represent void.

PowerFoam introduces an oriented internal face per cell, also described as a
dipole. Each primitive has:

```text
p_i       face center / cell center
n_i       normal
```

The plane is:

```text
(x - p_i) dot n_i = 0
```

This plane splits the cell into two half-spaces:

```text
inside/high-density side     sigma_i, radiance
outside/empty side           zero density
```

The exact sign convention is implementation-dependent. The loss stack includes
a normal-orientation penalty so the high-density face points consistently
relative to visible camera rays.

### 6.1 Why A Quaternion, Not Just A Normal?

The paper language often highlights the normal, but official code stores a full
quaternion. This matters because each primitive also needs a 2D coordinate
frame on its local plane. A normal alone gives the plane orientation, but not
the in-plane rotation.

Official code derives:

```text
n_i = first rotated basis axis
t_i = second rotated basis axis
b_i = third rotated basis axis
```

With quaternion `q = (w,x,y,z)`, official normal is:

```text
n_x = 1 - 2(y^2 + z^2)
n_y = 2(xy - zw)
n_z = 2(xz + yw)
```

Official tangent is:

```text
t_x = 2(xy + zw)
t_y = 1 - 2(x^2 + z^2)
t_z = 2(yz - xw)
```

Official bitangent is:

```text
b_x = 2(xz - yw)
b_y = 2(yz + xw)
b_z = 1 - 2(x^2 + y^2)
```

The current Metal texel path now has an explicit tangent/bitangent material
frame, so it no longer has the old `local_coord.xy`/normal-only flaw. It is
still not the official quaternion parameterization.

## 7. Detail Sites On The Dipole Plane

Each primitive has `k` detail sites:

```text
s_ij = (u_ij, v_ij) in R^2, j=1..k
```

Official default:

```text
k = num_texel_sites = 8
```

The official code stores these in local radius-normalized plane coordinates.
The world position of detail site `j` is:

```text
world_site_ij =
    p_i + r_i * (u_ij * t_i + v_ij * b_i)
```

So changing radius scales the local texture footprint. This is important:
texel sites are not world-space free points. They are local coordinates
attached to the primitive frame.

## 8. Soft Voronoi Texture On The Plane

PowerFoam uses the detail sites to define soft Voronoi interpolation on the
surface.

The paper equations write weights as an exponential in distance. The official
implementation uses squared radius-normalized distance:

```text
w_ij(x) = exp(-tau_tex * ||x - world_site_ij||^2 / r_i^2)
```

Official texture temperature:

```text
tau_tex = 10
```

For any value `v_ij` stored at the detail sites, the interpolated value is:

```text
V_i(x) = sum_j w_ij(x) v_ij / sum_j w_ij(x)
```

This is used for both:

```text
height/displacement
color/radiance
```

Our current Metal texel mode implements the color interpolation part over
learned local 2D sites. It does not yet implement height/displacement.

## 9. Displacement / Height Field

Each detail site stores a scalar height:

```text
h_ij in R
```

Official code scales it by radius:

```text
height_world_ij = h_ij * r_i
```

For a ray, first intersect the base dipole plane and compute a soft-Voronoi
height:

```text
h_i(x_base) =
    sum_j w_ij(x_base) height_world_ij / sum_j w_ij(x_base)
```

Then the final displaced surface point is offset along the primitive normal:

```text
x_final = x_base + h_i(x_base) * n_i
```

The official implementation effectively recomputes the ray-plane intersection
with this height offset. The point is not just used for color. It changes the
effective surface intersection and therefore can change the segment length and
opacity.

This is a major missing piece in the current Metal path. Our texel sites affect
color; they do not yet push geometry.

## 10. Spherical Voronoi Directional Color

The user question asked "Spherical reflection?" The paper is not using
"spherical reflection" as the core color model. It uses **Spherical Voronoi**:
a soft nearest-axis basis over directions on the sphere.

PowerFoam can support ray-traced reflection/refraction because it is an explicit
cell complex, but the per-primitive color representation is Spherical Voronoi
directional radiance, not a reflection model by itself.

### 10.1 Per Detail Site View-Dependent Color

Each detail site has `sv_dof` directional basis entries. Official default:

```text
sv_dof = 8
```

For each detail site `(i,j)` and directional component `m`, official code stores
a raw axis vector:

```text
a_ijm in R^3
```

It decodes:

```text
u_ijm = a_ijm / ||a_ijm||
tau_ijm = ||a_ijm||
```

So the raw vector encodes both:

- spherical axis direction
- softness/temperature

Given view direction from camera to texel point:

```text
v = normalize(x - camera_origin)
```

Spherical weight:

```text
W_ijm(v) = exp(-tau_ijm * ||v - u_ijm||)
```

Site color:

```text
c_ij(v) =
    0.5 + sum_m W_ijm(v) c_ijm / sum_m W_ijm(v)
```

Official code clamps negative final color components to zero.

Important code detail:

```text
texel_sites.view(-1, 3).detach()
```

The official code detaches texel world positions before the Spherical Voronoi
color query. That means:

- SV axis and RGB values get color gradients.
- The geometry does not get gradient through "move the texel point to change
  view direction and therefore color."
- Geometry still gets gradients through plane/displacement/intersection and
  texture interpolation.

This detach is a reproduction-critical detail.

### 10.2 Surface Color From Detail-Site Colors

After each detail site's view-dependent color `c_ij(v)` is known, the surface
hit color is another soft Voronoi interpolation:

```text
c_i(x, v) =
    sum_j w_ij(x) c_ij(v) / sum_j w_ij(x)
```

So color is nested:

```text
direction on sphere -> per-detail-site color
surface position    -> interpolation among detail sites
```

This is why "feature foam" in our Dynaworld sense can be generalized:
instead of each detail site storing RGB, each site could store arbitrary
features, and a decoder could map features plus view/time to RGB. But official
PowerFoam's published appearance model is RGB Spherical Voronoi.

## 11. Complete Per-Primitive Parameter Inventory

For each primitive:

```text
points:
    p_i, shape [N,3]
    unconstrained in world coordinates

radii:
    raw_r_i, shape [N]
    decoded as softplus(raw_r_i, beta=100)
    affects support sphere and power face locations

density:
    raw_sigma_i, shape [N]
    decoded as softplus(raw_sigma_i, beta=100)
    controls optical thickness of high-density side

quaternions:
    raw_q_i, shape [N,4]
    decoded by L2 normalization
    determines normal/tangent/bitangent frame

texel_sites:
    s_ij, shape [N,k,2]
    local radius-normalized plane coordinates
    world_site = p + r * (u * tangent + v * bitangent)

texel_height:
    h_ij, shape [N,k]
    scaled by radius
    gives normal-direction displacement

texel_sv_axis:
    a_ijm, shape [N,k,3*sv_dof]
    reshaped to [N,k,sv_dof,3]
    direction = normalized raw axis
    temperature = norm raw axis

texel_sv_rgb:
    c_ijm, shape [N,k,3*sv_dof]
    reshaped to [N,k,sv_dof,3]
    RGB color offsets around 0.5
```

Default sizes in official indoor/DL3DV config:

```text
k = num_texel_sites = 8
sv_dof = 8
init_points = 100,000
final_points = 500,000
```

Outdoor config:

```text
init_points = 100,000
final_points = 1,200,000
```

Per primitive scalar/vector count, ignoring optimizer state:

```text
center                         3
radius                         1
density                        1
quaternion                     4
texel_sites                    2*k
texel_height                   k
texel_sv_axis                  3*k*sv_dof
texel_sv_rgb                   3*k*sv_dof
```

With `k=8`, `sv_dof=8`:

```text
3 + 1 + 1 + 4 + 16 + 8 + 192 + 192 = 417 floats per primitive
```

This is appearance-heavy. PowerFoam trades fewer primitives for richer
per-primitive texture.

## 12. Initialization

Official static-scene initialization:

```text
points:
    from SfM points, random bounded, or random unbounded depending on config

radii:
    mean distance to KNN neighbors, capped by camera-dependent max projected size

quaternions:
    random normalized quaternions

density:
    raw density initialized around 0.1

texel_sites:
    random local offsets with scale about 0.1

texel_sv_axis:
    random raw vectors with scale about 2.0

texel_sv_rgb:
    zeros, so decoded color starts near 0.5

texel_height:
    zeros, so initial surface is undisplaced
```

This is not a random-color soup. Color starts neutral and view-dependent color
is learned through SV RGB coefficients.

Our local direct/video init has been adapted for small video overfits. It is
not a literal SfM static-scene initialization contract.

## 13. Optimization And "Movement"

The paper is a static-scene method. It does not model temporal movement as a
native dynamic representation.

However, during optimization, every primitive can move or change:

```text
p_i changes                         cell/surface center moves
r_i changes                         support and power faces move
sigma_i changes                     opacity/density changes
q_i changes                         normal and texture frame rotate
s_ij changes                        detail sites slide on local plane
h_ij changes                        local surface relief changes
a_ijm changes                       directional lobes rotate/sharpen/soften
c_ijm changes                       view-dependent color changes
```

Densification and pruning also change the set of primitives:

- low-contribution cells are removed
- high-error regions are resampled/duplicated
- duplicates are perturbed tangentially to the local normal

For Dynaworld dynamic video, a natural extension is to make these parameters
functions of time:

```text
p_i(t), r_i(t), sigma_i(t), q_i(t), s_ij(t), h_ij(t), a_ijm(t), c_ijm(t)
```

But that is our dynamic-foam extension, not the paper's baseline.

## 14. Training Objective

The paper objective:

```text
L = L_rgb
  + lambda_1 L_SSIM
  + lambda_2 L_normal
  + lambda_3 L_sparse
  + lambda_4 L_connect
```

Official train loop uses:

```text
rgb_loss       = pixel MSE
ssim_loss      = 1 - SSIM
normal_loss    = renderer normal error plus optional external normal supervision
contrib_loss   = sum primitive contributions
interpenetration_loss = overlap/connectivity penalty
```

### 14.1 Normal Loss

The supplementary formula is:

```text
L_normal(P_i) =
    sum_{r in train rays} T_r alpha_r max(n_i dot d_r, 0)^2
```

This penalizes back-facing/degenerate orientations, weighted by contribution.
Only visible/contributing primitives matter strongly.

### 14.2 Sparsity / Contribution Loss

The sparsity/contribution term is:

```text
L_sparse(P_i) =
    sum_{r in train rays} T_r alpha_r
```

This suppresses floaters and low-value redundant primitives before pruning.

### 14.3 Connectivity / Interpenetration Loss

The connectivity term is:

```text
L_connect(P_i) =
    sum_{j in Cech(i)} max(r_i + r_j - ||p_i - p_j||, 0)^2
```

This discourages excessive overlap between neighboring support spheres.

This is not just aesthetic. It makes the Cech graph sparser and rendering
cheaper.

### 14.4 Loss Schedules

Official default weights:

```text
normal_weight              0.1 -> 0.01 exponential decay
contribution_weight        0.1 -> 0.0001 exponential decay
interpenetration_weight    1e-4 -> 1e-7 exponential decay
SSIM weight                0.2
```

The paper also starts with downsampled images before full resolution:

```text
first 500 iterations downsampled
then full resolution
train to 30,000 iterations
densify from 1,000 to 24,000
```

## 15. Official Learning Rates

Indoor/DL3DV default:

```text
points_lr:          1e-3  -> 5e-5
density_lr:         1e0   -> 1e0
radii_lr:           5e-5  -> 5e-6
quaternions_lr:     1e-1  -> 1e-2
texel_sites_lr:     1e-2  -> 1e-3
texel_sv_axis_lr:   5e-2  -> 5e-3
texel_sv_rgb_lr:    5e-3  -> 5e-4
texel_height_lr:    5e-3  -> 5e-4
```

Outdoor changes some appearance/height rates:

```text
points_lr_final:    1e-4
radii_lr:           1e-4  -> 1e-5
texel_sv_axis_lr:   1e-2  -> 1e-3
texel_sv_rgb_lr:    2e-3  -> 2e-4
texel_height_lr:    1e-2  -> 1e-3
```

The important engineering point is that the parameter groups are very
different. A single LR for all PowerFoam parameters is not faithful:

- density is high LR
- radii are tiny LR
- quaternions are high LR
- SV axes are high-ish LR
- SV RGB/height are lower

## 16. Densification, Pruning, Resampling

Official training keeps EMAs:

```text
contrib_ema_i       primitive visibility/contribution
point_error_ema_i   photometric error attributed to primitive
```

Pruning:

```text
remove primitives with low contribution
```

Resampling/densification:

```text
valid_indices = primitives above contribution threshold
prob_i = clamp(point_error_ema_i, max=99th percentile)
sample duplicates from valid_indices proportional to prob_i
duplicate all parameter tensors and optimizer state
perturb duplicated centers tangentially:
    direction = random vector projected perpendicular to normal
    perturb = 0.05 * r_i * direction
```

The target point count grows exponentially between `densify_from` and
`densify_until`:

```text
target(i) = init_points * a^(i - densify_from)
a = (final_points / init_points)^(1 / (densify_until - densify_from - 1))
```

This is not incidental. PowerFoam needs adaptive capacity because the landscape
is local and underfit regions need new primitives.

## 17. Rasterization Pipeline In Official Code

Official rasterization is tiled:

1. Count which primitive spheres intersect each tile.
2. Prefix-sum counts to tile offsets.
3. Write tile primitive indices and sort keys.
4. Sort by packed tile id plus power distance.
5. Prefetch adjacency relative offsets.
6. Forward kernel:
   - one thread per pixel in an 8x8 tile
   - iterate tile's sorted primitive list
   - sphere interval
   - power-face clipping against adjacency
   - dipole/detail-plane intersection
   - detail-site texture/displacement
   - alpha/transmittance accumulation
   - output color, alpha, normals, depths, contribution, point error

The paper emphasizes that rasterization and ray tracing should produce
mathematically identical results. The rasterizer is an acceleration of exact
ray-cell integration, not a splatting approximation.

## 18. Backward Pipeline

Official backward is a replay backward:

1. Forward saves compact per-tile lists, offsets, early-stop counters, color,
   log transmittance, and needed primitive buffers.
2. Backward launches one thread per pixel.
3. It walks the saved tile primitive list in reverse.
4. It recomputes interval clipping and dipole/detail intersections.
5. It propagates through alpha/transmittance recurrence.
6. It routes endpoint gradients to the active boundary:
   - sphere endpoint -> current center/radius
   - power face endpoint -> current and neighbor center/radius
   - dipole/detail plane -> point/normal/texel/height
7. It uses atomics/scatter-adds to accumulate primitive gradients.

This is the relevant memory contract:

```text
Do not store grad per primitive per pixel.
Store compact forward traversal metadata.
Replay intersections in backward.
Accumulate into primitive tensors.
```

Our Metal path follows this philosophy in the streaming version, but without
the official tile candidate acceleration and without full SV/height math.

## 19. Ray Tracing And Reflection/Refraction

PowerFoam claims unified rasterization and ray tracing. The ray-tracing side
uses the foam's adjacency graph:

```text
current cell -> check neighboring faces -> step to next cell
```

This preserves Radiant Foam's efficient neighbor-to-neighbor traversal.

The project page shows reflection/refraction demos. That capability comes from
having an explicit spatial cell complex and ray-traversable surfaces. It is not
the same as the Spherical Voronoi color basis.

In short:

```text
Spherical Voronoi:
    view-dependent radiance/color basis on outgoing direction

Reflection/refraction:
    secondary ray transport made possible by ray-traceable foam geometry
```

They are related only in that both use directions. They are not the same
mechanism.

## 20. How Our Current Metal Path Maps To Full PowerFoam

Current Metal modes:

```text
constant:
    p, r, sigma, RGB/features per cell

linear:
    p, r, sigma, local linear feature coefficients

surface_linear:
    fixed camera-facing plane, local linear feature at surface

oriented_surface_linear:
    learned normal, local linear feature at surface

oriented_texel_surface:
    learned normal, differentiable tangent/bitangent material frame,
    learned local 2D sites, per-site RGB/features, soft local interpolation
```

What we now cover:

```text
bounded power cells
support sphere interval
neighbor radical-plane clipping
front-to-back power-distance order
volume compositing
learned oriented plane normal
learned material-frame tangent/bitangent coordinates for texel lookup
learned local detail-site positions
soft Voronoi site interpolation
replay backward for points/radii/density/features/normals/frame/texel sites
```

What is missing:

```text
official quaternion frame parameterization
detail-site height/displacement
spherical-Voronoi directional color
official AABB/Cech adjacency builder
official loss stack and schedules
contribution/point-error EMAs
densification/pruning/resampling
tiled visible-list rasterizer
ray-tracing backend and Steiner points
non-pinhole generic camera acceleration
```

The latest quality jump came from adding local detail sites. That matches the
paper's ablation story: more detail sites increase quality because high-frequency
texture/detail can be represented without exploding primitive count.

## 21. Minimal Full-Paper Parameter Schema For Our Next Implementation

A faithful Metal trainer should represent one static or per-frame PowerFoam
state as:

```python
PowerFoamState:
    points:          [T?, N, 3]        # or [N,3] static scene
    raw_radii:       [T?, N]
    raw_density:     [T?, N]
    raw_quat:        [T?, N, 4]
    texel_sites:     [T?, N, K, 2]
    texel_height:    [T?, N, K]
    texel_sv_axis:   [T?, N, K, M, 3]
    texel_sv_rgb:    [T?, N, K, M, 3]
```

Decoded:

```python
radii = softplus(raw_radii, beta=100)
density = softplus(raw_density, beta=100)
quat = normalize(raw_quat)
normal, tangent, bitangent = quat_to_frame(quat)
sv_axis = normalize(texel_sv_axis)
sv_temp = norm(texel_sv_axis)
world_texel_site = point + radius * (
    texel_sites[..., 0] * tangent + texel_sites[..., 1] * bitangent
)
world_texel_height = texel_height * radius
```

Render inputs to a full Metal kernel:

```text
points
radii
density
normal/tangent/bitangent or quaternion
texel local sites
texel heights
SV axes
SV temps
SV RGB
CSR adjacency
sorted ids / tile visible lists
rays/camera
```

## 22. Pseudocode: Forward For One Ray

This is the conceptual non-tiled version:

```python
T = 1.0
C = 0.0
A = 0.0

for cell in sorted_by_power_distance(camera_origin):
    interval = ray_sphere_interval(ray, p[cell], r[cell])
    if not interval.hit:
        continue

    t0, t1 = interval.t_near, interval.t_far

    for nbr in adjacency[cell]:
        ok, t0, t1 = clip_by_power_face(
            ray,
            t0,
            t1,
            p[cell],
            r[cell],
            p[nbr],
            r[nbr],
        )
        if not ok:
            break
    if not ok:
        continue

    # Dipole/detail surface
    base_hit = intersect_plane(ray, p[cell], normal[cell])
    height = soft_voronoi_height(base_hit, texel_sites[cell], texel_height[cell])
    surf_hit = intersect_plane(ray, p[cell] + height * normal[cell], normal[cell])

    # Clip the segment to high-density side of the dipole.
    t0, t1 = clip_interval_by_dipole_side(t0, t1, surf_hit, normal[cell], ray)
    if t1 <= t0:
        continue

    # View-dependent per-site color.
    site_rgb = []
    for site in texel_sites[cell]:
        v = normalize(world_site[cell, site] - camera_origin)
        site_rgb.append(spherical_voronoi_color(v, sv_axis, sv_temp, sv_rgb))

    x_query = ray.origin + surf_hit * ray.direction
    rgb = soft_voronoi_value(x_query, texel_sites[cell], site_rgb)

    optical = density[cell] * (t1 - t0)
    alpha = 1.0 - exp(-optical)
    C += T * alpha * rgb
    A += T * alpha
    T *= exp(-optical)

    if T < transmittance_threshold:
        break
```

The real official code is tiled and optimized, but the math is this shape.

## 23. Pseudocode: Backward Replay

Conceptual reverse pass:

```python
grad_T_after = -grad_alpha_out * exp(log_T_final)

for visited_cell in reverse(saved_or_replayed_order):
    recompute sphere interval
    recompute power-face clips and endpoint identities
    recompute dipole/detail intersection
    recompute rgb and alpha

    # color path
    accumulate grad SV RGB / SV axes from spherical Voronoi
    accumulate grad texel sites from local soft Voronoi
    accumulate grad height from displaced surface

    # alpha/transmittance path
    grad_optical = ...
    grad_density += grad_optical * segment_length
    grad_t_near += ...
    grad_t_far += ...

    # endpoint routing
    if endpoint came from sphere:
        grad p_i, r_i through ray-sphere root
    if endpoint came from power face:
        grad p_i, r_i, p_j, r_j through radical-plane crossing
    if endpoint came from dipole/detail plane:
        grad p_i, quaternion/frame, texel sites, heights

    grad_T_before = ...
```

The key invariant:

```text
backward memory is O(forward compact traversal), not O(N * H * W)
```

## 24. Failure Modes To Watch

### 24.1 Wrong Power-Face Sign

Correct face offset:

```text
h_ij = 0.5 * (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)
```

The signs on `r_i^2` and `r_j^2` are easy to flip. A mirrored reference will
not catch this if both implementation and reference share the mistake.

### 24.2 Normal Without Tangent/Bitangent Is Incomplete

If we use only a normal vector, texel coordinates do not have a learned
in-plane orientation. This can work for a small front-facing fit but is not the
paper representation.

### 24.3 Missing Height Makes Detail Sites Appearance-Only

Our current oriented texel-surface mode can place color/detail on a plane, but
cannot create displaced relief. The paper's detail sites are both appearance and
geometry anchors.

### 24.4 SV Color Gradients Should Not Move Geometry Through View Query

Official code detaches texel positions before SV color lookup. If we allow
geometry gradients through the directional color query, we may get a different
and potentially unstable method.

### 24.5 KNN Adjacency Is Not Cech

KNN can miss true overlapping neighbors and add arbitrary far neighbors. Missing
a true overlapping neighbor changes geometry. A Cech superset can add false
edges safely. These are not equivalent.

### 24.6 Single LR Is Not Faithful

Radii, density, quaternion, sites, SV axes, RGB, and height have very different
scales. A single optimizer LR can make "PowerFoam failed" when the issue is only
parameter scaling.

## 25. What To Implement Next

The next paper-faithful sequence should be:

1. Add quaternion frame to Metal trainer and kernel inputs.
2. Change texel sites from normal-only local xy to full tangent/bitangent frame.
3. Add texel height/displacement and gradients.
4. Add spherical-Voronoi color:
   - `sv_axis`, `sv_temp`, `sv_rgb`
   - detach geometry for SV color query, matching official code
5. Add normal/contribution/connectivity outputs from renderer.
6. Add official-ish regularizers and schedules.
7. Add contribution/error EMAs and resampling/densification.
8. Replace KNN adjacency with Cech/AABB or a conservative overlap graph.
9. Only then spend serious effort on the tiled candidate-list kernel.

The current Metal path proves the most important memory principle:

```text
we can train foam by replaying intersections in backward,
without storing per-cell per-pixel gradients.
```

But the current quality path is still an approximation to the full paper:

```text
full PowerFoam = bounded power cells
              + quaternion dipole frame
              + displaced detail-site surface
              + spherical-Voronoi directional radiance
              + Cech adjacency
              + adaptive densification/pruning
              + official regularizers
```

## 26. Short Answer To The Original Questions

Cells?
    Bounded power cells: sphere-clipped weighted Voronoi cells.

Voronoi?
    Three levels: 3D power diagram, 2D soft Voronoi texture/displacement, and
    Spherical Voronoi directional color.

Spherical reflection?
    Not exactly. Spherical Voronoi is a view-direction color basis. Reflection
    support comes from the ray-traceable foam geometry, not from the SV basis.

Color?
    Per detail site, color is a Spherical Voronoi function of view direction.
    Then surface color is a soft Voronoi interpolation over local detail sites.

Orientation?
    Official code uses quaternions. The quaternion gives normal, tangent, and
    bitangent. The normal defines the dipole surface; tangent/bitangent define
    the local texture coordinate frame.

Movement?
    The paper is static-scene optimization. Parameters move during training,
    and primitives are resampled/densified/pruned. Dynamic motion would be a
    Dynaworld extension where parameters become functions of time.

All free params?
    `p`, `r`, `sigma`, `q`, `s`, `h`, `SV axis/temp`, `SV RGB`.

Core equations?
    Power distance, radical-plane clipping, sphere interval, dipole plane,
    soft Voronoi displacement/color, Spherical Voronoi directional color, and
    alpha compositing.
