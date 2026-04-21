# Thinking About Initialization

This note is a long-form scratchpad on camera and Gaussian initialization in
Dynaworld. It is intentionally broader than the immediate 128px NaN issue
because the goal is not just to memorize one local clip. We need initialization,
coordinate normalization, and camera encoding choices that can survive
pretraining across many scenes and camera paths.

## Why This Note Exists

The corrected 128px/4fps prebaked-camera run fails at step 1 before optimizer
update. The decoded Gaussian tensors are finite, but the renderer produces NaNs
for late frames. Debug metrics show many initialized Gaussians are behind or
near the camera for the failing 128px camera windows.

That exposed an implicit contract in the current code:

- model Gaussian coordinates live in a fixed canonical volume
- camera poses, especially DUSt3R poses, may have arbitrary scale
- the renderer assumes the camera path and Gaussian volume are mutually
  compatible

The old stable prebaked-camera baseline was the 32px/2fps all-frame run in
commit `be87e96`. It did not test the new 128px/4fps camera trajectory.

## Current Coordinate Conventions

### CameraSpec

The project represents a camera as:

```python
CameraSpec(
    fx,
    fy,
    cx,
    cy,
    camera_to_world,
)
```

Rendering transforms a world-space point `x_w` into camera coordinates using
the inverse of `camera_to_world`. In the current renderer helper this is written
as:

```python
means_camera = (means3D - translation) @ rotation_cw
```

where `rotation_cw = camera_to_world[:3, :3]` and `translation =
camera_to_world[:3, 3]`.

Given the way `build_look_at_camera_to_world(...)` constructs the basis columns
as `(right, up, forward)`, multiplying by `rotation_cw` is effectively taking
dot products against camera axes. For the default orbit camera at `(0, 0, -r)`
looking toward origin, a point at the origin has camera depth `z_c = r`.

### Projection

For a camera-space point `(x_c, y_c, z_c)`, projection is:

```text
u = fx * x_c / z_c + cx
v = fy * y_c / z_c + cy
```

The renderer currently clamps depth for projection:

```python
z_safe = clamp(z_c, min=1e-4)
```

and masks opacity for points behind or on the near plane:

```python
front_mask = (z_c > 1e-4)
opacities = opacities * front_mask
```

Important subtlety: the projection/covariance/exponent math is still evaluated
for near/behind Gaussians before opacity masking can make them harmless. If
those intermediate values overflow or become indefinite, `opacity = 0` does not
save the computation.

## Current Known-Camera DynamicTokenGS

File:

```text
src/train/gs_models/dynamic_token_gs.py
```

This model is used by the prebaked-camera trainer:

```text
src/train/dynamicTokenGS.py
```

### Inputs

For known-camera training, each frame is loaded with a DUSt3R camera. The trainer
passes:

- image frame
- frame time
- `CameraSpec`

The model builds a Plucker ray grid from the provided camera:

```python
plucker_grid = build_plucker_ray_grid_batch(camera, image_size=width)
grounded_feature_map = image_encoder(image) + ray_proj(plucker_grid)
```

So known-camera DynamicTokenGS uses camera pose/intrinsics as conditioning
through Plucker rays.

### Gaussian Head Range

Current known-camera Gaussian head:

```python
xyz_raw = xyz_head(tokens)
xyz = [
    tanh(x_raw) * 1.5,
    tanh(y_raw) * 1.5,
    sigmoid(z_raw) * 2.0 + 0.5,
]
scales = exp(scale_raw) * 0.05
quats = normalize(rot_raw)
opacities = sigmoid(opacity_raw)
rgbs = sigmoid(rgb_raw)
```

Therefore the initial and reachable output coordinate range is:

```text
x, y in [-1.5, 1.5]
z in [0.5, 2.5]
scale_i > 0, roughly around 0.05 at raw 0
opacity in [0, 1], around 0.5 at raw 0
rgb in [0, 1], around 0.5 at raw 0
```

This is not a general world coordinate system. It is an implicit canonical
volume assumption: visible content is expected to be somewhere in front of the
first camera in roughly this depth range.

### Prebaked Camera Loading

File:

```text
src/train/sequence_data.py
```

Known-camera data loading does:

```python
base_pose_inv = inverse(records[0]["camera_to_world"])
pose = base_pose_inv @ pose
```

So all DUSt3R poses become relative to the first frame. This removes global
camera pose but does not normalize translation scale. DUSt3R can choose a pose
scale that is convenient for its reconstruction objective. That scale is not
guaranteed to match `[0.5, 2.5]` Gaussian depth.

Intrinsics are scaled from DUSt3R input pixels to training/render pixels:

```python
scale = target_size / camera_image_size
fx_train = fx_dust3r * scale
fy_train = fy_dust3r * scale
cx_train = cx_dust3r * scale
cy_train = cy_dust3r * scale
```

This part is a viewport conversion. It should preserve FoV.

## Current Implicit-Camera Model

Files:

```text
src/train/gs_models/implicit_camera.py
src/train/gs_models/dynamic_token_gs_implicit_camera.py
src/train/gs_models/dynamic_video_token_gs_implicit_camera.py
```

The implicit camera model uses a different camera scheme from the prebaked
known-camera model.

### Global Camera Head

Global camera defaults:

```python
base_fov_degrees = 60
base_radius = 3
max_fov_delta_degrees = 15
max_radius_scale = 1.5
```

The head is zero-initialized:

```python
raw = zero_init_mlp(camera_token)
fov = base_fov + tanh(raw[0]) * max_fov_delta
radius = base_radius * exp(tanh(raw[1]) * log(max_radius_scale))
```

At initialization:

```text
fov = 60 deg
radius = 3
```

Allowed global range:

```text
fov in [45 deg, 75 deg]
radius in [3 / 1.5, 3 * 1.5] = [2, 4.5]
```

The base camera is:

```python
make_orbit_camera(
    radius=radius,
    azimuth=0,
    elevation=0,
)
```

With current `make_orbit_camera`, azimuth 0 and elevation 0 gives:

```text
camera position = (0, 0, -radius)
camera forward = +z toward origin
```

So yes: the implicit camera scheme moves the camera away from the origin and
points it at the origin.

### Path Camera Head

Path head defaults:

```python
max_rotation_degrees = 5
max_translation_ratio = 0.2
```

It is also zero-initialized:

```python
raw = zero_init_mlp(path_token)
rotation_delta = tanh(raw[:3]) * 5 deg
translation_delta = tanh(raw[3:]) * (base_radius * 0.2)
```

At initialization:

```text
rotation_delta = 0
translation_delta = 0
```

With `base_radius = 3`, each translation component is bounded by:

```text
[-0.6, 0.6]
```

The vector norm is bounded by:

```text
sqrt(3) * 0.6 = 1.039
```

The camera transform composition is:

```python
composed = base_camera.camera_to_world @ delta_transform
```

Because the delta is right-multiplied, the path translation lives in base-camera
local axes. For the default base camera, local axes match world axes, but this
matters if base azimuth/elevation become learned later.

### Is The Camera Path A Smooth Chain?

No. Current implicit camera paths are not represented as a recursive chain like:

```text
C_t = C_{t-1} * delta_t
```

Instead, each frame predicts a direct residual from the same base camera:

```text
C_t = C_base * exp(delta_t)
```

The path head is conditioned by a path token and time features. Smoothness comes
from:

- the shared network/function over time
- zero initialization
- bounded `tanh` output
- optional temporal smoothness loss on adjacent predicted deltas

It is not guaranteed by the parameterization itself.

This is probably okay for overfit experiments. For pretraining, this choice is
important:

- Direct-from-base residuals avoid accumulated drift.
- Recursive chains model physical incremental motion but can accumulate errors.
- Direct functions can represent jumps and sharp turns more easily.
- Low-degree polynomial/spline paths impose smoothness but can underfit sudden
  handheld motion or cuts.

## Image-Implicit vs Video-Implicit Camera Path

### Image-Implicit

`DynamicTokenGSImplicitCamera` processes each frame image and time.

Token layout:

```text
token 0: global camera token
token 1: path camera token
tokens 2..: Gaussian tokens
```

The code creates time offsets:

```python
token_offsets[:, 1:, :] = time_proj(frame_times)
```

So the global camera token is not time-conditioned, but the path token and
Gaussian tokens are.

At forward:

```python
global_camera_token = mean(refined_tokens[:, 0, :])
splat_tokens = refined_tokens[:, 2:, :]
path_tokens = refined_tokens[:, 1, :]
base_camera = global_camera_head(global_camera_token)
delta_t = path_camera_head(path_tokens)
C_t = C_base * delta_t
```

This means global camera is shared across the batch, while path is per frame.

### Video-Implicit

`DynamicVideoTokenGSImplicitCamera` encodes a clip, then decodes each requested
time.

Token layout is again:

```text
token 0: global camera token
token 1: path camera token
tokens 2..: Gaussian tokens
```

For each decode time:

```python
time_offset = time_proj(t)
global_camera_token = refined_queries[:, 0, :]
path_token = refined_queries[:, 1, :] + time_offset
gs_tokens = refined_queries[:, 2:, :] + time_offset
```

The global camera token is clip-conditioned but not time-conditioned. The path
and Gaussian tokens are time-conditioned.

Because `time_proj` is an MLP, path smoothness is learned rather than explicit.
For normalized times in `[0, 1]`, an MLP with SiLU is continuous and smooth as a
function, but it can still learn high curvature. The temporal loss is currently
the main explicit preference for smoothness.

## Canonical Gaussian Initialization

There are two relevant Gaussian initialization styles.

### Known-Camera DynamicTokenGS

Known-camera z is constrained to positive depth:

```text
x, y in [-1.5, 1.5]
z in [0.5, 2.5]
```

This is camera-forward biased. It makes sense if the first camera is the
canonical view and all other cameras stay near it.

### Implicit-Camera Canonical Heads

Implicit camera heads use:

```python
xyz = tanh(raw_xyz) * scene_extent
```

So with `scene_extent = 1`:

```text
x, y, z in [-1, 1]
```

For the default implicit camera at `(0, 0, -3)` looking toward the origin, the
camera-space depth of this cube is approximately:

```text
z_camera in [2, 4]
```

because camera depth is roughly `z_world + radius`.

This is a cleaner object-centric coordinate system than the known-camera
positive-depth range. The camera starts outside the canonical scene and points
at it.

## Range Proofs And Safety Conditions

### Base Orbit Camera Safety

Assume:

- canonical object points lie inside a sphere of radius `S`
- camera base radius is `r`
- path translation vector has norm at most `T`
- camera points roughly toward origin
- ignore rotation residual for the simplest sufficient bound

The minimum possible camera depth is lower bounded by:

```text
z_min >= r - T - S
```

For implicit defaults:

```text
scene_extent = 1 cube -> enclosing sphere S = sqrt(3) = 1.732
base_radius = 3
max per-axis translation = 0.6
T <= sqrt(3) * 0.6 = 1.039
```

Then:

```text
z_min >= 3 - 1.039 - 1.732 = 0.229
```

This is above zero, but not by a lot. A near plane of `1e-4` still technically
passes, but numerically a larger practical margin would be healthier.

If we use the cube z extent instead of the enclosing sphere and assume no
rotation coupling:

```text
z_min >= r - translation_forward_max - scene_z_extent
      >= 3 - 0.6 - 1
      = 1.4
```

That is much better. The sphere bound is conservative because it allows any
point in the cube to align with the camera forward axis after arbitrary
rotation.

### Known-Camera DUSt3R Failure Condition

For the known-camera model, initialized points have:

```text
z_world in [0.5, 2.5]
```

If a relative camera has approximately the same orientation as the first camera
and translation `t_z`, then:

```text
z_camera ~= z_world - t_z
```

To keep all initialized Gaussians in front:

```text
min(z_world) - t_z > near
```

With `min(z_world) = 0.5`, this requires:

```text
t_z < 0.5
```

To keep at least some deepest initialized Gaussians in front:

```text
max(z_world) - t_z > near
```

With `max(z_world) = 2.5`, this requires:

```text
t_z < 2.5
```

The corrected 128px/4fps bake has late camera translations around:

```text
t_z ~= 2.4 to 2.56
```

Therefore the late cameras can be at or past the deepest initialized Gaussians.
This exactly matches the debug metrics: most Gaussians are near/behind camera
for frames `35-45`.

### Why The Renderer NaNs

Mathematically, if `cov3D` is positive semidefinite and the projection Jacobian
is finite, then:

```text
cov2D = J cov3D J^T
```

should also be positive semidefinite.

But for near/behind Gaussians:

- `z_safe = 1e-4`
- projected means can be millions of pixels away
- Jacobian entries can be enormous
- `cov2D` entries can reach `1e18` or higher
- determinant involves subtracting large products:

```text
det = a*d - b*c
```

This is numerically fragile. Even if the exact determinant should be
non-negative, floating point cancellation can make it huge-negative.

The current code clamps determinant only after computing it:

```python
det = clamp(raw_det, min=1e-6)
inv_cov = adjugate(cov2D) / det
```

If `cov2D` is huge and the raw determinant was effectively invalid, the inverse
covariance can become indefinite. The Gaussian exponent:

```text
power = -0.5 * dx^T inv_cov dx
```

can become huge positive, e.g. `1e31`. Then:

```text
alpha = opacity * exp(power)
```

For culled Gaussians `opacity = 0`, but `exp(power) = inf`, so:

```text
0 * inf = NaN
```

This proves the immediate failure is renderer/projection numeric instability,
but the underlying cause is a camera/scene scale mismatch.

## What Changed And What Broke

The committed stable baseline was:

```text
32px / 2fps / 23 frames / all frames per step
local_mac_overfit_prebaked_camera.jsonc at be87e96
```

It used `test_data/dust3r_outputs/test_video_small_all_frames`.

The new variants are:

```text
64px / 4fps / 46 frames
128px / 4fps / 46 frames
```

These were correctly regenerated from the original 30fps source and DUSt3R was
rerun. The duplicate-frame mistake is fixed.

The new 128px DUSt3R camera solve is not just "the same camera path at higher
resolution." It has:

```text
raw median fx: 662.79 at DUSt3R 224 input
median FoV: 19.18 deg
relative camera z translation up to about 2.56
```

The 64px solve has:

```text
raw median fx: 602.65
median FoV: 21.06 deg
relative camera z translation about [-0.28, 0.11]
```

The old 32px solve has:

```text
raw median fx: 396.68
median FoV: 31.53 deg
```

So the 128px failure is a new camera distribution problem.

## Camera Encoding: What We Currently Do

### Known Cameras

Known-camera DynamicTokenGS encodes camera through Plucker rays:

```text
ray direction d
ray moment m = origin x d
plucker = [d, m]
```

These 6 channels are projected by a `1x1` conv and added to image features.

Important scale issue:

- ray directions are unit scale
- moments scale with camera origin magnitude

If camera translations are arbitrary DUSt3R scale, Plucker moments are arbitrary
scale too. This means known-camera conditioning can change distribution just
because DUSt3R chose a different reconstruction scale.

For pretraining, Plucker moments probably need a normalization contract:

```text
m_normalized = (origin / scene_scale) x d
```

or equivalent.

### Implicit Cameras

Image-implicit and video-implicit camera models do not currently feed predicted
Plucker grids back into the Gaussian decoder in the same way. They predict:

- canonical scene Gaussians
- base camera parameters
- per-frame camera residuals

Then render with the predicted cameras.

This encourages an object-centric scene representation, which is probably the
right direction for pretraining. But it also means the model must learn camera
geometry mostly through reconstruction loss and camera regularizers, not through
explicit per-pixel ray conditioning.

## Should Camera Start As Raw Plucker Pointed At Origin?

There are two different uses of Plucker rays:

1. Conditioning the network with the camera/ray geometry.
2. Defining the render camera.

For implicit cameras, the render camera already starts as:

```text
position = (0, 0, -radius)
look at origin
```

This is equivalent to a raw canonical camera pointed at origin.

The question is whether the decoder should also see a Plucker grid for this
camera, then learn camera motion. Possible designs:

### Design A: Canonical Object, Camera Only In Renderer

Current implicit-camera direction.

```text
video/image -> canonical Gaussians
video/image -> camera path
render(canonical Gaussians, camera path)
```

Pros:

- object-centric representation
- less chance of camera leaking into geometry
- aligns with novel-view and pretraining goals

Cons:

- harder optimization, because camera and geometry are only coupled through
  render loss
- no per-pixel ray geometry available to the image encoder

### Design B: Feed Predicted Plucker Rays Into Decoder

```text
predict base/path camera
build Plucker rays
condition Gaussian decoder on image/video features + rays
render
```

Pros:

- ray-aware features can help localize geometry
- similar to known-camera DynamicTokenGS

Cons:

- risk of view-dependent memorization
- if camera prediction is wrong early, ray features are wrong too
- moments need normalization

### Design C: Use Canonical Base Plucker Plus Residual Motion Tokens

```text
base orbit camera Plucker grid
path residual tokens
Gaussian decoder gets canonical ray prior and motion features
```

Pros:

- stable initialization
- explicit camera prior
- less arbitrary than raw DUSt3R rays

Cons:

- may bias model toward orbit-like paths
- needs careful handling for non-orbit camera motion

My current bias: for pretraining, keep an object-centric canonical scene, but
make camera normalization explicit and optionally provide normalized Plucker
conditioning. Do not let arbitrary DUSt3R scale leak unnormalized into either
the renderer or the ray features.

## MLP Time Path vs Polynomial/Spline Path

Current path is MLP-conditioned:

```text
delta_t = path_head(path_token + time_proj(t))
```

or in image-implicit:

```text
path_token_t = refined path token for frame t
```

This is not a chain and not a spline. It is a learned function from features and
time to SE(3) residuals.

### MLP Advantages

- supports sharp turns
- supports non-polynomial paths
- can condition on visual content
- no accumulated drift from recursive updates
- easy to pretrain across many path families

### MLP Weaknesses

- smoothness is not guaranteed
- extrapolation outside trained time ranges is weak
- can hide high-frequency camera jitter
- temporal regularizer only applies where sampled

### Polynomial Basis

A polynomial camera path might be:

```text
delta(t) = a0 + a1 t + a2 t^2 + ...
```

Pros:

- naturally smooth
- compact
- interpretable

Cons:

- poor at sudden turns
- high-degree polynomials can oscillate
- global support means one local correction affects all times

### Spline Or Control-Point Basis

Use low-frequency control points:

```text
delta(t) = spline(control_points, t)
```

Pros:

- smooth with local control
- can bound velocity/acceleration
- good for handheld-ish paths

Cons:

- needs chosen number of controls
- harder with variable frame count
- still bad for cuts unless segmentation exists

### Hybrid

Maybe best long term:

```text
delta(t) = smooth_path_basis(t) + bounded_residual_mlp(t, features)
```

This lets the model represent common smooth motion while preserving capacity
for sharp turns. For pretraining this is attractive because the basis gives a
stable inductive bias and the residual supports diversity.

## What Should Be Normalized?

This is the core issue.

### 1. Camera Translation Scale

DUSt3R pose scale is arbitrary. Before feeding known cameras to training, we
probably need a canonical scale.

Possible normalization:

```text
camera_centers = translations after first-frame normalization
path_radius = robust_percentile(norm(camera_centers - center))
scale = target_camera_radius / max(path_radius, eps)
translations *= scale
```

But this alone is not enough. Need decide whether to also scale reconstructed
scene/depth, Gaussian init range, and Plucker moments.

### 2. Camera Center

Current known-camera normalization sets the first camera to identity. That is
not the same as centering the camera path around the scene.

Options:

- first-camera canonicalization: simple, current behavior
- path-centroid canonicalization: center camera trajectory
- scene-centroid canonicalization: center DUSt3R point cloud
- look-at-origin canonicalization: rotate/translate so scene center is origin
  and initial camera sees it

For pretraining, scene-centric canonicalization is likely better than
first-camera-only. First-camera-only can put the object anywhere relative to
later cameras.

### 3. Scene Scale

Gaussian heads have `scene_extent`. Known-camera head hard-codes:

```text
x,y extent 1.5
z extent 0.5..2.5
```

Implicit-camera heads use:

```text
scene_extent = 1
```

Pretraining needs a consistent relation between:

```text
scene_extent
camera radius
camera motion scale
near plane
Gaussian scale prior
Plucker moment scale
```

The invariant should be something like:

```text
most scene content lies in a unit-ish ball
default camera radius is about 3 scene radii
camera motion residuals are a small fraction of radius
near plane remains comfortably below min visible depth
```

### 4. Focal/FoV

DUSt3R focal estimates differ by variant:

```text
32px/2fps: median FoV 31.5 deg
64px/4fps: median FoV 21.1 deg
128px/4fps: median FoV 19.2 deg
```

Narrow FoV itself is not a NaN cause, but it magnifies projection sensitivity:

```text
u = fx * x / z
du/dz = -fx * x / z^2
```

Higher `fx` and small `z` are a bad combination. So camera normalization should
track both pose scale and focal scale.

### 5. Plucker Moment Scale

Plucker moment:

```text
m = o x d
```

If origin `o` is arbitrary scale, moment scale is arbitrary too.

For pretraining:

```text
o_normalized = (o - scene_center) / scene_scale
m = o_normalized x d
```

This makes ray conditioning comparable across scenes.

### 6. Time Scale

Current `frame_times` are normalized to `[0, 1]` for the loaded sequence or
clip. That is good for a single clip, but pretraining may need more:

- absolute FPS or frame delta as metadata
- normalized time for path shape
- physical time for velocity/acceleration losses

If frame rate varies, the same normalized `t` can represent different motion
speeds. For reconstruction over fixed frames this may be okay. For path priors,
it matters.

## Initialization Questions To Explore

### Question 1: Should Known-Camera Model Use Object-Centric XYZ?

Known-camera DynamicTokenGS uses positive depth `[0.5, 2.5]`, unlike implicit
models that use object-centric `[-E, E]`.

If known-camera data is normalized to a base orbit/scene-centric coordinate
system, maybe known-camera Gaussians should also use:

```text
xyz = tanh(raw) * scene_extent
```

and the first camera should be placed outside the scene.

That would align known-camera and implicit-camera pretraining.

### Question 2: Should Camera Pose Normalization Be Data-Level Or Model-Level?

Data-level normalization:

- canonicalizes every training item before model sees it
- simpler for pretraining
- makes Plucker moments stable

Model-level normalization:

- lets model learn scale
- may preserve real metric-like information when available
- harder to train

For DUSt3R pseudo-labels, data-level normalization seems safer because DUSt3R
scale is not metric.

### Question 3: Should The Renderer Be Robust Even If Normalization Fails?

Yes. Even with perfect normalization, pretraining will see bad camera estimates,
bad predictions, and outliers.

The renderer should not produce NaNs for culled Gaussians. This is separate from
the modeling fix.

Diagnostic fact:

```text
opacity mask after projection is too late to prevent 0 * inf
```

Potential robust renderer principles:

- cull behind/near Gaussians before expensive projection exponent math
- clamp exponent before `exp`
- avoid indefinite inverse covariance
- track culling metrics
- never let invalid invisible Gaussians poison visible pixels

But a robust renderer should not be used to hide a bad camera-scale contract.
Both are needed.

### Question 4: Should Initialization Be Randomized For Pretraining?

If every sequence starts with:

```text
camera radius = 3
scene_extent = 1
fov = 60
```

the model may overfit to that canonical camera. But if normalization maps
training scenes into that canonical frame, this is a feature, not a bug.

Potential strategy:

- canonicalize all scenes to object radius roughly 1
- canonicalize cameras to radius roughly 2.5 to 4
- randomize small global rotations or first-camera choice during pretraining
- expose normalized intrinsics/FoV explicitly

This gives invariance without making the model solve arbitrary scale from
scratch.

## Current Failure Reinterpreted As A Contract Violation

The 128px failure is not "the model initialized badly" in a random sense. It is
a contract violation:

```text
model expects visible content in its initialized canonical volume
camera path places camera through/past that volume
renderer assumes invisible/behind Gaussians remain numerically harmless
```

All three are involved:

1. DUSt3R pose scale is arbitrary and unnormalized.
2. Gaussian initialization depth range is fixed.
3. Renderer culling happens too late for numerical safety.

The first two are modeling/data normalization issues. The third is renderer
robustness.

## What I Would Measure Next Before Fixing

No fix in this note, but these are the metrics that should guide fixes.

For each camera bake/config:

```text
camera center min/max/median
camera center norm percentiles
camera z translation range after first-frame normalization
focal/FoV percentiles
point-cloud depth percentiles if DUSt3R points are available
fraction of initialized Gaussian volume visible per frame
min camera-space depth of initialized Gaussian box/corners
Plucker moment min/max after current encoding
```

For model initialization:

```text
initial xyz range
initial camera-space z range per frame
initial front Gaussian count per frame
initial power max per frame
initial alpha nonfinite count per frame
initial render nonfinite count per frame
```

For implicit-camera training:

```text
base radius
fov
translation_delta norm
rotation_delta norm
min depth of canonical scene corners under predicted cameras
temporal first/second differences of camera residuals
```

## Candidate Normalization Invariants

These are not chosen yet, but they are plausible invariants.

### Invariant A: Scene Unit Ball

```text
DUSt3R point cloud robust radius -> 1
camera centers scaled by same factor
Gaussian scene_extent = 1
base camera radius target = 3
```

Good for object-centric pretraining.

### Invariant B: Camera Radius Target

```text
median camera-center distance to scene center -> 3
scene scaled by same factor
```

Good if camera path is more reliable than point cloud bounds.

### Invariant C: First-Camera View Frustum Fit

Scale scene/camera so DUSt3R points in first camera mostly occupy:

```text
z in [1, 4]
x/z and y/z inside visible frustum
```

Good for stable rendering, but may overfit to first view.

### Invariant D: Hybrid Robust Scale

Use both point-cloud radius and camera path radius:

```text
scene_scale = robust max(point_radius, camera_path_radius / target_ratio)
```

This avoids pathological camera path scale overwhelming scene scale.

## Pretraining Implications

For one-scene overfit, many bad choices can still work if initialization happens
to be close. For pretraining, we need a contract the model can rely on.

The contract should probably be:

```text
1. Scenes are canonicalized into an object-centric coordinate frame.
2. Camera paths are normalized into that same frame.
3. Intrinsics/FoV are preserved as real viewport geometry.
4. Ray/Plucker encodings use normalized camera origins.
5. Camera path parameterization starts as a safe orbit-like prior.
6. The model can learn residual camera paths from visual evidence.
7. Renderer is numerically safe for outliers and bad early predictions.
```

This lets pretraining learn general geometry and camera behavior instead of
memorizing a particular DUSt3R scale or local clip initialization.

## Tentative Theory

The cleanest long-term formulation may be:

```text
canonical scene coordinates:
    object center = 0
    object radius ~= 1

base camera:
    look at origin
    radius ~= 3
    fov from normalized/inferred intrinsics

camera path:
    C_t = C_base * residual(t, visual_features)
    residual bounded relative to radius
    optional smooth basis + MLP residual

known camera data:
    DUSt3R poses normalized into this canonical frame
    Plucker origins normalized by scene scale

implicit camera data:
    same canonical frame and priors

renderer:
    robustly culls invalid Gaussians before exponent math
```

The important distinction is that normalization is not just an initialization
trick. It is the shared coordinate contract that allows pretraining across
different scenes, source resolutions, DUSt3R solves, and camera paths.

## Open Questions

- Should known-camera and implicit-camera models share the same object-centric
  Gaussian head range?
- Should positive-depth `[0.5, 2.5]` be retired in favor of `scene_extent` and
  canonical cameras?
- Should camera path residuals be direct-from-base, spline-based, or hybrid?
- Should camera path smoothness be enforced through architecture, loss, or both?
- How much Plucker conditioning helps pretraining vs encourages view-specific
  memorization?
- Should Plucker moments always be normalized by scene scale?
- Should DUSt3R point clouds be used to canonicalize scale, or only camera
  centers/intrinsics?
- Should each training example randomly choose a canonical reference camera, or
  always use the first frame?
- What distribution of camera radii/FoVs should pretraining expose?
- Can render diagnostics become part of dataset validation before training?

No code fix was applied as part of this note.

## Expansion Pass: Initialization Is A Gauge Choice

The first version of this note already had the local story: some 128px DUSt3R
camera windows put many of the initial Gaussians near or behind the camera, and
the dense renderer computes unstable projection math before opacity masking can
hide those Gaussians.

This expansion tries to make the larger issue explicit:

```text
initialization = choosing a gauge for an underdetermined reconstruction problem
```

The word "gauge" matters. In a monocular video, many 3D coordinate systems
produce exactly the same pixels. If we do not choose one coordinate system
deliberately, the dataset/tool/model/renderer stack chooses one accidentally.

For a single overfit run, accidental gauges can still work. The model only has
to adapt to one scene, one camera solve, one renderer, and one initialization
seed. For pretraining, accidental gauges become label noise in geometry space.
The model sees the same visual problem expressed in many unrelated coordinate
systems and has to spend capacity undoing that randomness.

The current code already contains several gauge choices:

```text
known-camera Gaussian head:
    x,y in [-1.5, 1.5]
    z in [0.5, 2.5]

implicit-camera Gaussian head:
    x,y,z in [-scene_extent, scene_extent]

implicit-camera base camera:
    camera center = (0, 0, -radius)
    target = origin
    radius initially 3

implicit-camera path:
    residual SE(3) composed after base camera
    translation bounded by radius * 0.2 per axis

DUSt3R known-camera data:
    first frame becomes identity-ish reference
    no translation scale normalization
```

These choices are individually reasonable, but they do not yet form one explicit
contract. The note below is the attempt to spell out what the contract might be.

## Notation

Use these symbols consistently:

```text
X_w       point in world coordinates
X_c       point in camera coordinates
C_w       camera center in world coordinates
R_cw      rotation with camera axes as columns in world coordinates
T_cw      4x4 camera_to_world transform
R_wc      inverse rotation = R_cw.T
K         intrinsics matrix
f_x,f_y   focal lengths in pixels
c_x,c_y   principal point in pixels
u,v       pixel coordinates
z_c       camera-space depth, positive in front of camera
d_w       world-space ray direction, unit length
m_w       Plucker moment = C_w cross d_w
s         scene scale
mu        scene center
```

The code stores `camera_to_world`, not `world_to_camera`.

The ideal transform is:

```text
X_w = R_cw X_c + C_w
X_c = R_cw.T (X_w - C_w)
```

Projection:

```text
u = f_x X_c.x / X_c.z + c_x
v = f_y X_c.y / X_c.z + c_y
```

Valid projective geometry requires:

```text
X_c.z > near
```

The renderer's current near plane is effectively:

```text
near = 1e-4
```

That is a numerical guard, not a modeling prior. A modeling prior should keep
most initialized mass much farther away than `1e-4`.

## Similarity Gauge Invariance

Monocular reconstruction is ambiguous up to a global similarity transform.

Let a scene and camera set be:

```text
points:        X_i
cameras:       C_t, R_t
intrinsics:    K_t
```

Apply a global transform:

```text
X_i' = s A X_i + b
C_t' = s A C_t + b
R_t' = A R_t
```

where:

```text
s > 0
A is any 3x3 rotation matrix
b is any 3-vector translation
```

Camera-space coordinates after the transform:

```text
X_c' = R_t'.T (X_i' - C_t')
     = (A R_t).T (s A X_i + b - s A C_t - b)
     = R_t.T A.T s A (X_i - C_t)
     = s R_t.T (X_i - C_t)
     = s X_c
```

Projection after the transform:

```text
u' = f_x (s X_c.x) / (s X_c.z) + c_x = u
v' = f_y (s X_c.y) / (s X_c.z) + c_y = v
```

So all global translations, rotations, and positive scales produce identical
pixels if cameras and points transform together.

This proof is the core reason raw DUSt3R scale is not a reliable training
coordinate. DUSt3R can output a valid camera solve in a scale that is unrelated
to our Gaussian head's fixed depth range. That solve can be perfectly good
projectively and still hostile to our renderer initialization.

## What "Move The Camera Away From Origin And Point At Origin" Means

The implicit-camera model has a clean base prior:

```text
C_base = (0, 0, -r)
target = (0, 0, 0)
forward = normalize(target - C_base) = (0, 0, 1)
right = cross(up, forward) = (1, 0, 0)
camera_up = cross(forward, right) = (0, 1, 0)
```

So:

```text
R_cw = [right, up, forward]
     = identity
```

for azimuth 0, elevation 0.

Then a point at origin has:

```text
X_c = R_cw.T (0 - C_base)
    = (0, 0, r)
```

Depth is exactly the radius. If `r = 3`, the origin is safely in front of the
camera at depth 3.

A point in an object-centric ball:

```text
||X_w|| <= E
```

has camera depth:

```text
z_c = r + X_w.z
```

for this initial camera orientation. The worst case is `X_w.z = -E`, so:

```text
z_min = r - E
```

Safety condition:

```text
r > E + near_margin
```

For the implicit defaults:

```text
r = 3
E = 1
z_min = 2
```

That is a very safe initial relationship.

For the known-camera prebaked model, the equivalent relationship is not
constructed by the model. The camera comes from data. The Gaussian head emits
points in:

```text
x,y in [-1.5, 1.5]
z in [0.5, 2.5]
```

If first-frame camera is identity and later-frame camera centers stay modest
relative to that volume, this works. If a later camera moves forward to or past
the Gaussian cloud, this fails.

## Direct Residual Path vs Chained Path

The current implicit `PathCameraHead` predicts one residual transform per frame:

```text
raw_t = MLP(path_token_t)
omega_t = tanh(raw_t[0:3]) * max_rotation
tau_t = tanh(raw_t[3:6]) * base_radius * max_translation_ratio
Delta_t = SE3(omega_t, tau_t)
T_t = T_base Delta_t
```

This means frame `t` is not computed from frame `t-1`.

It is:

```text
direct-from-base:
    T_t = T_base Delta(t)
```

not:

```text
chained:
    T_t = T_{t-1} Delta(t)
```

Consequences:

```text
direct-from-base:
    bounded absolute camera deviation
    no drift accumulation by construction
    temporal smoothness is learned/regularized, not guaranteed
    sharp turns are easy if MLP outputs sharp changes
    physical velocity/integration interpretation is weak

chained:
    natural motion interpretation
    local deltas can be small
    smoothness can be enforced on deltas
    drift accumulation is a real failure mode
    training errors early in sequence can poison later poses
```

For pretraining, direct-from-base is attractive because each frame stays inside a
known safety box. It is less expressive for long camera paths unless the max
translation/rotation grows or the base is updated.

One hybrid is:

```text
T_t = T_base ControlSpline(t) ResidualMLP(t)
```

where `ControlSpline(t)` is a smooth low-frequency path and `ResidualMLP(t)` is
small. This separates global path shape from high-frequency correction.

## SE(3) Composition Details

The code composes:

```text
T_t = T_base Delta_t
```

where `Delta_t` contains:

```text
R_delta = Exp(omega_t)
tau_t   = translation_delta
```

Because multiplication is on the right, `tau_t` is expressed in the base
camera's local coordinate frame before being mapped to world coordinates:

```text
C_t = C_base + R_base tau_t
R_t = R_base R_delta
```

At initial azimuth/elevation:

```text
R_base = I
C_t = (0,0,-r) + tau_t
```

At another base orbit:

```text
world translation = R_base tau_t
```

This matters for interpreting `max_translation_ratio`. A `tau_z` residual moves
along the base camera's forward axis, not necessarily global z.

If the camera is looking at the origin, positive local z moves the camera toward
the target:

```text
C_base = (0,0,-r)
forward = (0,0,1)
tau = (0,0,+a)
C = (0,0,-r+a)
distance to origin = r-a
```

negative local z moves it away:

```text
tau = (0,0,-a)
C = (0,0,-r-a)
distance to origin = r+a
```

The current bound at init is:

```text
a <= 0.2 r
```

So for `r = 3`:

```text
tau_i in [-0.6, 0.6]
```

If the object extent is 1, the worst camera-forward move gives:

```text
distance to origin = 2.4
front of object depth ~= 2.4 - 1 = 1.4
```

still safe.

This is an implicit proof that the implicit-camera default range is numerically
safe at initialization.

## Known-Camera Failure As A Missing SE(3) Bound

Known-camera mode does not have the above bound.

It receives:

```text
T_t = inverse(T_0_raw) T_t_raw
```

where raw poses come from DUSt3R.

This fixes:

```text
T_0 = I
```

but it does not fix:

```text
max_t ||C_t||
camera path radius relative to Gaussian cloud
near/far relation
```

The failure condition for a Gaussian `X` and camera `t` is:

```text
z_c(t, X) = r_t.forward dot (X - C_t) <= near
```

Since initialized `X` is model-produced and camera `C_t` is data-produced, both
sides need a shared scale. We currently have:

```text
X scale: fixed by Gaussian head
C_t scale: raw DUSt3R relative scale
```

That is the contract break.

## Intrinsics And FoV Math

For square image size `S` and symmetric focal `f`:

```text
f = 0.5 S / tan(0.5 fov)
fov = 2 atan(0.5 S / f)
```

The implicit-camera default:

```text
fov = 60 deg
f = 0.5 S / tan(30 deg)
  = 0.8660254 S
```

At `S=64`:

```text
f ~= 55.43 px
```

At `S=128`:

```text
f ~= 110.85 px
```

DUSt3R corrected outputs observed much narrower FoV:

```text
64/4fps raw median fx ~= 602.65 before scaling to training viewport
FoV ~= 21.06 deg at DUSt3R crop size

128/4fps raw median fx ~= 662.79 before scaling to training viewport
FoV ~= 19.18 deg at DUSt3R crop size
```

If scaled to `S=128`, a 19.18 degree FoV has:

```text
f = 0.5 * 128 / tan(9.59 deg)
  ~= 379 px
```

That is a narrow camera. Narrow FoV does not directly cause behind-camera
NaNs. It amplifies projection scale:

```text
u = f x / z
du/dx = f / z
du/dz = -f x / z^2
```

For large `f` and small `z`, tiny depth errors produce huge pixel/covariance
changes.

So narrow FoV is a multiplier on the real issue:

```text
near/behind Gaussians + high focal = explosive projection math
```

## Gaussian Projection Covariance Intuition

The dense renderer projects a 3D Gaussian to a 2D Gaussian. At a high level:

```text
Sigma_2d = J Sigma_3d J.T
```

where `J` is the Jacobian of perspective projection.

For:

```text
u = f_x x / z
v = f_y y / z
```

the Jacobian wrt camera coordinates is:

```text
J = [
    f_x / z,       0, -f_x x / z^2
          0, f_y / z, -f_y y / z^2
]
```

As `z -> 0+`, terms scale as:

```text
f / z
f x / z^2
```

The covariance can scale like:

```text
O(1 / z^2)
O(x^2 / z^4)
```

If `z` is clamped to `1e-4`, then:

```text
1 / z = 1e4
1 / z^2 = 1e8
1 / z^4 = 1e16
```

With focal in the hundreds:

```text
f / z ~= 1e6 to 1e7
f x / z^2 ~= 1e10 to 1e11
```

This is enough to create enormous 2D covariance entries. Inverting a bad or
indefinite 2D covariance then creates enormous inverse covariance entries.

The renderer exponent:

```text
power = -0.5 dx.T invCov2D dx
```

should be non-positive for a positive semidefinite inverse covariance. If
`invCov2D` becomes indefinite, `dx.T invCov2D dx` can be negative, so:

```text
power can become huge positive
exp(power) can overflow to inf
```

Then:

```text
alpha = opacity * exp(power)
```

If opacity has already been multiplied by zero for a behind Gaussian:

```text
0 * inf = NaN
```

That is the renderer-level reason "opacity mask after projection" is not enough.

## Renderer Robustness Is Separate From Initialization

There are two separate responsibilities:

```text
initialization/normalization:
    put most mass in a sane visible volume

renderer robustness:
    avoid NaNs even when mass is not sane
```

We should not use renderer robustness as an excuse to leave scale unnormalized.
But we also should not rely on perfect normalization for numerical safety. During
pretraining, bad examples and early predictions will happen.

The renderer should be safe under:

```text
some Gaussians behind camera
some tiny scales
some large scales
some narrow FoV cameras
some camera predictions that are temporarily wrong
```

Possible renderer safety contract:

```text
1. compute front_mask before expensive projection covariance terms
2. either skip invalid Gaussians or substitute harmless projected values
3. ensure covariance regularization keeps invCov2D positive semidefinite
4. clamp exponent before exp
5. never multiply 0 by inf
6. expose diagnostic counts for invalid geometry
```

This does not decide the camera initialization. It just prevents one bad frame
from destroying a debugging run.

## Gaussian Parameter Initialization Details

The known-camera Gaussian head uses `build_mlp`, not a zero-init final layer.
That means exact initial Gaussian locations are random functions of:

```text
token initialization
image encoder random weights
attention random weights
head random weights
input image
Plucker grid
```

The range is bounded by output nonlinearities, but the initial distribution is
not a hand-placed cloud.

Nominal bounds:

```text
x,y: tanh(raw) * 1.5
z:   sigmoid(raw) * 2 + 0.5
```

If `raw ~= 0`, then:

```text
x ~= 0
y ~= 0
z ~= 1.5
scale ~= 0.05
opacity ~= 0.5
rgb ~= 0.5
```

But with random raw values, the initialization has spread. It can put some
points near:

```text
z ~= 0.5
```

and some near:

```text
z ~= 2.5
```

The dangerous region for a later camera is any point whose camera-space depth is
near zero. If camera motion has scale comparable to the depth range, random
initial spread makes some points fail even if the median point is safe.

This suggests an important diagnostic:

```text
for each frame t:
    compute quantiles of z_c(t, X_i)
```

not just min/max. Useful quantiles:

```text
0.0% minimum
0.1%
1.0%
5.0%
50.0%
95.0%
99.0%
100.0% maximum
```

If only one point is behind the camera, renderer robustness should handle it.
If 40% of points are behind, normalization/init is wrong.

## Depth Range Design

For a camera looking at an object-centric scene, choose:

```text
object radius:      E
camera radius:      r
near safety margin: n
far desired:        F
```

For a spherical scene:

```text
z_min = r - E
z_max = r + E
```

Need:

```text
z_min > n
```

For good gradients, also want:

```text
z_min not too small
z_max / z_min not too large
```

The ratio controls how differently front/back parts of the scene project:

```text
projected scale ~ f / z
front/back projected-scale ratio = z_max / z_min
```

With:

```text
r = 3
E = 1
```

ratio:

```text
z_max / z_min = 4 / 2 = 2
```

reasonable.

With:

```text
r = 1.2
E = 1
```

ratio:

```text
2.2 / 0.2 = 11
```

unstable and visually extreme.

With:

```text
r = 10
E = 1
```

ratio:

```text
11 / 9 ~= 1.22
```

stable but almost orthographic; depth gradients weaken.

So a target camera radius of about:

```text
r ~= 2.5E to 4E
```

is a good default band.

## Choosing A Canonical Scene Extent

Possible canonical choices:

```text
unit ball:
    robust object radius = 1

unit cube:
    robust x,y,z half extents <= 1

camera-relative slab:
    first camera sees depth in [z_near_init, z_far_init]

hybrid:
    object radius ~= 1 and first camera target/radius fixed
```

Unit ball is attractive for pretraining because it is rotation invariant:

```text
||X|| <= 1
```

Unit cube is attractive for heads:

```text
x,y,z = tanh(raw) * extent
```

Camera-relative slab is attractive for known-camera data:

```text
z relative to first camera in [0.5, 2.5]
```

But camera-relative slab is not object-centric. If the first frame is a side
view, or if the object is not centered in the first view, the canonical scene
can be biased by camera choice.

For pretraining, the strongest prior is probably:

```text
canonical object coordinates:
    robust scene center = origin
    robust scene radius = 1

canonical camera coordinates:
    first/reference camera center around radius 3
    look direction roughly points to origin
```

This makes scene and camera both normalized, not one defined entirely by the
other.

## Gauge Fixing Options For Known DUSt3R Poses

Given DUSt3R cameras and possibly points, we need choose a transform:

```text
X_norm = (A (X_raw - mu)) / s
C_norm = (A (C_raw - mu)) / s
R_norm = A R_raw
```

Choice components:

```text
mu: center
s:  scale
A:  rotation/orientation
```

### Center Choices

First camera center:

```text
mu = C_0
```

Pros:

```text
T_0 translation becomes zero
simple
preserves current relative-pose behavior
```

Cons:

```text
object may not be near origin
camera path may be centered, not scene
Gaussian head has to learn offset to object
```

Mean camera center:

```text
mu = mean_t C_t
```

Pros:

```text
camera path centered
less sensitive to first frame
```

Cons:

```text
object can still be offset
for orbit paths, mean camera center may be near object
for forward motion, mean camera center may be inside path not object
```

Point cloud center:

```text
mu = robust_median_i X_i
```

Pros:

```text
object-centric
best for Gaussian volume
```

Cons:

```text
requires usable DUSt3R points
foreground/background ambiguity
outliers can dominate without robust filtering
```

Ray-intersection center:

```text
mu = least-squares point closest to camera optical axes
```

Pros:

```text
does not require dense points
approximates common look-at target
useful for object-centric clips
```

Cons:

```text
bad for forward-moving cameras or non-object-centric videos
ambiguous if optical axes are nearly parallel
```

Hybrid center:

```text
mu = robust blend(point_center, optical_axis_center, camera_center)
```

This is likely best long term, but it has more knobs.

### Scale Choices

Camera path radius:

```text
s = median_t ||C_t - mu||
```

Then normalized camera radius is about 1, unless multiplied by target radius.

To target radius 3:

```text
s = median_t ||C_t - mu|| / 3
```

Point radius:

```text
s = robust_quantile_i ||X_i - mu|| / E
```

where `E=1`.

Depth range:

```text
s = robust_median_visible_depth / z_target
```

where `z_target` might be 3.

Hybrid:

```text
s = max(
    robust_point_radius / E,
    robust_camera_radius / r_target,
    near_safety_requirement
)
```

The max version prevents either camera path or point cloud from becoming too
large for the canonical volume.

### Rotation Choices

First camera orientation:

```text
A = R_0.T
```

This makes first camera orientation identity. It is close to current behavior.

Pros:

```text
simple
first frame has canonical axes
image-to-world ray encoding stable for frame 0
```

Cons:

```text
object gravity/up is not canonical
camera path orientation depends on first frame
```

Average up/look orientation:

```text
look_axis = robust_mean_t forward_t
up_axis = robust_mean_t up_t
construct A to map look/up to canonical axes
```

Pros:

```text
less sensitive to first frame
canonicalizes video-level orientation
```

Cons:

```text
averaging rotations is tricky
bad for full orbit paths where mean forward can cancel
```

Object PCA:

```text
A = PCA axes of point cloud
```

Pros:

```text
object-centric shape orientation
```

Cons:

```text
sign flips
symmetry ambiguities
unstable for sparse/noisy point clouds
```

No rotation normalization:

```text
A = I
```

Pros:

```text
least intrusive
keeps DUSt3R/world orientation
```

Cons:

```text
pretraining sees arbitrary rotations
Gaussian head must learn rotational equivariance without architecture support
```

## First Frame Identity Is Not Enough

Current known-camera normalization:

```text
T_t_norm = inverse(T_0_raw) T_t_raw
```

This fixes:

```text
T_0_norm = I
```

It also maps the first camera center to:

```text
C_0_norm = 0
```

But that does not define an object-centric scene. It defines a first-camera
coordinate system. If the Gaussian head emits positive z `[0.5, 2.5]`, we are
implicitly saying:

```text
the object starts in front of first camera at depth roughly 1.5
```

If DUSt3R later frames are:

```text
C_t.z ~= 2.5 in first-camera coordinates
```

then the camera may have moved through the initialized object slab.

This is why first-frame identity can be stable for short/small camera motion and
fail for a larger corrected path.

## Pretraining Objective Implied By Normalization

If we train across examples with arbitrary scales:

```text
example A:
    same visual object occupies canonical depth 1.5

example B:
    same visual object occupies canonical depth 25

example C:
    same visual object occupies canonical depth 0.15
```

then the model has no consistent target. It can learn image reconstruction by
absorbing scale into:

```text
Gaussian positions
Gaussian scales
camera translation
opacity
focal/FoV
```

but those factors are entangled. The same pixels can be matched by many internal
solutions. Pretraining wants the internal solution to be reusable.

So normalization is not just about avoiding NaNs. It is about making the latent
geometry identifiable enough that weights learned on one example transfer to
another.

A useful framing:

```text
the dataset should remove known gauge freedoms
the architecture should encode remaining symmetries
the loss should regularize remaining ambiguities
```

## Camera Position Encoding Options

Current known-camera model encodes pose through a Plucker ray grid:

```text
ray direction: d_w
ray moment:    m_w = C_w cross d_w
plucker:       [d_w, m_w]
```

This is fed spatially through:

```text
grounded_feature_map = image_encoder(image) + ray_proj(plucker_grid)
```

The Gaussian decoder itself does not separately receive:

```text
camera center
camera orientation
intrinsics scalar
FoV scalar
```

except insofar as the Plucker grid contains them.

### What Plucker Encodes

For any ray:

```text
L = (d, m)
m = C cross d
```

Properties:

```text
d dot m = 0
||d|| = 1
```

If camera origin changes by `C`, moment magnitude is:

```text
||m|| = ||C cross d|| = distance from origin to the ray
```

So Plucker moment magnitude is directly tied to chosen world origin and scale.

If all camera centers are scaled by `s`:

```text
C' = s C
m' = C' cross d = s m
```

Directions stay unit length, moments scale.

Therefore raw Plucker is not scale-invariant.

For pretraining:

```text
raw Plucker without scene normalization = model sees arbitrary moment scale
```

### Normalize Plucker Moment By Scene Scale

Given canonical scale `s`, use:

```text
m_norm = (C_w - mu) cross d_w / s
```

If the whole scene is transformed by scale `a`, then:

```text
C_w' = a C_w
s' = a s
m_norm' = (a C_w cross d) / (a s) = m_norm
```

This makes Plucker moments scale-invariant under global scale.

But if `mu` changes, moments change. That is correct: Plucker moments encode ray
position relative to the chosen canonical origin. The origin must be meaningful.

### Encode Camera Scalars Separately

Alternative or supplement:

```text
camera token receives:
    normalized center C_norm
    rotation representation R
    log focal or FoV
    near/far estimate
    time t
```

Then the spatial feature map receives only directions or normalized rays.

Possible split:

```text
local per-pixel geometry:
    direction d_camera or d_world
    normalized image coordinates x=(u-cx)/fx, y=(v-cy)/fy

global camera metadata:
    C_norm
    R
    fov
    path index/time
```

This may be easier for the model than recovering global path facts from every
pixel's Plucker moment.

### Encode Rays In Camera Coordinates

Instead of world-space Plucker:

```text
d_camera = normalize([(u-cx)/fx, (v-cy)/fy, 1])
```

This encodes intrinsics and pixel direction but not camera pose. Pose can be
provided elsewhere.

Pros:

```text
scale-free
stable across datasets
separates viewpoint from ray shape
```

Cons:

```text
decoder must get pose another way
less direct grounding for world coordinates
```

For pretraining, separating intrinsics and pose may be cleaner than one
all-in Plucker grid.

## Raw Plucker Pointed At Origin?

The user's question: should camera position be a Plucker/raw representation
pointed at the origin and then make the camera move?

Possible interpretation:

```text
base camera:
    origin C_base = radius * direction_from_target
    look direction points to origin

ray representation:
    Plucker rays from that base camera

motion:
    camera path residual changes C_t and R_t
```

This can be good if:

```text
origin is a meaningful target
radius is normalized
Plucker moment is normalized
motion residual is bounded
```

It is bad if:

```text
origin is arbitrary first-camera origin
radius is raw DUSt3R scale
motion residual can move through the object
```

The phrase "pointed at origin" should not mean a hard assumption that every
camera always looks at origin. It can mean an initial prior:

```text
initial pose = look-at origin
learned residual = allowed to deviate
```

This is probably the right balance. Many object-centric videos roughly look at
the subject. But real camera paths can pan, cut, turn, or track a moving object.

## Camera Motion Scale

Current implicit path scale:

```text
translation_bound = base_radius * max_translation_ratio
```

At default:

```text
base_radius = 3
max_translation_ratio = 0.2
translation_bound = 0.6 per axis
```

Maximum Euclidean translation if every axis saturates:

```text
||tau|| <= sqrt(3) * 0.6 ~= 1.039
```

So the total residual vector could be about one scene extent if `scene_extent=1`.

This is larger than it looks because the ratio is per-axis, not radial norm.

If we want norm-bounded translation:

```text
tau = direction * radius * ratio * sigmoid_or_tanh_magnitude
```

or:

```text
tau = raw / max(||raw||, 1) * radius * ratio
```

Current per-axis tanh box:

```text
tau in [-a, a]^3
```

contains a sphere of radius `a` and is contained in a sphere of radius
`sqrt(3)a`.

For safety proofs, use:

```text
max_norm = sqrt(3) base_radius ratio
```

not:

```text
base_radius ratio
```

With:

```text
r = 3
ratio = 0.2
E = 1
```

worst possible direct movement toward object in any direction could reduce
camera distance by about:

```text
sqrt(3) * 0.6 = 1.039
```

Then:

```text
min distance to origin ~= 1.961
min depth for object radius 1 ~= 0.961
```

still safe, but the margin is smaller than the one-axis estimate.

If ratio were raised to 0.5:

```text
max_norm = sqrt(3) * 1.5 = 2.598
min distance to origin ~= 0.402
```

then an object radius 1 would not be safe.

So path translation ratio interacts directly with scene extent and base radius.

General safety inequality:

```text
r - sqrt(3) r q - E > n
```

where:

```text
q = max_translation_ratio
n = near margin
```

Solve:

```text
r (1 - sqrt(3) q) > E + n
```

Need:

```text
q < 1 / sqrt(3) ~= 0.577
```

and:

```text
r > (E+n) / (1 - sqrt(3)q)
```

For:

```text
q = 0.2
E = 1
n = 0.25
```

required:

```text
r > 1.25 / (1 - 0.3464)
  > 1.914
```

The default `r=3` passes.

This kind of inequality should probably be encoded in config validation or at
least diagnostic notes.

## Rotation Scale

Current implicit path rotation:

```text
omega_i in [-5 deg, 5 deg] per axis
```

Euclidean axis-angle norm bound:

```text
||omega|| <= sqrt(3) * 5 deg ~= 8.66 deg
```

Again, per-axis tanh means actual total rotation can exceed the nominal scalar.

Small-angle approximation:

```text
R_delta X ~= X + omega cross X
```

For a point at depth `z` and lateral coordinate `x`, a yaw rotation of `theta`
changes projected coordinate roughly by:

```text
delta u ~= f theta
```

At:

```text
f = 380 px
theta = 5 deg = 0.0873 rad
```

pixel shift:

```text
delta u ~= 33 px
```

At 128px resolution, that is large. So even "5 degrees" can be visually large
for narrow FoV cameras.

For implicit default 60 deg FoV at 128:

```text
f ~= 111 px
delta u ~= 9.7 px
```

less severe.

So path rotation bounds should maybe be expressed in image-space motion:

```text
max_pixel_motion ~= f * theta
theta_bound ~= max_pixel_motion / f
```

This would make residual rotation scale aware of FoV.

## FoV, Radius, And Apparent Object Size

Object apparent size is controlled by:

```text
apparent_radius_pixels ~= f * E / r
```

For a canonical object radius `E=1`:

```text
f = 111, r = 3 => apparent_radius ~= 37 px at S=128
f = 380, r = 3 => apparent_radius ~= 127 px at S=128
```

A narrow FoV camera at radius 3 would make a unit-radius object fill or exceed
the frame. That may or may not match the data.

This means if we preserve DUSt3R narrow FoV, we might need larger canonical
camera radius to keep object size comparable:

```text
r_target ~= f * E / apparent_radius_target
```

If target apparent radius is 40px at S=128:

```text
r_target ~= 380 * 1 / 40 = 9.5
```

That is much larger than current implicit default radius 3.

But if the object truly fills the frame, apparent radius 60px may be fine:

```text
r_target ~= 380 / 60 ~= 6.3
```

This suggests canonical radius should not be fixed independently of FoV and
object crop. It should satisfy:

```text
f * E / r ~= desired_projected_size
```

In normalized image coordinates:

```text
projected_radius_fraction = E / (r * tan(fov/2))
```

For square image, half-frame corresponds to normalized tangent radius:

```text
half_width_world_at_depth_r = r * tan(fov/2)
```

To fit object radius E with margin `k` of half-frame:

```text
E <= k r tan(fov/2)
r >= E / (k tan(fov/2))
```

If `k = 0.7`:

```text
fov = 60 deg => r >= 1 / (0.7 * 0.577) = 2.47
fov = 20 deg => r >= 1 / (0.7 * 0.176) = 8.12
```

This is a big insight: a fixed radius 3 is safe for 60 deg FoV but not an
object-size prior for 20 deg FoV.

For known-camera DUSt3R data, if FoV is 19 deg and Gaussian z range is 0.5-2.5,
then a cloud with x/y extent 1.5 is enormous in the image unless the camera is
farther away. The optimizer may collapse opacity/color to gray because initial
splats are wildly projected or culled.

## Why 64 Could Appear Faster Than 32

The earlier observation was strange:

```text
default old-looking config: ~4.6 it/s
64/4fps config:            ~6-7 it/s
```

Even though 64 has more pixels than 32.

Potential explanations:

```text
frames_per_step differs:
    64 config used FramesPerStep=4
    old config logged FramesPerStep=23 in the screenshot/run

renderer batch path differs:
    batch rendering can amortize Python overhead

MPS kernel launch overhead:
    very small tensor workloads can underutilize GPU

media logging timing:
    W&B media may affect apparent speed around log steps

data/camera failure:
    degenerate projections may produce different compute behavior
```

Compute per step is roughly:

```text
O(B * G * H * W)
```

where:

```text
B = frames rendered per step
G = Gaussian count
H,W = render size
```

Compare:

```text
32 old all frames:
    B=23, H*W=1024
    pixel-gaussian work ~= 23 * 1024 * G = 23552G

64 config 4 frames:
    B=4, H*W=4096
    work ~= 16384G
```

So 64/4-step can be less work per step than 32/all-frame.

Compare epoch-like coverage:

```text
32 all frames:
    23 frames per step, 23 total frames
    every step sees all frames

64 4fps:
    4 frames per step, 46 total frames
    one sweep ~= 11.5 steps
```

So iterations per second is not the whole story. Need metrics:

```text
frames/sec
pixels/sec
pixel-gaussians/sec
optimizer updates per sequence pass
```

For stability comparisons, use:

```text
same frames_per_step
same sequence length or same sampled frame count
same image size
same logging cadence
```

## Why Gray Render Is A Diagnostic Smell

A gray render can come from several sources:

```text
1. model colors initialized around sigmoid(0)=0.5
2. opacities low or splats mostly culled
3. alpha compositing leaves white background plus gray low-alpha splats
4. renderer NaNs get clamped or logged badly
5. optimizer finds mean color because geometry is unusable
6. camera projections put splats offscreen
```

If loss settles around a mean-color value and render media is gray, inspect:

```text
render finite fraction
alpha total per pixel
number of front Gaussians
onscreen projected means
2D covariance sizes
RGB mean/std
opacity mean/std
gradient norms for xyz, opacity, rgb
```

Mean-color solution often means:

```text
geometry path is not providing useful image gradients
```

The optimizer can reduce MSE by setting RGB to dataset mean even if geometry is
bad. That is not a renderer correctness proof.

## Initialization And Loss Landscape

MSE reconstruction gives gradients through rendering:

```text
loss = ||render(theta) - image||^2
```

For Gaussian positions, useful gradients require:

```text
splats visible
splats not fully saturated alpha
splats not infinitesimal/offscreen
camera projection finite
```

Bad initialization cases:

```text
behind camera:
    no useful gradients if culled
    possible NaNs if not robustly culled

very near camera:
    huge projected covariance/position sensitivity
    unstable gradients

very far camera:
    tiny motion/parallax gradients
    splats can become small

offscreen:
    little or no image gradient

too large opacity:
    early splats occlude all later splats

too small opacity:
    signal dominated by background

too large scale:
    blurry mean-color gradients

too small scale:
    sparse/noisy gradients
```

Good initialization is not merely "finite." It should place enough Gaussian
mass in front of every early camera with projected sizes that produce smooth
gradients.

## Scale Of 3D Gaussian vs Pixel Footprint

Approximate pixel footprint for isotropic 3D Gaussian scale `sigma` at depth
`z`:

```text
sigma_px ~= f * sigma / z
```

Current scale init near:

```text
sigma = 0.05
```

At:

```text
f = 111, z = 3 => sigma_px ~= 1.85 px
f = 380, z = 3 => sigma_px ~= 6.33 px
f = 380, z = 1 => sigma_px ~= 19 px
```

So narrow FoV/high focal makes the same 3D Gaussian much larger in pixels.

If radius is adjusted to preserve apparent object size:

```text
z ~= r ~= f * E / target_px
```

then:

```text
sigma_px ~= f * sigma / (f E / target_px)
         = target_px * sigma / E
```

The pixel footprint becomes independent of focal if radius scales with focal.

This is another argument that FoV, radius, scene extent, and Gaussian scale
should be normalized together.

## Camera Path Smoothness: MLP Is Not A Guarantee

The current path gets smooth-ish behavior because:

```text
time t is continuous-ish
MLP weights are shared
training data is adjacent frames
optional temporal losses can regularize outputs
```

But an MLP can approximate sharp functions. It is not mathematically forced to
be smooth unless:

```text
architecture has low bandwidth
input encoding is low-frequency
weights are regularized
loss penalizes derivatives
outputs are basis-limited
```

If time input is scalar `t` and MLP has normal activations, it has a spectral
bias toward low frequencies early in training. But after training it can still
represent high-frequency wiggles.

For camera path pretraining, there are different desired behaviors:

```text
smooth handheld drift:
    path should be smooth

object turntable:
    smooth orbit

camera cut:
    discontinuity is valid

fast pan:
    high velocity, maybe smooth acceleration

tracking shot:
    camera and object motion entangled

bad DUSt3R pose:
    apparent discontinuity may be solver error, not real camera
```

An unconditional smoothness prior can hurt cuts and sharp turns. A pure MLP can
overfit pose jitter. A good pretraining architecture may need:

```text
path mode token:
    smooth / cut / orbit / dolly / handheld

or robust loss:
    mostly smooth with sparse discontinuity allowance
```

## Polynomial Camera Paths

A polynomial path:

```text
C(t) = sum_{k=0}^K a_k t^k
```

Pros:

```text
simple
smooth
few parameters
global trajectory constraints
```

Cons:

```text
global basis: local correction affects entire sequence
bad for long sequences
bad for sharp turns/cuts
can oscillate near boundaries
rotation polynomial needs care
```

For short overfit clips, low-degree polynomial might be enough:

```text
K=1: linear dolly/pan
K=2: constant acceleration
K=3: smooth ease-in/ease-out
```

But for pretraining over diverse videos, a fixed low-degree polynomial is too
restrictive.

## Spline Camera Paths

A spline path:

```text
C(t) = sum_j B_j(t) P_j
```

where `B_j` are local basis functions and `P_j` are control points.

Pros:

```text
local control
smooth by construction
scales to longer sequences
can bound velocity/acceleration through control differences
```

Cons:

```text
more implementation complexity
need choose knot spacing
harder for discontinuities unless segmented
rotation spline requires SO(3) handling
```

A useful compromise:

```text
translation:
    cubic B-spline in R3

rotation:
    small axis-angle residual spline around base look-at orientation

per-frame residual:
    tiny MLP correction
```

This gives smooth default plus flexibility.

## Piecewise Smooth With Cuts

Real videos can have cuts. Pretraining should not assume all frames are a single
smooth camera path.

Possible path representation:

```text
segments = detected or learned
for each segment:
    smooth path
between segments:
    independent base pose
```

In overfit configs for one continuous video, we can ignore this. For pretraining
dataset scale, we need at least diagnostics:

```text
image difference spikes
DUSt3R pose jumps
low feature match confidence
```

If a clip has a cut and the model is forced into one smooth path, it may use
Gaussian deformation or opacity hacks to explain it.

## Time Encoding

Current time path uses scalar frame time:

```text
time_proj(frame_times)
```

Questions:

```text
is t normalized to [0,1]?
is t in seconds?
is t in frame index units?
does doubling FPS change t spacing?
```

For pretraining, time encoding should make physical sense:

```text
same video at 2fps vs 4fps should represent same path sampled more densely
```

If time is normalized by frame index:

```text
t_i = i / (N-1)
```

then 2fps and 4fps versions of the same duration both cover [0,1]. Good for
sequence-level path shape. But velocity in real seconds is hidden.

If time is seconds:

```text
t_i = i / fps
```

then same physical motion has same time values, but clips of different duration
have different ranges. The MLP must handle arbitrary domains.

Possible encoding:

```text
sequence_time = i / (N-1)
seconds_time = i / fps
delta_time = 1 / fps
```

Use all three where relevant:

```text
path shape uses sequence_time
motion magnitude/velocity regularization uses seconds_time
sampling awareness uses delta_time
```

## Relationship Between FPS And Camera Path

Doubling FPS from 2 to 4 should not double total camera motion. It should add
intermediate samples.

For a smooth path `C(t)` over physical time:

```text
2fps samples: C(0), C(0.5), C(1.0), ...
4fps samples: C(0), C(0.25), C(0.5), C(0.75), ...
```

If DUSt3R is run independently on each video, pose scale and even trajectory
shape can change due to:

```text
different frame matching graph
different image resolution
different feature density
different optimizer convergence
```

So the 64/4fps and 128/4fps cameras are not guaranteed to be scaled versions of
the old 32/2fps solve. They are new solves.

This is why dataset baking should store diagnostics:

```text
input source video path
ffmpeg crop/fps/scale command
frame count
DUSt3R command/commit
intrinsics stats
camera center stats
camera depth stats under initial Gaussian prior
pose scale normalization stats
```

## Camera Solver Confidence

DUSt3R outputs should ideally provide confidence for points/poses. If we have
confidence maps or matching scores, use them for normalization:

```text
point center:
    confidence-weighted robust median

point radius:
    confidence-weighted quantile

camera scale:
    ignore low-confidence frames

failure detection:
    flag frames whose camera path jumps when confidence is low
```

Without confidence, use robust statistics:

```text
median
MAD
trimmed quantiles
RANSAC-ish filtering
```

Do not use raw min/max except as diagnostics. Raw min/max are too sensitive to
outliers.

## Object Motion vs Camera Motion

Dynaworld has "dynamic" in the name. In videos, apparent motion can come from:

```text
camera motion
object motion
nonrigid deformation
lighting/exposure
rolling shutter
reconstruction noise
```

If camera normalization assumes all motion is camera path around a static object,
then moving foreground objects can distort the estimated canonical frame.

For pretraining, useful distinction:

```text
camera canonicalization:
    should use stable background/scene if available

object-centric canonicalization:
    should use foreground/object if task is object reconstruction
```

These can conflict. A clip of a moving subject against a static background has:

```text
background-defined camera path
foreground-defined object center
```

The Gaussian model might represent both, but initialization needs decide what
volume to fill.

Possible answer:

```text
canonical volume covers visible dynamic content
camera path normalized by cameras/background
object center estimated from masks or robust visible content
```

Without masks, a conservative approach is:

```text
center by robust point cloud
scale by max(point extent, camera path extent / target_radius)
```

## Known-Camera vs Implicit-Camera Should Share A Contract

Right now:

```text
known-camera:
    Gaussian z positive slab
    camera from DUSt3R relative to first frame

implicit-camera:
    Gaussian centered cube/ball
    camera learned as orbit around origin
```

This means the two branches do not use the same mental model of coordinates.

For pretraining and transfer, better:

```text
canonical scene:
    centered at origin, radius ~= 1

known camera:
    normalized DUSt3R cameras in canonical scene

implicit camera:
    predicted cameras in same canonical scene

Gaussian head:
    same coordinate range or compatible ranges
```

Known-camera positive-z slab may still be useful for a camera-centric model, but
it fights object-centric implicit cameras.

Potential migration:

```text
phase 1:
    normalize known-camera DUSt3R so old z slab is safe

phase 2:
    add object-centric known-camera head option

phase 3:
    share Gaussian head style between known and implicit camera models

phase 4:
    pretrain with mixed known/implicit camera supervision
```

## Camera-Centric vs Object-Centric Coordinates

Camera-centric:

```text
first camera is canonical
points mostly in positive z
great for single-view prediction
```

Object-centric:

```text
object/scene center is canonical
cameras orbit/move around it
great for multi-view/video consistency
```

For one image to 3D, camera-centric is common:

```text
predict depth in front of input camera
```

For video and pretraining across paths, object-centric is stronger:

```text
same object should have same canonical geometry from different starting views
```

But object-centric needs a way to choose orientation. If orientation is
ambiguous, the model can still choose a learned canonical orientation, or we can
randomize orientation during training to encourage equivariance.

Possible architecture split:

```text
single-view encoder predicts camera-centric provisional Gaussians
normalization module maps them into object-centric canonical frame
video/attention module refines object-centric geometry
renderer uses normalized cameras
```

## Initialization For Pretraining: Do Not Depend On Exact DUSt3R Scale

A dangerous pattern:

```text
model only trains if DUSt3R scale happens to match head init
```

Pretraining should tolerate:

```text
DUSt3R scale 0.1x
DUSt3R scale 1x
DUSt3R scale 10x
```

after normalization all become:

```text
camera radius ~= target
scene extent ~= target
Plucker moment scale ~= target
Gaussian scale ~= target
```

Test:

```text
take same cameras and multiply all translations by s in {0.1, 1, 10}
after normalization, training diagnostics should be identical or near-identical
```

If not, normalization leaked raw scale.

## Scale-Equivariant Model Alternative

Instead of fully normalizing data, make model scale-aware:

```text
predict canonical Gaussians
predict or consume scene scale
render in scaled coordinates
normalize losses accordingly
```

But rendering pixels are scale-invariant only if cameras and points scale
together. Gaussian scales also need scale:

```text
X' = sX
sigma' = s sigma
C' = sC
```

Opacity is not scale-invariant in full volumetric rendering unless density is
handled carefully. In splat alpha compositing, opacity is a learned 2D-ish
quantity, so scale equivariance is approximate.

Full scale-equivariant design is more complex than canonical normalization.

Recommendation:

```text
normalize data first
maybe expose original scale as metadata later
```

## FoV Normalization: Preserve Or Canonicalize?

Do not casually normalize FoV away. FoV is observable from perspective effects
and affects the actual image formation.

Options:

```text
preserve FoV:
    use true intrinsics
    adjust camera radius/scene scale accordingly

canonicalize FoV:
    warp/crop images or cameras into a standard FoV
    simpler model distribution
    can distort data or lose information

mixed:
    preserve intrinsics but encode normalized FoV token
    choose scene radius based on FoV/object fit
```

For our current data, preserve FoV seems right. But then radius/scene extent
must account for narrow FoV.

The key invariant is not:

```text
radius = 3 always
```

It is:

```text
projected canonical object size falls in a good range
and depth margins are safe
```

## Possible Canonical Fit Algorithm

Given:

```text
raw cameras T_t
intrinsics K_t
optional DUSt3R points P_i with confidence w_i
image size S
target object radius E=1
target projected half-frame fraction k=0.65
near margin n=0.25
```

Algorithm sketch:

```text
1. choose center mu
   if points available:
       mu = robust weighted median of high-confidence points
   else:
       mu = robust optical-axis intersection
   fallback:
       mu = median camera center + median forward * rough_depth

2. compute raw object radius
   r_obj_raw = q90_i ||P_i - mu||
   if no points:
       infer from camera frustum overlap or set unknown

3. compute raw camera distances
   d_t_raw = ||C_t - mu||
   d_med_raw = median_t d_t_raw
   d_min_raw = q05_t d_t_raw

4. compute FoV-derived target radius for each frame
   fov_t = 2 atan(0.5 S / f_t)
   r_fit_t = E / (k tan(fov_t/2))
   r_target = median_t r_fit_t

5. choose scale from camera distance
   s_cam = d_med_raw / r_target

6. choose scale from object radius
   s_obj = r_obj_raw / E

7. combine
   s = robust blend/max(s_cam, s_obj)

8. normalize
   C_t_norm = (C_t - mu) / s
   P_i_norm = (P_i - mu) / s

9. rotate if desired
   align reference camera or average up

10. validate
   all/most initialized Gaussian prior depths safe
   projected object radius within desired band
```

Potential problem:

```text
s_cam and s_obj can disagree
```

Disagreement is informative:

```text
DUSt3R point cloud scale/center may not match camera path
FoV may be wrong
object/background selection may be wrong
video may not be object-centric
```

Log this disagreement.

## Normalization Diagnostics To Store Per Dataset Item

For every baked sequence:

```text
source:
    source_video_path
    ffmpeg command
    fps
    output size
    frame count

intrinsics:
    fx min/median/max
    fy min/median/max
    fov_x min/median/max
    fov_y min/median/max
    cx,cy offsets from center

raw camera path:
    center min/max per axis
    center norm quantiles
    step translation quantiles
    relative rotation quantiles
    look-at-origin angle quantiles

normalization:
    center mu
    scale s
    rotation A
    scale source/method
    point radius before/after
    camera radius before/after

post-normalization:
    camera radius quantiles
    object radius quantiles
    Plucker moment quantiles
    initial prior depth quantiles
    projected Gaussian footprint estimate

renderer safety:
    front Gaussian count under init prior
    near/behind count under init prior
    worst frame id
```

The goal is to catch failures before training.

## Initial Gaussian Prior As Dataset Validation

Use the model's initial coordinate support, not just actual random initial
sample.

For known-camera current head support:

```text
X support = [-1.5,1.5] x [-1.5,1.5] x [0.5,2.5]
```

For a camera `t`, approximate support depth range by checking corners:

```text
z_corners = forward_t dot (corner_j - C_t)
z_min_support = min_j z_corners
z_max_support = max_j z_corners
```

If:

```text
z_min_support <= near_margin
```

then some allowed initial Gaussians can be invalid.

But actual random outputs may not hit corners. So also evaluate sampled initial
Gaussians:

```text
run model init on a few frames with no optimizer step
log z quantiles under all cameras
```

Dataset validation can use both:

```text
support-level safety:
    architecture/data compatibility

sample-level safety:
    current seed/model compatibility
```

For pretraining, support-level safety is stronger.

## Better Known-Camera Init Options

Current known-camera slab:

```text
x,y in [-1.5,1.5]
z in [0.5,2.5]
```

Option 1: normalize cameras to fit slab.

```text
keep head
normalize DUSt3R cameras so visible content is inside slab
```

Pros:

```text
minimal model change
preserves stable baseline shape
```

Cons:

```text
camera-centric forever
less aligned with implicit object-centric model
```

Option 2: object-centric head.

```text
x,y,z in [-1,1]
cameras normalized around object
```

Pros:

```text
matches implicit camera
better for pretraining
```

Cons:

```text
requires data normalization
may break existing baseline
```

Option 3: depth-conditioned head.

```text
predict normalized coordinates in canonical box
then transform by per-sequence camera/scene normalization
```

Pros:

```text
explicit contract
can preserve old render camera conventions
```

Cons:

```text
more moving parts
need store transform with sequence
```

Option 4: initialize Gaussians from rays/depth.

```text
for each token/pixel:
    choose ray from input frame
    initialize point at depth z0 along ray
```

Pros:

```text
always in front of input camera
spatially grounded
```

Cons:

```text
other frames can still see points behind if camera path scale bad
needs depth prior
may bias geometry to first view
```

## Ray-Depth Initialization

A ray-depth initialization can define:

```text
X_i = C_0 + z_i d_i
```

where:

```text
d_i = ray direction for sampled pixel/token
z_i in [z_near_init, z_far_init]
```

If `C_0` is first camera and `z_i` chosen in `[1,3]`, first-frame safety is
guaranteed.

For other frames:

```text
z_c(t, X_i) = forward_t dot (C_0 + z_i d_i - C_t)
```

No guarantee unless camera path is normalized.

Ray-depth init is useful but not a substitute for camera/scene normalization.

It might help image encoders:

```text
tokens correspond to image regions
initial splats project near their source pixels
```

But current token model does not have explicit token-pixel assignment. It uses
attention over image features, so ray-depth init would require architectural
changes.

## Multi-Frame Initialization Idea

Use known cameras to initialize Gaussians by triangulation-like priors:

```text
1. sample feature tracks or DUSt3R points
2. place initial Gaussians at normalized 3D points
3. initialize scales from local point spacing/depth uncertainty
4. initialize colors from projected image samples
```

Pros:

```text
strong geometry start
less gray-collapse risk
diagnostic-friendly
```

Cons:

```text
less pure learned model
depends on external reconstruction quality
harder for pretraining if point quality varies
```

For pretraining, maybe use this as teacher/initialization for known-camera
experiments, but keep learned initialization robust enough not to require it.

## Camera Path MLP Conditioning

Current implicit path token comes from image-conditioned refined tokens:

```text
path_token_t = refined_tokens[t, 1, :]
```

So path is not merely `MLP(t)`. It is visual features plus time offset through
attention.

This is expressive. It can infer camera motion from image evidence. But it also
means path predictions can be frame-local and jittery if not regularized.

Possible additions:

```text
sequence-level path latent:
    one global token summarizes whole clip

per-frame residual:
    local token predicts small correction

basis path:
    global token predicts control points

confidence:
    per-frame token predicts uncertainty
```

Architecture:

```text
global_camera_token = mean over frames
path_control_token = aggregate over frames
path_residual_token_t = per frame
```

Then:

```text
T_t = Base(global) * Spline(path_control, t) * Residual(path_residual_t)
```

This gives both sequence-level coherence and frame-level flexibility.

## Camera Uncertainty

For known DUSt3R cameras, not all poses are equally trustworthy.

For implicit cameras, predictions are uncertain early in training.

Represent uncertainty:

```text
camera_state:
    pose_mean
    pose_log_sigma
```

Use uncertainty for:

```text
loss weighting
regularization strength
debug diagnostics
renderer culling margin
```

For known cameras:

```text
pose_sigma from DUSt3R confidence or pose graph residual
```

For implicit:

```text
pose_sigma predicted by model
regularize not to inflate unless needed
```

This may be overkill now, but for pretraining noisy camera labels it can matter.

## Pose Regularization Ideas

For direct residual path:

```text
L_translation = mean_t ||tau_t||^2
L_rotation = mean_t ||omega_t||^2
```

For smoothness:

```text
L_vel = mean_t ||tau_{t+1} - tau_t||^2
L_acc = mean_t ||tau_{t+2} - 2tau_{t+1} + tau_t||^2
```

For rotations, use relative rotations:

```text
R_rel_t = R_t.T R_{t+1}
angle_t = log_SO3(R_rel_t)
L_rot_vel = mean_t ||angle_t||^2
```

For look-at prior:

```text
view_dir_t = R_t[:,2]
to_origin_t = normalize(-C_t)
L_look = mean_t (1 - dot(view_dir_t, to_origin_t))
```

Do not make look-at too strong for general videos. Use as init/weak prior.

For path radius:

```text
L_radius = mean_t (||C_t|| - r_target)^2
```

Again, only if object-centric origin is meaningful.

## Learning Camera vs Using Camera Labels

Modes:

```text
known camera:
    use DUSt3R cameras as fixed labels

learned camera:
    predict cameras from video

semi-known:
    initialize from DUSt3R and learn residual

self-calibrating:
    jointly optimize cameras and Gaussians from photometric loss
```

For pretraining, semi-known might be powerful:

```text
T_t = normalize(DUSt3R_T_t) * learned_residual_t
```

Residual bounded small:

```text
rotation residual <= few degrees
translation residual <= small fraction of radius
```

This lets model correct DUSt3R noise without inventing full camera path.

But if DUSt3R is systematically wrong, small residual cannot fix it.

Dataset diagnostics can decide:

```text
if camera solve confidence high:
    fixed or small residual
else:
    learned camera path with priors
```

## Initialization Across Resolutions

Image resolution should not change world geometry.

If source video is center-cropped and resized to 64 or 128:

```text
same pixels in normalized image coordinates
same physical FoV if intrinsics scaled correctly
same camera path in normalized world coordinates
```

But DUSt3R run at different resolution can output different raw pose scale and
intrinsics. Therefore:

```text
post-DUSt3R normalization should make 64 and 128 comparable
```

Test:

```text
normalize 64 cameras
normalize 128 cameras
compare:
    normalized camera center trajectories
    normalized radius quantiles
    normalized FoV
    relative rotations
    look-at-origin angles
```

If normalized trajectories disagree a lot, the issue is not just scale. It may
be DUSt3R resolution sensitivity or bad frame matching.

## Initialization Across FPS

Doubling FPS changes sequence length and adjacent frame baseline.

DUSt3R might perform better or worse:

```text
higher FPS:
    smaller inter-frame motion
    easier local matching
    more frames in pose graph
    more compute

lower FPS:
    larger baseline
    stronger parallax
    harder matching if motion too large
```

Camera path normalization should be invariant to sample density:

```text
same physical trajectory sampled more densely should have same normalized path
```

Diagnostics:

```text
arc length over time
mean step length
median speed = step length / delta_time
acceleration
pose graph jumps
```

For 2fps vs 4fps:

```text
step length should roughly halve
speed should roughly match
total arc length should roughly match
```

If 4fps total arc length is much larger, DUSt3R path may be jittering.

## Scale Of Plucker Moments Under Base Orbit

Base camera:

```text
C = (0,0,-r)
```

Central ray:

```text
d = (0,0,1)
m = C cross d = 0
```

Off-center ray:

```text
d = normalize(x, y, 1)
C cross d = (r y, -r x, 0) / sqrt(x^2+y^2+1)
```

Moment magnitude:

```text
||m|| = r sqrt(x^2+y^2) / sqrt(x^2+y^2+1)
```

At image corner for symmetric FoV:

```text
x = tan(fov/2)
y = tan(fov/2)
```

For fov 60:

```text
x=y=0.577
sqrt(x^2+y^2)=0.816
denominator=sqrt(1.666)=1.291
||m|| ~= 0.632 r
```

With r=3:

```text
||m|| corner ~= 1.897
```

For fov 20:

```text
x=y=0.176
sqrt=0.249
denominator=1.031
||m|| ~= 0.241 r
```

If radius scaled to 8 for narrow FoV:

```text
||m|| corner ~= 1.93
```

Interesting: if radius grows as `1/tan(fov/2)`, Plucker moment ranges can stay
similar. This supports the apparent-size invariant again.

## Focal-Aware Radius As Plucker Stabilizer

Given:

```text
r ~= E / (k tan(fov/2))
```

Corner moment magnitude:

```text
||m_corner|| ~= r * sqrt(2) tan(fov/2) / sqrt(1 + 2 tan^2(fov/2))
```

Substitute:

```text
||m_corner|| ~= E * sqrt(2) / (k sqrt(1 + 2 tan^2(fov/2)))
```

For small FoV:

```text
tan(fov/2) small
||m_corner|| ~= E * sqrt(2) / k
```

For `E=1, k=0.7`:

```text
||m_corner|| ~= 2.02
```

So focal-aware radius keeps Plucker moment magnitude bounded in a useful range.

If radius stays fixed at 3 while FoV shrinks, Plucker moments shrink, but object
projected size explodes. If radius grows, Plucker moments stay informative.

## Plucker Direction vs Moment Balance

Plucker channels contain:

```text
d in [-1,1], unit norm
m scale depends on origin/radius
```

If moment magnitude is much larger than direction:

```text
ray_proj may be dominated by position/origin
```

If moment magnitude is tiny:

```text
ray_proj mostly sees direction
camera translation is weakly encoded
```

For stable pretraining, want moment channel distribution in a consistent band:

```text
std(m) roughly comparable to std(d)
```

Potential normalization:

```text
plucker = [d, m / m_scale]
```

where:

```text
m_scale = scene_scale
or m_scale = target_camera_radius
or learned/recorded normalization constant
```

Need log:

```text
d mean/std/min/max
m mean/std/min/max
ray_proj activation mean/std
```

## Alternative Ray Encodings

1. Normalized pixel coordinates:

```text
[x=(u-cx)/fx, y=(v-cy)/fy]
```

Pros:

```text
intrinsics-aware
scale-free
small dimensional
```

Cons:

```text
no camera pose
```

2. Camera-frame ray direction:

```text
d_c = normalize([x,y,1])
```

3. World-frame direction:

```text
d_w = R_cw d_c
```

4. World Plucker:

```text
[d_w, C_w cross d_w]
```

5. Relative Plucker to scene center:

```text
[d_w, ((C_w - mu)/s) cross d_w]
```

6. Ray plus camera center:

```text
per pixel: d_w
global token: C_norm, R, fov
```

7. Positional camera grid:

```text
per pixel: expected point at canonical depth z0
P0 = C + z0 d
```

This encodes where a default-depth surface would live.

Potential experiment:

```text
compare Plucker vs direction+camera-token on same normalized data
```

## The "Depth Slab" Prior

Known-camera head:

```text
z in [0.5,2.5]
```

This is a depth slab in first camera coordinates. It imposes:

```text
all points have positive first-frame depth
```

That is helpful for single-view image-to-3D. But for a video where camera moves
around an object, first-camera depth is not the most natural coordinate.

Object-centric prior:

```text
z in [-1,1]
```

lets points exist behind the object center relative to first camera. The first
camera sees them at depth:

```text
z_camera = r + z_object
```

if camera radius is `r`.

This decouples object extent from first-camera near plane.

Migration idea:

```text
known camera model:
    internally predict object-centric xyz
    known camera normalization maps cameras around object
```

Then both known and implicit branches agree.

## Initial Opacity Considerations

Current:

```text
opacity = sigmoid(raw)
```

If raw near 0:

```text
opacity ~= 0.5
```

With hundreds of Gaussians and alpha compositing, 0.5 can be quite opaque.

For `G=512`, if many overlap, transmittance:

```text
T after k splats ~= product_i (1-alpha_i)
```

If `alpha=0.5`:

```text
T after 10 ~= 0.00098
```

So early splats can saturate visibility. In practice projected Gaussians may be
small/sparse, but opacity init matters.

For stable optimization:

```text
initial opacity maybe should be lower
```

e.g.

```text
opacity_raw_bias = logit(0.05) ~= -2.944
```

But too-low opacity may make gradients weak.

Potential schedule:

```text
low opacity init
warmup opacity or density
```

This interacts with gray collapse. If opacity starts high and colors start
around 0.5, the image can become gray before geometry aligns.

## Initial Color Considerations

Current:

```text
rgb = sigmoid(raw)
```

If raw near 0:

```text
rgb ~= 0.5 gray
```

Mean-color collapse can happen if:

```text
geometry gradients poor
rgb gradients easy
```

Options:

```text
initialize rgb from image features
sample source pixels for token colors
predict residual over image mean
regularize color variance
delay rgb learning until geometry stable
```

But color init is secondary to camera/geometry scale. Good color cannot fix
behind-camera projections.

## Initial Scale Considerations

Current:

```text
scale = exp(raw) * 0.05
```

If raw has std, scale can vary multiplicatively.

Projected scale:

```text
scale_px = f * scale / z
```

For stable learning, maybe target:

```text
scale_px initial in [1, 5] pixels
```

Then choose world scale:

```text
scale_world = scale_px_target * z / f
```

If:

```text
z ~= r
f ~= S / (2 tan(fov/2))
```

then:

```text
scale_world ~= scale_px_target * r * 2 tan(fov/2) / S
```

For fixed `S`, if `r` is adjusted by FoV to keep object apparent size constant,
world scale can stay proportional to object extent.

But if training resolution changes, pixel scale target changes. A Gaussian that
is 2px at 64 should maybe be 4px at 128 for same angular size, or 2px if we want
same raster footprint. This is a modeling choice:

```text
same world/angular scale:
    scale_px doubles with resolution

same raster optimization scale:
    scale_px constant across resolution
```

For multi-resolution training, probably use world/angular consistency, but
renderer speed/debug may prefer raster consistency.

## Token Count And Gaussian Density

Current examples use:

```text
128 tokens * 4 Gaussians/token = 512 Gaussians
```

At 32x32:

```text
pixels = 1024
pixels per Gaussian ~= 2
```

At 64x64:

```text
pixels = 4096
pixels per Gaussian ~= 8
```

At 128x128:

```text
pixels = 16384
pixels per Gaussian ~= 32
```

So going to 128 without increasing Gaussian count makes reconstruction much
harder. Loss/gray behavior can reflect undercapacity as well as camera failure.

But NaN at step 1 is not undercapacity. That is geometry/numerics.

For stability tests, separate:

```text
finite renderer at step 1
can optimize below mean-color loss
can reconstruct details
```

128 may pass the first after normalization but still need more Gaussians for
quality.

## What "Stable Baseline" Should Mean

Define baseline stability levels:

```text
level 0:
    starts and finishes without NaN

level 1:
    render media is finite and non-gray

level 2:
    loss decreases meaningfully below mean-color predictor

level 3:
    diagnostics show healthy front counts and projected sizes

level 4:
    repeated seeds work

level 5:
    neighboring configs/resolutions work
```

The old 32/2fps baseline likely satisfied level 0 and maybe level 2 for one
config. The new goal is stronger:

```text
same architecture/data contract should survive 64/128 and 4fps
```

## Mean-Color Loss Baseline

For MSE images in [0,1], compute:

```text
image_mean = mean over all training pixels/channels
loss_mean_color = mean((image - image_mean)^2)
```

If training loss converges near this value, geometry is not useful.

Log:

```text
Loss
MeanColorLoss
Loss / MeanColorLoss
```

A stable reconstruction should beat mean color clearly.

If render is all gray, likely:

```text
Loss ~= MeanColorLoss
```

This should become a standard diagnostic.

## Camera-Conditioned Mean Baseline

A more subtle baseline:

```text
per-frame mean color
```

Loss:

```text
mean_t mean_pixels((image_t - mean_color_t)^2)
```

If model samples frames and outputs different colors per frame, it might beat
global mean without learning geometry.

Also compute:

```text
center-crop blur baseline
low-res upsample baseline
```

These baselines help interpret loss values across 32/64/128.

## Normalization As Dataset Transform vs Runtime Transform

Data-level normalization:

```text
write normalized cameras/points to baked dataset
trainer just loads them
```

Pros:

```text
simple trainer
cached diagnostics
repeatable
same model sees clean data
```

Cons:

```text
if normalization policy changes, need rebake
harder to compare raw outputs
```

Runtime normalization:

```text
trainer loads raw cameras and normalizes on fly
```

Pros:

```text
easy to iterate
can log raw and normalized
```

Cons:

```text
more trainer complexity
risk inconsistent policies across scripts
```

Best approach:

```text
normalization utility library
bake command can write normalized data
trainer can optionally verify/recompute diagnostics
configs select normalization policy by name
```

Avoid hidden env vars. Put policy in JSONC config.

## Config Concepts For Normalization

Possible JSONC section:

```jsonc
"camera_normalization": {
  "enabled": true,
  "center": "points_robust",
  "scale": "hybrid_camera_object_fov",
  "orientation": "first_camera",
  "target_scene_extent": 1.0,
  "target_projected_half_fraction": 0.65,
  "target_camera_radius_fallback": 3.0,
  "near_margin": 0.25,
  "plucker_moment_scale": "scene_scale",
  "write_diagnostics": true
}
```

For old baseline:

```jsonc
"camera_normalization": {
  "enabled": false
}
```

This allows direct comparison without guessing.

## Initialization Config Concepts

Possible model section:

```jsonc
"gaussian_init": {
  "coordinate_mode": "camera_slab",
  "xy_extent": 1.5,
  "z_min": 0.5,
  "z_max": 2.5,
  "scale_world": 0.05,
  "opacity_init": 0.1,
  "rgb_init": "image_mean"
}
```

For object-centric:

```jsonc
"gaussian_init": {
  "coordinate_mode": "object_box",
  "scene_extent": 1.0,
  "scale_world": 0.03,
  "opacity_init": 0.05
}
```

Keep the actual implementation lean. The point is the config should record the
experiment's coordinate contract.

## Diagnostic Metric Sets

The requested `with_metrics(set_a=true|false,set_b=true|false)` idea can become:

```text
renderer metrics:
    finite counts
    depth quantiles
    power quantiles
    alpha quantiles
    covariance determinant quantiles
    front/near/behind counts

optimizer metrics:
    parameter norms
    gradient norms
    update ratios
    NaN/Inf counts

camera metrics:
    camera center/radius quantiles
    relative motion
    FoV/focal stats
    look-at-origin angle

gaussian metrics:
    xyz quantiles
    scale quantiles
    opacity quantiles
    color mean/std

loss baselines:
    global mean color loss
    per-frame mean color loss
```

Named sets:

```text
set_a = cheap per-step health
set_b = expensive renderer internals
set_c = camera/normalization diagnostics
set_d = optimizer/gradient diagnostics
```

For 128 NaN:

```text
set_b caught the failure before optimizer changed parameters
```

## A Better Failure Taxonomy

When a run fails, classify:

```text
data failure:
    bad video bake
    wrong FPS/source
    wrong camera directory
    stale DUSt3R output

camera failure:
    scale mismatch
    narrow FoV + close camera
    pose jump
    wrong coordinate convention

renderer failure:
    behind/near projection NaN
    covariance inversion unstable
    alpha overflow

optimization failure:
    gradient explosion
    bad LR/schedule
    opacity/color collapse

capacity failure:
    too few Gaussians
    too small tokens/model

logging illusion:
    media panel stale
    W&B step mismatch
```

The 128 step-1 NaN is:

```text
camera failure + renderer robustness failure
```

not:

```text
LR schedule
optimizer
capacity
```

because it happens before optimizer update.

## Learning Rate Is Probably Not The Step-1 NaN

If failure is at step 1 before or during first render, LR cannot be the root
cause. LR affects parameters after backprop/optimizer step.

LR can still matter for later:

```text
camera path residual explosion
Gaussian scale explosion
opacity saturation
```

But diagnostics showed decoded params finite and render internals nonfinite.

So for this specific failure:

```text
do not clip gradients first
do not lower LR first
fix/log geometry and renderer safety first
```

## Proof: Scale Normalization Preserves Projection

If we normalize both cameras and Gaussians:

```text
X_norm = (X_raw - mu) / s
C_norm = (C_raw - mu) / s
R_norm = R_raw
```

then:

```text
X_c_norm = R.T (X_norm - C_norm)
         = R.T ((X_raw - mu)/s - (C_raw - mu)/s)
         = (1/s) R.T (X_raw - C_raw)
         = X_c_raw / s
```

Projection:

```text
u_norm = f (X_c_raw.x/s) / (X_c_raw.z/s) + c
       = u_raw
```

So scale/center normalization does not change rendered pixels if points and
cameras are transformed together. It only changes internal coordinates and
Gaussian world scales.

Gaussian scales must also transform:

```text
sigma_norm = sigma_raw / s
```

For learned Gaussian heads, this means the head should emit normalized scales
appropriate for normalized coordinates.

## Proof: Rotation Normalization Preserves Projection

Let:

```text
X_norm = A (X_raw - mu)
C_norm = A (C_raw - mu)
R_norm = A R_raw
```

Then:

```text
X_c_norm = R_norm.T (X_norm - C_norm)
         = (A R_raw).T (A X_raw - A C_raw)
         = R_raw.T A.T A (X_raw - C_raw)
         = R_raw.T (X_raw - C_raw)
         = X_c_raw
```

Projection unchanged.

So global rotation normalization is also a gauge choice that preserves pixels.

## Important Exception: Learned Priors Are Not Gauge-Invariant

Even though projection is unchanged under global similarity transforms, the
model is not invariant to them by default.

The model has fixed priors:

```text
tanh coordinate bounds
scale multiplier 0.05
Plucker ray projection weights
attention weights
regularizers
optimizer LR
```

These priors make one gauge easier than another. Normalization should choose the
gauge where those priors are meaningful.

This is why "DUSt3R scale is arbitrary but mathematically equivalent" is not
enough. It is equivalent for pixels, not for our parameterization.

## Camera Pose Parameterization Options

Rotation representations:

```text
axis-angle:
    compact, current path uses this
    small-angle friendly
    singular near pi

quaternion:
    smooth-ish with normalization
    double cover q and -q

6D rotation:
    common neural representation
    maps to orthonormal basis
    larger output

Lie algebra residual:
    best for small residual around known/base pose
```

Translation representations:

```text
raw xyz:
    simple, but scale-sensitive

radius + angles:
    natural for orbit/look-at
    singular at poles

look-at target + radius:
    object-centric
    can decouple target motion from camera radius

velocity integration:
    physical path
    drift risk

spline control points:
    smooth and bounded
```

For initialization, residual Lie algebra around a safe base is good. For broad
camera paths, add a low-frequency path basis.

## Look-At Parameterization

Instead of predicting full rotation directly, predict:

```text
C_t
target_t
roll_t
```

Then:

```text
forward_t = normalize(target_t - C_t)
right_t = normalize(cross(up_or_roll, forward_t))
up_t = cross(forward_t, right_t)
```

Pros:

```text
strong object-centric prior
camera naturally points at scene
easy radius interpretation
```

Cons:

```text
harder for cameras not looking at target
up/roll handling
singular if camera equals target
```

Hybrid:

```text
look-at base rotation + small residual rotation
```

This is essentially current implicit base camera plus residual, but the path
could also predict moving target:

```text
C_t = orbit/path center
target_t = origin + small target residual
R_t = look_at(C_t, target_t) * roll/residual
```

This may generalize better than unconstrained rotation for object-centric clips.

## Camera Target Motion

A path can be decomposed into:

```text
camera center C_t
look target L_t
roll rho_t
fov_t
```

For a tracking shot, target may move:

```text
L_t != 0
```

If we force target origin, the model might explain object motion through
deforming Gaussians. That may be wrong.

For dynamic scenes, allow:

```text
target_t = target_base + bounded smooth residual
```

with:

```text
||target_residual|| <= small fraction of scene extent
```

This gives object tracking without throwing away object-centric prior.

## Camera FoV Dynamics

Current global camera head predicts one FoV for implicit model, not per-frame
zoom.

Real videos can have zoom/focus changes. DUSt3R intrinsics may vary per frame.

Options:

```text
fixed FoV per sequence:
    simpler
    good for most phone clips if no zoom

per-frame FoV residual:
    handles zoom
    can absorb geometry errors

known intrinsics:
    use DUSt3R/camera metadata
```

For pretraining, per-frame FoV should be tightly regularized if allowed:

```text
L_fov_smooth
L_fov_delta
```

Otherwise the model can cheat by changing focal length instead of moving
geometry/camera.

## Near Plane Should Be A Modeling Parameter

Renderer near clamp `1e-4` is numerical. But model diagnostics should use a
larger semantic near margin:

```text
near_numeric = 1e-4
near_model = 0.1 or 0.25 in canonical units
```

Require:

```text
most initialized Gaussians have z_c > near_model
```

This leaves room for gradients and covariance.

If canonical camera radius is 3 and object extent 1:

```text
near_model = 0.25
```

is easy.

If normalized data has:

```text
z_min_quantile < 0.25
```

flag it even if renderer won't NaN.

## Far Plane / Too Far Diagnostics

Too far is less catastrophic but weakens gradients.

Define:

```text
far_model maybe 10 or 20 canonical units
```

If most Gaussians are:

```text
z_c > far_model
```

then:

```text
projected scales tiny
parallax weak
position gradients small
```

For canonical object radius 1 and camera radius 3:

```text
z in [2,4]
```

is ideal. Values in:

```text
[1,8]
```

are probably acceptable. Values near:

```text
0 or 100
```

are bad.

## Camera Path Arc Length

Normalize camera path not just by radius but by arc length:

```text
arc_length = sum_t ||C_{t+1} - C_t||
```

For a turntable orbit of radius r and angle span theta:

```text
arc_length = r theta
```

If normalized radius is 3 and theta=30 deg:

```text
arc_length = 3 * 0.524 = 1.57
```

If DUSt3R normalized path has arc length 20 for a short clip, likely scale or
jitter issue.

Metrics:

```text
arc_length / median_radius
max_step / median_radius
step_length_quantiles
```

These are dimensionless and good across scenes.

## Look-At-Origin Angle

For each camera:

```text
view_dir_t = R_t[:,2]
to_origin_t = normalize(-C_t)
angle_t = acos(dot(view_dir_t, to_origin_t))
```

If object-centric normalization is good for object-centric video, angle should
usually be modest:

```text
0-15 deg: strong look-at
15-45 deg: plausible
>60 deg: camera not looking at origin or origin wrong
```

For known-camera DUSt3R after first-frame normalization, if origin is first
camera rather than object center, this angle is not meaningful. After
object-centric normalization it becomes meaningful.

## Frustum Fit Diagnostics

For canonical scene sphere radius E at origin and camera radius r/FoV:

Horizontal fit requires:

```text
angular_radius = asin(E / r)
angular_radius < fov_x / 2 * margin
```

Equivalently:

```text
E / sqrt(r^2 - E^2) < tan(fov/2) * margin
```

For rough simpler version:

```text
E < k r tan(fov/2)
```

Log:

```text
fit_ratio = E / (r tan(fov/2))
```

If:

```text
fit_ratio < 0.5:
    object small in frame

fit_ratio ~= 0.7:
    good margin

fit_ratio > 1:
    object radius exceeds half-frame
```

This should be computed per frame.

## Dataset Split Risk

If pretraining data is normalized with one method and evaluation/overfit data
with another, conclusions will be noisy.

Every dataset item should declare:

```text
normalization_version
normalization_policy
source_camera_solver
source_video_transform
```

Training configs should check compatibility:

```text
if model expects normalized cameras but data says raw:
    fail loudly
```

Do not silently train normalized model on raw cameras.

## Versioning Normalization

Normalization is part of data semantics. Treat it like schema.

Example:

```text
camera_schema_version: 2
coordinate_frame:
    type: object_centric_similarity_normalized
    center: points_robust_q
    scale: hybrid_object_camera_fov
    orientation: first_camera_axes
```

If a new normalization policy is introduced:

```text
write new output directory
do not overwrite old camera json silently
```

This avoids "old bad data" confusion.

## Bad Data Clearing Checklist

Before rerunning training after a bake mistake:

```text
1. confirm source video path is original base video
2. confirm ffmpeg command is stored next to output
3. confirm frame count and fps with ffprobe
4. remove or archive old DUSt3R output directory
5. rerun DUSt3R
6. inspect per_frame_cameras timestamp/schema
7. run camera diagnostics
8. run one-step renderer diagnostics
9. only then run 1000-step training
```

This checklist belongs in workflow notes or a script eventually.

## What The 128 Failure Teaches

The failure is not "128 is too high resolution."

The failure is:

```text
128/4fps DUSt3R cameras define a camera trajectory whose scale/position is
incompatible with the current known-camera Gaussian init slab.
```

128 may have contributed through:

```text
narrower DUSt3R FoV estimate
different DUSt3R pose scale
higher focal after scaling
more explosive projection terms
```

But the root is coordinate contract mismatch.

If normalized cameras put the object at canonical radius and the renderer culls
invalid Gaussians safely, 128 should at least start without NaN.

Quality is a separate issue.

## Ideas For Immediate Experiments

No fix requested in this note, but future experiments:

```text
E0: run current 128 with renderer fail-fast metrics
    status: done, failure localized to projection/near-camera

E1: render initial random Gaussians under all 128 cameras without training
    output depth quantile table and worst frames

E2: normalize 128 cameras by simple translation scale so median radius=3
    keep Gaussian head same
    see if NaN disappears

E3: normalize by FoV fit radius
    compare front counts/projected footprints

E4: object-centric known-camera head with scene_extent=1
    compare to slab

E5: renderer robust cull before covariance/exponent
    verify diagnostics still show bad geometry but no NaN

E6: opacity init lower
    see if gray collapse changes after geometry is finite

E7: increase Gaussian count for 128
    separate capacity from stability

E8: same video 64 vs 128 normalized trajectory compare
    determine DUSt3R resolution sensitivity
```

Do these in order. Do not jump straight to gradient clipping.

## Theories To Keep In Mind

Theory A:

```text
Narrow FoV from DUSt3R is real.
The source video/crop has telephoto-like framing or DUSt3R infers narrow FoV.
Need larger canonical radius.
```

Theory B:

```text
Narrow FoV is a DUSt3R artifact at low resolution.
Camera path/intrinsics need smoothing or external intrinsics prior.
```

Theory C:

```text
The model's current Gaussian slab is first-camera-centric and fundamentally
misaligned with multi-frame object-centric training.
```

Theory D:

```text
Renderer robustness bug hid behind stable baseline because old cameras did not
stress near/behind Gaussians.
```

Theory E:

```text
Gray render is an optimization attractor after geometry invalidity, not a W&B
media bug.
```

All can be partially true.

## Mathematical Invariants Worth Targeting

Invariant 1:

```text
median camera radius / scene radius ~= 3 to 8 depending on FoV
```

Invariant 2:

```text
projected scene radius fraction ~= 0.5 to 0.8 of half-frame
```

Invariant 3:

```text
initial Gaussian projected sigma ~= 1 to 6 px
```

Invariant 4:

```text
initial Gaussian depth quantile 1% > near_model
```

Invariant 5:

```text
Plucker moment std in a stable band, maybe 0.3 to 2
```

Invariant 6:

```text
camera step length / radius has sane quantiles
```

Invariant 7:

```text
FoV and radius jointly explain apparent object scale
```

Invariant 8:

```text
render is finite even when invariants fail
```

## Questions For Future Notes

More detailed questions to answer with experiments:

```text
1. What are depth quantiles for the actual initialized Gaussians on 32/64/128?
2. What are support-depth bounds for the whole Gaussian head range?
3. How much does DUSt3R pose scale change across 64 vs 128?
4. Is the 128 narrow FoV consistent frame-to-frame?
5. Does simple median-radius normalization remove NaN?
6. Does FoV-aware normalization improve loss vs median-radius only?
7. Does object-centric Gaussian head beat camera slab after normalization?
8. Does lower opacity init avoid gray collapse?
9. Does renderer robust culling change gradients for valid Gaussians?
10. How many Gaussians are needed for 128 to beat blur/mean baselines?
11. Are Plucker moments dominating feature activations?
12. Does normalizing Plucker moments help convergence?
13. Can a path spline reduce jitter vs per-frame MLP?
14. Does direct-from-base path limit long camera sequences?
15. Do camera cuts exist in planned pretraining data?
```

## Short Current Belief

My current best mental model:

```text
The project needs one canonical coordinate contract:

    scene centered at origin
    scene radius around 1
    camera path normalized around that scene
    camera radius chosen with FoV/object fit in mind
    Plucker moments normalized by the same scale
    Gaussian heads emit coordinates/scales in that canonical space
    renderer robustly ignores invalid geometry before unstable math

The old stable baseline worked because the old data happened not to violate the
contract too badly. The corrected 128/4fps data violated it immediately.
```

The most important pretraining point:

```text
Do not make the model learn away arbitrary DUSt3R gauges.
Choose the gauge once, log it, validate it, and train every example in that
shared frame.
```

## Backtracking And Assumption Audit

This note should not become dogma. A lot of the reasoning above is a working
model, not a settled proof of how the system must be built. Future work should
actively try to falsify the assumptions here.

The right stance:

```text
derive a hypothesis
write the assumptions that make it true
design a cheap diagnostic that can break it
keep the result even if it invalidates the hypothesis
```

### Assumption: The 128 Failure Is Primarily Camera Scale

Why we believe it:

```text
decoded Gaussian params were finite
failure happened at step 1
diagnostics showed many near/behind Gaussians
renderer internals overflowed during projection/exponent math
```

What could invalidate it:

```text
camera scale normalization does not reduce near/behind counts
NaNs persist after all Gaussians are demonstrably in front
same failure occurs with identity cameras
render covariance math is wrong even for safe points
```

Test:

```text
render current initialized Gaussians with identity/default camera
render same Gaussians with 128 DUSt3R cameras
render synthetic safe Gaussian cloud with 128 DUSt3R cameras
render current Gaussians after simple camera scale normalization
```

Interpretation:

```text
if synthetic safe cloud fails:
    renderer math bug independent of initialization

if current cloud fails only under 128 cameras:
    camera/initial cloud compatibility issue

if normalized cameras still fail with near/behind counts low:
    covariance/inversion bug or focal/covariance regularization issue
```

### Assumption: DUSt3R Scale Is Arbitrary And Needs Normalization

This is theoretically true for monocular reconstruction, but practical DUSt3R
may have implicit priors that make scale somewhat consistent for a given
pipeline.

What could invalidate the urgency:

```text
across many clips/resolutions, DUSt3R camera scale already lands in a narrow
range compatible with the Gaussian head
```

Test:

```text
collect camera path radius and depth stats across baked sequences
compare raw scale distributions before normalization
compare failure rate raw vs normalized
```

Possible outcome:

```text
raw DUSt3R scale may be consistent enough for a local benchmark
but still not a sound pretraining contract
```

So even if raw scale works sometimes, normalization can still be the right
long-term design. But we should know whether it is fixing the present bug or
future-proofing.

### Assumption: Object-Centric Coordinates Are Better Than First-Camera Slab

Why we believe it:

```text
video/multiview training should represent scene geometry independent of the
first sampled camera
implicit-camera branch already uses object-centric origin/radius
pretraining benefits from one canonical scene frame
```

What could invalidate or weaken it:

```text
dataset is mostly forward-facing single-view-ish clips
object center is hard to estimate without masks
camera-centric slab trains faster and transfers better
object-centric orientation ambiguity causes more harm than first-camera bias
```

Test:

```text
train matched configs:
    first-camera slab + normalized camera scale
    object-centric box + normalized cameras

compare:
    startup finite rate
    loss vs mean baseline
    render quality
    cross-view consistency
    transfer/pretraining validation
```

Possible backtrack:

```text
keep camera-centric slab for known-camera overfit/debug
use object-centric only for implicit/pretraining branch
or bridge them with explicit transforms
```

### Assumption: FoV-Aware Radius Is Necessary

The derivation says apparent object size depends on:

```text
apparent_size ~ f * E / r
```

So narrow FoV should imply larger radius if scene extent is fixed.

What could invalidate it:

```text
DUSt3R narrow FoV estimate is wrong or unstable
the scene/object extent E estimated from points changes with FoV in compensating
ways
the video crop truly fills frame and target projected size should be large
the Gaussian model benefits from a different apparent-size prior
```

Test:

```text
log fit_ratio = E / (r tan(fov/2)) for raw and normalized cameras
compare normalized variants:
    fixed radius target
    FoV-aware radius target
    point-cloud radius only
```

Possible backtrack:

```text
use FoV-aware radius only as diagnostic, not as normalization rule
preserve radius target 3 if empirical convergence is better
```

### Assumption: Plucker Moment Normalization Helps

Why we believe it:

```text
m = C cross d
if C scale changes, m scale changes
ray_proj sees arbitrary channel scale
```

What could invalidate it:

```text
ray_proj learns to normalize moment scale easily
raw moment magnitude carries useful absolute camera-distance signal
normalizing moments removes a cue needed by current architecture
```

Test:

```text
run same normalized camera data with:
    raw Plucker
    scene-scale-normalized moment
    direction-only plus camera token

log:
    ray_proj activation stats
    convergence
    camera/geometry metrics
```

Possible backtrack:

```text
use LayerNorm/standardization on ray features instead of hand-normalizing
or feed both raw and normalized moments
```

### Assumption: MLP Camera Path Is Too Unconstrained

The note argues MLP smoothness is not guaranteed.

What could invalidate the concern:

```text
for current clip lengths, spectral bias plus temporal loss is enough
MLP path fits sharp turns better than splines without causing jitter
the bottleneck is camera scale, not temporal parameterization
```

Test:

```text
after scale/renderer stability is fixed, compare:
    direct per-frame MLP residual
    low-degree polynomial
    cubic spline controls
    spline + residual MLP

measure:
    pose jitter
    render loss
    novel/interpolated frame render
```

Possible backtrack:

```text
do not change path architecture until normalized-camera baseline is stable
```

### Assumption: Renderer Robustness Should Be Fixed Separately

Why:

```text
bad geometry should not produce NaN
diagnostics are easier when failures are finite
pretraining will hit outliers
```

What could go wrong:

```text
robust culling hides real geometry bugs
clamping exponent/covariance changes gradients for valid splats
renderer changes break parity with intended 3DGS math
```

Test:

```text
create synthetic render cases:
    all splats safely in front
    some behind
    some near
    narrow FoV
    tiny/large scales

verify:
    safe case unchanged or nearly unchanged
    invalid cases finite
    diagnostics count invalid splats
```

Backtrack rule:

```text
if a robustness patch changes safe-case renders materially, it is too invasive
or needs to be configurable
```

### Assumption: Gray Means Geometry Failure

Gray render strongly suggests geometry/color/opacity collapse, but do not assume
one cause.

Other causes:

```text
W&B media panel stale or showing wrong step
render video composed with alpha/background issue
RGB head initialized/biased incorrectly
opacity too low everywhere
all splats offscreen but finite
training sampling logs only hard frames
```

Test:

```text
save raw render tensor stats
save local PNG/MP4 outside W&B
log alpha sum and color std
compare Render_GT_vs_Pred panels at the exact step
```

Possible backtrack:

```text
if local renders are not gray, debug logging/media path before model
```

### Assumption: Current Coordinate Convention Interpretation Is Correct

The note assumes:

```text
camera_to_world columns are right/up/forward
means_camera = (means - translation) @ rotation_cw
positive z is forward
```

What could invalidate it:

```text
row/column convention mismatch in some path
DUSt3R pose matrices use opposite convention
relative-pose multiplication order is wrong
renderer helper silently transposes differently
```

Test:

```text
unit tests with hand-built cameras:
    camera at (0,0,-3) looking at origin => origin depth 3
    point to right projects right
    point up projects up/down as expected by image convention

DUSt3R sanity:
    project known point/cloud into images
    verify camera centers and viewing directions visually
```

Backtrack:

```text
if convention test fails, rewrite this note's derivations with the corrected
transform before making normalization changes
```

### Assumption: First-Frame Relative Poses Are Stable

Current known-camera data uses:

```text
T_t = inverse(T_0) T_t_raw
```

What could be wrong:

```text
composition side/order may not match camera_to_world semantics
using first frame as reference amplifies a bad first pose
first frame may not be representative
```

Test:

```text
compare first-frame-relative vs median-camera-relative vs object-centric
plot camera centers and forward vectors
render DUSt3R point cloud from normalized cameras if available
```

Backtrack:

```text
if first frame is bad, choose a reference frame by confidence/centrality or use
similarity normalization independent of any single frame
```

### Assumption: The Note's Math Uses The Right Simplifications

Several derivations assume:

```text
spherical object radius
symmetric square FoV
camera looking at origin
isotropic Gaussian scale
small-angle rotations
principal point at image center
```

These are simplifications, not reality.

What to do:

```text
use derivations for intuition and initial thresholds
use actual tensor diagnostics for decisions
```

For real cameras:

```text
use f_x and f_y separately
use actual camera forward vectors
use actual Gaussian covariance
use actual point/camera quantiles
use actual principal point offsets
```

Backtrack rule:

```text
if a simple formula and actual diagnostic disagree, trust the diagnostic and
update the formula/assumption.
```

### Invalidated Ideas Should Stay In Notes

Do not delete wrong hypotheses from loose notes. Mark them:

```text
status: invalidated
why: diagnostic result
replacement hypothesis: ...
```

This is important because the failed path may become relevant again under a
different config.

Suggested format:

```text
Hypothesis:
    ...

Why it seemed plausible:
    ...

Test:
    ...

Result:
    ...

Status:
    supported / weakened / invalidated / unresolved

Next:
    ...
```

### Current Highest-Value Falsification Tests

Before implementing a big normalization redesign, the best tests are:

```text
1. Hand-camera projection unit sanity.
2. Synthetic safe Gaussian render under narrow FoV.
3. Current random init depth quantiles for 32/64/128 cameras.
4. Simple median-radius normalized 128 render smoke.
5. Raw vs normalized Plucker channel stats.
6. Local render media save independent of W&B.
```

These tests can invalidate large chunks of this note quickly. That is good.

## Expansion Pass 2: Turn The Theories Into Testable Programs

This section is intentionally more procedural. The earlier sections argue about
geometry and priors. This section asks:

```text
What exact tests would make us trust or reject those arguments?
What exact quantities should a script print?
What exact normalization policies should be compared?
```

The main goal is to avoid vague future debugging. "Camera scale is wrong" is too
loose. A useful note says:

```text
compute these tensors
print these quantiles
expect these inequalities
if they fail, this hypothesis weakens
```

## Minimal Geometry Test Suite

Before changing initialization again, create a tiny geometry test suite that
does not depend on real videos.

### Test 1: Look-At Camera Sign Convention

Construct:

```text
image_size = 128
camera = make_orbit_camera(radius=3, azimuth=0, elevation=0, fov=60)
```

Expected:

```text
C = (0,0,-3)
R_cw = identity
origin camera coords = (0,0,3)
point (1,0,0) projects right of center
point (0,1,0) projects below or above depending image y convention, but
consistently with renderer and ray builder
point (0,0,-4) is behind camera
```

Exact values for central projection:

```text
S = 128
f = 0.5*S/tan(30deg) = 110.851
cx = cy = 64

X = (0,0,0):
    X_c = (0,0,3)
    u = 64
    v = 64

X = (1,0,0):
    X_c = (1,0,3)
    u = 110.851 * 1/3 + 64 = 100.950

X = (-1,0,0):
    u = 27.050
```

If this fails, all later derivations using sign/depth convention need revision.

### Test 2: Camera Ray Consistency

For the same camera, build rays. The center ray should point roughly:

```text
d_center ~= (0,0,1)
origin ~= (0,0,-3)
```

The ray through pixel center should pass through world origin at depth 3:

```text
origin + 3 * d_center ~= (0,0,0)
```

For a pixel at `u = cx + f/3`, expected normalized camera direction:

```text
d_camera = normalize([1/3, 0, 1])
```

World direction should equal camera direction for identity orientation.

This checks:

```text
build_camera_rays
projection
renderer camera transform
```

are mutually consistent.

### Test 3: Plucker Central Ray

For base camera:

```text
C = (0,0,-3)
d_center = (0,0,1)
m = C cross d = (0,0,0)
```

For right-ish ray:

```text
d = normalize([a,0,1])
C cross d = (0, -3a/sqrt(1+a^2), 0)
```

So moment y should be negative for positive x ray under current coordinates.

This catches cross-product sign mistakes.

### Test 4: Similarity Transform Invariance

Use a synthetic point cloud and camera. Render or project points before and
after:

```text
X' = s A X + b
C' = s A C + b
R' = A R
```

Expected:

```text
pixel coordinates identical up to numerical tolerance
depths scaled by s
projected covariance identical if Gaussian scales also multiplied by s
```

The last condition matters:

```text
Sigma_3d' = s^2 A Sigma_3d A.T
J' = J / s
J' Sigma_3d' J'.T = J Sigma_3d J.T
```

If Gaussian scale is not scaled with positions/cameras, render changes. That is
expected and important.

### Test 5: Near-Plane Robustness

Synthetic Gaussians:

```text
X_safe = (0,0,3)
X_near = (0,0,1e-6)
X_behind = (0,0,-1)
```

with identity camera.

Expected after a robust renderer patch:

```text
render finite
safe Gaussian contributes
near/behind Gaussians counted as invalid
invalid Gaussians do not produce inf/NaN power or alpha
```

Before a robust patch, this test should reproduce the failure mechanism. That
is useful as a regression test.

## Quantile Tables Beat Single Scalars

For camera/initialization diagnostics, print tables like:

```text
metric                         p00     p01     p05     p50     p95     p99     p100
camera_radius
camera_step
fov_x_degrees
look_at_origin_angle
init_depth_all_gaussians
init_projected_sigma_px
plucker_moment_norm
```

Single min/max values are useful for worst-case failures, but quantiles tell
whether the whole distribution is bad.

Example interpretation:

```text
init_depth p00 = -0.8, p01 = 1.5:
    one or few outliers; renderer robustness may be enough

init_depth p00 = -0.8, p50 = 0.1:
    initialization/normalization is badly wrong

init_depth p05 = 0.2, p50 = 3:
    marginal near-plane risk
```

The 128 failure had many near/behind Gaussians in the failing frame window, so
it was not just a one-point outlier.

## Define Camera Normalization Policies Explicitly

Use named policies so logs are searchable.

### Policy P0: Raw First-Frame Relative

This is the current known-camera behavior:

```text
T_t_norm = inverse(T_0_raw) T_t_raw
scale = raw DUSt3R scale
```

Expected:

```text
works only if DUSt3R scale happens to match Gaussian head
good baseline for comparing old behavior
```

Do not delete this policy. It is the control.

### Policy P1: First-Frame Relative Plus Median Camera Radius

Start with current relative poses, then scale translations:

```text
C_t_rel = C_t after first-frame relative transform
r_med = median_t ||C_t_rel - C_center||
C_t_norm = C_t_rel * (r_target / r_med)
```

Need choose `C_center`. Options:

```text
C_center = 0
C_center = mean camera center
C_center = optical-axis center
```

P1a:

```text
C_center = 0
```

This is questionable because origin is first camera, not object.

P1b:

```text
C_center = mean camera center
```

Better for path scale, still not necessarily object center.

Expected:

```text
simple scale fix
may remove NaN if scale is main issue
may not improve object framing
```

Falsifies:

```text
if P1 removes NaN immediately, raw scale mismatch was enough to explain startup
if P1 does not reduce near/behind counts, center/orientation also wrong
```

### Policy P2: Optical-Axis Center Plus Radius Target

Estimate common look target `mu` from camera rays.

Each camera optical axis:

```text
line_t(lambda) = C_t + lambda f_t
```

where:

```text
f_t = camera forward vector
```

Find point `mu` minimizing squared distance to all lines:

```text
min_mu sum_t ||(I - f_t f_t.T)(mu - C_t)||^2
```

This is linear:

```text
A = sum_t (I - f_t f_t.T)
b = sum_t (I - f_t f_t.T) C_t
mu = solve(A, b)
```

If `A` is ill-conditioned, optical axes are nearly parallel or geometry is weak.
Use damping:

```text
mu = solve(A + lambda I, b)
```

Then:

```text
r_med = median_t ||C_t - mu||
s = r_med / r_target
C_t_norm = (C_t - mu) / s
```

Orient optionally so first camera axes become canonical.

Expected:

```text
good for object-centric videos where cameras roughly look at subject
bad for forward motion where optical axes are parallel
```

Diagnostics:

```text
condition number of A
distance from each optical axis to mu
look-at-origin angles after normalization
```

### Policy P3: Point-Cloud Robust Center And Radius

Use DUSt3R points:

```text
mu = robust point center
s = robust point radius / E
```

Normalize:

```text
X_norm = (X_raw - mu) / s
C_norm = (C_raw - mu) / s
```

Robust center choices:

```text
coordinate median
geometric median
trimmed mean after confidence filtering
```

Robust radius:

```text
q90 ||X_i - mu||
q95 ||X_i - mu||
median + k*MAD
```

Expected:

```text
best if point cloud tracks the actual subject/scene
bad if points include huge background/outlier structure
```

Diagnostics:

```text
point confidence distribution
radius q50/q90/q95/q99
camera radius after point normalization
frustum fit ratio
```

### Policy P4: Hybrid Point-Camera-FoV Fit

Use both points and cameras:

```text
mu = robust point center if reliable else optical-axis center
E_raw = robust point radius
r_raw = median_t ||C_t - mu||
r_fit_t = E / (k tan(fov_t/2))
r_target = median_t r_fit_t
s_obj = E_raw / E
s_cam = r_raw / r_target
s = blend_or_max(s_obj, s_cam)
```

`max` is conservative:

```text
s = max(s_obj, s_cam)
```

But max can make one side too small:

```text
if point cloud has huge background, object becomes tiny
if camera radius estimate bad, points become huge
```

Robust blend:

```text
s = sqrt(s_obj * s_cam)
```

or:

```text
s = exp(weighted_mean(log s_obj, log s_cam))
```

with reliability weights.

Expected:

```text
best long-term candidate
more failure modes
requires good diagnostics
```

### Policy P5: Frustum-Only Fit

Ignore points. Choose scale so current Gaussian support fits all camera frusta.

Given desired support `B` in normalized coordinates, solve scale/center such
that:

```text
for most frames, B is in front and roughly in frame
```

This is tricky because support is model prior, not data scene.

Could be useful as emergency compatibility:

```text
make cameras compatible with current Gaussian head
```

But it may ignore actual object location. Use only as debug/control.

### Policy P6: Do Not Normalize But Make Renderer Robust

Control:

```text
raw cameras
robust renderer
```

Expected:

```text
no NaN
may still optimize to gray/mean
near/behind diagnostics still bad
```

This separates:

```text
numerical survival
from
good coordinate contract
```

### Policy P7: Learn Per-Sequence Similarity

Instead of fixed normalization from data, add trainable:

```text
scale a
translation b
rotation A
```

and render:

```text
X_render = a A X_model + b
T_render = transform cameras consistently or inverse-transform points
```

Pros:

```text
model can find gauge that fits photometric loss
```

Cons:

```text
reintroduces gauge ambiguity
can hide data issues
bad for pretraining if unconstrained
```

Maybe useful for overfit debugging, not as main pretraining path.

## Normalization Policy Evaluation Matrix

Run each policy on 32/64/128:

```text
policy   finite_step1   depth_p01   depth_p50   fit_ratio_p50   loss_100   gray?   notes
P0
P1a
P1b
P2
P3
P4
P6
```

Important:

```text
same seed
same frames_per_step
same W&B disabled for smoke tests
same diagnostics
```

First pass can use one-step forward/render only. Do not run full 1000 steps for
every policy until the geometry diagnostics narrow the choices.

## Geometric Median For Point Center

Coordinate median:

```text
mu = [median x, median y, median z]
```

is robust and cheap but not rotation invariant.

Geometric median:

```text
mu = argmin_p sum_i w_i ||p - X_i||
```

is rotation invariant. Weiszfeld iteration:

```text
mu_{k+1} = sum_i w_i X_i / max(||X_i - mu_k||, eps)
           --------------------------------------------
           sum_i w_i / max(||X_i - mu_k||, eps)
```

Use confidence weights if available.

Stop when:

```text
||mu_{k+1} - mu_k|| < tol
```

Need robust filtering before/after:

```text
discard points with low confidence
discard points beyond q99 after initial center
recompute
```

Possible problem:

```text
if point cloud includes large background, geometric median may land between
foreground and background
```

No center estimate is perfect without masks.

## Robust Scale From MAD

Given radii:

```text
r_i = ||X_i - mu||
```

Median absolute deviation:

```text
m = median_i r_i
mad = median_i |r_i - m|
```

Scale estimate:

```text
r_robust = m + k * mad
```

For Gaussian-like distributions, `1.4826 * MAD` estimates standard deviation,
but point radii are positive and not Gaussian. Quantiles may be simpler:

```text
r_robust = q90(r_i)
```

Compare:

```text
q80, q90, q95, q99
```

If:

```text
q99 / q90 >> 1
```

there are strong outliers.

If:

```text
q90 / q50 >> 1
```

point cloud has long tails or mixed foreground/background.

## Optical Axis Least Squares Derivation

Distance from point `p` to line `(C, f)`:

```text
d_line(p)^2 = ||(I - f f.T)(p - C)||^2
```

Since:

```text
P = I - f f.T
P is symmetric
P^2 = P
```

objective:

```text
J(p) = sum_t ||P_t(p - C_t)||^2
     = sum_t (p - C_t).T P_t (p - C_t)
```

Derivative:

```text
dJ/dp = 2 sum_t P_t (p - C_t)
```

Set zero:

```text
(sum_t P_t) p = sum_t P_t C_t
```

So:

```text
p = A^{-1} b
A = sum_t P_t
b = sum_t P_t C_t
```

If all forward vectors are parallel:

```text
A has nullspace along the common direction
```

The closest point along that direction is underdetermined. Condition number
tells us whether the estimate is reliable.

Weighted version:

```text
A = sum_t w_t P_t
b = sum_t w_t P_t C_t
```

Weights can come from:

```text
pose confidence
image sharpness
frame inclusion
DUSt3R confidence
```

## Kabsch/Procrustes For Aligning Camera Paths

To compare 64 vs 128 normalized trajectories, align centers by similarity:

Given matched camera centers:

```text
A_i = centers from 64
B_i = centers from 128
```

Find:

```text
s,R,t = argmin sum_i ||s R A_i + t - B_i||^2
```

Umeyama algorithm gives closed form.

After alignment, report:

```text
RMSE
median error
max error
scale ratio
rotation angle
```

If 64 and 128 camera paths are identical up to similarity:

```text
normalization can make them comparable
```

If not:

```text
DUSt3R produced genuinely different camera paths
```

That matters for interpreting training differences.

## Path Smoothness Metrics

For camera centers:

```text
v_t = (C_{t+1} - C_t) / dt
a_t = (v_{t+1} - v_t) / dt
```

Dimensionless:

```text
speed_norm = ||v_t|| / median_radius
accel_norm = ||a_t|| / median_radius
```

For rotations:

```text
R_rel_t = R_t.T R_{t+1}
theta_t = acos((trace(R_rel_t)-1)/2)
angular_speed = theta_t / dt
```

For 2fps vs 4fps:

```text
if same physical path:
    speed distributions similar
    step lengths roughly scale with dt
    total arc length similar
```

If 4fps path has large alternating acceleration:

```text
pose jitter
```

Add:

```text
jitter = median ||C_{t+1} - 2C_t + C_{t-1}||
```

normalized by radius.

## Pose Graph Sanity Without Images

A camera path can be invalid even if each pose matrix is finite.

Check:

```text
det(R) close to +1
R.T R close to I
translation finite
focal positive
principal point in image-ish range
```

For every camera:

```text
abs(det(R)-1) < 1e-3
||R.T R - I|| < 1e-3
fx > 0, fy > 0
0 <= cx <= W, 0 <= cy <= H, or at least near
```

If DUSt3R writes camera_to_world with scale/shear in rotation block, projection
math assumptions break.

## Rotation Averaging Caution

Do not average rotation matrices elementwise unless re-orthonormalizing.

Bad:

```text
R_mean = mean_t R_t
```

This is not generally a rotation.

Better:

```text
average quaternions with sign alignment
or use Lie algebra around a reference:
    delta_t = log(R_ref.T R_t)
    delta_mean = mean delta_t
    R_mean = R_ref Exp(delta_mean)
```

For this project, avoid fancy rotation averaging until needed. First-camera
orientation or no orientation normalization is easier to reason about.

## Camera Coordinate Frame From First Camera

Current first-frame relative transform likely does:

```text
T_t_rel = inverse(T_0) T_t
```

For camera-to-world matrices, check meaning.

Let:

```text
X_world = T_t X_cam_t
```

To express world in first-camera coordinates:

```text
X_cam0 = inverse(T_0) X_world
```

Then camera t to cam0 coordinates:

```text
X_cam0 = inverse(T_0) T_t X_cam_t
```

So:

```text
T_t_rel = inverse(T_0) T_t
```

is correct if:

```text
T_t is camera_to_world
relative world is first camera coordinates
```

But the renderer expects `camera_to_world` in whatever world coordinate is used.
If relative world is first-camera coordinates, then `T_t_rel` is camera t to
relative world. Good.

Need test:

```text
T_0_rel should be identity
origin point at (0,0,z) should project in first frame center
```

## Hand-Derived Camera Center From Matrix

For camera_to_world:

```text
C = T[:3,3]
R_cw = T[:3,:3]
forward = R_cw[:,2]
right = R_cw[:,0]
up = R_cw[:,1]
```

For world_to_camera:

```text
X_c = R_wc X_w + t_wc
C = -R_wc.T t_wc
```

If a matrix is actually world_to_camera but treated as camera_to_world, camera
centers and depths will be wrong.

DUSt3R loader should be tested with known projection or docs/metadata.

## Coordinate Convention Visual Debug

Generate a simple 3D axes render:

```text
red point:    +X
green point:  +Y
blue point:   +Z
white point:  origin
```

Render with first camera and save PNG.

Expected under current camera:

```text
+X appears right
+Y appears one vertical direction consistently
+Z appears near center but different depth
```

Also render camera frusta as lines in a simple matplotlib 3D plot:

```text
camera centers
forward rays
origin
Gaussian support box
```

This often reveals errors faster than tables.

## Gaussian Support Box vs Camera Frusta

For current known-camera head, support box corners:

```text
x in {-1.5, 1.5}
y in {-1.5, 1.5}
z in {0.5, 2.5}
```

For every camera, transform corners to camera coordinates and project.

Log:

```text
min_depth
max_depth
fraction corners in front
u_min,u_max,v_min,v_max
box_fit_ratio
```

If the support box is mostly behind late cameras, the architecture/data contract
is definitely broken independent of random initialization.

For object-centric support:

```text
x,y,z in {-1,1}
```

same diagnostic.

This should be a standalone debug utility.

## Exact Frustum Planes For Fit

For each camera, a point is inside image bounds if:

```text
0 <= f_x x/z + c_x <= W
0 <= f_y y/z + c_y <= H
z > near
```

Equivalent inequalities:

```text
x >= -c_x z / f_x
x <= (W - c_x) z / f_x
y >= -c_y z / f_y
y <= (H - c_y) z / f_y
z > near
```

These define frustum planes in camera coordinates.

For support box corners, exact projected min/max is okay but not sufficient
because perspective over a box can have extrema at corners if z positive and
box convex? Usually corners suffice for conservative checks under positive z,
but if box crosses near plane, all bets are off.

So first require:

```text
all or most corners z > near_model
```

then use projected bounds.

## Expected Object Size From Support Box

For object-centric sphere radius E:

```text
angular_radius = asin(E / r)
pixel_radius = f * tan(angular_radius)
             = f * E / sqrt(r^2 - E^2)
```

For `r >> E`:

```text
pixel_radius ~= f E / r
```

The exact formula matters when `r` is close to `E`.

If:

```text
r = 2, E = 1, f = 111
```

exact:

```text
pixel_radius = 111 / sqrt(3) = 64.1
```

approx:

```text
55.5
```

Difference is meaningful.

For `r=3`:

```text
exact = 111 / sqrt(8) = 39.2
approx = 37.0
```

close.

Use exact for diagnostics if easy.

## Gaussian Scale And Covariance Regularization

Current scale:

```text
sigma_i = exp(raw_i) * 0.05
```

3D covariance from scales/quaternion:

```text
Sigma_3d = R diag(sigma^2) R.T
```

Projection:

```text
Sigma_2d = J Sigma_3d J.T + epsilon I
```

If `epsilon` is too small, tiny projected Gaussians can produce large inverse
covariances. If too large, renders blur.

Diagnostics:

```text
det(Sigma_2d)
eig_min(Sigma_2d)
eig_max(Sigma_2d)
condition_number
```

For safe positive definite 2D covariance:

```text
eig_min > 0
det > 0
condition_number not insane
```

If raw determinant goes negative, the covariance math or numerical precision is
bad. True covariance should be positive semidefinite before numerical errors.

Potential fixes:

```text
symmetrize covariance:
    Sigma = 0.5 * (Sigma + Sigma.T)

add regularization:
    Sigma += eps I

eigendecomp clamp:
    eigvals = clamp(eigvals, min=eps, max=max_var)

avoid computing invalid points:
    skip if z <= near_model
```

Backtracking note:

```text
If covariance is indefinite for safe z and normal scales, then the renderer
projection math itself needs review, not just initialization.
```

## Precision: fp32 vs Lower Precision

Earlier question: should renderer cast inputs to fp32?

For projection/covariance:

```text
perspective division
covariance construction
matrix inverse/determinant
exp(power)
cumprod alpha
```

are numerically sensitive. fp32 is safer.

Lower precision might help speed but can worsen:

```text
small determinant accuracy
near-plane division
alpha transmittance cumprod
exp overflow/underflow
```

Potential mixed precision:

```text
keep camera transform, projection, covariance inverse in fp32
compute large pixel grid alpha/rgb accumulation in fp16/bf16 if stable
```

But MPS support/performance may vary.

Do not assume lower precision fixes speed. Test:

```text
fp32 renderer
mixed projection fp32 + raster fp16
full fp16/bf16 if supported
```

Metrics:

```text
it/s
render diff vs fp32
NaN rate
loss convergence
```

For the current NaN, lower precision is likely worse, not better.

## Batched Renderer Performance Model

Dense renderer work:

```text
means2D:   O(BG)
dx/power:  O(BGHW)
alpha:     O(BGHW)
cumprod:   O(BGHW)
rgb sum:   O(BGHW)
```

Python overhead in frame loop:

```text
O(B) launches and function calls
```

Batched render reduces Python/frame overhead and can improve GPU occupancy.

But memory:

```text
alpha shape = B x G x H x W
```

For:

```text
B=4, G=512, H=W=128
elements = 4*512*16384 = 33,554,432
fp32 memory ~= 134 MB for alpha alone
```

Several intermediates can multiply this. For:

```text
B=23, H=W=128
elements ~= 193M
fp32 alpha ~= 773 MB
```

So full all-frame batched 128 render may be memory-heavy.

Tiling or chunking over Gaussians/frames may be needed:

```text
frame batch size
gaussian chunk size
pixel tile size
```

Speed/stability config should log:

```text
effective B
H,W
G
renderer path
peak memory if available
```

## Why Tiny Res Can Be Slow On MPS

At very small H/W:

```text
kernel launch overhead
Python overhead
attention overhead
W&B/media overhead
```

can dominate `O(GHW)` math.

At larger H/W:

```text
GPU math utilization improves
```

until memory dominates.

So 64/4fps can be faster in it/s than 32/all-frame if:

```text
fewer frames per step
better batching
less Python overhead
```

Throughput should be normalized:

```text
frames/sec = B * it/s
pixels/sec = B * H * W * it/s
gaussian_pixel/sec = B * G * H * W * it/s
```

Add these to logs if speed is a goal.

## Training Schedule And Stability

Learning rate schedule matters after render is finite.

Questions to record per run:

```text
optimizer type
base lr
warmup?
decay?
separate lrs for camera/gaussian/color?
gradient clipping?
```

Potential issue:

```text
same lr for xyz, scale, opacity, rgb, camera can be suboptimal
```

For example:

```text
rgb can learn fast
xyz/camera should maybe move slower
scale/opacity can destabilize alpha
```

Possible param groups:

```text
xyz lr
scale lr
opacity lr
rgb lr
camera lr
backbone lr
```

But do not add this complexity until:

```text
step-1 render finite
normalization diagnostics sane
mean-color baseline beaten
```

## Separating Geometry From Appearance During Warmup

Potential training phases:

```text
phase 0: fixed safe initialization, render diagnostics only
phase 1: learn colors/opacities with xyz/camera frozen
phase 2: unfreeze xyz/scales
phase 3: unfreeze camera residuals
```

Alternative:

```text
freeze rgb initially and let geometry/opacity align
```

Which is better is not obvious.

Risks:

```text
freeze geometry:
    if init geometry bad, colors just memorize mean

freeze color:
    if colors gray, geometry gradients may be weak/ambiguous
```

For current project, first priority is coordinate compatibility. Warmups are
secondary experiments.

## Mean-Color Collapse Mechanism

If geometry is bad, rendered image may be approximately:

```text
render ~= sum_i weights_i rgb_i + background
```

If weights are broad or uninformative, gradient wrt rgb pushes colors toward:

```text
argmin_c mean ||c - image||^2 = mean(image)
```

Thus RGB can quickly become mean color.

Opacity can also learn to reduce damage:

```text
if background white and image has bright sky, low opacity can reduce loss
if rgb mean close to image mean, broad alpha can reduce loss
```

This is why beating mean-color baselines matters.

## Camera-Geometry Cheating Modes

Photometric loss alone allows cheats:

```text
move camera instead of moving Gaussians
scale scene and camera together
increase FoV instead of moving camera back
inflate Gaussian scales to blur
lower opacity to reveal background
use per-frame Gaussians instead of consistent scene
encode view-specific appearance in tokens
```

Regularization/architecture should decide which cheats are allowed.

For pretraining:

```text
consistent scene geometry across frames matters
camera path should explain viewpoint changes
dynamic tokens should explain actual object motion, not camera gauge noise
```

## Dynamic Scene Gauge Ambiguity

If Gaussians are time-dependent:

```text
X_i(t)
```

then camera motion and scene motion are even more ambiguous:

```text
camera moves right + object static
can look like
camera static + object moves left
```

Need priors:

```text
camera path smooth/low-dimensional
object motion local/sparse/smooth
background mostly static
```

Initialization should not put camera and object motion on equal footing if one
is known from DUSt3R.

For known-camera training:

```text
trust cameras enough to learn dynamic scene
or learn small camera residuals with penalty
```

For implicit:

```text
camera head needs strong priors to avoid absorbing object motion
```

## Pretraining Data Curriculum

Possible curriculum:

```text
stage 1:
    static-ish, object-centric, good DUSt3R confidence, normalized cameras

stage 2:
    mild dynamic content, smooth camera paths

stage 3:
    more complex motion, wider FoV/radius variation

stage 4:
    noisy cameras, learned residual correction
```

Do not start pretraining with arbitrary raw videos and expect the model to learn
the gauge. It will learn shortcuts.

Dataset filters:

```text
camera normalization diagnostics pass
enough front/visible support
no severe pose jumps unless cut-aware
FoV in plausible range
point/camera scale agreement
mean-color baseline not too low/high
```

## Randomizing Gauge During Pretraining

Contrary idea: after canonical normalization, randomly apply similarity
transforms during training to force equivariance.

For example:

```text
random yaw rotation
random small scale jitter
random translation jitter
```

If cameras and Gaussians transform together, pixels unchanged. But model inputs
include cameras/rays and outputs Gaussians. Random gauge augmentation may teach
the model to be less brittle.

Risks:

```text
makes target coordinate less deterministic
harder for model to learn canonical geometry
can conflict with fixed Gaussian head bounds
```

Maybe useful later:

```text
train in canonical gauge first
then add small gauge jitter to improve robustness
```

## Canonical Orientation Ambiguity

Even with center/scale fixed, orientation can be arbitrary. If object has
symmetry, PCA axes can flip or rotate.

Options:

```text
use first camera orientation:
    deterministic, view-dependent

use gravity/up if available:
    stable for videos with metadata or horizon

use object PCA:
    object-centric but sign/symmetry ambiguous

do not canonicalize rotation:
    leave orientation to raw DUSt3R/world

randomize orientation:
    encourage equivariance
```

For now, first camera orientation is pragmatic for known-camera overfit. For
pretraining, view-dependent first-camera orientation means same object from
different starting views has different canonical geometry. That may or may not
matter depending on objective.

If goal is reconstruct video, not category-level canonical object, first-camera
orientation can be okay. If goal is reusable object priors, object-centric
orientation matters more.

## What Does "Generalizes" Mean Here?

Need define target generalization:

```text
G1: same training code works across configs/resolutions/videos
G2: pretrained weights initialize new overfit clips faster
G3: model predicts good geometry for unseen videos without per-scene optimize
G4: learned latent space has consistent object/scene coordinates
G5: novel-view interpolation works
```

Different initialization choices serve different goals.

For G1:

```text
renderer robustness + diagnostics + simple normalization may be enough
```

For G3/G4:

```text
canonical object-centric coordinate contract becomes much more important
```

Keep this distinction visible. Do not over-engineer pretraining invariants if
the immediate target is local overfit stability, but do not build local hacks
that block pretraining.

## A Possible Two-Layer Coordinate System

Use both:

```text
camera frame:
    convenient for image encoding and initial ray/depth features

canonical scene frame:
    used for persistent Gaussians and cameras
```

Architecture:

```text
image encoder sees camera-frame rays
pose encoder sees camera-to-canonical transform
Gaussian decoder emits canonical Gaussians
renderer uses canonical cameras
```

This avoids feeding huge raw Plucker moments directly into image features while
still giving pose.

Token data:

```text
per-pixel:
    normalized pixel coords / camera ray dir

per-frame:
    normalized camera center
    rotation 6D or quaternion
    fov
    time

per-sequence:
    normalization scale metadata if useful
```

This may be cleaner for pretraining than current `image_features + ray_proj`.

## Architectural Split: Geometry Tokens vs Appearance Tokens

Current token model uses shared tokens to produce:

```text
xyz, scale, rotation, opacity, rgb
```

Possible split:

```text
geometry tokens:
    xyz, scale, rotation, opacity

appearance tokens:
    rgb/view-dependent color
```

Why:

```text
geometry should be camera/scale consistent
appearance may be view/light dependent
```

Camera encoding might affect them differently:

```text
geometry sees normalized pose
appearance sees view direction
```

This is beyond immediate initialization, but gray collapse suggests color can
dominate when geometry is weak.

## Explicit Depth Prior From Images

Another initialization route:

```text
predict coarse depth map
lift image features to 3D
initialize Gaussians around lifted points
```

Known camera:

```text
X(u,v) = C + depth(u,v) d(u,v)
```

Depth can be:

```text
constant canonical depth
DUSt3R depth
monocular depth estimator
learned depth head
```

Pros:

```text
front-of-camera guaranteed for source frame
spatial correspondence
better color init
```

Cons:

```text
requires depth supervision/prior
can overfit source view
does not solve camera scale across frames alone
```

Maybe useful for pretraining after camera normalization.

## Gaussian Count Scaling With Resolution

If target is similar visual detail per pixel, Gaussian count should scale with
image area:

```text
G proportional H*W
```

Old:

```text
32x32, G=512
```

Equivalent densities:

```text
64x64: 2048 Gaussians
128x128: 8192 Gaussians
```

That is much more expensive.

If keeping G=512 at 128:

```text
expect coarse reconstruction only
```

This does not explain NaN, but it does set expectations for loss/visual quality.

Alternative:

```text
train 128 with same G for stability only
train quality configs with higher G or hierarchical splats
```

## Hierarchical / Multi-Resolution Gaussian Idea

For speed and stability:

```text
coarse Gaussians:
    large scales, low count, explain broad color/layout

fine Gaussians:
    smaller scales, higher count, added after warmup
```

Training schedule:

```text
start 32/64 coarse
upsample/add tokens for 128
```

This mirrors image pyramids and can reduce gray collapse.

But first fix camera normalization.

## Camera Pose Residual Over DUSt3R

For known-camera training, instead of fixed cameras:

```text
T_t = T_dust3r_norm_t Exp(delta_t)
```

where `delta_t` is small se(3) residual.

Regularize:

```text
L_delta = ||translation_delta||^2 / sigma_t^2 + ||rotation_delta||^2 / sigma_r^2
L_smooth = temporal residual smoothness
```

Benefits:

```text
corrects small DUSt3R errors
can reveal whether camera labels are limiting optimization
```

Risks:

```text
photometric loss may move cameras to cheat
more ambiguity with dynamic objects
```

Use after fixed-camera normalized baseline.

## Camera-Intrinsics Residual

Similarly:

```text
f_t = f_dust3r_t * exp(delta_log_f_t)
cx_t = cx_dust3r_t + delta_cx_t
cy_t = cy_dust3r_t + delta_cy_t
```

Regularize strongly.

This can test whether DUSt3R's narrow FoV is harmful. If optimization pushes
FoV wider consistently, maybe DUSt3R intrinsics are wrong or the model prior
prefers wider cameras.

But intrinsics residual can cheat geometry. Use diagnostics and bounds.

## Per-Frame vs Shared Intrinsics

Phone video usually has fixed intrinsics across frames unless digital zoom or
crop changes.

DUSt3R may output varying intrinsics. We currently use median focal mode in some
configs.

Options:

```text
per-frame intrinsics:
    flexible, follows solver

shared median focal:
    stable, reduces noise

known source intrinsics:
    best if available
```

For generated center-crop resized video, if original camera intrinsics unknown,
shared median may be okay. But if DUSt3R's focal estimate is biased narrow, a
shared biased median preserves the bias.

Diagnostics:

```text
focal variation over frames
correlation between focal and pose jumps
render stability with per-frame vs median focal
```

## Principal Point

Current center-cropped square videos likely have principal point near center.

If DUSt3R principal point differs:

```text
cx,cy offsets
```

can affect rays and Plucker moments.

For low-res crops, principal point estimation may be noisy.

Possible policy:

```text
force cx=cy=S/2 for synthetic/resized center crops
use median focal
```

Test:

```text
compare DUSt3R cx/cy to center
run diagnostics/render with centered principal point
```

Do not assume principal point noise is irrelevant. At 64px, a few pixels is a
large angular offset.

## Original Video Crop Geometry

We corrected video generation to:

```text
center crop square from original base video
fps=4
scale to 64 or 128
```

Center crop changes FoV:

```text
cropping removes horizontal or vertical field depending original aspect
resizing preserves cropped FoV
```

If original video is portrait 2160x3840 and we crop square:

```text
crop size = 2160x2160
vertical FoV is reduced from original full portrait height
horizontal FoV maybe same as original width
```

DUSt3R sees only the crop. Its focal estimate corresponds to cropped video.

Need store:

```text
original source resolution
crop box
output resolution
fps
```

This affects any attempt to use original camera metadata.

## Video Source Stability

If future data has rolling shutter, stabilization, or digital zoom:

```text
camera path is not a simple pinhole rigid camera
```

Symptoms:

```text
DUSt3R pose jitter
per-frame intrinsics variation
photometric residuals even with good geometry
```

Initialization cannot solve this, but diagnostics can flag it.

## Handling Bad Frames

If a few frames have bad cameras:

```text
skip them?
downweight them?
learn residual?
robust loss?
```

For overfit baseline:

```text
skip/inspect bad frames may be okay
```

For pretraining:

```text
data loader should filter or mark frame confidence
```

Frame-level diagnostics:

```text
near/behind init count by frame
pose jump by frame
focal outlier by frame
image difference by frame
DUSt3R confidence by frame
```

Output top-k worst frames with indexes. The 128 failure frames around 35-40
were a useful clue.

## Loss Masking For Invalid Geometry?

If some frames have invalid cameras, do not just mask their loss silently. That
can hide data bugs.

Maybe:

```text
if invalid count > threshold:
    fail fast in debug mode

if pretraining large data:
    skip sample and log dataset issue
```

Training through invalid geometry is not helpful unless the model is supposed to
learn camera correction.

## Differentiability Of Normalization

Data normalization can be nondifferentiable:

```text
median
quantile
RANSAC
confidence filtering
```

That is fine if normalization is preprocessing.

If normalization is inside model/training and affects gradients, be careful.
For now, keep normalization outside gradient path.

## The Role Of Background

3D Gaussian splatting with white background:

```text
out = render + (1 - accumulated_alpha) * white
```

If images contain sky/bright background, low alpha/white background can produce
low-ish loss. If images contain dark content, background mismatch stronger.

Mean-color and background baselines should include:

```text
white background loss
black background loss
global mean loss
per-frame mean loss
```

If white background loss is already close to model loss, the model may not need
learn much to appear stable numerically.

## Alpha Ordering

Current dense renderer likely composites Gaussians in parameter order, not depth
sorted. True 3DGS usually sorts by depth or uses tile sorting.

If order is arbitrary:

```text
opacity/occlusion behavior is not physically correct
```

For initialization/stability, arbitrary order can cause:

```text
early gray splats occlude later useful splats
```

Diagnostics:

```text
depth-sorted vs unsorted render on same params
```

If depth sorting improves gray collapse or gradients, renderer order matters.

This is separate from NaN, but relevant to baseline quality.

## Alpha Saturation Metric

Log:

```text
accumulated_alpha_mean
accumulated_alpha_p95
accumulated_alpha_p99
transmittance_final_mean
```

Cases:

```text
alpha near 0 everywhere:
    model mostly background

alpha near 1 everywhere:
    saturated/opaque; early splats dominate

alpha moderate with spatial variation:
    healthier
```

Also log:

```text
number of splats contributing > threshold per pixel
```

This can diagnose too-high opacity init.

## Projected Gaussian Footprint Metrics

For each projected covariance:

```text
eigvals lambda1, lambda2
sigma_px_major = sqrt(max(lambda))
sigma_px_minor = sqrt(min(lambda))
area_px = 2*pi*sqrt(det(Sigma_2d)) maybe
```

Log quantiles:

```text
sigma_major p50/p95/p99
sigma_minor p50/p95/p99
area p50/p95
```

Bad:

```text
sigma_major >> image size
sigma_minor near 0
negative/NaN eigvals
```

Healthy initial:

```text
sigma maybe 1-10 px depending resolution
not hundreds
```

This directly tests FoV/radius/scale interaction.

## Offscreen Metrics

A Gaussian can be in front but offscreen.

Log:

```text
means2D u/v quantiles
fraction with 0 <= u < W and 0 <= v < H
fraction within expanded bounds [-margin, W+margin]
```

If most Gaussians are offscreen, loss gradients may be weak and render may be
background/gray.

For current 128 issue, check:

```text
front count
onscreen count among front
```

A camera can have many front Gaussians but few useful onscreen splats.

## Initial Depth Distribution From Head Support

For support box `B`, camera depth is linear:

```text
z_c = f_t dot (X - C_t)
```

The extrema over a box occur at corners. For quick support safety:

```text
z_min_box = min_corners z_c
z_max_box = max_corners z_c
```

But actual tanh/sigmoid heads do not sample uniformly over the support.

Approximate actual distribution:

```text
sample random raw from observed head raw distribution
transform through tanh/sigmoid
or run current initialized model
```

Both support and actual matter:

```text
support failure:
    architecture allows invalid points

actual failure:
    current seed/input produces invalid points
```

For pretraining, architecture-level support should be safe under normalized
cameras when possible.

## Data Normalization Cannot Make Every Possible Head Output Safe

If head support is large:

```text
x,y,z in [-10,10]
```

no reasonable camera normalization can ensure all points are visible.

Safety should target:

```text
likely initial distribution
intended scene support
not arbitrary saturated tanh corners if they are unreachable early
```

But if tanh can saturate during training, renderer robustness must handle
out-of-support/pathological points.

## LR And Gauge Scaling

If coordinates are scaled, gradients scale.

For a coordinate parameter `X_norm` transformed to render coordinate:

```text
X_render = s X_norm
```

gradient:

```text
dL/dX_norm = s dL/dX_render
```

So changing normalization scale can effectively change xyz learning rate if
parameters are in normalized coordinates but render scale differs.

If we keep all training in normalized coordinates and render normalized cameras,
scale is fixed. Good.

If we introduce per-sequence scale transforms inside model, need LR awareness.

Another reason to canonicalize data before model.

## Opacity/Scale Gauge

Gaussian scale and opacity can trade off:

```text
larger Gaussian with lower opacity
smaller Gaussian with higher opacity
```

can produce similar color coverage.

Regularizers may be needed:

```text
scale prior
opacity sparsity/entropy
alpha coverage target
```

Initialization should avoid starting at a degenerate extreme.

## Camera Radius And Scale Gauge

Object size in image depends on:

```text
E / r
```

If both scene extent and camera radius multiply by same scale, pixels unchanged.

But Gaussian scale prior `0.05` fixes an absolute scale in model coordinates.
Thus canonical `E=1, r~3` makes `0.05` meaningful.

If data normalization uses `E=10, r=30`, same pixels but Gaussian scale init is
too small in world units unless also scaled.

Again: choose canonical coordinate range to match model priors.

## The "Same Pixels, Different Optimization" Principle

Many transforms preserve rendered pixels exactly if all variables transform
consistently. But optimization changes because:

```text
parameter bounds
initial values
learning rates
regularization
finite precision
renderer epsilons
activation nonlinearities
```

Therefore:

```text
projective equivalence does not imply training equivalence
```

This is the core reason initialization/normalization is worth this much thought.

## Proposed Debug Utility Layout

Potential files:

```text
src/train/debug_geometry.py
src/train/debug_metrics.py
src/train/camera_normalization.py
```

Responsibilities:

```text
debug_geometry.py:
    projection sanity
    camera path stats
    support box stats
    Plucker stats
    frustum fit stats

camera_normalization.py:
    named normalization policies P0-P4
    transform cameras/points
    emit normalization report

debug_metrics.py:
    runtime renderer/optimizer metrics
```

Keep trainer lean:

```text
if config["camera_normalization"]["enabled"]:
    sequence = normalize_sequence(sequence, policy_config)

if with_metrics.camera:
    log_camera_diagnostics(...)
```

## Report Format

Write JSON plus human text.

JSON:

```json
{
  "schema_version": 1,
  "policy": "P2_optical_axis_radius",
  "source_sequence": "...",
  "image_size": 128,
  "frame_count": 46,
  "intrinsics": {
    "fov_x_degrees": {"p50": 19.2}
  },
  "camera_path": {
    "radius": {"p01": 2.8, "p50": 3.0, "p99": 3.4}
  },
  "init_support": {
    "depth": {"p01": 1.2, "p50": 3.0}
  }
}
```

Text:

```text
Camera normalization report: P2
Frames: 46, size: 128
FoV x median: ...
Radius median: ...
Worst init depth frame: ...
Verdict: PASS/FAIL with reasons
```

Reports should be committed or stored with baked data for reproducibility.

## Pass/Fail Thresholds Draft

Debug thresholds, not laws:

```text
intrinsics:
    5 deg < fov_x < 120 deg
    fx,fy finite positive

rotation:
    |det(R)-1| < 1e-3
    ||R.T R - I|| < 1e-3

camera path:
    radius p50 in [1.5, 12] canonical units
    max step/radius not huge unless cut

init depth:
    p01 > 0.1
    p05 > 0.25 preferred

projected sigma:
    p50 in [0.5, 20] px
    p99 < maybe 2*image_size

onscreen:
    at least 10-30% front Gaussians near screen early

Plucker:
    moment p99 not enormous, maybe < 10 after normalization
```

These thresholds should be revised after data.

## Result Recording Discipline

When an experiment changes belief, update:

```text
agent_notes/loose_notes/YYYY-MM-DD_HH-MM-SS_experiment_slug.md
agent_notes/key_learnings.md if surprising/high-signal
```

For each experiment:

```text
command
config
git commit/status
data dirs
W&B URL if any
diagnostic summary
what changed in belief
what remains unknown
```

Do not only record successful fixes. Failed assumptions are the useful part.

## "Stable Enough To Run All" Checklist

Before telling user to run all configs:

```text
1. data source verified
2. no stale bad DUSt3R dirs
3. config points to expected sequence dir
4. camera diagnostics pass
5. one-step render finite
6. mean-color baseline logged
7. W&B media local render checked
8. full run completes once
9. second seed or repeated run does not immediately fail
```

For the current state, 128 fails at item 5. So do not call it stable yet.

## Concrete Hypothesis Tree For Current 128 Run

Root:

```text
128 step-1 render NaN
```

Branch A: invalid camera/scene compatibility.

Evidence:

```text
many near/behind Gaussians
camera z min/max bad
```

Tests:

```text
camera normalization smoke
support box depth diagnostics
```

Branch B: renderer cannot handle invalid points.

Evidence:

```text
0*inf => NaN after opacity mask
power huge positive
det negative
```

Tests:

```text
synthetic near/behind splats
pre-cull invalid points
```

Branch C: covariance math wrong even for valid points.

Evidence needed:

```text
negative determinant for safe z points
NaN with synthetic safe cloud
```

Tests:

```text
safe synthetic covariance eigenvalue check
```

Branch D: data/video mismatch.

Evidence:

```text
wrong video source
stale output
wrong camera dir
```

Current status:

```text
bad 2fps-derived videos were corrected
but still verify configs point to corrected dirs
```

Branch E: W&B/logging artifact.

Evidence against:

```text
actual training NaN diagnostics, not just media gray
```

Still test local render for gray issue.

## Update The Stable Baseline Definition In Docs

The phrase "stable baseline" should refer to:

```text
config path
data directory
video source
git commit
renderer path
frames_per_step
image size
fps
seed if relevant
```

Example:

```text
stable_old_32_2fps_all_frames:
    commit: be87e96
    config: src/train_configs/local_mac_overfit_prebaked_camera.jsonc
    data: test_data/dust3r_outputs/test_video_small_all_frames
    frames: 23
    size: 32
    fps: 2
    frames_per_step: all/23
```

Then new baselines:

```text
candidate_64_4fps:
    corrected source video
    data dir
    normalization policy
    metrics status

candidate_128_4fps:
    currently failing raw P0 at step 1
```

This prevents memory drift.

## Things That Should Not Be Conflated

Keep separate:

```text
video resolution:
    64 vs 128 pixels

source FPS:
    2 vs 4 frames/sec

sequence length:
    23 vs 46 frames

frames per optimizer step:
    all vs 4 sampled

DUSt3R processing resolution:
    may differ from training render size

camera FoV:
    inferred intrinsics, not same as image size

camera path scale:
    translation units from DUSt3R

Gaussian coordinate range:
    model head prior
```

A speed or stability observation is uninterpretable unless these are named.

## What To Avoid Next

Avoid:

```text
blind gradient clipping
blind LR reduction
blind lower precision renderer
changing many architecture pieces before one-step diagnostics
overwriting baked data without versioning
trusting W&B media alone
calling gray non-NaN training "stable"
```

Prefer:

```text
small synthetic tests
one-step real-data smoke
named normalization policies
quantile diagnostics
local render artifacts
explicit baseline labels
```

## If We Backtrack Entirely

Suppose future tests show:

```text
coordinate normalization does not improve 128
renderer robust culling fixes NaN and optimization works
object-centric head performs worse
raw DUSt3R scale is stable across data
```

Then update belief:

```text
the immediate issue was renderer invalid-point handling, not data gauge
normalization remains a pretraining cleanliness idea but not urgent for local
baselines
```

Even then, keep:

```text
diagnostics
renderer robustness
data provenance
baseline labels
```

because those are useful independent of the normalization theory.

Suppose future tests show:

```text
simple normalization fixes NaN but loss still gray
```

Then:

```text
startup failure solved
next bottleneck likely capacity/opacity/color/render ordering/camera quality
```

Do not keep changing normalization to solve every downstream problem.

Suppose future tests show:

```text
64 and 128 DUSt3R paths disagree after similarity alignment
```

Then:

```text
resolution-dependent camera solving is a data issue
consider shared source-frame camera solve or external intrinsics/poses
```

## One-Line Current Next Best Action

After this note, the next concrete engineering step should probably be:

```text
write a debug geometry utility that prints camera/support/depth/Plucker
quantiles for a config before training, then run it on 32, 64, and 128.
```

That will either support or break the main theory quickly.

## Expansion Pass 3: Mathematical And Implementation Appendices

This pass is for details that are too specific for the main argument but useful
when turning the note into code.

## Appendix A: Projection Derivatives

Perspective projection:

```text
u = f_x x / z + c_x
v = f_y y / z + c_y
```

Jacobian wrt camera-space point:

```text
du/dx = f_x / z
du/dy = 0
du/dz = -f_x x / z^2

dv/dx = 0
dv/dy = f_y / z
dv/dz = -f_y y / z^2
```

Matrix:

```text
J = [
    f_x/z,     0, -f_x*x/z^2
        0, f_y/z, -f_y*y/z^2
]
```

Near-plane sensitivity:

```text
||J|| grows like O(1/z^2) when x,y not near zero
```

If point is on optical axis:

```text
x = y = 0
||J|| grows like O(1/z)
```

So off-axis near points are worse than central near points.

Projected covariance:

```text
Sigma_2d = J Sigma_3d J.T
```

If isotropic:

```text
Sigma_3d = sigma^2 I
```

then:

```text
Sigma_2d = sigma^2 J J.T
```

Approx center ray:

```text
Sigma_2d ~= sigma^2 diag((f_x/z)^2, (f_y/z)^2)
```

Pixel std:

```text
sigma_u ~= sigma f_x / z
sigma_v ~= sigma f_y / z
```

This is the main scale formula.

## Appendix B: Positive Definite 2D Covariance

For 2x2 covariance:

```text
S = [[a,b],
     [b,c]]
```

Positive definite iff:

```text
a > 0
det = ac - b^2 > 0
```

Eigenvalues:

```text
lambda_1,2 = 0.5 * ((a+c) +/- sqrt((a-c)^2 + 4b^2))
```

Condition number:

```text
kappa = lambda_max / lambda_min
```

Inverse:

```text
S^{-1} = (1/det) [[c, -b],
                  [-b, a]]
```

If `det` is tiny or negative:

```text
inverse explodes or becomes indefinite
```

Renderer diagnostic should include:

```text
a_min
c_min
det_min
lambda_min_min
condition_p99
negative_det_count
```

If `negative_det_count > 0` for valid front points, investigate covariance
construction.

## Appendix C: Exponent Safety

Renderer exponent:

```text
power = -0.5 dx.T invCov dx
```

For positive semidefinite invCov:

```text
dx.T invCov dx >= 0
power <= 0
exp(power) <= 1
```

Therefore:

```text
power > 0
```

is a direct signal that `invCov` is not positive semidefinite or numerical
errors are severe.

Useful diagnostics:

```text
power_max
count(power > 0)
count(power > 20)
count(power > 80)
```

Why 80:

```text
exp(80) ~= 5.54e34
```

Near fp32 max:

```text
fp32 max ~= 3.4e38
exp(88.7) ~= fp32 max
```

So:

```text
power > 80
```

is already catastrophic.

Safe clamping:

```text
power_clamped = clamp(power, max=0, min=-max_power_abs)
```

But clamping positive power to zero hides covariance invalidity. Better in
debug:

```text
count and fail before clamping if positive power for valid points
```

In production robust mode:

```text
invalid covariance => zero contribution for that Gaussian/pixel or Gaussian
```

## Appendix D: SO(3) Exponential Map

Axis-angle vector:

```text
omega in R3
theta = ||omega||
a = omega / theta
```

Skew matrix:

```text
[a]_x = [
    0, -a_z, a_y
    a_z, 0, -a_x
    -a_y, a_x, 0
]
```

Rodrigues:

```text
R = I + sin(theta)[a]_x + (1 - cos(theta))[a]_x^2
```

Small angle:

```text
R ~= I + [omega]_x
```

Current implementation uses this pattern.

Gradient/numerical notes:

```text
theta near 0 needs special case
theta near pi can be ambiguous for log map, less relevant for small residuals
```

If path residual is bounded near 5 degrees per axis, exponential map is safe.

## Appendix E: SO(3) Log Map

For comparing rotations:

```text
theta = acos((trace(R)-1)/2)
```

Axis:

```text
omega = theta / (2 sin theta) * [
    R_32 - R_23,
    R_13 - R_31,
    R_21 - R_12
]
```

Small angle approximation:

```text
omega ~= 0.5 * [
    R_32 - R_23,
    R_13 - R_31,
    R_21 - R_12
]
```

Use for:

```text
rotation step metrics
rotation smoothness loss
camera path alignment
```

Clamp acos input:

```text
cos_theta = clamp((trace-1)/2, -1, 1)
```

## Appendix F: SE(3) Exponential Map

Full SE(3) exponential for twist:

```text
xi = [rho, omega]
```

where:

```text
omega = rotation vector
rho = translation-like vector
```

Then:

```text
R = Exp(omega)
t = V rho
```

with:

```text
V = I
    + (1 - cos theta)/theta^2 [omega]_x
    + (theta - sin theta)/theta^3 [omega]_x^2
```

Current code does simpler:

```text
R = Exp(omega)
t = translation_delta
Delta = [R,t]
```

This is not exactly `Exp_se3([rho,omega])` unless small angle or if `t` is meant
as post-rotation local translation. That is okay if we interpret it as direct
transform components.

For small residuals:

```text
V rho ~= rho
```

Difference is minor. If residual rotations grow, exact SE(3) may matter.

## Appendix G: Quaternion Rotation Representation

Unit quaternion:

```text
q = [w,x,y,z]
||q|| = 1
```

Rotation matrix formulas are standard. Benefits:

```text
easy normalize
good for interpolation via slerp
```

Problems:

```text
q and -q same rotation
normalization can have tiny gradients if norm near 0
```

For path residuals, axis-angle is currently simpler.

For rotation averaging/interpolation, quaternions may be useful.

## Appendix H: 6D Rotation Representation

Neural 6D rotation:

```text
a = raw[0:3]
b = raw[3:6]
r1 = normalize(a)
r2 = normalize(b - dot(r1,b) r1)
r3 = cross(r1,r2)
R = [r1,r2,r3]
```

Pros:

```text
continuous representation for neural nets
no quaternion sign ambiguity
```

Cons:

```text
less naturally bounded as small residual
not as interpretable for path scale
```

Maybe useful for direct camera orientation prediction. Less useful for small
residual around look-at base.

## Appendix I: Cubic B-Spline Basis

Uniform cubic B-spline for segment parameter `u in [0,1]`:

```text
B0(u) = (1 - 3u + 3u^2 - u^3) / 6
B1(u) = (4 - 6u^2 + 3u^3) / 6
B2(u) = (1 + 3u + 3u^2 - 3u^3) / 6
B3(u) = u^3 / 6
```

Point:

```text
C(u) = B0 P0 + B1 P1 + B2 P2 + B3 P3
```

Properties:

```text
C2 continuous
local support over 4 control points
smooth by construction
```

Velocity:

```text
dB0/du = (-3 + 6u - 3u^2)/6
dB1/du = (-12u + 9u^2)/6
dB2/du = (3 + 6u - 9u^2)/6
dB3/du = (3u^2)/6
```

Use:

```text
control points predicted by sequence token
per-frame t samples spline
small residual MLP adds correction
```

Bounding:

```text
if control points are bounded in a ball/box, spline lies in their convex hull
```

This is useful for camera safety.

## Appendix J: Bezier Curve

Cubic Bezier:

```text
C(u) = (1-u)^3 P0
     + 3(1-u)^2 u P1
     + 3(1-u) u^2 P2
     + u^3 P3
```

Pros:

```text
simple for short sequence
endpoints explicit
```

Cons:

```text
global over sequence
less local than B-spline
```

For short overfit video:

```text
Bezier camera center + look-at target residual
```

could be a clean experiment.

## Appendix K: Piecewise Linear Plus Smooth Residual

Simpler than splines:

```text
C_t = lerp(control_j, control_{j+1}, u) + residual_t
```

Regularize:

```text
||residual_t|| small
control step smoothness
```

Pros:

```text
easy
local
supports sharp-ish turns at knots
```

Cons:

```text
velocity discontinuities unless smoothed
```

May be enough for debugging before full B-spline.

## Appendix L: Umeyama Similarity Alignment

Given point sets:

```text
A_i, B_i, i=1..N
```

Compute means:

```text
mu_A = mean A_i
mu_B = mean B_i
```

Centered:

```text
A'_i = A_i - mu_A
B'_i = B_i - mu_B
```

Covariance:

```text
Sigma = (1/N) sum_i B'_i A'_i.T
```

SVD:

```text
Sigma = U D V.T
```

Rotation:

```text
S = diag(1,1,det(U V.T))
R = U S V.T
```

Scale:

```text
var_A = (1/N) sum_i ||A'_i||^2
s = trace(D S) / var_A
```

Translation:

```text
t = mu_B - s R mu_A
```

This aligns A to B:

```text
B_hat_i = s R A_i + t
```

Use for 64-vs-128 camera trajectory comparison.

## Appendix M: Weighted Umeyama

If frames have confidence weights:

```text
w_i >= 0
sum w_i = 1
```

Means:

```text
mu_A = sum w_i A_i
mu_B = sum w_i B_i
```

Covariance:

```text
Sigma = sum_i w_i B'_i A'_i.T
```

Variance:

```text
var_A = sum_i w_i ||A'_i||^2
```

Same SVD formula.

Use if some camera frames are low confidence.

## Appendix N: Debug CLI Sketch

Possible command:

```bash
uv run python src/train/debug_geometry.py \
    src/train_configs/local_mac_overfit_prebaked_camera_128_4fps.jsonc \
    --policy raw \
    --sample-init \
    --save-report test_data/debug_reports/128_raw.json
```

Output:

```text
Config: ...
Sequence: ...
Frames: 46
Image size: 128
Policy: raw

Intrinsics:
  fov_x deg p50=...

Camera path:
  radius p50=...
  step/radius p95=...

Gaussian support:
  depth p01=...
  worst frame=...

Sampled init:
  depth p01=...
  near/behind count max=...

Plucker:
  moment norm p50=...

Verdict:
  FAIL: sampled init has 195 near/behind gaussians in frame ...
```

Flags:

```text
--policy raw|median_radius|optical_axis|point_cloud|hybrid
--no-sample-init
--support camera_slab|object_box
--render-smoke
--save-report path
--save-plots dir
```

No trainer changes needed for first version.

## Appendix O: Report Schema Draft

Top-level:

```json
{
  "schema": "dynaworld.debug_geometry.v1",
  "config_path": "...",
  "sequence_dir": "...",
  "image_size": 128,
  "frame_count": 46,
  "normalization_policy": {...},
  "source": {...},
  "intrinsics": {...},
  "camera_path": {...},
  "support_diagnostics": {...},
  "sample_init_diagnostics": {...},
  "plucker_diagnostics": {...},
  "verdict": {...}
}
```

Quantile object:

```json
{
  "min": 0.0,
  "p01": 0.1,
  "p05": 0.2,
  "p50": 1.0,
  "p95": 2.0,
  "p99": 3.0,
  "max": 4.0
}
```

Verdict:

```json
{
  "status": "fail",
  "reasons": [
    "init_depth_p01 <= 0.1",
    "near_behind_count_max > 0"
  ],
  "worst_frames": [37, 38, 39, 40]
}
```

## Appendix P: Debug Geometry Functions

Potential functions:

```python
def camera_centers(cameras) -> Tensor[N,3]
def camera_forwards(cameras) -> Tensor[N,3]
def fov_from_intrinsics(cameras, width, height) -> dict
def project_points(points, camera) -> Tensor[P,2], depth
def transform_points_to_camera(points, camera) -> Tensor[P,3]
def support_box_corners(mode, params) -> Tensor[8,3]
def support_depth_stats(corners, cameras) -> dict
def optical_axis_center(cameras, weights=None, damping=1e-6) -> center, stats
def plucker_stats(cameras, image_size, scene_scale=1.0) -> dict
def quantiles(tensor, qs=(0,.01,.05,.5,.95,.99,1)) -> dict
```

Keep them pure. Trainer can import later.

## Appendix Q: Config Compatibility Checks

Before training:

```text
config image_size == loaded frame size
config sequence_dir exists
per_frame_cameras.json exists
number of images == number of cameras
intrinsics finite positive
camera_to_world shape 4x4
camera schema/normalization matches model expectation
```

If config says:

```jsonc
"expected_video": "test_video_small_128_4fps.mp4"
```

then loader can verify metadata if available.

This would have caught some stale-data confusion earlier.

## Appendix R: Data Directory Naming

Use names that encode source:

```text
test_video_small_64_4fps_from_original_center_crop
test_video_small_128_4fps_from_original_center_crop
```

Current shorter names are workable, but generated-from-wrong-source mistake
shows naming/provenance matters.

At minimum write:

```text
source_manifest.json
ffmpeg_command.txt
dust3r_command.txt
camera_diagnostics.json
```

in each output dir.

## Appendix S: Local Render Artifact Names

For render smoke:

```text
debug_outputs/
  2026-04-20_128_raw_step0/
    gt_grid.png
    pred_grid.png
    alpha_grid.png
    depth_hist.png
    camera_path.png
    report.json
```

This avoids depending only on W&B.

## Appendix T: Expected Diagnostic Outcomes For Current Data

Hypothetical expected, based on observed metrics:

```text
32 raw:
    should pass finite startup
    depth margins probably okay enough
    FoV around 31 deg
    frames=23

64 raw corrected:
    starts and completes
    may have gray/quality issue
    FoV around 21 deg
    frames=46

128 raw corrected:
    fail sampled init/render smoke
    worst late frames 35-40
    many near/behind gaussians
    FoV around 19 deg
```

If debug geometry does not show this pattern, then our interpretation of
observed runs is incomplete.

## Appendix U: Expected Outcome For Simple Scale Normalization

If raw 128 failure is mostly scale:

```text
P1/P2 normalization should:
    increase init_depth_p01 above near_model
    reduce near/behind count to zero or near zero
    avoid renderer NaN
```

But may still:

```text
render gray
have bad offscreen fraction
have huge projected sigma due to narrow FoV
```

If P1 fixes NaN but not quality:

```text
move to FoV-aware/object fit and opacity/capacity tests
```

If P1 does not fix NaN:

```text
center/orientation/convention or renderer covariance bug likely
```

## Appendix V: Expected Outcome For Renderer Robust Culling

If robust culling is added:

```text
raw 128 should not NaN
diagnostics should still report invalid geometry
loss may stay near mean baseline
```

This is acceptable. Robust renderer is not supposed to turn bad geometry into
good geometry.

Success criterion:

```text
safe synthetic renders unchanged
invalid synthetic renders finite
raw 128 fail-fast can be disabled and run produces diagnostics instead of NaN
```

## Appendix W: How To Interpret Improvements

If a change improves it/s:

```text
check frames/sec and pixels/sec
make sure frames_per_step did not change
make sure logging cadence did not change
```

If a change improves loss:

```text
compare to mean-color and blur baselines
inspect media
check geometry diagnostics did not get worse
```

If a change removes NaNs:

```text
check whether invalid geometry was fixed or just masked
```

If a change makes render not gray:

```text
check alpha, color std, projected footprint, and local artifact
```

## Appendix X: Possible Research Threads

1. Canonicalization as amortized gauge fixing:

```text
model learns in canonical coordinates
data pipeline fixes similarity gauge
```

2. Camera as latent variable with weak DUSt3R prior:

```text
DUSt3R gives pose prior
photometric training refines
```

3. Ray-conditioned token fields:

```text
image tokens grounded by normalized ray geometry
decoder emits canonical 3D tokens
```

4. Dynamic 3DGS disentanglement:

```text
separate camera motion, object rigid motion, nonrigid deformation
```

5. Multi-resolution curriculum:

```text
low-res stable geometry first
high-res detail later
```

These are not immediate fixes, but they frame why initialization matters.

## Appendix Y: Red-Team Questions

Ask these before accepting a fix:

```text
Did we accidentally tune only to this one clip?
Does the fix work if camera scale is multiplied by 10 before normalization?
Does it work for wide FoV and narrow FoV?
Does it work for 2fps and 4fps?
Does it work with W&B disabled and local artifacts?
Does it preserve old 32 baseline behavior?
Does it hide invalid geometry instead of reporting it?
Does it make pretraining coordinate targets more or less consistent?
```

If any answer is unknown, record it.

## Appendix Z: Current Standing Recommendation

Do the next work in this order:

```text
1. Write debug_geometry utility.
2. Run raw diagnostics on 32/64/128 configs.
3. Add synthetic projection/renderer tests.
4. Try simple normalization policy in diagnostics only.
5. Add renderer robust invalid-point handling with safe-case parity test.
6. Run one-step training smokes.
7. Only then run full 64/128 training.
```

This ordering preserves the ability to backtrack. It tests assumptions before
turning them into architecture.

## Expansion Pass 4: Plucker Rays As Camera Representation

The note talked about Plucker rays as conditioning, but not enough about the
stronger idea: use Plucker geometry as the camera representation itself, or as
the differentiable camera-direction/path representation.

That deserves a separate section because Plucker rays are attractive:

```text
they represent 3D lines without Euler-angle singularities
they are naturally differentiable tensor features
they combine camera direction and position relative to origin
they are already what the known-camera model feeds into the image features
```

But there is an important distinction:

```text
a single Plucker ray represents one 3D line
a pinhole camera is a constrained bundle of rays sharing one origin
an SE(3)+intrinsics camera is a low-dimensional way to generate that bundle
```

So Plucker rays can be an excellent camera *encoding*, but a raw unconstrained
Plucker grid is not automatically a valid camera.

## Plucker Basics

A 3D line can be represented by:

```text
L = (d, m)
```

where:

```text
d = unit line direction
m = p cross d
```

for any point `p` on the line.

Constraints:

```text
||d|| = 1
d dot m = 0
```

The second constraint follows:

```text
d dot (p cross d) = 0
```

For a camera ray:

```text
p = C
d = ray direction from camera center
m = C cross d
```

where `C` is camera center.

Scale note:

```text
(d,m) and (a d, a m) represent the same projective line if homogeneous
Plucker coordinates are used
```

In our code, `d` is normalized, so scale is fixed by:

```text
||d|| = 1
```

Then `m` has units of world distance.

## A Single Plucker Ray Does Not Recover Camera Center

Given:

```text
m = C cross d
```

with unit `d`, compute:

```text
d cross m = d cross (C cross d)
          = C (d dot d) - d (d dot C)
          = C - d(d dot C)
```

This is the point on the line closest to the origin, not necessarily the camera
center. The component of `C` along the ray direction is lost for that one line.

So:

```text
one Plucker ray = line in 3D
one Plucker ray != full camera pose
```

To recover a camera center, we need a bundle of rays that should intersect at
one common point.

For a valid pinhole camera:

```text
for every pixel ray i:
    m_i = C cross d_i
```

All rays share the same `C`.

Given many predicted Plucker rays, estimate `C` by:

```text
m_i = C cross d_i
```

Using skew matrix:

```text
[d_i]_x C = -m_i
```

because:

```text
C cross d_i = - d_i cross C = -[d_i]_x C
```

Actually:

```text
[d_i]_x C = d_i cross C = - C cross d_i = -m_i
```

Stack over rays and solve least squares:

```text
A C = b
A_i = [d_i]_x
b_i = -m_i
```

If the predicted rays are a valid camera bundle, one `C` explains them all.
Bundle inconsistency can be measured by residual:

```text
mean_i ||C cross d_i - m_i||
```

This gives a nice differentiable camera-validity loss.

## Plucker Bundle Camera

A camera can be represented by its whole ray bundle:

```text
R(u,v) = (d(u,v), m(u,v))
```

For a pinhole camera:

```text
d(u,v) = R_cw normalize([(u-cx)/fx, (v-cy)/fy, 1])
m(u,v) = C cross d(u,v)
```

So the Plucker grid encodes:

```text
intrinsics through the pattern of directions over pixels
rotation through direction orientation
translation through moment field
```

This is why it is a powerful conditioning signal.

But a generic neural network outputting six channels per pixel:

```text
[d_x,d_y,d_z,m_x,m_y,m_z]
```

may violate:

```text
||d|| = 1
d dot m = 0
all rays share one origin
directions form a calibrated pinhole grid
neighboring directions vary smoothly with image coordinates
```

A raw Plucker-grid camera head is highly differentiable but overparameterized.

## Why Plucker Feels More Differentiable Than Euler Cameras

Euler angles have singularities/gimbal issues. Plucker rays avoid Euler angles
because ray directions are just normalized vectors.

For a ray direction:

```text
d = normalize(raw_d)
```

This is differentiable except near:

```text
raw_d = 0
```

Moment:

```text
m = C cross d
```

is bilinear and differentiable.

If we optimize a ray bundle directly, every pixel gets a direct differentiable
ray representation. No matrix inversion or angle composition is required for
the encoding itself.

This is attractive for neural fields because the model can condition on:

```text
where this pixel ray lives in 3D line space
```

without needing to interpret camera matrices.

## Differentiability Is Not The Only Requirement

For camera learning we also need:

```text
validity
low dimensionality
identifiability
stable optimization
good priors
```

SE(3)+intrinsics:

```text
6 pose params + maybe 1-4 intrinsics params
always produces a valid pinhole camera if parameterized carefully
```

Plucker ray bundle:

```text
6 * H * W values if dense
many constraints required to be a valid pinhole camera
can represent non-pinhole/warped cameras
```

For a 128x128 image:

```text
6 * 128 * 128 = 98,304 camera ray values
```

versus:

```text
~7-10 camera pose/intrinsics values
```

The Plucker bundle is massively overparameterized if the intended camera is
pinhole.

Overparameterization can be good for robustness or nonrigid/rolling-shutter
cameras, but bad for pretraining if the model can cheat by warping rays instead
of learning geometry.

## Plucker As Output vs Plucker As Derived Feature

Two regimes:

```text
derived Plucker:
    model predicts/uses camera pose
    code derives valid Plucker rays

predicted Plucker:
    model predicts ray grid directly
    losses enforce camera validity
```

Derived Plucker is what we currently do for known cameras:

```text
CameraSpec -> build_plucker_ray_grid_batch -> image conditioning
```

It preserves a valid camera by construction because the source is a camera pose.

Predicted Plucker would be a new camera head:

```text
image/video -> plucker grid -> render/ray condition
```

Need losses:

```text
direction unit norm
d dot m = 0
common origin consistency
pinhole direction-grid consistency
intrinsics smoothness/calibration consistency
temporal smoothness
```

This could be interesting, but it is not the simplest baseline fix.

## Camera Direction As Plucker

If the main need is "camera direction," a Plucker ray can represent the central
camera ray:

```text
d_center = camera forward direction
m_center = C cross d_center
```

This central Plucker ray captures:

```text
where the optical axis line is in space
which way it points, if d is oriented
```

But it does not capture:

```text
roll about the optical axis
FoV/focal length
principal point
full image-plane basis
camera center along the optical axis from that one line
```

So a central Plucker ray is useful but incomplete.

To recover a full camera, add:

```text
up/right direction or roll
FoV/intrinsics
camera radius/center constraint
```

Possible representation:

```text
central Plucker line + roll + FoV
```

If the line is constrained to pass through a camera center on a known radius
sphere or known look-at target, then it becomes more complete.

## Plucker Look-At Camera Parameterization

For an object-centric camera looking near origin:

```text
central ray line passes through C and origin
```

Then:

```text
d_center = normalize(origin - C)
m_center = C cross d_center = 0
```

Wait: if the central ray goes through the origin, the line also passes through
origin. The closest point to origin is origin, so its Plucker moment relative to
origin is:

```text
m = 0 cross d = 0
```

This is a subtle issue.

For a perfect look-at-origin central ray, Plucker moment around the origin is
zero no matter the camera radius. The central Plucker ray alone loses radius.

Off-center rays still have nonzero moments:

```text
m(u,v) = C cross d(u,v)
```

and the full bundle encodes radius through how off-center rays miss the origin.

Therefore:

```text
central Plucker is not enough for an origin-look-at camera
full ray bundle or explicit radius is needed
```

This is a strong reason to be careful with "Plucker for camera direction."

## Full Bundle Encodes Radius For Look-At Origin

Base camera:

```text
C = (0,0,-r)
central d = (0,0,1)
central m = 0
```

Ray with camera coordinate direction:

```text
d = normalize([x,y,1])
```

Moment:

```text
m = C cross d
  = (r y, -r x, 0) / sqrt(x^2+y^2+1)
```

So radius appears in off-center moments:

```text
||m|| proportional r
```

The ray bundle, not the central ray, carries camera distance.

This also means Plucker moment statistics depend on:

```text
radius
FoV
chosen origin
```

again motivating normalization.

## Plucker Camera Validity Losses

If predicting Plucker ray grids directly:

Direction norm:

```text
L_norm = mean_pixels (||d|| - 1)^2
```

Orthogonality:

```text
L_orth = mean_pixels (d dot m)^2
```

Common origin:

```text
C_hat = least_squares_origin(d_i, m_i)
L_origin = mean_i ||C_hat cross d_i - m_i||^2
```

Pinhole smoothness:

```text
neighboring directions should lie on a projective grid
```

A weaker version:

```text
L_smooth_d = mean_neighbors ||d_i - d_j||^2
L_smooth_m = mean_neighbors ||m_i - m_j||^2
```

But smoothness alone permits non-pinhole warps.

Pinhole calibration loss:

Fit a pinhole camera to predicted rays:

```text
estimate C_hat
estimate R_hat, f_hat, cx_hat, cy_hat
reconstruct rays
L_pinhole = mean ||L_pred - L_reconstructed||^2
```

This is more complex but directly measures whether the bundle is camera-like.

## Plucker Camera As Relaxed Camera Model

A predicted Plucker grid can model:

```text
rolling shutter
lens distortion
non-central cameras
local ray warps
bad camera calibration
```

This could be useful for real internet videos.

But it also permits cheats:

```text
warp rays to match images without learning correct 3D
encode optical flow in ray changes
hide object motion as camera-ray deformation
```

For pretraining, the risk is large. A relaxed camera model can reduce loss while
destroying the consistency of learned 3D.

Possible compromise:

```text
pinhole SE(3) camera generates base Plucker grid
small residual Plucker warp predicts corrections
regularize residual heavily
```

Formula:

```text
d_pred = normalize(d_base + delta_d)
m_pred = m_base + delta_m
```

Loss:

```text
L_residual = ||delta_d||^2 + ||delta_m||^2
L_origin/L_pinhole to keep bundle valid-ish
```

This could handle calibration defects without giving total freedom.

## Plucker For Camera Path Smoothness

Instead of smoothing SE(3) params, smooth the induced ray bundles:

```text
L_ray_smooth = mean_t,p ||Plucker_t(p) - Plucker_{t-1}(p)||^2
```

This measures image-ray motion directly.

But raw Plucker difference mixes:

```text
rotation
translation
FoV
origin scale
```

Need normalize moments:

```text
L_t(p) = [d_t(p), m_t(p)/s]
```

Then:

```text
||L_t - L_{t-1}||
```

is a dimensionless camera-ray path smoothness.

This may be more meaningful than smoothing raw translation/rotation separately,
especially if FoV varies.

Potential issue:

```text
camera dolly along central axis may produce subtle central-ray changes but
larger off-center moment changes
```

Full-bundle smoothing catches that.

## Plucker For Comparing Cameras Across Resolutions

To compare 64 and 128 cameras, sample a common normalized grid:

```text
q_j in normalized image coordinates, e.g. 16x16 grid
```

For each camera set, generate Plucker rays at those normalized coordinates.

Compare after similarity normalization:

```text
direction angular error
moment error normalized by scene scale
common-origin/radius differences
```

This avoids comparing pixel grids of different size directly.

Metrics:

```text
mean angle(d64, d128)
median ||m64 - m128||
max ||m64 - m128||
```

This can reveal if 64 and 128 DUSt3R cameras differ only by scale or also by
orientation/FoV.

## Plucker And Intrinsics

Camera-frame directions:

```text
d_c(u,v) = normalize([(u-cx)/fx, (v-cy)/fy, 1])
```

For two cameras with same pose but different focal:

```text
central ray same
off-center ray directions differ
Plucker moments differ because d differs
```

So full Plucker grid encodes intrinsics. If we only encode central Plucker ray,
we lose intrinsics.

If using Plucker as camera representation, decide whether intrinsics should be:

```text
implicit in ray bundle
or explicit scalar token
```

Explicit scalar token is lower-dimensional and easier to regularize. Ray bundle
is more directly useful to the image encoder.

Best current guess:

```text
use explicit SE(3)+intrinsics for the camera state
derive normalized Plucker ray bundles for conditioning and diagnostics
optionally learn small Plucker residuals later
```

## Plucker And Differentiable Rendering

Our current dense 3DGS renderer projects 3D Gaussians using camera matrices:

```text
X_c = R.T (X_w - C)
u = f x/z + c
```

A pure Plucker renderer would instead reason about distance from Gaussian center
to ray:

```text
distance from point X to line (d,m)
```

For unit `d`, line point closest to origin:

```text
p0 = d cross m
```

Distance:

```text
dist(X, line)^2 = ||X - p0||^2 - (d dot (X - p0))^2
```

or:

```text
dist = ||X cross d - m||
```

if `d` unit and `m = p cross d`.

This is elegant and differentiable:

```text
pixel contribution could be based on distance from Gaussian to each camera ray
```

But classic splatting also needs:

```text
depth ordering
projected covariance/footprint
visibility/occlusion
```

Ray-distance rendering may be a different renderer, closer to ray marching or
point splatting by ray distance. It could avoid some perspective covariance
singularities, but it is not a drop-in replacement for current 3DGS projection.

Interesting future direction:

```text
ray-space Gaussian rendering using Plucker ray distance
```

Immediate baseline:

```text
keep matrix camera renderer, derive Plucker features from valid cameras
```

## Plucker Distance Formula

For line:

```text
L = (d,m), ||d||=1, m=p cross d
```

For point `X`, the vector:

```text
X cross d - m
```

equals:

```text
X cross d - p cross d = (X - p) cross d
```

Magnitude:

```text
||(X-p) cross d|| = distance from X to line
```

because `||d||=1`.

So:

```text
dist^2 = ||X cross d - m||^2
```

This is very differentiable and does not divide by depth.

But it does not by itself know whether the point is in front of the camera:

```text
depth = d dot (X - C)
```

Need camera center or a point on the ray plus ray orientation.

For central/pinhole camera, all rays are oriented away from camera. If only a
line is represented without `C`, front/behind is ambiguous. With bundle-derived
`C_hat`, depth is available.

## Plucker Avoids Some Singularities But Not All Camera Problems

Plucker line distance avoids:

```text
division by z in projection
Euler angle singularities
```

But still has issues:

```text
line orientation normalization
common-origin constraints
front/behind ambiguity if center not explicit
occlusion/depth ordering
scale of moment channel
degenerate central look-at-origin moment
```

Therefore:

```text
Plucker is not magic, but it is a very good differentiable camera/ray language.
```

## Proposed Plucker Experiments

Experiment PL1: normalized Plucker diagnostics.

```text
for 32/64/128:
    build Plucker grids from current cameras
    report direction stats
    report moment stats raw and scene-scale-normalized
    report central/off-center moment behavior
```

Experiment PL2: Plucker path smoothness.

```text
sample 16x16 normalized ray grid per frame
compute mean Plucker delta frame-to-frame
compare 2fps vs 4fps
```

Experiment PL3: direction+camera token vs Plucker conditioning.

```text
replace or augment ray_proj(plucker) with:
    camera-frame directions only
    world directions only
    normalized Plucker
    Plucker + explicit camera token
```

Experiment PL4: common-origin consistency for predicted residual ray warp.

```text
derive base Plucker from SE(3)
predict small delta Plucker
measure common-origin residual
```

Experiment PL5: ray-distance renderer prototype.

```text
not baseline-critical
small synthetic scene only
compare stability near z=0 against projection renderer
```

## Should We Use Plucker For Current Camera Initialization?

For the immediate 128 failure:

```text
Plucker is probably best as a diagnostic/encoding, not the first fix.
```

Reason:

```text
the renderer consumes CameraSpec matrices
DUSt3R gives camera matrices
known-camera trainer already derives Plucker from cameras
failure is camera/scene scale plus renderer invalid projection
```

But Plucker should be part of the next diagnostic utility:

```text
raw moment quantiles
normalized moment quantiles
Plucker path smoothness
64-vs-128 Plucker camera comparison
common-origin sanity if any predicted ray grids are introduced
```

For future implicit camera design:

```text
SE(3)+intrinsics state
derived Plucker grid for conditioning
optional residual Plucker warp
Plucker-bundle validity losses if predicting rays directly
```

This captures the differentiability advantage without giving up valid-camera
structure too early.

## Revised Belief After Plucker Audit

The earlier note slightly underweighted Plucker. Better statement:

```text
Plucker rays are likely the right differentiable language for camera-conditioned
features and camera diagnostics.

SE(3)+intrinsics are still likely the right compact state for a pinhole camera
baseline.

A full predicted Plucker bundle is a powerful but dangerous relaxed camera
model that should come after a valid-camera baseline is stable.
```

The most interesting future hybrid:

```text
valid SE(3)+intrinsics camera
    -> normalized Plucker bundle
    -> image/geometry conditioning
    -> small learned Plucker residual for real-video imperfections
    -> common-origin/pinhole regularization
```

That gives differentiability, expressiveness, and a path back to valid camera
geometry.

## Expansion Pass 5: Central Ray Plus FoV As A Compact Bundle Generator

The stronger version of the Plucker idea is:

```text
a camera ray bundle is just a central ray expanded by FoV
```

This is mostly right for a calibrated pinhole camera, with a few required
qualifiers.

For a square, centered, no-skew pinhole camera, the ray bundle can be generated
from:

```text
camera center C
central direction f
roll/up orientation around f
FoV
aspect ratio, if not square
principal point, if not centered
```

If those assumptions are fixed:

```text
square image
principal point at center
zero skew
known up/roll convention
```

then a camera can be close to:

```text
central oriented ray + FoV
```

If the camera is also constrained to look at origin with known/learned radius:

```text
C = -r f
```

then:

```text
direction f + radius r + FoV + roll/up convention
```

is enough to generate the full pinhole ray bundle.

## Why "Single Ray Plus FoV" Is Almost Enough

Let:

```text
f = unit central camera direction in world coordinates
C = camera center
theta = horizontal/vertical FoV for square image
```

Build an orthonormal camera frame:

```text
right = normalize(cross(up_hint, f))
up = normalize(cross(f, right))
```

For normalized image coordinates:

```text
a in [-1,1] horizontal
b in [-1,1] vertical
```

square FoV:

```text
x = a * tan(theta/2)
y = b * tan(theta/2)
```

Ray direction:

```text
d(a,b) = normalize(f + x right + y up)
```

Plucker ray:

```text
m(a,b) = C cross d(a,b)
L(a,b) = (d(a,b), m(a,b))
```

That is the whole ray bundle.

So if the representation contains:

```text
C, f, theta, roll/up
```

then Plucker bundle construction is differentiable and compact.

## What A Central Plucker Ray Alone Misses

A central Plucker line:

```text
L_center = (f, m_center)
m_center = C cross f
```

represents the optical axis line.

It does not by itself determine:

```text
where along the line the camera center is
roll around the line
FoV
principal point
aspect ratio
```

If the central line is known to pass through the origin:

```text
m_center = 0
```

then it loses even more:

```text
all cameras at any radius looking at origin along same axis have the same
central Plucker line
```

So the compact camera should not be just:

```text
central Plucker ray
```

It should be something like:

```text
central oriented ray with explicit center/radius + FoV + roll/up
```

or:

```text
look-at direction + radius + FoV + roll/up
```

## Differentiable Central-Ray Camera Parameterization

A useful camera head could output:

```text
raw_direction in R3
raw_radius in R
raw_fov in R
raw_roll in R
optional target offset in R3
```

Then:

```text
f = normalize(raw_direction)
r = base_radius * exp(bounded_raw_radius)
theta = base_fov + bounded_delta_fov
target = target_base + bounded_target_offset
C = target - r f
```

Build camera frame:

```text
right0 = normalize(cross(up_hint, f))
up0 = cross(f, right0)
apply roll around f:
    right = cos(rho) right0 + sin(rho) up0
    up = -sin(rho) right0 + cos(rho) up0
```

Then generate:

```text
R_cw = [right, up, f]
CameraSpec(C, R_cw, FoV)
Plucker grid derived from CameraSpec
```

This keeps the compact, valid pinhole camera structure while using the "central
ray expanded by FoV" view.

## Why This May Be Better Than Generic SE(3) Residuals

Generic SE(3) residual:

```text
rotation_delta + translation_delta
```

does not encode that cameras usually look at a scene/target. It can move and
rotate independently.

Central-ray look-at parameterization:

```text
target
direction/radius
roll
FoV
```

makes the geometry more interpretable:

```text
direction controls viewing angle
radius controls distance/apparent scale
target controls what is being tracked
roll controls horizon/camera twist
FoV controls cone width
```

For object-centric videos, this is probably a better prior than arbitrary
translation/rotation residuals.

It also makes the ray bundle construction explicit:

```text
central ray + FoV => full Plucker bundle
```

## Remaining Caveats

Central-ray-plus-FoV assumes:

```text
pinhole camera
centered principal point
known aspect ratio
known or predicted roll
one central camera center
```

Real data may violate:

```text
principal point offset
lens distortion
rolling shutter
digital stabilization
non-object-centric camera target
```

So the design ladder could be:

```text
1. central ray + radius + FoV + roll valid pinhole camera
2. derive Plucker bundle for conditioning/render diagnostics
3. optionally add small residual principal point/FoV
4. optionally add small Plucker residual warp for real-video imperfections
```

This keeps the base camera valid and differentiable, while leaving a path to
more expressive ray bundles later.

## Revised Camera Representation Shortlist

For implicit camera pretraining, compare:

```text
A. current orbit base + SE(3) residual
B. central direction + radius + FoV + roll + target residual
C. full SE(3)+intrinsics direct prediction
D. derived Plucker bundle from B or C
E. derived Plucker bundle + small learned Plucker residual
```

My updated suspicion:

```text
B with D is the most conceptually clean object-centric camera prior.
```

It says:

```text
predict the central camera ray and cone
expand it into the full Plucker bundle
render with the equivalent CameraSpec
condition image features on the derived normalized Plucker bundle
```

That is probably the version of the user's Plucker idea that best matches this
project.
