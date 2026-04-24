# Camera Ray Contract

## Goal

Represent phone, cinema, and common fisheye lenses as differentiable camera
parameters that map to rays before rendering:

```text
camera params -> Torch ray generator -> rays -> differentiable renderer -> image
camera params -> Torch 3DGS projection -> 2D splat packet -> existing rasterizers
```

The renderer-facing contract is ray-first. Projection-model details should stay
inside the camera ray generator.

## Data Shape

Single camera:

```text
CameraSpec:
  fx, fy, cx, cy: scalar float or scalar tensor
  camera_to_world: [4, 4]
  lens_model: "pinhole" | "radial_tangential" | "opencv_fisheye"
  distortion:
    pinhole: None or []
    radial_tangential: [k1, k2, p1, p2, k3]
    opencv_fisheye: [k1, k2, k3, k4]
```

Ray output:

```text
origins_world: [H, W, 3]
dirs_world:    [H, W, 3], unit vectors
```

Batch output:

```text
origins_world: [B, H, W, 3]
dirs_world:    [B, H, W, 3], unit vectors
```

Plucker output:

```text
plucker: [..., 6] = concat(direction, moment)
moment = origin cross direction
```

Projected 3DGS packet:

```text
means2D:  [G, 2] or [B, G, 2]
invCov2D: [G, 2, 2] or [B, G, 2, 2]
cov2D:    [G, 2, 2] or [B, G, 2, 2]
opacities:[G, 1] or [B, G, 1], zeroed for near/behind-camera splats
rgbs:     [G, 3] or [B, G, 3]
```

## Decoder Split

Implicit-camera models should keep camera identity fixed across a clip:

```text
global camera token -> fixed lens/intrinsics/base radius
path camera token   -> per-frame pose motion only
```

The base camera starts as an orbit camera looking at the origin. At zero
residuals, the camera center is on the negative z axis and the optical axis
points toward the origin. Path tokens may move the camera center and rotate the
camera, but they should not emit lens or intrinsics.

Current configurable global heads:

```text
camera.global_head = "legacy_orbit"
  decodes: fov, radius
  lens: pinhole only
  purpose: old simple baseline

camera.global_head = "central_lens"
  decodes: fov, radius, fx/fy aspect, principal point, lens distortion
  lens: pinhole | radial_tangential | opencv_fisheye
  purpose: new central-camera ray model
```

The active video-token decoder now decodes the global camera token once per
clip, then reuses that fixed camera identity for every per-frame path decode.

Example config:

```jsonc
"camera": {
  "global_head": "central_lens",
  "lens_model": "opencv_fisheye",
  "base_fov_degrees": 90.0,
  "base_radius": 3.0,
  "max_fov_delta_degrees": 30.0,
  "max_radius_scale": 1.5,
  "max_aspect_log_delta": 0.05,
  "max_principal_point_delta": 0.02,
  "distortion_max_abs": [0.1, 0.03, 0.01, 0.005],
  "base_distortion": [0.0, 0.0, 0.0, 0.0],
  "max_rotation_degrees": 5.0,
  "max_translation_ratio": 0.2
}
```

## Shared Pixel Normalization

For pixel coordinate `(u, v)`:

```text
x_d = (u - cx) / fx
y_d = (v - cy) / fy
```

`x_d, y_d` are distorted normalized image coordinates. Each lens model maps
them to a camera-frame unit ray.

## Pinhole

Pinhole assumes the normalized pixel coordinate is already undistorted:

```text
dir_camera = normalize([x_d, y_d, 1])
```

This covers tele/main lenses well and is the baseline for debugging.

## Radial-Tangential Distortion

This is the OpenCV/Brown-Conrady family, useful for common video/cinema lenses
and phone lenses with moderate distortion. Forward distortion is:

```text
r2 = x^2 + y^2
radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3

x_d = x*radial + 2*p1*x*y + p2*(r2 + 2*x^2)
y_d = y*radial + p1*(r2 + 2*y^2) + 2*p2*x*y
```

Ray generation needs the inverse mapping `(x_d, y_d) -> (x, y)`. The Torch
implementation uses a fixed-point iteration, then:

```text
dir_camera = normalize([x, y, 1])
```

## OpenCV Fisheye

This is a central fisheye/Kannala-Brandt style model. It is useful for iPhone
ultra-wide, action cameras, and common fisheye-like footage.

Let:

```text
r_d = sqrt(x_d^2 + y_d^2)
theta_d = r_d
theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
```

Ray generation solves this scalar equation for `theta` with Newton iterations,
then:

```text
unit_x = x_d / r_d
unit_y = y_d / r_d

dir_camera = [
  sin(theta) * unit_x,
  sin(theta) * unit_y,
  cos(theta)
]
```

At the principal point, `r_d = 0`, so the direction is `[0, 0, 1]`.

Forward projection for splatting uses:

```text
x = X/Z
y = Y/Z
r = sqrt(x^2 + y^2)
theta = atan(r)
theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)

x_d = (theta_d / r) * x
y_d = (theta_d / r) * y
```

with the `r -> 0` limit set to scale `1`.

## World Transform

All current models are central cameras: all pixel rays share one camera center.

```text
R = camera_to_world[:3, :3]
C = camera_to_world[:3, 3]

dir_world = normalize(R @ dir_camera)
origin_world = C
```

## Renderer Projection

The production rasterizer path is still projected-splat based:

```text
world Gaussian -> camera-frame Gaussian -> lens projection + Jacobian
cov2D = J cov3D J^T
2D splat packet -> dense / tiled / taichi / fast-mac rasterizer
```

`render.camera_projection` selects projection dispatch:

```text
auto            -> legacy pinhole for pinhole cameras, camera_model otherwise
legacy_pinhole  -> old pinhole path; errors for non-pinhole CameraSpec
camera_model    -> CameraSpec-aware Torch projection
```

The all-pinhole camera-model batch path delegates to the legacy projection for
parity. Non-pinhole projection is vectorized Torch with analytic
`d(pixel)/d(camera_xyz)` Jacobians.

## Experimental Metal Projection

The `third_party/fast-mac-gsplat/variants/v8_project3d/` fork copies the v5
projected-2D Metal rasterizer and adds a pinhole projection op:

```text
3D Gaussian + pinhole CameraSpec fields -> Metal projection -> v5-style raster
```

This is for benchmarking whether moving the current pinhole `3D -> 2D` packet
build out of Torch saves enough time to wire into production training. The v8
path is now training-ready for pinhole benchmarks: forward projection runs in
Metal, raster backward runs through the copied v5 Metal kernels, and projection
gradients run through a per-splat Metal VJP kernel. It does not implement
non-pinhole radial/fisheye projection.

## Current Limits

This is a central-camera contract. It does not represent non-central optics,
rolling shutter, per-pixel ray warps, or learned optical-flow-like ray fields.
Those can be added later as a small residual ray correction after the valid
central camera is stable. Strong fisheye distortion makes a projected 3D
Gaussian less exactly elliptical near the image edge; the current rasterizers
still approximate each projected splat with one local 2D ellipse.
