# Camera Contract

This file defines the renderer-facing camera format and the current adapter
status for each validation source.

## Renderer CameraSpec

The renderer consumes `src/train/camera.py::CameraSpec`:

```text
fx, fy, cx, cy: intrinsics in pixels for the current render viewport
camera_to_world: 4x4 transform
lens_model: pinhole | radial_tangential | opencv_fisheye
distortion: optional lens coefficients
```

Projection math in the differentiable rasterizer is:

```text
p_camera = (p_world - camera_center) @ camera_to_world[:3, :3]
depth = p_camera.z
u = fx * (p_camera.x / depth) + cx
v = fy * (p_camera.y / depth) + cy
```

Depth is valid only when `depth > near_plane`. The camera-frame `+z` axis is
the optical axis. The `camera_to_world[:3, 0:3]` columns are the camera-local
`+x`, `+y`, and `+z` axes expressed in world coordinates.

Important convention hazard: image pixel `v` increases downward, and the
renderer maps camera-local `+y` to increasing `v`. Dataset adapters must choose
the sign of their local `+y` axis accordingly or the render will be vertically
flipped.

Supported lens models:

- `pinhole`: no distortion.
- `radial_tangential`: OpenCV/Brown-Conrady coefficients
  `[k1, k2, p1, p2, k3]`.
- `opencv_fisheye`: OpenCV/Kannala-Brandt coefficients
  `[k1, k2, k3, k4]`.

`camera_for_viewport(...)` rescales intrinsics when frames are loaded at one
resolution but rendered at another. The camera pose is not changed by viewport
resizing.

## Scene Scale

The rasterizer itself does not impose a unit scale. It only requires the
Gaussian positions, Gaussian scales, and camera translations to use the same
world units.

The current token-GS heads usually emit a local normalized world:

- `x,y` in roughly `[-xy_extent, xy_extent]`
- `z` in `[z_min, z_max]`
- common local configs use `xy_extent=1.5`, `z_min=0.5`, `z_max=1.5`
  or `2.5`, and implicit camera radius around `3.0`

External calibrated datasets use their own metric-ish capture coordinates. We
need a dataset adapter layer that normalizes each scene to a stable local scale
before passing cameras and splats to the renderer.

## Dataset Adapter Status

### DeepView Video

Source files:

- MP4s: `data/external/deepview_video/extracted/<scene>/<scene>/camera_*.mp4`
- calibration: `models.json`

DeepView provides enough information for `CameraSpec`:

- `position`: camera center in DeepView world coordinates.
- `orientation`: rotation vector. The README example converts it to a
  world-to-camera rotation matrix `R`.
- `focal_length`, `principal_point`, `width`, `height`.
- `projection_type="fisheye"`.
- `radial_distortion`, compatible with `opencv_fisheye` after padding to four
  coefficients.

Adapter rule:

```text
R_world_to_camera = rotvec_to_matrix(orientation)
camera_to_world[:3, :3] = R_world_to_camera.T
camera_to_world[:3, 3] = position
fx = fy = focal_length
cx, cy = principal_point
lens_model = opencv_fisheye
distortion = [k1, k2, k3_or_0, k4_or_0]
```

Current state: DeepView calibration is preserved in manifests, but
`load_multicam_val_sample` does not yet return a `CameraSpec`.

### Neural 3D Video

Source files:

- MP4s: `cam*.mp4`
- calibration: `poses_bounds.npy`

The local data has camera poses, but the adapter is not implemented. `poses`
must be parsed carefully because LLFF-style pose arrays often use a different
axis ordering from this renderer. This is a priority adapter before Neural 3D
Video should be used for quantitative camera-conditioned rendering.

Current state: frames load, camera specs do not.

### ViVo

Source files:

- compact RGB MP4s: `data/external/vivo/rgb_mp4/<scene>/<split>/<camera>.mp4`
- per-frame metadata: original `*.jpg.meta.json`
- scene calibration: `calibration.json`

The raw metadata includes RGB intrinsics and extrinsics. For example, per-frame
metadata contains `imageMetadata.intrinsics` and
`imageMetadata.extrinsics`. The scene-level `calibration.json` also contains
camera calibration, but the raw metadata and the compact landscape MP4s need a
rotation/crop audit before converting to renderer coordinates.

Current state: frames load and timestamp alignment works, camera specs do not.

### AIST Dance DB

Source files:

- MP4s from the official refined 10Mbps CSV.
- AIST++ camera bundle under
  `data/external/aist_dance_db/cameras/extracted/cameras/`, including
  `mapping.txt` and per-setting JSON files with OpenCV-style intrinsics,
  radial distortion, rotations, and translations.

Current state: fixed camera IDs, synchronized videos, and local AIST++ camera
parameters are available. Treat AIST as visual GT frame-pair validation until
the AIST++ camera payload is converted into canonical renderer-facing
`CameraSpec` objects and the axis/scale convention is checked.

### YouTube

YouTube clips are monocular and uncalibrated. They are for local train/smoke and
qualitative checks only, not GT camera-conditioned novel-view validation.

## Required Next Step

Add canonical camera adapters and update `load_multicam_val_sample` to return:

```text
source_camera_spec: CameraSpec | None
target_camera_spec: CameraSpec | None
source_camera_raw: dataset-native camera payload
target_camera_raw: dataset-native camera payload
camera_adapter_status: ready | missing | approximate
```

The validation evaluator should only compute camera-conditioned novel-view
metrics on samples with `camera_adapter_status="ready"`.
