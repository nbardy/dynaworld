# Camera Ray Contract Implementation Notes

## Context

The user asked whether there is a unified camera model for iPhone lenses,
fisheye, and common video/cinema lenses, then asked to write the shape/data,
define the key math, and move toward Torch code that turns cameras into rays for
a future ray-based differentiable renderer.

## What Changed

- Extended `src/train/camera.py` so `CameraSpec` remains backward-compatible
  with existing pinhole fields, but now also carries:
  - `lens_model`
  - `distortion`
- Added a central-camera Torch ray generator supporting:
  - `pinhole`
  - `radial_tangential`
  - `opencv_fisheye`
- Added inverse distortion math:
  - fixed-point inverse for OpenCV/Brown-Conrady radial-tangential distortion
  - Newton inverse for OpenCV fisheye/Kannala-Brandt theta polynomial
- Added `build_plucker_from_rays` so Plucker construction is downstream of ray
  generation, not tied to pinhole math.
- Updated camera clone paths in `runtime_types.py`, `rendering.py`, and
  `gs_models/implicit_camera.py` so lens metadata is preserved.
- Added `research_notes/camera_ray_contract.md` as the durable shape/math
  contract.
- Verification caught a fisheye autograd edge case: forward rays were finite at
  the principal point, but `torch.where(x/r, 0)` still evaluated the hidden
  `0/0` division path and produced NaN gradients for `fx`. The implementation
  now clamps the fisheye radius before square root/division.

## Follow-Up Change In Same Session

The camera model was made configurable through `camera.global_head` and
`camera.lens_model`.

- `legacy_orbit` preserves the old simple behavior: camera token decodes
  `fov/radius`, lens is pinhole, path token decodes per-frame motion.
- `central_lens` decodes fixed per-clip intrinsics and lens params:
  `fov/radius/aspect/principal point/distortion`.
- The video-token implicit-camera model now decodes the global camera token once
  per clip and reuses it for all frame decodes. This prevents lens/intrinsics
  from changing per frame through the time-conditioned query path.
- The pose-to-Plucker config now uses `central_lens` with `pinhole` lens params
  so it exercises the new head without asking the current matrix splat renderer
  to handle distortion it does not consume yet.

## Important Design Choice

The current implementation deliberately uses a central-camera superset rather
than a fully free per-pixel ray field. This covers the practical target set
well: iPhone tele/main/ultra-wide, common cinema lenses, action-camera-like
fisheye. It does not yet represent non-central optics, rolling shutter, or
learned ray warps. That is intentional because a free ray field can hide
geometry and motion errors.

## Follow-Up Ideas

- Add a camera-token head that decodes `fx/fy/cx/cy`, lens family logits, and
  bounded distortion coefficients.
- Prefer config/metadata-selected lens family first; let the model learn
  continuous parameters before letting it choose the lens family.
- When a ray renderer exists, consume `build_camera_rays` directly. Current
  project renderers still use pinhole projection matrices for splatting.
