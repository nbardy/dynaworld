# PowerFoam Geometry Boundary

## Context

PowerFoam Direct and PowerFoam Metal each had the same camera-ray helper code,
while Dynamic PowerFoam imported pinhole rays and surface-frame helpers from the
full Metal trainer. That kept pure geometry utilities tied to a large trainer
module and repeated the multiview ray-grid code.

## Changes

- Added `src/train/powerfoam_geometry.py`.
- Moved pure helpers there:
  - `make_pinhole_rays(...)`
  - `powerfoam_rays_from_camera(...)`
  - `powerfoam_rays_from_camera_grid(...)`
  - `stable_tangent_from_normals(...)`
  - `orthonormal_surface_frame(...)`
- Updated PowerFoam Direct and PowerFoam Metal to import/re-export the shared
  ray helpers for compatibility.
- Updated Dynamic PowerFoam to import pinhole rays and surface-frame helpers
  directly from `powerfoam_geometry.py`.
- Updated two raytrace diagnostics to import `powerfoam_rays_from_camera(...)`
  from the geometry module.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for
  `powerfoam_geometry.py`, PowerFoam Direct, PowerFoam Metal, Dynamic
  PowerFoam, the touched Dynamic PowerFoam test, and the touched raytrace
  diagnostics.
- Focused tests passed:
  `test_powerfoam_metal_camera_rays_include_camera_pose`,
  `test_powerfoam_normals_from_ray_depth_orients_against_rays`,
  `test_dynamic_powerfoam_default_rays_match_fixed_pinhole`,
  `test_dynamic_powerfoam_implicit_camera_rays_are_per_frame_and_trainable`,
  and `test_token_dynamic_powerfoam_features_mps_raster_backward_smoke`.
- Re-export smoke passed: `train_powerfoam_metal.make_pinhole_rays` is
  `powerfoam_geometry.make_pinhole_rays`, and the Direct/Metal
  `powerfoam_rays_from_camera_grid` exports point at the same helper.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is a pure geometry/helper extraction. It does not change PowerFoam Direct,
PowerFoam Metal, Dynamic PowerFoam, or raytrace kernel behavior.
