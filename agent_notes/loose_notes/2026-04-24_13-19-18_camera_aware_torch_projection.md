# Camera-Aware Torch Projection

Implemented the production-side Torch preprojection path for `CameraSpec`
lenses without touching the fast-mac Metal shader. The important split is that
the existing renderers consume projected 2D splat packets, not 3D cameras, so
lens correctness can happen before dense/tiled/taichi/fast-mac rasterization.

Added `src/train/renderers/projection.py` with:

- `project_points_camera`: camera-frame point to pixel plus
  `d(pixel)/d(camera_xyz)`.
- `project_gaussians_2d_camera`: single-frame 3DGS projection via
  `CameraSpec`.
- `project_gaussians_2d_camera_batch`: batch wrapper, delegating all-pinhole
  batches to the legacy projector for parity.

The non-pinhole path uses local ellipse linearization:

```text
world Gaussian -> camera Gaussian -> lens projection + Jacobian
cov2D = J cov3D J^T
```

Implemented analytic Jacobians for radial-tangential and OpenCV fisheye
projection. The fisheye zero-radius case is guarded with the `theta_d/r -> 1`
limit to avoid NaN gradients at the principal point.

Renderer wrappers now accept a projection mode:

```text
auto            -> legacy pinhole for pinhole cameras, camera_model otherwise
legacy_pinhole  -> old path, errors for non-pinhole cameras
camera_model    -> CameraSpec-aware Torch projection
```

Verification run:

- `python3 -m compileall` on touched modules.
- Exact pinhole parity for single and batch projection.
- Finite gradient checks for radial-tangential and fisheye projection,
  including a principal-point fisheye splat.
- Dense render smoke through camera-model projection.
- Fast-mac and Taichi preprojection helper shape/finite checks.
- Tiled render smoke through camera-model projection.

Current caveat: strong nonlinear fisheye still approximates each projected
Gaussian as one local 2D ellipse, so edge splats can deviate from the true
warped footprint. A future ray renderer or splat subdivision would address that.
