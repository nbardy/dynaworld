# Fast-Mac v8 Project3D Forward Fork

Worker 2 scope: create an experimental fast-mac variant that starts from v5,
adds a Metal-side 3D-to-2D pinhole projection op, and provides a benchmark
against the current Torch projection plus v5 Metal raster path. Worker 1 owns
the production Torch camera-aware renderer files, so those were only imported
by the benchmark and not edited.

Implemented `third_party/fast-mac-gsplat/variants/v8_project3d/` as a source
copy of v5 with package/custom-op namespace changed to
`torch_gsplat_bridge_v8_project3d` / `gsplat_metal_v8_project3d`. Build outputs
and copied caches were removed after verification so the new variant is source
only.

Added a forward-only `project_pinhole_forward` custom op. It accepts flattened
`means3d`, `scales`, `quats`, `opacities`, batched `camera_to_world`, batched
`fx/fy/cx/cy`, and small shape/near-plane metadata. The Metal kernel mirrors
the existing Torch pinhole math: build Gaussian covariance from quat+scale,
transform means/covariance world-to-camera, project center with
`u = fx*x/z + cx`, form the pinhole projection Jacobian, convert projected
covariance to packed inverse-conic `[a,b,c]`, zero opacity behind the near
plane, and emit true camera-space depth for later sorting by the copied v5
rasterizer.

Public Python additions:

- `project_pinhole_gaussians(...)`
- `rasterize_pinhole_gaussians(...)`

The projected-2D v5 API remains intact as `rasterize_projected_gaussians(...)`.

Backward status: 3D projection backward is intentionally not implemented in
this pass. The convenience raster path raises if projection inputs require grad
while grad mode is enabled. Color gradients can still flow through the copied
v5 raster backward because colors bypass the projection op.

Benchmark added at `src/benchmarks/fast_mac_project3d_benchmark.py`. It compares:

1. `project_for_fast_mac_batch` from the current train renderer + v5 raster.
2. v8 Metal pinhole projection + v8 copied raster.

Default cases are `smoke:64:512:1` and `realistic_128_8192:128:8192:1`. It
prints timing rows and image max/mean absolute error. On the first case it also
compares color-gradient max/mean error unless `--skip-grad-check` is passed.

Verification:

- Python syntax check passed for the v8 package and benchmark.
- `python setup.py build_ext --inplace` in the v8 variant succeeded with the
  repo `.venv` Python.
- Runtime benchmark could not execute in this shell because PyTorch reports
  `torch.backends.mps.is_built() == True` but `is_available() == False`.

Run from repo root on an MPS-available shell:

```bash
python src/benchmarks/fast_mac_project3d_benchmark.py --build-v8
```
