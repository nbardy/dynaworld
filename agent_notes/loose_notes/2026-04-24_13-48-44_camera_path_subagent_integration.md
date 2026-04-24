# Camera Path Subagent Integration

Integrated the two-subagent camera/rendering pass.

Worker 1 landed the production path in Torch: `CameraSpec` now carries a
central lens model, implicit-camera decoders can switch between `legacy_orbit`
and `central_lens`, and dense/tiled/taichi/fast-mac wrappers route through
`render.camera_projection`. The production fast-mac path still projects in
Torch and sends the same 2D splat packet to v5 Metal rasterization.

Worker 2 landed the experiment under
`third_party/fast-mac-gsplat/variants/v8_project3d/`: a v5 fork with a
forward-only Metal pinhole projection op plus
`src/benchmarks/fast_mac_project3d_benchmark.py`. The benchmark compares
current Torch projection + v5 raster against v8 Metal projection + v8 raster.
The v8 3D projection has no backward pass; it is timing/profiling scaffolding,
not train-ready renderer plumbing.

Local verification after integration:

- `compileall` passed for the camera/rendering/train files, v8 Python wrapper,
  and benchmark.
- Pinhole projection parity passed exactly against the legacy projector.
- Radial-tangential and OpenCV fisheye projection/render smokes produced finite
  outputs and finite point gradients.
- Small video-token forward smokes passed for `central_lens` with pinhole,
  radial-tangential, and OpenCV fisheye lens models.
- `git diff --check` passed on the touched top-level files and the v8 submodule
  paths.
- The v8 runtime benchmark could not execute in this shell because PyTorch has
  MPS built but reports `torch.backends.mps.is_available() == False`.
