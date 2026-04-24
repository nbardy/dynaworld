# Fast-Mac v8 Project3D Metal VJP

Implemented the pure Metal pinhole projection backward that replaced the
temporary Torch-rematerialized training path.

Added `project_pinhole_backward` to the v8 custom op. It takes raster-side
gradients for projected means, conics, projected opacities, and depths, then
runs a one-thread-per-splat reverse pass through:

- pinhole mean projection
- conic inverse from projected covariance
- projected covariance `J C J^T`
- world-to-camera covariance transform
- quaternion/scale covariance construction
- camera intrinsics and `camera_to_world`

Camera/intrinsic gradients use atomic adds across splats in the batch; per-splat
means/scales/quats/opacities write directly.

Benchmark after warmup:

```text
128px/8192 forward_backward: v5 21.8107ms, v8 Metal VJP 11.0773ms
256px/65536 forward_backward: v5 41.2138ms, v8 Metal VJP 22.3111ms
```

Image parity stayed tight. Full gradient max diff on the first benchmark case
was dominated by camera pose atomic reduction order; per-splat gradients were
near fp32 noise (`means3d max ~4.8e-5`, `scales max ~1.8e-4`, `quats max
~2.3e-5` in a diagnostic run).
