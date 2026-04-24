# Fast-Mac v8 Project3D Training Backward

Follow-up to the forward-only v8 project3d fork. The user clarified that the
Metal experiment needs to be training-ready and benchmark forward plus backward,
not just eval forward.

Changed the v8 Python wrapper so `rasterize_pinhole_gaussians(...)` no longer
raises when 3D projection inputs require gradients. The forward still calls the
Metal `project_pinhole_forward` op. In backward, the wrapper receives gradients
from the copied v5 raster backward for `means2d`, conics, colors, and projected
opacities, then rematerializes the pinhole projection math in Torch autograd to
propagate gradients into 3D means, scales, quats, opacities, intrinsics, and
`camera_to_world`.

This is a hybrid training path, not a pure hand-derived Metal projection
backward kernel. That is deliberate for this pass: it lets us benchmark whether
Metal projection forward is worth production integration before carrying the
risk of a large analytic projection-backward shader.

Extended `src/benchmarks/fast_mac_project3d_benchmark.py`:

- reports `forward_eval` rows for current v5 and v8 project3d
- reports `forward_backward` rows for current v5 and v8 project3d
- compares full gradient parity on the first case unless `--skip-grad-check`
  is passed

Local verification:

- Python compile checks passed for the v8 wrapper and benchmark.
- `git diff --check` passed for the benchmark and v8 wrapper.
- CPU smoke of the Torch rematerialized projection backward produced finite
  gradients for means, scales, quats, opacities, camera intrinsics, and
  `camera_to_world`.
- Runtime benchmark still cannot execute in this shell because PyTorch reports
  MPS is built but unavailable.
