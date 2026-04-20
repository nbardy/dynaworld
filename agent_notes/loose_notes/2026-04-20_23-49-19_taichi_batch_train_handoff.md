# Taichi Batch Train Handoff

Status:

- The Taichi fork has a pushed native batch API:
  `taichi_splatting.rasterizer.rasterize_batch`.
- The API is documented near the top of
  `third_party/taichi-splatting/README.md`.
- The pushed submodule commit is `3851e15 Add native batched Metal rasterization`.
- Dynaworld's submodule pointer commit is
  `4756b4a Update Taichi native batch rasterizer pointer`.

API shape:

```python
from taichi_splatting.rasterizer import rasterize_batch

raster = rasterize_batch(
    gaussians2d,  # [B, G, 7] = x, y, axis_x, axis_y, sigma_x, sigma_y, opacity
    depth,        # [B, G, 1], nonnegative, front-to-back order key
    features,     # [B, G, C]
    image_size=(width, height),
    config=config,
)

# raster.image: [B, H, W, C]
# raster.image_weight: [B, H, W]
```

Validation already run:

- CPU native batch vs looping single-image `rasterize`:
  - forward image max diff: `0.0`
  - alpha max diff: `0.0`
  - packed Gaussian grad max diff: `3.7e-9`
  - feature grad max diff: `0.0`
- MPS/Metal native batch backward:
  - output shape `(3, 32, 32, 3)`
  - packed/features gradients finite
- Dynaworld adapter smoke:
  - output shape `(2, 3, 32, 32)`
  - finite output
  - finite `xyz` and RGB gradients

Speed probe on Apple Silicon / MPS after separate warmup:

```text
B=4, 64x64,  G=512:  batch fwd 15.993ms vs loop fwd 64.844ms = 4.05x
B=4, 64x64,  G=512:  batch fwd+bwd 37.645ms vs loop fwd+bwd 108.780ms = 2.89x
B=4, 128x128, G=512: batch fwd 20.703ms vs loop fwd 73.141ms = 3.53x
B=4, 128x128, G=512: batch fwd+bwd 39.856ms vs loop fwd+bwd 151.560ms = 3.80x
B=4, 128x128, G=4096: batch fwd 21.384ms vs loop fwd 73.639ms = 3.44x
B=4, 128x128, G=4096: batch fwd+bwd 48.696ms vs loop fwd+bwd 116.102ms = 2.38x
```

Train-readiness notes:

- The Taichi fork itself is ready for a train smoke.
- The local Dynaworld train adapter has been updated to call native batch from
  `render_gaussian_frames(..., mode="taichi")`, but those parent train files are
  still in the dirty worktree alongside broader trainer/debug-metric edits.
- Before handing to a train engineer through a clean PR, isolate and commit the
  train-facing adapter files or reapply them on a clean branch:
  - `src/train/renderers/taichi.py`
  - `src/train/rendering.py`
  - the config that selects `"renderer": "taichi"`
  - whichever existing `dynamicTokenGS.py` batch-render call changes are needed
    in that branch
- `pytest` was not installed in the `uv` env, so the new pytest file in the
  Taichi fork was validated by running the equivalent direct Python check.

Caveats:

- This is still the simple `metal_reference` / `pixel_reference` renderer, not
  the final fused threadgroup-memory sort+raster path.
- It supports float32 today.
- `sort_backend="torch"` / `"auto"` is the recommended Mac path.
- The old looped single-image MPS path can trip PyTorch version-counter checks
  when passing `packed[b]` slice views into Taichi autograd; native batch avoids
  that call pattern.
