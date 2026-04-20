# Fast Mac gsplat fastpath handoff

User handed over a second Torch/Metal gsplat package:

- `/Users/nicholasbardy/Downloads/bindings (1).cpp`
- `/Users/nicholasbardy/Downloads/gsplat_metal (1).mm`
- `/Users/nicholasbardy/Downloads/gsplat_fast_kernels.metal`
- `/Users/nicholasbardy/Downloads/rasterize (1).py`
- `/Users/nicholasbardy/Downloads/torch_metal_gsplat_fastpath.tar.gz`

Created `submodules/fast-mac-gsplat-fastpath/`, extracted the tarball there, and copied the original tarball to `source_artifacts/torch_metal_gsplat_fastpath.tar.gz`. The loose standalone files matched the extracted versions by SHA before local edits.

This handoff is not the same scaffold as the first `fast-mac-gsplat` submodule. It has a real Torch/MPS fastpath:

- Python wrapper package `torch_gsplat_bridge_fast`
- Torch MPS `argsort` used for depth order before the custom op
- MPS `cumsum` used for tile offsets
- PyTorch internal `DynamicMetalShaderLibrary`
- Metal kernels for tile counting, bin emission, local bitonic sort/render, and backward

It still has caveats, but no intentional `TORCH_CHECK(false)` in the actual MPS path. Throws remain for unsupported backend/fallback and tile overflow.

Local fixes:

- Added/updated project packaging as `fast-mac-gsplat-fastpath`.
- Changed Torch custom-op C++ return types from `std::vector<torch::Tensor>` to fixed `std::tuple<...>` to match fixed tuple schemas.
- Fixed a correctness bug in the Metal kernels: tensors shaped `[G, 3]` are tightly packed by Torch, but Metal `float3*` has 16-byte alignment/stride. The original kernel read splats after index 0 incorrectly. Replaced `device float3*` reads for conics/colors with `device float*` plus manual `index * 3` loads.
- Removed the suspicious `+1` from the lower x/y bounds in the tile snugbox. That was not the main multi-splat mismatch, but it was inconsistent with the CPU reference.

Validation after fixes:

- `uv run python setup.py build_ext --inplace` succeeded.
- `import torch_gsplat_bridge_fast` registered `torch.ops.gsplat_metal_fast`.
- Tiny MPS forward/backward smoke passed with nonzero gradients.
- 16x16, 4-splat forward/backward matched a direct CPU Torch reference at around `1e-8` max absolute error:
  - image max error `5.74e-08`
  - means grad max error `4.29e-10`
  - conics grad max error `1.05e-08`
  - colors grad max error `3.01e-09`
  - opacities grad max error `1.02e-08`
- Synthetic projected 4096x4096 / 65536 splat forward smoke:
  - forward `13.31ms`
  - output shape `(4096, 4096, 3)` on MPS
- Synthetic projected 4096x4096 / 65536 splat forward+backward smoke:
  - forward `9.89ms`
  - backward `99.58ms`
  - total `109.47ms`
  - gradients were nonzero for means, conics, colors, and opacities

Interpretation:

- Ready for local experiments as a projected 2D Torch/MPS differentiable rasterizer.
- Not yet integrated into Dynaworld's benchmark matrix or training code.
- Bench numbers are synthetic projected 2D splats with small random radii, not a full 3D scene-quality benchmark.
- Requires MPS tensors and float32. CPU tensors are not the intended path.
- Relies on PyTorch internal MPS headers, so PyTorch version pinning/testing matters.
- Has a tile capacity limit: excessive splats in one tile can throw.

Cleaned generated validation artifacts from the submodule after testing: local `.venv`, build directory, extension `.so`, egg-info, `__pycache__`, and `uv.lock`.

Follow-up after user asked about backward being too slow:

- Rebuilt locally for measurement and separated direct custom-op forward/backward timing from Python loss/autograd timing.
- Found the `~10x` backward/forward impression is not fixed; synchronized direct-op ratio depends heavily on overlap.
- Added a low-risk optimization: forward writes tile-local IDs back into `binned_ids` after local bitonic sort, so backward can skip re-sorting the same tile IDs.
- Correctness after that patch still matched the 16x16 / 4-splat CPU reference at about `1e-8`.
- Synchronized direct-op smoke after sorted-ID reuse:
  - 4096x4096 / 65,536 splats / sigma 1-5 px: forward `9.9ms`, backward `31.4ms`, ratio `3.18x`
  - 4096x4096 / 65,536 splats / sigma 3-8 px: forward `15.5ms`, backward `93.4ms`, ratio `6.02x`
  - 1024x1024 / 65,536 splats / sigma 1-5 px: forward `6.36ms`, backward `30.0ms`, ratio `4.73x`
- This confirms tile sorting was duplicated work, but medium/dense overlap is still dominated by backward recomputation plus global atomic gradient accumulation.
