# fast-mac-gsplat

Archived scaffold note: this was the older Torch/Metal binding handoff. The
active working renderer now lives in `third_party/fast-mac-gsplat` and the
GitHub repo `nbardy/fast-mac-gsplat`.

Provides a fast forward and backward Torch-compatible differentiable Gaussian Splat Renderer written in pure Metal.

This repo was built to lower the barrier to entry for world model research. The core component of world models at the moment is Gaussian splats. Gaussian splats are a cheap 3D primitive that can be rendered insanely fast and are differentiable; this allows them to be used for all manners of learned representations and AI optimization. Modern Macs are incredible machines for deep learning. They may not have the scale needed for training large-scale foundation models, but for training small models, doing the engineering of developing new architectures and optimizers, and establishing baselines, they are incredibly powerful.

The biggest bottleneck for this is fast differentiable Gaussian splat rendering on Macs.

Traditional Gaussian splat algorithms have been:

1. Torch-only. Vectorized Torch is not low level enough for the sort of random memory access patterns and kernel programming needed for fast splats.
2. CUDA-only.
3. Render-only, with no backward pass.

This repo changes that and brings modern fast differentiable Gaussian splat rendering to the Mac, allowing individual researchers to run experiments on commodity hardware.

## Current Status

This handoff is the cleaner Torch-compatible binding direction for the successful raw Metal rasterizer work. It is intended to replace the current benchmark bridge that goes Torch -> NumPy -> MLX -> NumPy -> Torch.

What is here:

- `torch_gsplat_bridge/rasterize.py`: Torch-facing Python API and autograd wrapper.
- `csrc/bindings.cpp`: Torch custom operator registration.
- `csrc/metal/gsplat_metal.mm`: Objective-C++ Metal backend entrypoints.
- `csrc/metal/gsplat_kernels.metal`: Metal kernels for tight bounds, pair emission, tile forward rasterization, and tile backward rasterization.
- `csrc/shared/common.h`: shared C++ metadata contracts.

Important: the Objective-C++ bridge is still a scaffold. `metal_forward` and `metal_backward` currently stop with explicit `TORCH_CHECK(false, ...)` calls because the macOS-specific MPS tensor to `MTLBuffer` interop, GPU scan/sort path, command submission, and backward aux-state plumbing still need to be completed on an Apple machine.

The MLX prototype has already shown the core Metal rasterization approach can be fast. This package is the next step: a direct Torch-compatible extension that avoids the MLX bridge overhead.

## Tensor Contract

Inputs:

- `means2d`: `[G, 2]` float32
- `conics`: `[G, 3]` float32 storing `[a, b, c]`
- `colors`: `[G, 3]` float32
- `opacities`: `[G]` float32
- `depths`: `[G]` float32

Output:

- `image`: `[H, W, 3]` float32

Backward returns gradients for:

- `means2d`
- `conics`
- `colors`
- `opacities`

Depth gradients are intentionally zero in this version because the sort order is treated as piecewise constant.

## Python API

```python
import torch

from torch_gsplat_bridge import ProjectedGaussianRasterizer
from torch_gsplat_bridge.rasterize import RasterConfig

cfg = RasterConfig(height=4096, width=4096, tile_size=16)
rasterizer = ProjectedGaussianRasterizer(cfg)

image = rasterizer(means2d, conics, colors, opacities, depths)
loss = image.square().mean()
loss.backward()
```

## Build

This package is source-only and should be built on macOS with Xcode command line tools installed.

```bash
python -m pip install -e .
```

The extension currently registers the Torch custom op surface, but the Metal host bridge must be completed before this package can render through Torch directly.
