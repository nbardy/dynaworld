# Third Party Renderer Inventory

This directory holds external or handoff renderer code. Dynaworld's internal
baseline renderers stay in `src/train/renderers/`; benchmark adapters stay in
`src/benchmarks/`.

## Active submodules

- `dust3r/`: DUSt3R upstream submodule used for camera/geometry preprocessing.
- `taichi-splatting/`: Taichi gsplat fork with the Mac/Metal experiments. Taichi
  implementation work should live in this submodule, not in Dynaworld source.
- `fast-mac-gsplat/`: Torch-compatible Metal gsplat renderer for Apple Silicon.
  This is the working fastpath handoff published as `nbardy/fast-mac-gsplat`.

## Vendored handoffs

- `raw-metal-mlx-gsplat/`: MLX/Metal projected gsplat rasterizer handoff. This
  remains vendored source for benchmarking and comparison against the Torch
  fastpath.
- `fast-mac-gsplat-scaffold/`: Older Torch/Metal binding scaffold. It is kept
  only as an archived handoff reference; the active repo is `fast-mac-gsplat/`.

## Main Repo Boundaries

- `src/train/renderers/`: small internal Torch baseline renderers and shared
  projection math.
- `src/benchmarks/`: benchmark harnesses and adapters that call third-party
  renderers.
- `shader_experiments/`: scratch-only area. Serious renderer handoffs should be
  moved here into `third_party/`.
