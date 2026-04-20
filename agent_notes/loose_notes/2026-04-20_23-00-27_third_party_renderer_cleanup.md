# Third-party renderer cleanup

User asked to consolidate external renderer handoffs under one vendor area and make the working Torch/Metal renderer the `fast-mac-gsplat` repo.

Actions:

- Created GitHub repo `nbardy/fast-mac-gsplat` from the working Torch/MPS/Metal fastpath.
- Renamed the package metadata in that standalone repo from `fast-mac-gsplat-fastpath` to `fast-mac-gsplat`.
- Added `third_party/fast-mac-gsplat` as a real Git submodule pointing to `git@github.com:nbardy/fast-mac-gsplat.git`.
- Removed the previously committed plain-folder copy at `submodules/fast-mac-gsplat-fastpath`.
- Moved the MLX/Metal raw handoff from `shader_experiments/raw_metal_fast_rasterizer` to `third_party/raw-metal-mlx-gsplat`.
- Moved the older Torch/Metal binding scaffold from `submodules/fast-mac-gsplat` to `third_party/fast-mac-gsplat-scaffold` and marked it archived in its README.
- Updated `src/benchmarks/raw_metal_mlx_bridge.py` to import the MLX raw-metal code from `third_party/raw-metal-mlx-gsplat`.
- Added `third_party/README.md` documenting the boundaries:
  - internal renderers stay in `src/train/renderers`
  - benchmark adapters stay in `src/benchmarks`
  - external renderer implementations live under `third_party`
  - scratch-only code should not be promoted from `shader_experiments` without moving it to `third_party`
- Removed the loose Taichi `shader_experiments/snugbox_rasterizer.py` scratch file from the active tree so Taichi implementation work stays in `third_party/taichi-splatting`.

Verification:

- `uv run python -m py_compile src/benchmarks/raw_metal_mlx_bridge.py`
- `git submodule status` shows `third_party/fast-mac-gsplat` at commit `338c0f9`.

Notes:

- `third_party/dust3r` and `third_party/taichi-splatting` still show pre-existing submodule dirtiness outside this cleanup scope.
- The active fast Mac renderer repo is private on GitHub unless visibility is changed later.
