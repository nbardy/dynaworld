# Fast Mac Gsplat Submodule

Imported the chief scientist's Torch/Metal handoff into a standalone source
submodule at `submodules/fast-mac-gsplat/`.

Source handling:

- Extracted `torch_metal_gsplat_bridge.tar.gz` into the submodule.
- Kept the original tarball under `submodules/fast-mac-gsplat/source_artifacts/`
  for provenance.
- Verified the standalone downloaded `setup.py`, `rasterize.py`,
  `bindings.cpp`, `gsplat_metal.mm`, and `gsplat_kernels.metal` matched the
  corresponding extracted files before local edits.

README:

- Replaced the scaffold README with a product README using the user's wording
  with small grammar fixes.
- Added a current-status section so future agents do not confuse this with the
  already-running MLX benchmark path. This package is the direct Torch custom-op
  direction, but `metal_forward`/`metal_backward` still intentionally throw until
  MPS tensor to `MTLBuffer` interop, GPU scan/sort, command submission, and aux
  plumbing are implemented.

Packaging fixes:

- Renamed the distribution from `torch_metal_gsplat_bridge` to
  `fast-mac-gsplat`; the Python import remains `torch_gsplat_bridge`.
- Added minimal `[project]` metadata to `pyproject.toml` so `uv` treats the
  submodule as a project.
- Changed extension sources in `setup.py` from absolute to relative paths,
  because editable builds reject absolute source paths.
- Kept the include directory absolute because Torch's temporary build directory
  otherwise could not find `csrc/shared/common.h`.
- Changed custom-op C++ return types from `std::vector<torch::Tensor>` to fixed
  `std::tuple<...>` return types to match the Torch operator schemas.

Validation:

- `uv run python setup.py --name` from the submodule prints `fast-mac-gsplat`.
- `uv run python -m py_compile torch_gsplat_bridge/rasterize.py torch_gsplat_bridge/__init__.py` passes.
- `uv run python setup.py build_ext --inplace` builds the Objective-C++/C++
  extension on this Mac.
- Importing the built extension registers `torch.ops.gsplat_metal` without the
  previous schema-mismatch abort.
- A tiny MPS forward call reaches the expected scaffold RuntimeError:
  `metal_forward scaffold generated successfully... MTLBuffer/MPS interop and command submission still need to be completed`.

Cleaned generated `.venv`, build output, `.so`, egg-info, and `__pycache__`
from the submodule after validation.
