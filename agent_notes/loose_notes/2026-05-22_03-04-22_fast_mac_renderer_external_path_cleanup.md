# Fast-Mac Renderer External Path Cleanup

## Context

`src/train/renderers/fast_mac.py` still rebuilt the same
`third_party/fast-mac-gsplat/variants/...` root for every active variant and
carried its own compiled-bridge origin guard. That was the same helper boundary
as the new train-local `external_paths.py` cleanup, but inside the core
renderer wrapper.

## Changes

- Added one `FAST_MAC_VARIANTS_DIR` root and `_fast_mac_variant_path(...)` helper
  in `renderers.fast_mac`.
- Replaced repeated `Path(__file__).resolve().parents[3] / "third_party" /
  "fast-mac-gsplat" / "variants"` expressions with variant-name calls.
- Routed `_ensure_variant_on_path(...)` through
  `external_paths.ensure_module_path(...)`, preserving wrong-compiled-bridge
  origin protection while removing the renderer-local `sys.modules` and
  `sys.path` logic.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for
  `external_paths.py`, `renderers/fast_mac.py`, and the benchmark modules that
  import Fast-Mac.
- Import smoke for `renderers.fast_mac` passed and confirmed the variants root,
  v5 path, and default feature variant.
- Import smoke for `fast_mac_project3d_benchmark.py` passed.
- Import smoke for `splat_renderer_benchmark.py` and
  `splat_renderer_accuracy.py` passed.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This did not run the renderer benchmarks. It is an import/bootstrap cleanup
only. Variant dispatch, package names, and renderer-specific behavior remain
local to `renderers.fast_mac`.
