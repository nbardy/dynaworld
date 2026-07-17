# PowerFoam Adjacency Boundary

## Context

Dynamic PowerFoam and several Dynamic Foam diagnostics imported
`build_csr_adjacency(...)` from `train_powerfoam_metal.py`. The implementation
is mostly pure CPU/Torch CSR construction, but one mode
(`regular_triangulation`) needs the Metal extension.

Keeping this inside the full Metal trainer forced helper-only callers to import
a large training/runtime module.

## Changes

- Added `src/train/powerfoam_adjacency.py`.
- Moved pure adjacency helpers there:
  - `dense_overlap_mask(...)`
  - `build_csr_adjacency(...)`
  - `csr_adjacency_stats(...)`
- Kept `regular_triangulation` support by lazy-loading
  `torch_powerfoam_metal.make_regular_triangulation_adjacency` only when that
  mode is requested.
- Updated `train_powerfoam_metal.py` to import/re-export the adjacency helpers
  for compatibility.
- Updated Dynamic PowerFoam and Dynamic Foam diagnostics to import adjacency
  helpers directly from `powerfoam_adjacency.py`.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for
  `powerfoam_adjacency.py`, PowerFoam Metal, Dynamic PowerFoam, and the touched
  Dynamic Foam diagnostics.
- Re-export smoke passed:
  `train_powerfoam_metal.build_csr_adjacency is
  powerfoam_adjacency.build_csr_adjacency` and
  `train_powerfoam_metal.csr_adjacency_stats is
  powerfoam_adjacency.csr_adjacency_stats`.
- Focused tests passed:
  `test_powerfoam_metal_knn_adjacency_has_fixed_degree_and_no_self_edges`,
  `test_powerfoam_metal_cech_aabb_is_dense_overlap_superset`,
  `test_powerfoam_metal_cech_aabb_fixes_knn_missed_power_face`, and
  `test_token_dynamic_powerfoam_features_mps_raster_backward_smoke`.
- `test_powerfoam_metal_regular_triangulation_matches_unweighted_delaunay_edges`
  skipped because the optional SciPy dependency was unavailable in this run.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is a CSR adjacency helper extraction only. It does not change PowerFoam
training behavior, adjacency semantics, kernels, or benchmark results.
