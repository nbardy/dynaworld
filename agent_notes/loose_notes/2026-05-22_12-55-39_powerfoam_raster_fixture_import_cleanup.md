# PowerFoam Raster Fixture Import Cleanup

## Context

After the Token-GS, precomputed-feature, and multicam trainer splits, the next
live wrapper audit found that `tests/test_powerfoam_direct.py` still imported
`FoamRasterConfig` and
`rasterize_power_foam_quaternion_height_sv_texel_surface` from
`train_powerfoam_metal.py` for two Metal fixture checks. Those symbols are not
trainer policy; they are exported by `torch_powerfoam_metal`.

## Change

- Added `_import_powerfoam_metal_height_sv_rasterizer()` in
  `tests/test_powerfoam_direct.py`.
- The helper calls `ensure_third_party_path("powerfoam-metal")` and imports
  `FoamRasterConfig` plus
  `rasterize_power_foam_quaternion_height_sv_texel_surface` directly from
  `torch_powerfoam_metal`.
- The two Metal fixture checks now use that helper instead of importing through
  `train_powerfoam_metal.py`.

This leaves `train_powerfoam_metal.py` as a trainer/model surface. The remaining
test and diagnostic imports from it are structural `MetalPowerFoamVideo` uses;
they should not be moved without a deliberate model-class split.

## Validation

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m py_compile tests/test_powerfoam_direct.py
```

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_loads_canonical_origin_parity_fixture \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_camera_local_fixture_shared_backward \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present \
  -q -rs
# 3 passed
```

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py -q
# 44 passed, 1 skipped
```

## Handoff

- Do not remove the `train.py` registry re-export surface yet; dynamic-load tests
  still use it intentionally.
- The next safe cleanup is import-scan driven: prefer moving pure helpers to
  existing owner modules, but leave model-class trainer imports alone unless the
  trainer class itself is being split.
