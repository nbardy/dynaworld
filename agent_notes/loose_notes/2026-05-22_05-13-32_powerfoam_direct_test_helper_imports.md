# PowerFoam Direct Test Helper Imports

## Context

The trainer unification pass had already split most PowerFoam Metal helper
logic into light modules, but `tests/test_powerfoam_direct.py` still imported
many pure helpers through `train_powerfoam_metal.py`. That made the full Metal
trainer look like the helper namespace even for CPU-safe contracts such as ray
construction, multiview flattening, point-cloud initialization, adjacency, and
objective math.

## Change

Rerouted pure helper imports in `tests/test_powerfoam_direct.py` to their owning
modules:

- `powerfoam_geometry.py` for pinhole/camera rays.
- `powerfoam_training.py` for multiview sample flattening.
- `powerfoam_metal_config.py` for Metal loss/render/train defaults.
- `powerfoam_objectives.py` for SSIM, scheduled Metal loss weights,
  contribution/normal losses, normal-map targets, and alpha/background
  compositing.
- `powerfoam_point_cloud.py` for point-cloud initialization.
- `powerfoam_raster_config.py` for `FoamRasterConfig` construction.
- `powerfoam_adjacency.py` for CSR adjacency and adjacency stats.

The remaining `train_powerfoam_metal.py` imports in that test file are
structural: `MetalPowerFoamVideo` for model behavior tests and raw Metal raster
fixture symbols for extension parity tests.

## Validation

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile tests/test_powerfoam_direct.py
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q
```

Result: `44 passed, 1 skipped`.

## Handoff

This closes the obvious PowerFoam Direct test-as-helper-import cleanup. The
next unification pass should look for real duplicate trainer bodies or
script-level dispatch bypasses, not for these pure PowerFoam helper imports.
