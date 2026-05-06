# Foam Implementation Status Audit

User asked whether we had fully implemented PowerFoam or Radiant Foam/RadFoam
in Metal, and whether only the dynamic/feature foam forks were complete.

## What I Checked

- Refreshed official repos:
  - `/tmp/powerfoam_official` at `96392252ebd0059fe6ca98881b62e12295d9242f`
  - `/tmp/radfoam_official` at `3e7b52cf74e37ab2ab5e695f53570f515f537e3d`
- Inventoried local foam files under:
  - `third_party/powerfoam-metal/`
  - `third_party/dynamic-powerfoam-metal/`
  - `src/train/powerfoam_direct.py`
  - `src/train/train_powerfoam_metal.py`
  - `src/train/train_dynamic_powerfoam_metal.py`
  - `research_notes/foam_papers/`
- Compared local implementations to upstream scene/raster/tracing surfaces.

## Result

We do not have a full official PowerFoam implementation locally.

We do not have a RadFoam/RadiantFoam Metal implementation locally at all.

What we do have:

- a slow Torch PowerFoam direct reference that is closest to paper primitive
  math, but not the official scalable training system
- a trainable partial PowerFoam Metal raster/backward core
- a dynamic PowerFoam Metal namespace fork whose kernels are still the same
  per-frame bounded-cell raster core
- a Dynaworld feature-foam fork that rasterizes F-channel features and
  colorizes them after alpha normalization

The strongest "foam moved a lot" result is still the motion-honesty probe
`xk5hwatb`, not an official PowerFoam/RadFoam reproduction:

- eval L1 `0.08901`
- mean temporal screen motion `2.92 px/frame`
- p95 temporal screen motion `8.46 px/frame`
- dynamic features disabled, so fit quality collapsed

## Validation

Confirmed local Python 3.11 extensions exist:

- `third_party/powerfoam-metal/torch_powerfoam_metal/_C.cpython-311-darwin.so`
- `third_party/dynamic-powerfoam-metal/torch_dynamic_powerfoam_metal/_C.cpython-311-darwin.so`

Ran:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/powerfoam_direct.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_powerfoam_direct.py \
  tests/test_dynamic_powerfoam_metal.py

PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/backward_check.py
PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/linear_texture_check.py
PYTHONPATH=src/train .venv/bin/python third_party/dynamic-powerfoam-metal/tests/backward_check.py
PYTHONPATH=src/train .venv/bin/python third_party/dynamic-powerfoam-metal/tests/linear_texture_check.py
```

All passed.

## Durable Doc

Added:

```text
research_notes/foam_papers/foam_implementation_status_2026-05-02.md
```

Also updated `research_notes/foam_papers/powerfoam_rasterizer_notes.md` with a
pointer because that older note still described the first forward-only Metal
prototype and could mislead future readers.

