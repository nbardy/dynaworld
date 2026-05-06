# Dynamic PowerFoam Naive Fork

## Context

User asked to fork the PowerFoam shaders and implement two naive dynamic
variants: a quick per-frame smooth diagnostic and a splat-like time-curve model
where alpha/position/etc. are functions of frame time.

## What Changed

- Forked `third_party/powerfoam-metal` to `third_party/dynamic-powerfoam-metal`.
- Renamed the Python package to `torch_dynamic_powerfoam_metal`.
- Renamed the Torch custom-op namespace from `powerfoam_metal` to
  `dynamic_powerfoam_metal`.
- Renamed the Metal shader entry points and files:
  - `dynamic_powerfoam_kernels.metal`
  - `dynamic_powerfoam_streaming_kernels.metal`
  - `dynamic_powerfoam_tiled_kernels.metal`
- Added `src/train/train_dynamic_powerfoam_metal.py`.
- Added configs:
  - `src/train_configs/local_mac_dynamic_powerfoam_metal_per_frame_smooth_1024_smoke.jsonc`
  - `src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_1024_smoke.jsonc`
- Registered `arch: dynamic_powerfoam_metal` in `src/train/train.py`.
- Added CPU decode/backward tests in `tests/test_dynamic_powerfoam_metal.py`.
- Added dispatch coverage in `tests/test_powerfoam_direct.py`.
- Added rows to `BASELINES.md`.

## Variant A: Per-frame Smooth

This is intentionally not a true dynamic representation. It is the current
per-frame PowerFoam table, but routed through the dynamic trainer/forked op and
with temporal acceleration penalties on decoded centers/radii/densities/features.

Purpose: quantify how much simply smoothing the current per-frame table changes
the tiny same-source fit.

Result:

- Config: `src/train_configs/local_mac_dynamic_powerfoam_metal_per_frame_smooth_1024_smoke.jsonc`
- 1024 cells, 16 frames, 64 px, 120 steps
- Train loop wall: 3.63 s
- Final eval L1: 0.02002
- Final eval MSE: 0.00151

This roughly matches the existing material-frame PowerFoam row
(`ecysgsk8`: L1 0.01984, MSE 0.00143, 3.08 s train loop).

## Variant B: Gaussian RBF Dynamic Decoder

This is the first true naive dynamic model. It stores canonical/base raw
PowerFoam parameters plus `K=8` normalized Gaussian time-basis coefficients for
centers, radii, density, and texel RGB features. Normals and texel sites are
static in the default config.

The Metal kernel is unchanged at the math level: Python decodes the selected
frame into ordinary per-frame cells, then calls the forked Metal rasterizer.
This keeps the existing replay backward and avoids any instance-by-pixel
gradient tensor.

Result:

- Config: `src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_1024_smoke.jsonc`
- 1024 cells, 16 frames, 64 px, 120 steps
- Train loop wall: 4.07 s
- Final eval L1: 0.05155
- Final eval MSE: 0.00803

Takeaway: with only 8 time controls, the shared dynamic model is much lower
capacity than the per-frame table. This is a useful failure signal. Next checks
should try more basis functions, per-field LR changes, dynamic normals/sites,
or piecewise-linear controls before adding fused time decoding to Metal.

## Validation

- `uv run python -m py_compile src/train/train_dynamic_powerfoam_metal.py src/train/train.py tests/test_dynamic_powerfoam_metal.py tests/test_powerfoam_direct.py`
- Built forked extension:
  `( cd third_party/dynamic-powerfoam-metal && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )`
- Manual decode/backward tests passed for both dynamic modes.
- Dynamic fork `reference_check.py` passed.
- Dynamic fork `backward_check.py` passed.
- Dynamic fork `linear_texture_check.py` passed, including oriented texel-surface material-frame gradients.
- 1-step MPS smokes passed for both modes at 64 cells / 32 px / 2 frames.
- Full local 1024-cell 120-step runs passed for both checked-in configs.

`pytest` is still unavailable in this environment, so tests were run manually.
