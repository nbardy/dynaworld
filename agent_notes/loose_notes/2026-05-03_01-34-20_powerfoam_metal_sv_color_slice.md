# PowerFoam Metal SV Color Slice

Date: 2026-05-03 01:34:20

## Context

The previous slice added height-displaced texel surfaces to the Metal streaming
path. The next missing PowerFoam primitive was spherical-Voronoi
view-dependent texel color. This note records the implementation, validation,
and the remaining 4K performance blocker.

## Implemented

- Added Metal streaming `feature_mode == 6`:
  - per-texel layout: `u, v, height, sv_axis[D,3], sv_rgb[D,3]`
  - frame footer remains `normal, tangent, bitangent`
  - `sv_dof` is passed through `meta_i32`
- Added SV color evaluation in Metal:
  - view direction is from ray origin to the texel world site
  - axes are normalized with temperature equal to raw-axis norm
  - output color is weighted raw SV RGB plus `0.5`, clamped at zero
- Added Metal replay backward for:
  - SV RGB
  - SV axes/temperatures
  - spatial texel interpolation around SV colors
  - height displacement and height-query gradients inherited from mode 5
- Matched the official detach semantics for the SV color query: the view
  direction used inside SV color does not route gradients back into geometry.
- Added Python APIs:
  - `rasterize_power_foam_oriented_height_sv_texel_surface`
  - `rasterize_power_foam_quaternion_height_sv_texel_surface`
- Added trainable trainer mode:
  `feature_mode="quaternion_height_sv_texel_surface"`.
- Added config:
  `src/train_configs/local_mac_powerfoam_metal_quaternion_height_sv_texel_surface_video_1024_smoke.jsonc`.
- Added small SV init jitter so the default image-init smoke breaks identical
  lobe symmetry and actually moves SV axes, not only SV RGB.
- Extended benchmark harness with `--foam-height-sv-texel-surface`.

## Validation

Passed:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/powerfoam-metal
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile \
  src/train/train_powerfoam_metal.py \
  third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py \
  third_party/powerfoam-metal/tests/linear_texture_check.py \
  tests/test_powerfoam_direct.py
PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/linear_texture_check.py
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q
```

New SV parity maxima from `linear_texture_check.py`:

- `oriented_height_sv_texel_surface` explicit frame:
  - features max error: `8.530914783477783e-07`
  - alpha max error: `1.4901161193847656e-06`
  - texel SV axis grad max error: `2.9103830456733704e-10`
  - texel SV RGB grad max error: `3.958120942115784e-09`
- `quaternion_height_sv_texel_surface`:
  - features max error: `9.238719940185547e-07`
  - alpha max error: `1.6093254089355469e-06`
  - texel SV axis grad max error: `4.0745362639427185e-10`
  - texel SV RGB grad max error: `6.28642737865448e-09`
  - quaternion grad max error: `6.705522537231445e-08`

1-step trainer smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled .venv/bin/python \
  src/train/train_powerfoam_metal.py /tmp/powerfoam_quaternion_height_sv_texel_1step_smoke.jsonc
```

Result:

- step 0 eval L1: `0.03431244194507599`
- step 1 eval L1: `0.03338077664375305`
- `state_mean_quaternion_delta`: `0.0012472760863602161`
- `state_mean_texel_height_delta`: `4.184159479336813e-06`
- `state_mean_texel_sv_axis_delta`: `0.000185119773959741`
- `state_mean_texel_sv_rgb_delta`: `0.0012483583996072412`

## 4K Benchmark

Command:

```bash
PYTHONPATH=src/train .venv/bin/python \
  third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py \
  --cells 1024,4096 \
  --resolutions 4096x4096 \
  --feature-dim 3 \
  --neighbors 32 \
  --warmup 1 \
  --iters 2 \
  --foam-backward \
  --foam-height-sv-texel-surface \
  --json
```

Saved JSON:

`outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_4k_1024_4096_2026-05-03.json`

Results:

- 1024 cells, 4096x4096: forward `2840.227 ms`, backward `12515.852 ms`,
  total `15356.079 ms`.
- 4096 cells, 4096x4096: forward `8192.781 ms`, backward `23298.422 ms`,
  total `31491.203 ms`.

Conclusion: the full height+SV primitive is accurate and trainable in the
current streaming Metal path, but it is not remotely fast enough at 4K. The
tiled candidate-list/replay path remains mandatory for the user's 4K speed
requirement.

## Remaining

- Official auxiliary outputs: normal distance, normal, depth/contribution/error,
  visibility.
- Cech/AABB adjacency instead of KNN as the correctness path.
- Static posed-camera SfM trainer.
- Densification/pruning/resampling.
- Ray tracing backend.
- Tiled/replay path fast enough for 4K.
