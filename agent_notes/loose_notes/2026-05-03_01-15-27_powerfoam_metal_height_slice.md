# PowerFoam Metal Height Slice

Date: 2026-05-03 01:15:27

## Context

The previous Metal slice added strict quaternion frame plumbing for the existing
oriented texel-surface primitive. The next missing paper primitive was per-site
height displacement. This note records what was implemented and what remains
before the implementation can be called full PowerFoam.

## Implemented

- Added Metal streaming `feature_mode == 5` with per-texel layout:
  `u, v, height, feature...` and the existing 9-float frame footer
  `normal, tangent, bitangent`.
- Added height-displaced surface clipping in forward.
- Extended the replay backward for height endpoints:
  - direct height value gradients
  - height interpolation weight gradients
  - texel site gradients from the height query
  - tangent/bitangent gradients from the height query
  - center/radius gradients from the height query local coordinate
  - base endpoint gradients when the height query uses the pre-surface near
    endpoint
- Added the missing color-sample local-coordinate gradient path for mode 5, so
  color interpolation through the displaced near sample contributes to
  `t_near`, points, and radii.
- Exposed Python APIs:
  - `rasterize_power_foam_oriented_height_texel_surface`
  - `rasterize_power_foam_quaternion_height_texel_surface`
- Added trainable `feature_mode="quaternion_height_texel_surface"` in
  `src/train/train_powerfoam_metal.py`.
- Added config:
  `src/train_configs/local_mac_powerfoam_metal_quaternion_height_texel_surface_video_1024_smoke.jsonc`.
- Extended `third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py`
  with `--foam-height-texel-surface`.

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

New height parity maxima from `linear_texture_check.py`:

- `oriented_height_texel_surface` explicit frame:
  - features max error: `9.238719940185547e-07`
  - alpha max error: `1.4901161193847656e-06`
  - points grad max error: `1.7182901501655579e-07`
  - radii grad max error: `1.043081283569336e-07`
  - densities grad max error: `2.421438694000244e-08`
  - texel sites grad max error: `6.28642737865448e-09`
  - texel heights grad max error: `3.725290298461914e-08`
  - texel features grad max error: `8.381903171539307e-09`
  - normals grad max error: `2.0954757928848267e-08`
  - tangents grad max error: `3.14321368932724e-09`
  - bitangents grad max error: `2.3865140974521637e-09`
- `quaternion_height_texel_surface`:
  - features max error: `7.525086402893066e-07`
  - alpha max error: `1.6093254089355469e-06`
  - points grad max error: `1.0058283805847168e-07`
  - radii grad max error: `2.253800630569458e-07`
  - densities grad max error: `5.21540641784668e-08`
  - texel sites grad max error: `6.810296326875687e-09`
  - texel heights grad max error: `4.470348358154297e-08`
  - texel features grad max error: `1.4668330550193787e-08`
  - quaternions grad max error: `5.960464477539063e-08`

1-step trainer smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled .venv/bin/python \
  src/train/train_powerfoam_metal.py /tmp/powerfoam_quaternion_height_texel_1step_smoke.jsonc
```

Result:

- step 0 eval L1: `0.03372238203883171`
- step 1 eval L1: `0.033242788165807724`
- `state_mean_quaternion_delta`: `0.0012468267232179642`
- `state_mean_texel_height_delta`: `4.189235369267408e-06`

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
  --foam-height-texel-surface \
  --json
```

Saved JSON:

`outputs/benchmarks/powerfoam_metal_height_texel_surface_4k_1024_4096_2026-05-03.json`

Results:

- 1024 cells, 4096x4096: forward `1770.273 ms`, backward `3746.958 ms`,
  total `5517.232 ms`.
- 4096 cells, 4096x4096: forward `6426.381 ms`, backward `9280.640 ms`,
  total `15707.021 ms`.

Conclusion: height mode is accurate and trainable in the current streaming
path, but the fast-at-4K requirement is still not satisfied. Tiled candidate
lists plus replay backward remain the performance blocker.

## Remaining

- Spherical-Voronoi view-dependent color in Metal.
- Official detach semantics for the SV color query.
- Official auxiliary outputs: normal distance, normal, depth/contribution/error,
  visibility.
- Cech/AABB adjacency instead of KNN as the correctness path.
- Static posed-camera SfM trainer.
- Densification/pruning/resampling.
- Tiled/replay path fast enough for 4K.
