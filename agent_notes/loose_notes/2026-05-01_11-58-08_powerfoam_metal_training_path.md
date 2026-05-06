# PowerFoam Metal training path

Date: 2026-05-01 11:58

## What changed

Implemented a trainable Metal path for the bounded power-cell foam core:

- Wired `powerfoam_stream_forward` and `powerfoam_stream_backward_global_atomic`
  into the `torch_powerfoam_metal` extension.
- Added a Python autograd wrapper for `rasterize_power_foam`, returning gradients
  for points, radii, densities, and per-cell features.
- Added a backward parity check against a Torch autograd reference.
- Added benchmark support for `--foam-backward`.
- Added `src/train/train_powerfoam_metal.py`, a lean MPS trainer that optimizes
  per-frame centers/radii/densities/RGB features directly through the Metal op.
- Added configs:
- `src/train_configs/local_mac_powerfoam_metal_video_64_smoke.jsonc`
- `src/train_configs/local_mac_powerfoam_metal_video_256_smoke.jsonc`
- `src/train_configs/local_mac_powerfoam_metal_video_1024_smoke.jsonc`
- Registered `arch: powerfoam_metal` in `src/train/train.py`.

Follow-up iteration in the same session:

- Replaced the Python per-cell KNN adjacency loop in `train_powerfoam_metal.py`
  with a vectorized `torch.cdist + topk` CSR builder.
- Added config-driven optimizer LR multipliers for points/radii/density/features.
- Added W&B/console drift metrics for center, xy, z, radius, density, and feature
  movement from initialization.

Second follow-up:

- Added a trainable linear feature mode to the Metal streaming renderer.
- New API: `rasterize_power_foam_linear(points, radii, densities, features, ...)`
  where `features` is `[N, C, 4]` for base/x/y/z coefficients.
- The backward replays the stream and includes the local-coordinate color
  gradient into features, centers, radii, and interval endpoints.
- Added `third_party/powerfoam-metal/tests/linear_texture_check.py`.
- Added trainer config knob `model.feature_mode: "constant" | "linear"`.
- Added `src/train_configs/local_mac_powerfoam_metal_linear_video_1024_smoke.jsonc`.

Third follow-up:

- Added a fixed camera-facing surface-linear feature mode to the Metal streaming
  renderer.
- New API: `rasterize_power_foam_surface_linear(...)`, using the same
  `[N, C, 4]` feature layout as midpoint-linear but clipping the bounded power
  interval by a `-z` plane through the cell center and sampling color at that
  surface point.
- Backward routes both the surface endpoint gradient and the local-coordinate
  color gradient into centers/radii/features without materializing
  `N x H x W`.
- Added `src/train_configs/local_mac_powerfoam_metal_surface_linear_video_1024_smoke.jsonc`.
- Extended `linear_texture_check.py` and the benchmark harness with
  `--foam-surface`.

## Expansion Pass 1: Learned Surface Normals In Metal

Context:

- The direct Torch PowerFoam path clips by per-cell normals derived from
  quaternions, while the first Metal surface path hard-coded a camera-facing
  `-z` plane.
- The next minimal paper-alignment step was learned orientation, not tiled
  scheduling or more cells.

Implementation decision:

- Added an oriented-surface-linear mode that appends learned `[N,3]` normals to
  the existing flattened feature buffer rather than adding another saved tensor.
- The Python wrapper normalizes normals before the custom autograd call, so the
  Metal backward returns gradients with respect to normalized normals and
  PyTorch handles the normalization VJP.
- The Metal backward routes both alpha endpoint clipping and local-coordinate
  color sampling through
  `t_surface = dot(center-origin,n) / dot(dir,n)`.
- This preserves the important memory property: replay per pixel, atomic-add to
  parameters, no `N x H x W` gradient tensor.

Falsification test:

- Extended `linear_texture_check.py` to compare oriented-surface forward and
  gradients against a Torch autograd reference.
- Result: oriented surface max errors were below `1e-6` for outputs and about
  `2e-8` for normal gradients, so the added math is locally correct.

## Verification

Build:

```bash
rtk env MAX_JOBS=4 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace
```

Parity:

```bash
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/reference_check.py
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/backward_check.py
```

Results:

- Forward reference check:
  - features max error `5.871e-06`
  - alpha max error `7.868e-06`
- Backward check:
  - output max error `2.302e-06`
  - alpha max error `3.278e-06`
  - points grad max error `2.459e-06`
  - radii grad max error `2.921e-06`
  - densities grad max error `1.155e-07`
  - features grad max error `4.657e-08`

Training smoke:

```bash
rtk env PYTHONPATH=src/train WANDB_MODE=offline uv run python src/train/train.py src/train_configs/local_mac_powerfoam_metal_video_64_smoke.jsonc
```

Result:

- 64 cells, 64px, 16 frames, 50 steps on MPS.
- step 0 eval `L1 = 0.07646`
- step 50 eval `L1 = 0.06346`

Random render harness:

```bash
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/render_random_png.py --cells 256 --height 128 --width 128 --out outputs/powerfoam_metal/random_foam_256_128.png
```

Result:

- Wrote `outputs/powerfoam_metal/random_foam_256_128.png`
- Wrote `outputs/powerfoam_metal/random_foam_256_128_alpha.png`
- 256 cells, 128x128, overlap adjacency, avg degree `9.74`.

Higher-capacity probe:

```bash
rtk env PYTHONPATH=src/train WANDB_MODE=offline uv run python -c 'from config_utils import load_config_file; from train_powerfoam_metal import run_training; cfg=load_config_file("src/train_configs/local_mac_powerfoam_metal_video_64_smoke.jsonc"); cfg["model"]["cells"]=256; cfg["model"]["neighbor_count"]=32; cfg["train"]["steps"]=120; cfg["logging"]["output_dir"]="outputs/powerfoam_metal/local_mac_powerfoam_metal_video_256_probe"; cfg["logging"]["wandb_enabled"]=False; run_training(cfg)'
```

Result:

- 256 cells, 64px, 16 frames, 120 steps on MPS.
- step 0 eval `L1 = 0.05469`
- step 120 eval `L1 = 0.03615`
- train-loop elapsed about `1.1s` excluding full-video logging.

Online W&B:

- Run: `https://wandb.ai/nbardy/dynaworld/runs/ytlgs1dm`
- Name: `powerfoam-metal-video-256-trainable-20260501`
- Final eval `L1 = 0.03615`, `MSE = 0.00358`.

Second online W&B after drift logging/capacity probe:

- Run: `https://wandb.ai/nbardy/dynaworld/runs/fqo5hqlp`
- Name: `powerfoam-metal-video-1024-drift-20260501`
- 1024 cells, 64px, 16 frames, 120 steps.
- Final eval `L1 = 0.03016`, `MSE = 0.00224`.
- Final drift:
  - mean center delta `0.02273`
  - p95 center delta `0.04248`
  - max center delta `0.08674`
  - mean radius delta `0.01238`
  - mean feature delta `0.05528`

Third online W&B after linear feature mode:

- Run: `https://wandb.ai/nbardy/dynaworld/runs/zyzr9j2c`
- Name: `powerfoam-metal-linear-video-1024-20260501`
- 1024 cells, 64px, 16 frames, 120 steps.
- Final eval `L1 = 0.02786`, `MSE = 0.00188`.
- Final drift:
  - mean center delta `0.02169`
  - p95 center delta `0.03981`
  - max center delta `0.07928`
  - mean radius delta `0.01196`
  - mean feature delta `0.05063`
- Visual preview improves scalar fit and foreground shape slightly, but the
  result is still visibly cell-based. Linear midpoint texture is a useful
  stepping stone, not a full PowerFoam surface/SV replacement.

Fourth online W&B after fixed surface-linear mode:

- Run: `https://wandb.ai/nbardy/dynaworld/runs/c3zeymsh`
- Name: `powerfoam-metal-surface-linear-video-1024-20260501`
- 1024 cells, 64px, 16 frames, 120 steps.
- Final eval `L1 = 0.02378`, `MSE = 0.00179`.
- Final drift:
  - mean center delta `0.01896`
  - p95 center delta `0.03829`
  - max center delta `0.11858`
  - mean xy delta `0.01642`
  - mean z delta `0.00752`
  - mean radius delta `0.00683`
  - mean density delta `0.00799`
  - mean feature delta `0.02558`
- Train-loop elapsed about `2.50s` excluding W&B/media upload overhead.
- Preview is now aligned with the target and beats constant/midpoint-linear
  scalar loss, but the alpha panel is still speckled/underfilled and the mode
  is a fixed camera-facing plane, not full learned PowerFoam geometry.

Surface-density calibration:

- `density_init=16.0` was a bad surface-linear default: step 0 eval
  `L1 = 0.24886`, step 120 eval `L1 = 0.07900`, visibly stippled coverage.
- `density64_radius072`, 60 steps: step 0 eval `L1 = 0.04317`, step 60 eval
  `L1 = 0.02539`.
- `density64_radius100`, 60 steps: step 0 eval `L1 = 0.03146`, step 60 eval
  `L1 = 0.02559`.
- `density128_radius100`, 60 steps: step 0 eval `L1 = 0.03075`, step 60 eval
  `L1 = 0.02727`.
- Promoted `density_init=64.0` for the checked-in surface-linear config.

Fifth online W&B after learned oriented-surface mode:

- Run: `https://wandb.ai/nbardy/dynaworld/runs/a0b48elm`
- Name: `powerfoam-metal-oriented-surface-linear-video-1024-20260501`
- 1024 cells, 64px, 16 frames, 120 steps.
- Promoted config: `normal_lr_multiplier=0.5`, `normal_init_jitter=0.05`.
- Final eval `L1 = 0.02367`, `MSE = 0.00174`.
- Final drift:
  - mean center delta `0.01847`
  - p95 center delta `0.03694`
  - max center delta `0.08503`
  - mean xy delta `0.01598`
  - mean z delta `0.00737`
  - mean radius delta `0.00672`
  - mean density delta `0.00779`
  - mean feature delta `0.03472`
  - mean normal delta `0.12080`
  - mean normal z `-0.98839`
- Train-loop elapsed about `2.67s` excluding W&B/media upload overhead.
- Visual preview remains close to the fixed-surface run. The scalar gain over
  fixed surface is real but tiny on this front-facing same-source clip.

Oriented-normal calibration:

- Default normal LR `0.05`, no jitter: final eval `L1 = 0.02379`, normal delta
  `0.0133`; basically tied with fixed surface.
- Normal LR `0.5`, no jitter: final eval `L1 = 0.02372`, normal delta `0.1218`.
- Normal LR `0.5`, jitter `0.05`: final eval `L1 = 0.02367`, normal delta
  `0.1208`; this became the checked-in oriented-surface config.
- Interpretation: learned normals are wired and trainable, but on this tiny
  mostly fronto-parallel video they do not unlock a large quality jump by
  themselves. The missing quality is likely detail sites / SV color / capacity,
  not just orientation.

Adjacency builder timing:

- Before vectorized KNN:
  - 256 cells / k=32: `6.28ms`
  - 512 cells / k=32: `7.66ms`
  - 1024 cells / k=32: `19.77ms`
- After vectorized KNN:
  - 256 cells / k=32: median `0.28ms`
  - 512 cells / k=32: median `0.74ms`
  - 1024 cells / k=32: median `2.77ms`

Geometry-LR probe:

- 512 cells, 80 steps, point LR multiplier `0.5`, radius/density multipliers
  `0.1`.
- Final eval worsened to `L1 = 0.05763`, `MSE = 0.01026`.
- Centers moved much more (`mean = 0.08834`, `max = 0.33389`), so the current
  failure is not simply frozen centers; aggressive geometry motion destabilizes
  the core-cell renderer.

Benchmark:

```bash
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py --cells 256,1024 --resolutions 128x128 --warmup 1 --iters 3 --foam-backward --compare-gs --gs-backward --json
```

Representative medians:

- PowerFoam Metal 256 cells: forward `5.11ms`, backward `5.15ms`, total `10.06ms`.
- GS v5 features 256 splats: forward `8.34ms`, backward `2.68ms`, total `11.32ms`.
- PowerFoam Metal 1024 cells: forward `5.04ms`, backward `4.35ms`, total `8.28ms`.
- GS v5 features 1024 splats: forward `3.36ms`, backward `1.97ms`, total `5.24ms`.

Later five-iteration rerun medians at `128x128`:

- PowerFoam Metal 256 cells: forward `2.72ms`, backward `2.22ms`, total `4.63ms`.
- GS v5 features 256 splats: forward `7.14ms`, backward `3.86ms`, total `11.00ms`.
- PowerFoam Metal 1024 cells: forward `6.18ms`, backward `6.49ms`, total `13.02ms`.
- GS v5 features 1024 splats: forward `7.16ms`, backward `3.98ms`, total `9.75ms`.

Clean sequential constant-vs-linear benchmark at `128x128`, warmup 3, iters 8:

- PowerFoam Metal constant 256 cells: forward `2.31ms`, backward `1.94ms`, total `4.42ms`.
- PowerFoam Metal linear 256 cells: forward `2.23ms`, backward `2.02ms`, total `4.38ms`.
- PowerFoam Metal constant 1024 cells: forward `4.69ms`, backward `5.00ms`, total `9.67ms`.
- PowerFoam Metal linear 1024 cells: forward `5.00ms`, backward `5.07ms`, total `10.21ms`.

Clean sequential constant-vs-linear-vs-surface benchmark at `128x128`, warmup
3, iters 8, rerun after surface-linear support:

- PowerFoam Metal constant 256 cells: forward `3.81ms`, backward `2.99ms`, total `6.91ms`.
- PowerFoam Metal linear 256 cells: forward `3.56ms`, backward `3.10ms`, total `7.13ms`.
- PowerFoam Metal surface-linear 256 cells: forward `4.58ms`, backward `3.46ms`, total `8.15ms`.
- PowerFoam Metal constant 1024 cells: forward `5.51ms`, backward `6.87ms`, total `12.38ms`.
- PowerFoam Metal linear 1024 cells: forward `6.59ms`, backward `5.33ms`, total `11.69ms`.
- PowerFoam Metal surface-linear 1024 cells: forward `12.47ms`, backward `11.13ms`, total `23.96ms`.
- The surface-linear quality win is not free: at 1024 cells it is roughly 2x
  the total fwd+bwd time of the midpoint-linear path in this harness.

Clean oriented-surface benchmark at `128x128`, warmup 3, iters 8:

- PowerFoam Metal oriented-surface 256 cells: forward `5.17ms`, backward `5.34ms`, total `10.61ms`.
- PowerFoam Metal oriented-surface 1024 cells: forward `10.98ms`, backward `11.76ms`, total `22.52ms`.
- Learned normals land in the same slow class as fixed surface. They did not
  add an obvious extra timing cliff, but surface clipping itself is still much
  slower than midpoint-linear.

## Expansion Pass 2: Oriented Texel-Surface Metal Mode

Implemented the next missing paper-shaped piece in the Metal path: learned local
detail sites on each learned surface. This is not the official PowerFoam SV
texture stack yet, but it changes the feature model from one linear color plane
per cell to `S` learned local sites with normalized Gaussian interpolation:

- Python API:
  - `rasterize_power_foam_oriented_texel_surface(points, radii, densities, texel_sites, texel_features, normals, ...)`
  - `texel_sites` shape `[N,S,2]`, radius-normalized in local surface coordinates.
  - `texel_features` shape `[N,S,C]`.
- Metal feature layout:
  - flattened `[site_x, site_y, C features] * S`, then appended `[normal_x, normal_y, normal_z]`.
  - `feature_mode = 4`.
  - `meta_f32[5] = texel_temperature`.
- Backward:
  - replay per pixel/cell, no `N x H x W` gradient tensor.
  - gradients route to centers, radii, densities, texel sites, texel RGB, and normals.
- Trainer:
  - new config `src/train_configs/local_mac_powerfoam_metal_oriented_texel_surface_video_1024_smoke.jsonc`.
  - initializes from `initialize_full_powerfoam_from_video(..., num_texel_sites=4, sv_dof=1)`.
  - logs center/radius/density/feature/normal/texel-site drift to W&B.

Parity and smoke checks:

```bash
rtk env PYTHONPATH=src/train uv run python -m py_compile src/train/train_powerfoam_metal.py third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py third_party/powerfoam-metal/tests/linear_texture_check.py tests/test_powerfoam_direct.py
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/reference_check.py
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/backward_check.py
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/linear_texture_check.py
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q
```

Results:

- `reference_check.py`: features max error `5.87e-06`, alpha max error `7.87e-06`.
- `backward_check.py`: points/radii/density/features gradients matched the scalar reference path.
- `linear_texture_check.py` now covers linear, surface-linear, oriented-surface-linear, and oriented-texel-surface.
- Oriented texel-surface parity:
  - features max error `8.42e-07`
  - alpha max error `1.43e-06`
  - points grad max error `1.16e-07`
  - radii grad max error `1.49e-07`
  - densities grad max error `3.17e-08`
  - texel-sites grad max error `3.49e-09`
  - texel-features grad max error `1.21e-08`
  - normals grad max error `1.77e-08`
- `tests/test_powerfoam_direct.py`: `8 passed`.

Offline then online 1024-cell video smoke:

- Config: `src/train_configs/local_mac_powerfoam_metal_oriented_texel_surface_video_1024_smoke.jsonc`.
- W&B: `https://wandb.ai/nbardy/dynaworld/runs/t4enmpcc`.
- Step 0 eval: `L1 = 0.04518`, `MSE = 0.00426`.
- Step 120 eval: `L1 = 0.01899`, `MSE = 0.00132`.
- Train-loop elapsed: `2.86s` excluding W&B/media upload.
- Final drift:
  - mean center delta `0.01757`
  - p95 center delta `0.03571`
  - max center delta `0.09140`
  - mean radius delta `0.00695`
  - mean density delta `0.00866`
  - mean feature delta `0.03146`
  - mean normal delta `0.12532`
  - mean normal z `-0.98760`
  - mean texel-site delta `0.02774`
- This beats the previous oriented-surface run (`a0b48elm`, `L1 = 0.02367`,
  `MSE = 0.00174`) and visually removes much of the coarse block/grid look.

Texel-surface benchmark at `128x128`, warmup 3, iters 8:

```bash
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py --cells 256,1024 --resolutions 128x128 --warmup 3 --iters 8 --foam-backward --foam-texel-surface --compare-gs --gs-backward --json
```

- PowerFoam Metal oriented-texel 256 cells: forward `4.58ms`, backward `2.41ms`, total `6.99ms`.
- GS v5 features 256 splats: forward `3.06ms`, backward `1.29ms`, total `4.36ms`.
- PowerFoam Metal oriented-texel 1024 cells: forward `3.75ms`, backward `4.44ms`, total `8.16ms`.
- GS v5 features 1024 splats: forward `3.11ms`, backward `1.97ms`, total `5.06ms`.
- Quality moved strongly, but the new mode is still slower than GS on this
  benchmark. The backward shape is still the right one: replay and atomics,
  not an instance-per-pixel grad buffer.

## Caveats

This is the trainable Metal **core** foam path, not full PowerFoam:

- learned oriented surface normals exist in the Metal path, but there is still
  no quaternion frame, two-sided dipole model, or official regularizer stack
- learned local detail sites exist, but no detail-site heights
- no spherical-Voronoi view-dependent texture
- no contribution/normal/point-error outputs
- no densification/pruning/resampling
- no official Cech/AABB adjacency builder
- no tiled high-throughput replay kernel

It does prove the critical training contract for the simple bounded power-cell
renderer: gradients flow through centers, radii, densities, and features without
materializing an `N x H x W` gradient tensor.
