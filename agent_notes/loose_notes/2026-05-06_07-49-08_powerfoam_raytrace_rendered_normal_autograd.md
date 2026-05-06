# PowerFoam Raytrace Rendered Normal Autograd

Date: 2026-05-06 07:49 Asia/Ho_Chi_Minh

## Context

The remaining completion audit blockers are still paper-scale DeepView heldout
quality, not the basic Metal core. The official PowerFoam scan showed a
specific paper-mechanism gap: upstream training can supervise a rendered normal
map, while the local Metal training path only exposed RGB, alpha, and scalar
normal-distance.

The global stronger normal-distance run was negative, so the next mechanism
should not be another scalar weight sweep. The missing prerequisite is a
differentiable rendered-normal output from the actual height+SV raytrace path.

## Implementation

Added a differentiable rendered-normal output to the raytrace height+SV path:

- `powerfoam_raytrace_forward` now writes `out_normal [B,H,W,3]` using the same
  `weight * n_aux` accumulation as the existing aux forward path.
- `raytrace_forward` now returns `(out, alpha, normal_distance, normal, steps)`.
- `raytrace_height_sv_backward` accepts `grad_out_normal` and routes it through:
  - the opacity/transmittance scalar path via `dot(grad_out_normal, n_aux)`;
  - the normal feature slot via `weight * grad_out_normal`.
- `raytrace_power_foam_oriented_height_sv_texel_surface(..., return_normal=True)`
  exposes the low-level output.
- `MetalPowerFoamVideo.forward(..., return_rendered_normal=True)` exposes it for
  the train model when `use_raytrace=True`.

This is intentionally raytrace-first because the selected real-scene PowerFoam
rows use the raytrace height+SV path. Tiled rendered normals already exist in
the aux path, but not as a differentiable train output.

## Verification

Build:

```bash
( cd third_party/powerfoam-metal
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Result: extension rebuilt and copied
`third_party/powerfoam-metal/torch_powerfoam_metal/_C.cpython-311-darwin.so`.

Runtime checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_raytrace_rendered_normal_backprops -q
```

Result: `1 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with scipy \
  python third_party/powerfoam-metal/tests/raytrace_check.py
```

Result: passed; existing raytrace feature/alpha/normal-distance forward/backward
parity still holds.

Combined focused pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_normal_distance_loss_backprops_through_tiled_primitive \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_raytrace_rendered_normal_backprops -q
```

Result: `2 passed`.

Low-level smoke printed:

```text
normal_grad_sum = 0.7736170
density_grad_sum = 0.0142017
```

so the rendered-normal loss is not a dead output; it reaches both normal/frame
parameters and opacity.

## Current Boundary

This closes the first official-style normal-supervision prerequisite, not the
whole paper-quality gate. The completion audit still fails the selected clean
DeepView row at heldout PSNR/SSIM `12.6689 / 0.1000` versus required
`13.0 / 0.15`.

The next implementation step should wire an explicit `normal_supervision`
training loss:

1. request/compute detached median-depth normals;
2. compare differentiable rendered normals against those targets under a valid
   depth mask;
3. log `normal_distance_loss` separately from `normal_map_supervision_loss`.

