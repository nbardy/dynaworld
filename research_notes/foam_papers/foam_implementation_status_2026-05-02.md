# Foam Implementation Status: PowerFoam, Feature Foam, Dynamic Foam, RadFoam

Original date: 2026-05-02

Status refresh: 2026-05-05. This file is now a routed status snapshot, not the
primary source of truth. For current acceptance gates and measured rows, use:

- `TODO/powerfoam_full_reproduction_todo.md`
- `BASELINES.md`

This note answers a specific confusion: which foam systems are fully
implemented locally, which are partial reproductions, and whether we ever
ported RadFoam / Radiant Foam to Metal.

Action plan:

```text
TODO/powerfoam_full_reproduction_todo.md
```

## Short Verdict

- **Full official PowerFoam:** not implemented locally.
- **Full official RadFoam / Radiant Foam:** not implemented locally.
- **PowerFoam Torch direct reference:** implemented as the slow correctness
  reference for the paper primitive and posed-camera smoke paths, but not the
  official scalable training system.
- **PowerFoam Metal:** implemented as a trainable bounded-cell Metal raster and
  raytrace core with replay backward, `cech_aabb` correctness adjacency,
  quaternion height+SV primitive math, static posed-camera trainer plumbing,
  and verified synthetic 4K forward+backward artifacts. It is still a partial
  PowerFoam reproduction, not the official CUDA/Warp system.
- **Dynamic PowerFoam Metal:** implemented as a local experimental fork. Time
  is decoded in Python before each frame calls the Metal rasterizer; there is no
  fused dynamic kernel.
- **Feature foam:** implemented as our Dynaworld fork: PowerFoam-style bounded
  cells raster arbitrary feature channels, then a colorizer maps features to
  RGB. This is not in the upstream PowerFoam or RadFoam repos.

## Refreshed Upstream Repos

Checked on 2026-05-02:

```text
/tmp/powerfoam_official 96392252ebd0059fe6ca98881b62e12295d9242f
/tmp/radfoam_official   3e7b52cf74e37ab2ab5e695f53570f515f537e3d
```

PowerFoam upstream is Python plus Warp/CUDA. It trains static posed-camera
scenes from COLMAP/SfM or random initial points. Its scene state includes:

- points
- radii
- density
- quaternions
- texel sites
- spherical-Voronoi axes/RGB
- texel heights
- AABB/Cech adjacency
- rasterizer and raytracer backends
- densification/resampling/pruning and contribution/error stats

RadFoam upstream is CUDA/C++ plus Python bindings. It is not a PowerFoam
rasterizer. It builds and updates a Delaunay triangulation / adjacency graph,
traces rays through that structure, and optimizes:

- primal points
- density
- DC color attributes
- SH color attributes
- triangulation and AABB-tree state
- densification/pruning state

There is no local `third_party/radfoam-metal`, no local `train_radfoam_metal.py`,
and no RadFoam row in `BASELINES.md`.

## Local Implementations

| Path | Exists | Metal | Trainable | Full official reproduction | Notes |
|---|---:|---:|---:|---:|---|
| `src/train/powerfoam_direct.py` | yes | no | yes | no | Slow Torch reference. Closest local paper-math path: bounded cells, quaternion frame, detail sites, heights, spherical-Voronoi color hooks, and posed-camera smoke paths. Not official scale/system. |
| `third_party/powerfoam-metal/` | yes | yes | yes | no | MPS custom op with streaming/tiled/raytrace paths and replay backward. Supports full quaternion height+SV primitive mode, Cech/AABB adjacency inputs, normal-distance gradients, and synthetic 4K benchmark artifacts. |
| `src/train/train_powerfoam_metal.py` | yes | yes | yes | no | Static posed-camera Metal trainer with point-cloud init, train/heldout metrics, official-style LR groups/losses, and grow/prune/resample plumbing. Current quality rows are short local probes, not paper-scale PowerFoam. |
| `third_party/dynamic-powerfoam-metal/` | yes | yes | yes | no | Namespace fork of the Metal raster core for dynamic experiments. Kernels are essentially the same raster core. |
| `src/train/train_dynamic_powerfoam_metal.py` | yes | yes | yes | no | Python time decoding plus Metal per-frame raster. Includes token feature foam. |
| RadFoam / Radiant Foam Metal | no | no | no | no | Not ported. Only scanned/planned in notes. |

## What PowerFoam Metal Actually Has

Local Metal package:

```text
third_party/powerfoam-metal/
third_party/dynamic-powerfoam-metal/
```

Implemented:

- MPS Torch extension and Metal kernels
- bounded power-cell ray interval clipping
- front-to-back compositing
- arbitrary feature channels
- streaming/tiled/raytrace replay backward for trainable primitive state
- local-linear feature mode
- fixed surface-linear mode
- oriented surface-linear mode
- oriented texel-surface feature mode
- strict quaternion texel-surface mode
- detail-site height/displacement in Metal
- spherical-Voronoi view-dependent color in Metal
- gradients for texel sites, heights, features, normals, tangents,
  bitangents, quaternions, SV axes, and SV RGB
- CPU `cech_aabb` correctness adjacency in the local trainer
- trainable full height+SV raytrace backend with normal-distance output/gradient
- static posed-camera trainer path with point-cloud init and heldout logging
- contribution/error EMAs plus grow/prune/resample plumbing in the local trainer
- saved synthetic 4K verifier for full height+SV raytrace `cech_aabb`
  forward+backward artifacts

Not implemented relative to official PowerFoam:

- local CUDA/Warp official fixture generation; the official parity generator is
  wired but needs a CUDA/Warp host
- differentiable arbitrary depth-quantile and external normal-supervision losses
  if those losses are selected
- paper-scale static multiview schedule/quality acceptance
- dense paper-clean COLMAP/SfM reconstruction; current clean DeepView artifacts
  are mostly two-view tracks and remain below paper-quality evidence
- paper-scale grow/prune/resampling acceptance runs

## What Feature Foam Actually Is

Feature foam is our fork, not upstream PowerFoam or RadFoam.

Current token feature foam does:

```text
cell token
  -> decoded bounded-cell state over RBF time
  -> Metal oriented texel-surface rasterizer
  -> F-channel feature image + alpha
  -> alpha normalization
  -> FeatureToColor
  -> RGB
```

The standard F32 config uses 1024 cells, 4 texel sites per cell, F32 features,
dynamic features, and dynamic densities. That is expressive enough to fit by
repainting a mostly fixed lattice.

Measured standard F32 feature run:

```text
W&B: 0v67kicc
eval L1: 0.03427
eval MSE: 0.00381
```

Motion-audited rerun after diagnostics:

```text
eval L1: 0.03446
mean temporal screen motion: 0.039 px/frame
p95 temporal screen motion: 0.109 px/frame
mean temporal feature delta: 0.02370
```

Conclusion: high-quality feature foam did not learn much temporal motion.

## Did We Ever Get Foam Moving A Lot?

Yes, but only in a motion-honesty probe, not in an official PowerFoam/RadFoam
reproduction.

Config:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_motion_probe.jsonc
```

W&B:

```text
https://wandb.ai/nbardy/dynaworld/runs/xk5hwatb
```

Result:

```text
eval L1: 0.08901
eval MSE: 0.01899
mean temporal screen motion: 2.92 px/frame
p95 temporal screen motion: 8.46 px/frame
mean temporal feature delta: 0.0
```

This proves the dynamic representation can move when appearance shortcuts are
removed, but the fit quality collapses.

## Validation Run During This Audit

Local `.so` files exist for Python 3.11:

```text
third_party/powerfoam-metal/torch_powerfoam_metal/_C.cpython-311-darwin.so
third_party/dynamic-powerfoam-metal/torch_dynamic_powerfoam_metal/_C.cpython-311-darwin.so
```

Commands run on 2026-05-02:

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

All passed. The Metal backward parity checks were within about `1e-6` to `1e-8`
for rendered features/alpha and gradients in the tested modes.

## Bottom Line

The local wording should be:

```text
PowerFoam Metal core: implemented as a partial trainable raster/raytrace/backward core.
Feature foam: implemented as our experimental feature-raster fork.
Dynamic feature foam: implemented as our experimental Python-time-decoded fork.
Full official PowerFoam reproduction: not implemented.
RadFoam / Radiant Foam Metal port: not implemented.
```
