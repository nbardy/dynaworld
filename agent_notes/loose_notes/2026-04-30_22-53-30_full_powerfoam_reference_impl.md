# Full PowerFoam reference implementation

## Why this happened

The first local `powerfoam_direct` trainer only implemented a small foam-like
subset: sphere/power-cell clipping plus one constant RGB per cell. That was not
the official PowerFoam model and produced blurry cell blobs.

The official implementation in `/tmp/powerfoam_official` uses:

- points, radii, density;
- quaternions, normals, tangents, bitangents;
- local texel/detail sites per cell;
- per-texel height/displacement;
- spherical-Voronoi view-dependent texel color;
- Cech/power adjacency rebuilt from an AABB tree;
- separate optimizer rates per parameter family;
- densification/resampling over long training.

## What changed locally

`src/train/powerfoam_direct.py` now implements a Torch reference path with the
main official attribute structure:

- `PowerFoamInitialization`
- camera-facing quaternion init
- quaternion-derived normal/tangent/bitangent frames
- overlap power adjacency with KNN fallback
- KNN/radius-scale radius init
- local texel sites sampled from image patches
- spherical-Voronoi texel color
- texel-height plane query before segment integration
- official-style plane clipping and density compositing
- optimizer param groups for points/radii/density/quats/texels/view-color/height

This is still not the final fast implementation. The remaining gap to official
PowerFoam is the production CUDA/Warp/Metal tiled renderer, AABB/Cech builder,
long-run densification/resampling, normal supervision, and raytrace backend.

## Run evidence

Tiny full single-frame smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python src/train/train.py \
  src/train_configs/local_mac_powerfoam_direct_full_tiny_smoke.jsonc
```

Result:

- 1 frame, 64px, 64 cells, 8 texel sites, 8 SV dof
- step 0 eval L1: `0.0613622069`
- step 2 eval L1: `0.0463245064`

128px single-image reference:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python src/train/train.py \
  src/train_configs/local_mac_powerfoam_direct_single_image_128_smoke.jsonc
wandb sync wandb/offline-run-20260430_224217-sgrodq6f
```

Result:

- W&B: `sgrodq6f`
- 1 frame, 128px, 128 cells, 8 texel sites, 8 SV dof
- step 0 eval L1: `0.0720376968`
- step 50 eval L1: `0.0377219804`
- step 100 eval L1: `0.0368901901`
- This slightly beats the earlier direct GS single-image baseline L1
  `0.0381829441`.

16-frame 128px single-LR failure:

- W&B: `vaxqqjb6`
- 16 frames, 128px, 128 cells, 8 texel sites, 8 SV dof
- step 0 eval L1: `0.0713407770`
- step 50 eval L1: `0.1245370805`
- step 100 eval L1: `0.1165450737`
- Interpretation: the full-attribute init is good, but a single large LR
  destabilized per-frame foam training.

Short 16-frame full smoke after optimizer param groups:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python src/train/train.py \
  src/train_configs/local_mac_powerfoam_direct_video_full_tiny_smoke.jsonc
```

Result:

- 16 frames, 64px, 64 cells, 8 texel sites, 8 SV dof
- step 0 eval L1: `0.0642575696`
- step 20 eval L1: `0.0565612949`
- This validates the full-attribute path on video at smoke scale.

## Next

The next implementation step is not more Python reference work. It is porting
this full attribute/raster path into the Metal kernel plan:

- tile/candidate list renderer;
- per-pixel reverse replay accumulating only per-cell/texel gradients, not
  `N*H*W`;
- overlap/Cech adjacency builder;
- optional densification/resampling.

The Python reference is now good enough to be the correctness oracle for that
Metal path.
