# PowerFoam reproduction gap fixes

Date: 2026-04-30 23:04

## Context

The reproduction audit found that the local Torch PowerFoam path was useful but
not paper/code parity. The easiest high-confidence fixes were local math and
loss/stat plumbing, not the full Metal rasterizer.

## Changes made

- Added `PowerFoamRenderResult` while preserving old four-value unpacking.
- Switched radii and density to official-style `softplus(beta=100)` plus a
  matching inverse-softplus initializer.
- Changed texel-site parameters to be radius-normalized local offsets:
  `site_world = point + radius * (u * tangent + v * bitangent)`.
- Detached world texel sites before spherical-Voronoi color lookup, matching
  official PowerFoam's color function call.
- Added renderer stats needed by the official loss stack:
  normal-distance, normal accumulation, contribution, point error, and visible
  primitive mask.
- Added SSIM, normal, contribution, and interpenetration loss hooks to the
  direct trainer.
- Added official-style exponential decay for the normal/contribution/
  interpenetration regularizer weights.
- Fixed the camera-facing quaternion sign. The earlier quaternion rotated the
  official +x normal to `+z`, even though the comment and intended init were
  `-z`. This made the new normal loss penalize step 0.
- Added focused tests for the radius-scaled texel sites, SV detach behavior,
  camera-facing quaternion sign, and render-stat shapes.

## Verification

Commands:

```bash
rtk .venv/bin/python -m py_compile src/train/powerfoam_direct.py src/train/train_powerfoam_direct.py
rtk uv run --with pytest python -m pytest tests/test_powerfoam_direct.py tests/test_sequence_data_single_frame.py -q
rtk env PYTHONPATH=src/train WANDB_MODE=offline uv run python src/train/train.py src/train_configs/local_mac_powerfoam_direct_full_tiny_smoke.jsonc
rtk env PYTHONPATH=src/train WANDB_MODE=offline uv run python src/train/train.py src/train_configs/local_mac_powerfoam_direct_video_full_tiny_smoke.jsonc
rtk git diff --check
```

Results:

- Tests: `6 passed`.
- Single-frame 64px full tiny smoke:
  - step 0: `eval_l1 = 0.0482663`, `eval_mse = 0.0052037`
  - step 2: `eval_l1 = 0.0392696`, `eval_mse = 0.0039687`
  - train `normal_loss = 0.0` after the quaternion sign fix.
- 16-frame 64px video full tiny smoke:
  - step 0: `eval_l1 = 0.0502943`, `eval_mse = 0.0058220`
  - step 20: `eval_l1 = 0.0461599`, `eval_mse = 0.0047499`

## Still missing

- Official Cech/AABB adjacency construction.
- Official LR schedules.
- Contribution/error EMA-driven densification, pruning, and resampling.
- Static SfM/multiview scene trainer; the current path is still a per-frame
  Dynaworld video adapter.
- Tiled Metal rasterizer with compact primitive lists and replay backward.
- Tiny-scene parity tests against the official CUDA/Warp implementation.
