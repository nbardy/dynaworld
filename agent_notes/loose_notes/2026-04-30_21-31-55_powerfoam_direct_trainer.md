# PowerFoam Direct Trainer

## Context

The user asked for the leanest trainer, architecture, and config that initializes
random PowerFoam and fits it directly to the local 128x128 small test video.

## Added

- `src/train/powerfoam_direct.py`
  - `DirectPowerFoamVideo`
  - vectorized Torch PowerFoam renderer
  - fixed KNN adjacency from random initialization
  - independent per-frame RGB foam cells
- `src/train/train_powerfoam_direct.py`
  - loads `test_data/test_video_small_128_4fps.mp4`
  - trains with L1/MSE plus tiny radius regularization
  - writes preview PNGs, render MP4s, side-by-side MP4s, and final checkpoint
- `src/train_configs/local_mac_powerfoam_direct_128_smoke.jsonc`
- `tests/test_powerfoam_direct.py`
- `arch=powerfoam_direct` dispatch in `src/train/train.py`

## Why CPU

The first 128px/96-cell MPS smoke took about 23 seconds for one step because the
temporary Torch renderer launched many small ops inside Python loops. After
vectorizing neighbor clipping and reducing the smoke config to 32 cells / 8
neighbors, CPU ran one 128px training step in about 0.57 seconds. The config
therefore defaults to CPU until the custom Metal backward exists.

## Run

```bash
PYTHONPATH=src/train uv run python src/train/train.py src/train_configs/local_mac_powerfoam_direct_128_smoke.jsonc
```

Observed 2026-04-30:

```text
frames=16 render_size=128 cells=32 neighbors=8 steps=100 device=cpu
step 100 sampled-frame L1=0.097700 MSE=0.016155
step 100 full-video eval L1=0.130358 MSE=0.032068
```

Outputs:

```text
outputs/powerfoam_direct/local_mac_powerfoam_direct_128_smoke/
```

## Caveat

This is a scaffold for representation and loss plumbing, not the final renderer
performance path. PyTorch autograd still stores more loop intermediates than the
planned custom Metal reverse replay will. The Metal kernel still needs the
streaming or tiled custom backward before this can become the real trainer
backend.
