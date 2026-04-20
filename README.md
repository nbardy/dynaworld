# dynaworld

Small local TokenGS and dynamic Gaussian-splat experiments.

## Setup

```bash
uv sync
git submodule update --init --recursive
```

## Camera Prebake

From the repo root:

```bash
./get_camera.sh
```

Default inputs and outputs live under `test_data/`.

## Train

Single-image baseline:

```bash
uv run python train_scripts/tokenGS.py
```

Dynamic training with DUSt3R camera prebake:

```bash
./train_full_dynamic_with_camera_prebake_all_frames.sh
```
