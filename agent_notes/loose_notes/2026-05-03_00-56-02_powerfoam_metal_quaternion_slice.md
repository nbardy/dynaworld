# PowerFoam Metal Quaternion Slice

User asked to complete proper PowerFoam on Metal: accurate forward/backward,
fast, trainable, and eventually 4K. That is larger than the existing partial
Metal core, so this session took the next concrete implementation slice instead
of writing more planning prose.

Implemented:

```text
third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py
third_party/powerfoam-metal/torch_powerfoam_metal/__init__.py
third_party/powerfoam-metal/tests/linear_texture_check.py
src/train/train_powerfoam_metal.py
src/train_configs/local_mac_powerfoam_metal_quaternion_texel_surface_video_1024_smoke.jsonc
tests/test_powerfoam_direct.py
TODO/powerfoam_full_reproduction_todo.md
```

What changed:

- Added `quaternion_frames(...)` to the Metal Python package.
- Added `rasterize_power_foam_quaternion_texel_surface(...)`, which derives
  official PowerFoam normal/tangent/bitangent frames from quaternions and then
  calls the existing Metal oriented texel-surface op.
- Added parity coverage proving Metal forward/backward agrees with the Torch
  reference and that gradients flow back into quaternions.
- Added trainable `feature_mode="quaternion_texel_surface"` in
  `train_powerfoam_metal.py`, with optimizer group and state-drift logging for
  quaternions.
- Added a checked-in 1024-cell local smoke config for the new mode.

Validation:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile \
  src/train/train_powerfoam_metal.py \
  third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py \
  third_party/powerfoam-metal/tests/linear_texture_check.py

PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q

PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/linear_texture_check.py

PYTHONPATH=src/train WANDB_MODE=disabled .venv/bin/python \
  src/train/train_powerfoam_metal.py /tmp/powerfoam_quaternion_texel_1step_smoke.jsonc

PYTHONPATH=src/train .venv/bin/python \
  third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py \
  --cells 1024,4096 \
  --resolutions 4096x4096 \
  --feature-dim 3 \
  --neighbors 32 \
  --warmup 1 \
  --iters 2 \
  --foam-backward \
  --foam-texel-surface \
  --json
```

Observed checks:

- `tests/test_powerfoam_direct.py`: 8 passed.
- Quaternion texel surface parity max errors:
  - features: `6.482e-07`
  - alpha: `1.431e-06`
  - points grad: `9.313e-08`
  - radii grad: `2.161e-07`
  - density grad: `5.215e-08`
  - texel sites grad: `3.653e-09`
  - texel features grad: `1.630e-08`
  - quaternions grad: `5.635e-08`
- 1-step 1024-cell train smoke completed on MPS:
  - step-0 eval L1: `0.033682`
  - step-1 eval L1: `0.033211`
  - quaternion delta after one step: `0.001247`
- Low-level 4K current-path benchmark saved at:
  `outputs/benchmarks/powerfoam_metal_texel_surface_4k_1024_4096_2026-05-03.json`
  - `1024` cells at `4096x4096`: `1784.8 ms` forward,
    `3331.4 ms` backward, `5116.2 ms` total.
  - `4096` cells at `4096x4096`: `6543.4 ms` forward,
    `8681.7 ms` backward, `15225.1 ms` total.
  - This proves the current streaming path is not the final fast 4K answer;
    the tiled candidate-list path is still required.

Still not full PowerFoam:

- no Metal texel-height displacement yet
- no spherical-Voronoi color path in the Metal/static trainer yet
- no Cech/AABB adjacency builder
- no static COLMAP/SfM multiview trainer
- no densification/pruning/resampling
- no ray-tracing backend
- no 4K PowerFoam forward/backward/trainable benchmark matrix
