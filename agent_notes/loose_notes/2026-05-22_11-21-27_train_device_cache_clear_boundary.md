# Train Device Cache Clear Boundary

## Context

Continuation of the modular trainer/code cleanup goal. A live scan found that
device resolution and synchronization had mostly been routed through
`train_devices`, but accelerator cache clearing still had two copies:

- `video_feature_cache._clear_accelerator_cache(...)`
- `src/benchmarks/benchmark_memory.clear_device_cache(...)`

Both performed Python GC plus backend-specific MPS/CUDA cache clearing. The
benchmark helper additionally synchronized the device afterward.

## Changes

- Added `train_devices.clear_torch_device_cache(device, sync=False)`.
- Routed `video_feature_cache._clear_accelerator_cache(...)` through the new
  helper without synchronization, preserving its previous call shape.
- Kept `benchmark_memory.clear_device_cache(...)` as the public benchmark API,
  but changed it to delegate to `clear_torch_device_cache(..., sync=True)`.
- Added focused `tests/test_train_devices.py` coverage for CUDA clear+sync and
  CPU skip behavior.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  src/train/train_devices.py \
  src/train/video_feature_cache.py \
  src/benchmarks/benchmark_memory.py \
  tests/test_train_devices.py \
  tests/test_benchmark_memory.py
rtk env PYTHONPATH=src/train:src/benchmarks uv run --with pytest python -m pytest \
  tests/test_train_devices.py tests/test_benchmark_memory.py -q
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_video_feature_cache.py -q
rtk git diff --check -- \
  src/train/train_devices.py \
  src/train/video_feature_cache.py \
  src/benchmarks/benchmark_memory.py \
  tests/test_train_devices.py
```

Results: train-device plus benchmark-memory tests passed (`14 passed`), and
video-feature-cache tests passed (`2 passed`). `py_compile`, `git diff
--check`, and the trailing-whitespace scan were clean for the touched files.
