# Train Device Helper

## Context

The PowerFoam-family trainers and STAR UVT runtime each had a local device
resolver. The policies looked similar but were not identical:

- PowerFoam Direct, PowerFoam Metal, and Dynamic PowerFoam Metal used
  `auto -> mps if available else cpu`.
- Dynamic Gauge Foam used `auto -> mps if available else cuda if available else
  cpu`.
- STAR UVT used the Dynamic Gauge-style auto fallback and additionally raised
  when an explicit unavailable CUDA/MPS device was requested.

## Change

- Added `src/train/train_devices.py`.
- Added `resolve_torch_device(value, auto_cuda=False, validate_requested=False)`.
- Added `sync_torch_device(device)`.
- Updated PowerFoam Direct, Dynamic Gauge Foam, PowerFoam Metal, Dynamic
  PowerFoam Metal, and `star_uvt_runtime.py` to delegate to the shared helper.
- Follow-up: routed Token-GS profile timing synchronization through
  `sync_torch_device(...)` as well, and made the helper preserve the previous
  `hasattr(torch, "mps")` guard.
- Preserved caller-specific behavior with explicit flags instead of changing the
  `auto` semantics globally.
- Added `tests/test_train_devices.py` for MPS preference, CPU fallback when
  CUDA auto is disabled, CUDA auto fallback, and explicit CUDA validation.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_devices.py \
  src/train/star_uvt_runtime.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_train_devices.py \
  tests/test_star_uvt_runtime.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_devices.py tests/test_star_uvt_runtime.py tests/test_train_logging.py -q
```

Result: `17 passed in 2.67s`.

Follow-up validation after routing Token-GS profile timing:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_devices.py \
  src/train/train_video_token_implicit_dynamic.py \
  tests/test_train_devices.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_devices.py tests/test_temporal_sampling.py tests/test_train_artifacts.py -q
```

Result: `18 passed in 5.96s`.

## Remaining

There are still two standalone colorizer probe CLIs with local `_resolve_device`
helpers. I left them alone because they are not trainer entrypoints and have a
separate argparse/debug-tool contract. If they become part of the canonical
train surface, route them through `train_devices.py` then.
