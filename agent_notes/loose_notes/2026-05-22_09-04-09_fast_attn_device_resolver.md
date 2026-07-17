# Fast-attention device resolver cleanup

## Context

`src/train/fast_attn.py` owns the active Token-GS attention backend helpers.
Its `pick_device()` still had a local CUDA-first `auto` device branch even
though `train_devices.resolve_torch_device(...)` now owns that policy.

## Change

- `fast_attn.pick_device()` now delegates to
  `resolve_torch_device("auto", auto_cuda=True, auto_prefer_cuda=True)`.
- The base Token-GS trainer and browser export keep the legacy CUDA-first
  behavior through the same `pick_device()` API.
- `tests/test_train_devices.py` now asserts `pick_device()` follows the shared
  CUDA-first auto policy.

## Validation

- `py_compile` covered `fast_attn.py`, `tests/test_train_devices.py`, the base
  Token-GS trainer, and browser export.
- `tests/test_train_devices.py -q` passed.

## Follow-up

Keep `configure_fast_attn(...)` and `fast_attn_context(...)` in `fast_attn.py`.
Those are attention-backend policy, not generic device selection.
