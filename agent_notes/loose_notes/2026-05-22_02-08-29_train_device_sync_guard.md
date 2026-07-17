# Train Device Sync Guard Cleanup

## Context

`train_devices.sync_torch_device(...)` is the shared synchronization boundary
used by trainer timing, STAR UVT runtime wrappers, renderer benchmarks, and
probe scripts. It already skipped MPS synchronization when MPS was unavailable,
but CUDA synchronization was unconditional for CUDA devices.

That asymmetry means caller code could still need CUDA availability guards even
when using the shared helper.

## Changes

- Updated `sync_torch_device(...)` to call `torch.cuda.synchronize(device)` only
  when `torch.cuda.is_available()`.
- Updated the CUDA sync test to make CUDA availability explicit.
- Added a focused test that unavailable CUDA is skipped without calling
  `torch.cuda.synchronize(...)`.

## Validation

- `rtk env PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_train_devices.py tests/test_star_uvt_runtime.py -q`
  passed: `12 passed`.
- `rtk uv run python -m py_compile src/train/train_devices.py src/train/star_uvt_runtime.py tests/test_train_devices.py tests/test_star_uvt_runtime.py`
  passed.

## Notes

This is a shared-helper hardening slice, not a trainer math change. It keeps the
same behavior on available CUDA devices and makes unavailable CUDA symmetric
with unavailable MPS.
