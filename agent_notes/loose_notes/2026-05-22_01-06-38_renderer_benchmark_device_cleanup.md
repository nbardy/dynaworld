# Renderer Benchmark Device Cleanup

## Context

Continuing the modularization goal, a fresh helper scan found reusable renderer
benchmark scripts still carrying local device and synchronization helpers. These
were not deep WorldFoam probes; they are generic benchmark CLIs under
`src/benchmarks`, so they should share the same device primitive as the trainer
profiling scripts.

## What changed

- `train_devices.resolve_torch_device(...)` now has an explicit
  `auto_prefer_cuda` option.
- `splat_renderer_benchmark.py` and `splat_renderer_accuracy.py` now call
  `resolve_torch_device(..., auto_cuda=True, auto_prefer_cuda=True)` directly.
  This preserves their historical CUDA-before-MPS `auto` behavior while moving
  the policy into the shared helper.
- Those two renderer benchmark scripts now use `sync_torch_device(...)` for
  local timing fences instead of local CUDA/MPS sync branches.
- `trainer_phase_benchmark.py` no longer defines a one-line local
  `sync_device(...)` wrapper; it calls `sync_torch_device(...)` directly.

## Validation

- `py_compile` passed for `train_devices.py`, `tests/test_train_devices.py`,
  `splat_renderer_benchmark.py`, `splat_renderer_accuracy.py`, and
  `trainer_phase_benchmark.py`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest
  tests/test_train_devices.py -q` passed: `8 passed`.
- `splat_renderer_benchmark.py --help`, `splat_renderer_accuracy.py --help`,
  and `trainer_phase_benchmark.py --help` passed.
- Targeted scan found no local `pick_device(...)`, local sync helper, or direct
  `torch.mps.synchronize(...)` / `torch.cuda.synchronize(...)` calls in those
  three benchmark scripts.
- `git diff --check` passed.
- Targeted trailing-whitespace scan passed.

## Note

This intentionally leaves `fast_attn.pick_device(...)`, STAR UVT runtime
wrappers, and deep WorldFoam timing probes alone. They have different caller
contracts or are one-off low-level experiments; forcing them through the helper
would add churn without proving the active trainer interface cleaner.
