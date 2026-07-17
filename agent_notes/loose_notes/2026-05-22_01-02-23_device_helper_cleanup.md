# Device Helper Cleanup

## Context

Continuing the trainer-interface modularization pass, a fresh scan found one
live mismatch between docs and code: DUSt3R export already imported
`resolve_torch_device(...)`, but still wrapped it in a local `pick_device(...)`.
The same scan found selected benchmark/probe scripts still calling
`torch.mps.synchronize()` directly.

## What changed

- `train_devices.sync_torch_device(...)` now checks `torch.backends.mps.is_available()`
  before synchronizing an MPS device. This makes the shared helper safe for
  benchmark scripts that keep an MPS-oriented sync function but may be imported
  on non-MPS machines.
- `run_dust3r_video.py` now calls `resolve_torch_device(args.device,
  auto_cuda=True)` directly and no longer defines a local `pick_device(...)`
  wrapper.
- `src/benchmarks/depth_aware_dof_demo.py` now uses
  `resolve_torch_device("auto", auto_cuda=False)` for its MPS-then-CPU policy
  and `sync_torch_device(...)` for timing fences.
- `src/benchmarks/mac_renderer_stack_compare.py` and
  `src/benchmarks/fast_mac_project3d_benchmark.py` now route local MPS timing
  fences through `sync_torch_device(...)` instead of calling
  `torch.mps.synchronize()` directly.

I did not chase every direct MPS synchronization under `research_experiments/`.
Most remaining hits are deep WorldFoam kernel probes or tests; those should be
left alone unless they become reusable trainer/benchmark entrypoints.

## Validation

- `py_compile` passed for the patched device helper, test, DUSt3R export, and
  benchmark scripts.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest
  tests/test_train_devices.py -q` passed: `7 passed`.
- `src/benchmarks/depth_aware_dof_demo.py --help` passed.
- Targeted scan found no local `pick_device(...)`, `pick_metal_device(...)`,
  direct `torch.mps.synchronize(...)`, or direct `torch.cuda.synchronize(...)`
  calls in the patched files.
- `git diff --check` passed.
- `run_dust3r_video.py --help` was attempted but blocked by the local
  environment missing `matplotlib`; the file still passes `py_compile`.
