# Gauge common device resolver cleanup

## Context

Gauge Fields intentionally stays under `research_experiments/gauge_fields/`
instead of being folded into the main trainer registry. Its `common.py` already
owns Gauge-local path setup and artifact helpers, but it still copied the same
`auto -> cuda/mps/cpu` device-selection branch that `train_devices.py` now
owns.

## Change

- `research_experiments/gauge_fields/common.py` imports
  `train_devices.resolve_torch_device`.
- `common.resolve_device(...)` now delegates to
  `resolve_torch_device(name, auto_cuda=True, auto_prefer_cuda=True)`.

## Preserved behavior

Gauge keeps its CUDA-first `auto` policy:

1. CUDA when available.
2. MPS when CUDA is unavailable.
3. CPU fallback.

Explicit device strings still pass through without availability validation,
matching the previous `torch.device(name)` behavior.

## Validation

- `py_compile` covered `common.py` plus Gauge train/probe entrypoints that
  import `resolve_device`.
- A focused Python import check compared `common.resolve_device("auto")` to
  the shared helper with the same CUDA-first policy and checked explicit
  `cpu` resolution.
