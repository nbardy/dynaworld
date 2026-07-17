# Gauged UVT interval-gated split q-UVT segments

## Context

The affine projective chart to q-UVT bridge reached the existing STAR UVT Metal
renderer. The next risk was segment leakage: if a curved/revolving trace is
split into several affine charts, each lowered q-UVT tube mathematically extends
for all sensor times unless the renderer also knows its chart domain.

Soft temporal precision is not the right solution. It suppresses samples outside
the chart, but also attenuates valid samples near chart endpoints. The correct
object is an explicit chart-domain sidecar.

## Work Changed

Extended:

```text
ProjectiveTraceUVTBridge
```

with:

```text
active_start: tuple[int, ...]
active_stop: tuple[int, ...]
```

Each lowered tube segment now carries its exact sample-domain interval:

```text
active_start <= sample_index < active_stop
```

Added CPU oracle:

```text
render_projective_trace_uvt_bridge_reference(..., use_window_gates=True)
```

This evaluates the q-UVT bridge while applying the interval gate before sorting
and compositing.

## New Test

Extended:

```text
tests/test_star_uvt_projective_correctness.py
```

with:

```text
test_projective_split_q_uvt_bridge_window_gates_prevent_segment_leakage
```

The test uses a curved projective trace, splits it into local degree-1 windows,
lowers the windows into q-UVT tubes, and compares:

```text
gated q-UVT bridge render == dense per-frame projective render
ungated q-UVT bridge render visibly differs
```

This proves the sidecar is semantically necessary for split orbit charts.

## Evidence

Focused suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py -q
```

Result:

```text
32 passed in 0.74s
```

## Current Model

The q-UVT bridge now has three levels:

```text
single affine projective chart -> existing q-UVT renderer contract
affine chart on MPS -> existing Metal render parity
split curved trace -> multiple q-UVT segments + explicit interval sidecar
```

The CPU oracle enforces the interval sidecar. The current Metal renderer does
not yet consume it.

## Next Gate

Either:

1. add Metal support for the q-UVT interval sidecar, so split affine chart
   segments can render without leakage on the hot path, or
2. add a direct nonlinear/projective atlas-cell Metal evaluator.
