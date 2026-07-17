# Gauged UVT span-gated Metal bridge for interval sidecars

## Context

The previous gate added explicit `[active_start, active_stop)` interval
sidecars to lowered q-UVT chart segments. The CPU oracle proved the gates are
necessary: split affine segments match dense rendering when gated, and leak
when ungated.

The next question was whether those interval sidecars could reach Metal before
writing a native shader-side gate buffer.

## Work Changed

Added:

```text
projective_trace_uvt_bridge_active_spans(...)
render_projective_trace_uvt_bridge_metal_gated(...)
```

Location:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

The span helper partitions the frame axis at every interval boundary. The Metal
wrapper then:

```text
for each [start, stop) span:
    select tubes active for the entire span
    call existing render_uvt_tubes(...) on that active set
    copy rendered[start:stop] into the final image
```

This preserves the exact interval sidecar semantics without modifying the Metal
shader. It is not yet the final native gate-buffer implementation, but it is a
real Metal-rendered bridge and gives a clean acceptance target.

## New Test

Extended:

```text
tests/test_star_uvt_projective_correctness.py
```

with:

```text
test_projective_split_q_uvt_bridge_interval_gates_reach_metal_if_available
```

The test skips if MPS or `star_uvt_v0.render` is unavailable. On this machine it
ran and passed. It compares the span-gated Metal wrapper against the gated CPU
q-UVT bridge reference on the curved split-chart scene.

## Evidence

Targeted test:

```text
1 passed in 1.59s
```

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
33 passed in 0.97s
```

## Current Model

The implementation ladder is now:

```text
CPU atlas reference
-> affine projective chart to q-UVT bridge
-> existing q-UVT Metal renderer for single affine charts
-> explicit interval sidecar for split curved traces
-> span-gated Metal wrapper for interval sidecars
```

The remaining hot-path gap is performance, not semantics. The current wrapper
uses one Metal render call per active-set span. If chart count is small relative
to frame count, this can still be sublinear in frames for world-side work. A
native shader-side interval gate buffer would avoid repeated full-frame launches
and become the cleaner production path.

## Next Gate

Either:

1. add native shader-side interval gate buffers for q-UVT tubes, or
2. add the first nonlinear/projective atlas-cell Metal evaluator.
