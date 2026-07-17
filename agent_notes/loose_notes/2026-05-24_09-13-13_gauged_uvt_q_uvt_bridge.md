# Gauged UVT affine chart to q-UVT bridge

## Context

The projective compiler had a CPU atlas reference renderer, but no route back
into the existing STAR UVT renderer contract. The next clean bridge was not a
large Metal rewrite. Accepted degree-1 projective chart windows are exactly
affine center traces, so they can be represented by existing `ma/q_uvt` tubes.

## Work Changed

Added:

```text
ProjectiveTraceUVTBridge
projective_trace_windows_to_uvt_tubes(...)
```

Location:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

The helper lowers each accepted degree-1 projective chart window into:

```text
ma: [u_center(t_c), v_center(t_c), t_c]
q_uvt: screen-space Gaussian precision with velocity cross terms
depth0: depth at chart center time
depth_beta: affine depth slope in sensor time
opacity/color: repeated from source primitive attributes
```

For a chart fit

```text
u(t) = u0 + u1 * ((t - c) / scale)
v(t) = v0 + v1 * ((t - c) / scale)
z(t) = z0 + z1 * ((t - c) / scale)
```

the bridge uses:

```text
ma = [u0, v0, c]
velocity_uv = [u1, v1] / scale
depth_beta_t = z1 / scale
lambda_u = lambda_v = 1 / sigma_px^2
q_ut = -lambda_u * velocity_u
q_vt = -lambda_v * velocity_v
q_tt = temporal_precision + lambda_u * velocity_u^2 + lambda_v * velocity_v^2
```

This exactly represents a Gaussian footprint centered on the affine projective
chart in the existing STAR UVT quadratic form.

## New Test

Extended:

```text
tests/test_star_uvt_projective_correctness.py
```

with:

```text
test_projective_affine_charts_lower_to_existing_q_uvt_renderer_contract
```

The test compiles two stable-depth affine projective traces, lowers them to the
existing q-UVT contract, renders with `brute_force_render_uvt_tubes(...)`, and
matches `render_projective_trace_tile_time_atlas_reference(...)`.

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
30 passed in 0.67s
```

## Current Model

This bridge is the correct fast path for affine/gauged local projective charts.
It does not solve nonlinear projective charts directly. Degree-2 or rational
charts need either subdivision into degree-1 windows, atlas-cell evaluation, or
a richer Metal kernel that evaluates projective centers directly.

## Next Gate At This Point

Run or add a guarded MPS parity smoke: lower affine projective charts into
q-UVT tensors, render through `render_uvt_tubes(...)` on MPS when available, and
compare against the CPU atlas reference / CPU brute-force q-UVT render.

## Follow-Up In Same Heartbeat

Added that guarded MPS parity smoke:

```text
test_projective_affine_q_uvt_bridge_matches_metal_renderer_if_available
```

It ran on this machine and passed. The focused suite is now:

```text
31 passed in 0.69s
```

The next gate moves beyond affine chart lowering: either nonlinear/projective
atlas-cell Metal evaluation, or explicit interval gating for split affine q-UVT
segments.
