# Anisotropic q-UVT Projective Compiler

## Goal Memory

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Context

The prior Metal bridge made `ProjectiveTraceCellTraceAtlas.spatial_precision_uv`
real in CPU/reference rendering, quadrature reference rendering, the support
tail certificate, and interval Metal forward/backward. The remaining mismatch
was q-UVT production: source `q_uvt` already has an anisotropic UV precision
block, but the compatibility compiler and backend still rejected anything that
was not the scalar `sigma_px^{-2} I` special case.

## Math

For q-UVT spatial precision:

```text
P = [[q_uu, q_uv],
     [q_uv, q_vv]]
```

the alpha support above a positive threshold satisfies:

```text
alpha_peak exp(-0.5 d^T P d) >= alpha_threshold
d^T P d <= R^2
R^2 = 2 log(alpha_peak / alpha_threshold)
```

The existing tile support index accepts one scalar square padding, so the safe
ellipse-to-rectangle bound is:

```text
max |du| <= sqrt(R^2 (P^-1)_00)
max |dv| <= sqrt(R^2 (P^-1)_11)
support_uv_padding = max(user_padding, max_i(max_u_i, max_v_i))
```

This is the concrete "screen fiber metric" move: keep the gauge-local
quadratic form instead of falling back to an isotropic approximation.

## Code Changes

- `uvt_tubes_to_projective_trace_cell_atlas(...)` now has:
  - `require_isotropic_spatial=True` by default, preserving old behavior.
  - `auto_support_padding_from_alpha=False` by default.
  - opt-in anisotropic lowering with support padding from the alpha ellipse.
- `ProjectiveCellIntervalBackendConfig` now has:
  - `allow_anisotropic_spatial_precision=False` by default.
  - when true, the backend passes `require_isotropic_spatial=False` and computes
    the same minimum alpha support padding before building the atlas, so trainer
    state reports the padding it actually needs.
- Live atlas updates now allow anisotropic precision under the same config
  flag and still reject by default.

Files touched:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
src/train/star_uvt_projective_interval_backend.py
tests/test_star_uvt_projective_uvt_producer.py
EXPERIMENTS.md
research_notes/gauged_uvt_trace_atlas/README.md
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
```

## Verification

Focused producer/backend test:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_uvt_producer.py -q
28 passed in 8.30s
```

Broad projective suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_visibility_support_bridge.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_binning.py -q
143 passed in 16.99s
```

Compile check:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/star_uvt_projective_interval_backend.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py \
  tests/test_star_uvt_projective_uvt_producer.py
```

## Boundary

This does not yet mean the learned source-view trainer naturally learns
anisotropic q-UVT precision. That path still calls the lock/init helper that
forces isotropic projective precision. What is now true is narrower and useful:
if q-UVT tensors already contain an SPD anisotropic UV precision block, the
projective compiler/backend can opt in, build support with the correct ellipse
bound, render through interval Metal, and reuse live atlas cells without
silently projecting the footprint back to scalar `sigma_px`.
