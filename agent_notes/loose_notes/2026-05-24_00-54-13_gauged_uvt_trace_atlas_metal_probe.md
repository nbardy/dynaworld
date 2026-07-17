# 2026-05-24 Gauged UVT Trace Atlas Metal Probe

## Context

User pushed back on "residual checks plus fallback" as too weak for revolving
cameras and asked whether fiber bundles / a UVT screen fiber with respect to a
camera gauge is the right formal object. We formalized UVT as a camera-ray
bundle:

```text
pi: E_Gamma -> B,   B = Omega x T
Gamma: E_Gamma -> M
UVT trace = pi_* Gamma^* world_primitive
```

Then user asked to make a new theory/plans folder with up to 10 subfolders for
theories and to try implementing some of it in Metal.

## What Changed

Created `research_notes/gauged_uvt_trace_atlas/` with ten subtheory folders:

- `00_bundle_foundations`
- `01_camera_gauge_choices`
- `02_gaussian_fiber_pushforward`
- `03_projective_rational_traces`
- `04_revolving_camera_atlas`
- `05_visibility_strata`
- `06_exposure_and_rolling`
- `07_adjoint_training`
- `08_worldfoam_bridge`
- `09_metal_acceptance_plan`

Also linked it from `research_notes/README.md`.

Implemented the first narrow Metal probe in STAR UVT:

```text
projective_trace_eval(coeffs, times, eps) -> [N, S, 4]
```

where `coeffs` is `[N,9]` quadratic homogeneous camera-time coefficients:

```text
h_u(t) = u0 + u1 t + u2 t^2
h_v(t) = v0 + v1 t + v2 t^2
h_z(t) = z0 + z1 t + z2 t^2
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
```

Output channel layout:

```text
[u, v, h_z, valid_sign]
```

with `valid_sign = 1/-1/0` for positive denominator / negative denominator /
near-zero invalid denominator.

Files touched:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/bindings.cpp`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/shared/common.h`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py`
- `tests/test_star_uvt_projective_trace.py`

## Tests

Focused test:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_trace.py -q
```

Result:

```text
4 passed in 4.63s
```

This includes the MPS/Metal parity path after rebuilding the extension.

Built STAR UVT extension with:

```bash
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Renderer regression smoke:

```bash
PYTHONPATH=src/train uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/uvt_pair_benchmark.py \
  --scenes single_static
```

Result:

```text
max_rgb_error = 5.960464477539063e-08
mean_rgb_error = 1.1123878485008731e-09
overflow_tile_count = 0
unstable_tile_fraction = 0.0
```

## Interpretation

This is not a full revolving-camera renderer yet. It is Gate A from the new
Metal acceptance plan: prove that homogeneous/projective camera-time traces can
be represented and evaluated in Metal with Torch parity. That is the first
usable kernel primitive for the richer gauged UVT atlas.

The next meaningful implementation gates are:

1. Fit affine/quadratic UVT local charts from projective trace samples and
   record residuals as chart validity certificates.
2. Add orbit-window splitting driven by denominator safety and rational center
   approximation error.
3. Add rational support bounds so binning can use projective traces without
   enumerating every frame.
4. Extend visibility metadata with depth uncertainty and denominator min/max.
5. Only then integrate rational traces into the renderer hot path.

## Important Caveat

The projective trace kernel evaluates center/depth trajectories, not footprint
covariance or visibility. For full orbit support, this must be paired with
Schur-complement/sigma-point support fitting and visibility strata. The value of
this pass is that the denominator/rational gauge now exists as tested GPU code
instead of only as notes.
