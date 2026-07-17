# Gate 0.9 MPS Composite VJP Status

Date: 2026-05-12 01:49 +0700

Scope: first fixed-segment backward proof for the World Foam Lane 2 toy
compositor. This remains an isolated research artifact, not training
readiness.

## Commands Run

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_composite_vjp_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_9_mps_composite_vjp_smoke.json
```

## Result

Saved artifact:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_9_mps_composite_vjp_smoke.json
```

The VJP smoke reports:

- `status=ok`
- `max_rgb_abs_error=1.1920928955078125e-07`
- `max_alpha_abs_error=1.7881393432617188e-07`
- `max_depth_abs_error=3.5762786865234375e-07`
- `max_rgba_gradient_abs_error=1.9073486328125e-06`
- `finite_difference_max_abs_error=0.0003147125244140625`
- `loss_abs_error=3.4033663922627966e-06`
- `mps_composite_vjp_wall_clock_ms=0.7583416499983286`

## Boundary

This is useful because the toy compositor now has a real MPS backward smoke for
site RGBA values through the shared segment tape.

This is not geometry or topology differentiation. Boundary cuts, owners,
sorting, site positions, site weights, camera projection, real-video image
formation, and trainer integration remain open.
