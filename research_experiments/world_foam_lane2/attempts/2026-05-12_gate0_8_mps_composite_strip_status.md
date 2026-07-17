# Gate 0.8 MPS Composite Strip Status

Date: 2026-05-12 01:39 +0700

Scope: first forward-only alpha/depth compositor proof for the World Foam Lane
2 scaffold. This remains an isolated research artifact, not a renderer
promotion.

## Commands Run

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_composite_strip_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip_smoke.json \
  --ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip.ppm
```

## Result

Saved artifacts:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip_smoke.json
dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip.ppm
```

The composite strip smoke reports:

- `status=ok`
- `rgb_shape=[16,17,3]`
- `alpha_shape=[16,17]`
- `depth_shape=[16,17]`
- `max_rgb_abs_error=1.7881393432617188e-07`
- `max_alpha_abs_error=1.7881393432617188e-07`
- `max_depth_abs_error=3.5762786865234375e-07`
- `mps_composite_wall_clock_ms=1.0734333001892082`
- `shared_forward_boundary_scan_ratio=0.0625`

## Boundary

This is useful because World Foam now has a real MPS forward compositor smoke
for RGB, accumulated alpha, and expected depth, all using the shared
screen-time candidate tape.

This is not enough for renderer comparison. There is still no density-gradient
VJP, geometry gradient, trainer hook, real-video camera/image formation, or
heldout-camera metric.
