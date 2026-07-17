# Gate 0.7 MPS RGB Strip Status

Date: 2026-05-12 01:23 +0700

Scope: first image-shaped MPS smoke for the World Foam Lane 2 scaffold. This is
still an isolated research artifact, not a renderer promotion.

## Commands Run

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_rgb_strip_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip_smoke.json \
  --ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip.ppm
python3 dynaworld/src/benchmarks/world_foam_gate0_paired_benchmark.py \
  --star-comparison-json \
  dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle/comparison_report.json \
  --out-json \
  dynaworld/research_experiments/world_foam_lane2/results/gate0_7_paired_with_star_dynamic_heldout_pilot.json
```

## Result

Saved artifacts:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip_smoke.json
dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip.ppm
dynaworld/research_experiments/world_foam_lane2/results/gate0_7_paired_with_star_dynamic_heldout_pilot.json
```

The RGB strip smoke reports:

- `status=ok`
- `strip_image_shape=[16,17,3]`
- `max_rgb_abs_error=4.76837158203125e-07`
- `color_gradient_max_abs_error=4.57763671875e-05`
- `loss_abs_error=1.0043974384643661e-05`
- `mps_rgb_strip_wall_clock_ms=1.0758187501778593`
- `shared_forward_backward_boundary_scan_ratio=0.03125`

The paired heldout-pilot artifact now has a
`world_foam_mps_rgb_strip_smoke` row beside the Gate 0 event rows, Gate 0.6
shared-replay row, STAR-UVT heldout-pilot row, and dynamic-splat baseline row.

## Boundary

This is useful because World Foam now produces finite, nonconstant RGB-shaped
MPS output and matching site-RGB signal gradients on the shared-replay toy.

This is not yet enough for renderer comparison. The strip uses one shared RGB
replay kernel, but there is still no alpha/depth compositor, geometry gradient,
trainer hook, or heldout-camera metric.
