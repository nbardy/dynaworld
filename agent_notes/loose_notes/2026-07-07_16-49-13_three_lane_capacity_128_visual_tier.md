# Three-Lane Capacity 128px Visual Tier

Date: 2026-07-07 16:49 KST.

## Context

The active goal is to get WorldFoam, WorldTubes, and base dynamic 3DGS running
through Metal-backed trainers/renderers on test data and inspect the visual
outputs. The previous 128px medium tier was runnable but under-capacity.

This pass added a capacity tier to the shared comparison harness.

## Implemented

Runner tier:

```text
research_experiments/world_foam_lane2/run_visual_compare_three_lanes.py
--tier capacity
```

New configs:

```text
src/train_configs/visual_compare_worldfoam_dynamic_powerfoam_metal_128_16f_80step_2048cells.jsonc
src/train_configs/visual_compare_star_uvt_worldtubes_metal_128_16f_60step_2048tubes.jsonc
src/train_configs/visual_compare_dynamic_gsplat_fast_mac_metal_128_16f_60step_4096gs.jsonc
```

Focused tests now cover the capacity tier:

```text
research_experiments/world_foam_lane2/test_run_visual_compare_three_lanes.py
```

## Verification

Focused tests:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_run_visual_compare_three_lanes.py \
  tests/test_token_gs_eval_media.py \
  -q
```

Result: `11 passed`.

Compile check:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/world_foam_lane2/run_visual_compare_three_lanes.py \
  src/train/token_gs_trainer.py
```

Capacity dry run:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_capacity_128_dry_run.json
```

First capacity all-lane run:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_capacity_128_all_lanes.json
```

Result: WorldFoam and STAR passed; dynamic 3DGS failed because
`max_fast_pairs=4096` exceeded the compiled fast-mac `v6_refined` cap of 2048.

Fix: kept 4096 explicit Gaussians but set runtime `max_fast_pairs` back to
2048 in the dynamic capacity config.

Dynamic rerun:

```text
outputs/visual_comparisons/2026-07-07_dynamic_gsplat_capacity_128_rerun.json
```

Result: dynamic 3DGS passed with all declared artifacts.

Clean all-lane summary:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_capacity_128_clean_all_lanes.json
```

Result: overall `status=ok`; all three lanes exited 0 and all declared
artifacts existed.

## Metrics

```text
WorldFoam eval PSNR mean: 17.7454, eval L1: 0.0801
STAR UVT final PSNR:      21.7685, final L1: 0.0546, render median: 12.266ms
Dynamic 3DGS final line:  Loss 0.1141, recon 0.1112
```

Runner elapsed times from the clean run:

```text
WorldFoam / dynamic PowerFoam: 32.81s
WorldTubes / STAR UVT:         17.08s
Base dynamic 3DGS / fast-mac:  89.57s
```

## Visual Read

WorldFoam stays nonblank and scene-aligned but does not meaningfully recover
dog structure; the color-only fixed-geometry setup still smears the dog.

STAR UVT is the best 128px visual in this tier: still blurred, but dog-shaped
and temporally coherent. It benefits from 2048 tubes and 60 steps.

Dynamic 3DGS improves loss but remains diffuse/splattery at 4096 Gaussians and
60 steps.

## Next

Do not blindly scale all three again. Use representation-specific next rows:

```text
STAR: more tubes/steps can be explored first.
WorldFoam: needs a quality bridge beyond color-only fixed geometry.
Dynamic 3DGS: needs better initialization/camera/loss schedule, and current fast-mac v6_refined runtime cap must stay <= 2048 unless rebuilt.
```
