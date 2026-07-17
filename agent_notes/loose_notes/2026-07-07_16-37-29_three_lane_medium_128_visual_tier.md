# Three-Lane Medium 128px Visual Tier

Date: 2026-07-07 16:37 KST.

## Context

The active goal is still to get WorldFoam, WorldTubes, and base dynamic 3DGS
running through Metal-backed trainer/render paths on test data and inspect the
visual outputs.

The prior 64px tiny gate was green. This pass made the next step concrete: a
128px/16f medium tier in the same comparison harness.

## Implemented

Runner:

```text
research_experiments/world_foam_lane2/run_visual_compare_three_lanes.py
```

The runner now supports:

```text
--tier tiny
--tier medium
```

The stable lane names stay the same across tiers:

```text
worldfoam_dynamic_powerfoam_metal
worldtubes_star_uvt_metal
dynamic_gsplat_fast_mac_metal
```

New medium configs:

```text
src/train_configs/visual_compare_worldfoam_dynamic_powerfoam_metal_128_16f_40step.jsonc
src/train_configs/visual_compare_star_uvt_worldtubes_metal_128_16f_20step.jsonc
src/train_configs/visual_compare_dynamic_gsplat_fast_mac_metal_128_16f_20step.jsonc
```

Report:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_medium_128_report.md
```

## Verification

Focused tests:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_run_visual_compare_three_lanes.py \
  tests/test_token_gs_eval_media.py \
  -q
```

Result: `9 passed`.

Config parse and compile checks passed:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/world_foam_lane2/run_visual_compare_three_lanes.py \
  src/train/token_gs_trainer.py
```

Medium dry-run summary:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_medium_128_dry_run.json
```

Medium all-lane run:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_medium_128_all_lanes.json
```

Result: overall `status=ok`; all three lanes exited 0 and all declared
artifacts existed.

## Artifacts

WorldFoam / dynamic PowerFoam:

```text
outputs/visual_comparisons/worldfoam_dynamic_powerfoam_metal_128_16f_40step/preview_step_0040.png
outputs/visual_comparisons/worldfoam_dynamic_powerfoam_metal_128_16f_40step/render_step_0040.mp4
outputs/visual_comparisons/worldfoam_dynamic_powerfoam_metal_128_16f_40step/side_by_side_step_0040.mp4
```

WorldTubes / STAR UVT:

```text
outputs/visual_comparisons/star_uvt_worldtubes_metal_128_16f_20step_contact.jpg
outputs/visual_comparisons/star_uvt_worldtubes_metal_128_16f_20step_sbs.mp4
```

Base dynamic 3DGS / fast-mac:

```text
outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_128_16f_20step/preview_step_0020.png
outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_128_16f_20step/render_step_0020.mp4
outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_128_16f_20step/side_by_side_step_0020.mp4
```

## Metrics

```text
WorldFoam eval PSNR mean: 17.8816, eval L1: 0.0774
STAR UVT final PSNR:      16.9816, final L1: 0.1142, render median: 12.297ms
Dynamic 3DGS final line:  Loss 0.2863, recon 0.2820
```

Runner elapsed times:

```text
WorldFoam / dynamic PowerFoam: 21.96s
WorldTubes / STAR UVT:         12.08s
Base dynamic 3DGS / fast-mac:  35.01s
```

## Visual Read

WorldFoam at 128px is nonblank and scene-aligned but under-capacity: it keeps
coarse background/color and loses the dog into a dark smear.

STAR UVT at 128px is temporally coherent and preserves the moving dog better
than the raw dynamic-3DGS lane, but it is strongly blurred with 1024 tubes and
20 steps.

Dynamic 3DGS at 128px is nonblank but noisy/splattery after 20 steps.

## Implication

The medium tier proves the runnable Metal-backed training/rendering shape at a
larger resolution. It does not prove representation quality ordering.

Next useful step: use the same `--tier medium` harness for a longer and/or
capacity-scaled 128px row. Do not add another renderer family before the
existing three canonical lanes have a fairer capacity/step ladder.
