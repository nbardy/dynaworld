# Three-Lane Metal Visual Compare

Date: 2026-07-07 16:16 KST.

## Context

Active goal: implement enough code to get WorldFoam, WorldTubes, and base
dynamic 3DGS running through Metal-backed trainers on test data so we can render
and inspect visuals.

The repo already had:

- `dynamic_powerfoam_metal` / PowerFoam-style Metal trainer with disk media.
- `star_uvt_video_overfit` / WorldTubes STAR UVT Metal tile trainer with
  contact sheet and side-by-side video outputs.
- `tokengs_video_implicit_camera` dynamic-3DGS trainer using `fast_mac`.

The missing piece was a single harness that could plan/run the three lanes and
verify produced visual artifacts.

## Implemented

Added:

```text
research_experiments/world_foam_lane2/run_visual_compare_three_lanes.py
research_experiments/world_foam_lane2/test_run_visual_compare_three_lanes.py
src/train_configs/visual_compare_star_uvt_worldtubes_metal_64_16f_20step.jsonc
src/train_configs/visual_compare_dynamic_gsplat_fast_mac_metal_64_16f_20step.jsonc
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_report.md
```

The runner supports dry-run planning, per-lane execution, per-lane stdout/stderr
capture, declared artifact checks, W&B offline media discovery, lane subsetting,
and a JSON summary.

The runner now launches trainer subprocesses with:

```text
uv run --with imageio --with moviepy python src/train/train.py <config>
```

This fixed real media-path failures:

- STAR UVT trained but initially failed when writing side-by-side video because
  `imageio` was missing.
- dynamic 3DGS trained but initially failed when W&B tried to encode video
  because `moviepy` was missing.

## Verification

Test gate:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest research_experiments/world_foam_lane2/test_run_visual_compare_three_lanes.py -q
```

Result: `5 passed`.

Dry-run summary:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_dry_run.json
```

Initial full run summary:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_summary.json
```

Result: WorldFoam ok; STAR and dynamic 3DGS failed on media dependencies.

Failed-lane rerun summary:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_failed_lanes_rerun.json
```

Result: STAR ok; dynamic 3DGS ok.

## Visual Artifacts

WorldFoam / dynamic PowerFoam:

```text
outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_video_1024_16f_40step_smoke/preview_step_0040.png
outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_video_1024_16f_40step_smoke/render_step_0040.mp4
outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_video_1024_16f_40step_smoke/side_by_side_step_0040.mp4
```

STAR UVT / WorldTubes:

```text
outputs/visual_comparisons/star_uvt_worldtubes_metal_64_16f_20step_contact.jpg
outputs/visual_comparisons/star_uvt_worldtubes_metal_64_16f_20step_sbs.mp4
```

Dynamic 3DGS / fast-mac:

```text
wandb/offline-run-20260707_161242-60v2gzdp/files/media/images/Render_GT_vs_Pred_15_41a56dfda6f740706ae0.png
wandb/offline-run-20260707_161242-60v2gzdp/files/media/videos/Render_Video_15_e95345a369bbe5364fa4.mp4
wandb/offline-run-20260707_161242-60v2gzdp/files/media/videos/Render_GT_Video_15_ff09b5bc6b2ae5ae8f0e.mp4
```

## Read

WorldFoam/PowerFoam is nonblank and captures coarse dog/background structure.
STAR UVT is soft but coherent and reports `final_psnr=21.785`. Dynamic 3DGS is
nonblank but noisy/splattery after only 20 steps.

## Original Next, Later Superseded

Add direct disk media output for the `tokengs` / fast-mac trainer so future
comparison summaries do not need to scrape W&B offline media. Then rerun all
three lanes in one clean invocation and promote that as the canonical tiny
visual compare gate.

## Follow-up: Dynamic 3DGS Direct Disk Media

Added direct disk media support to the token-GS trainer:

```text
src/train/token_gs_trainer.py
tests/test_token_gs_eval_media.py
```

`logging.output_dir` is now normalized for the trainer, and validation media
can be written even when W&B is disabled. The dynamic visual-compare config now
sets:

```text
wandb_enabled: false
output_dir: outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_64_16f_20step
```

Verification:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_run_visual_compare_three_lanes.py \
  tests/test_token_gs_eval_media.py \
  -q

PYTHONPATH=src/train uv run python -m py_compile \
  src/train/token_gs_trainer.py \
  research_experiments/world_foam_lane2/run_visual_compare_three_lanes.py
```

Result: `7 passed`; compile and `git diff --check` passed.

Direct dynamic-lane rerun:

```text
outputs/visual_comparisons/2026-07-07_dynamic_gsplat_direct_disk_media_rerun.json
```

Result: `status=ok`, with all declared artifacts present:

```text
outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_64_16f_20step/preview_step_0020.png
outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_64_16f_20step/render_step_0020.mp4
outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_64_16f_20step/side_by_side_step_0020.mp4
```

Visual read stayed the same: target is clear; render is nonblank but noisy and
splattery at 20 steps. This is fine for the tiny gate and should not be sold as
quality parity.

New next step: rerun all three lanes in one clean invocation, then make a
larger 128px/16f visual tier before writing any paper-quality comparison claim.

## Follow-up: Clean All-Lane Gate

Clean all-lane rerun:

```text
outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_clean_all_lanes.json
```

Result: overall `status=ok`. All three lanes exited 0 and all declared media
artifacts existed.

Lane timings from the runner:

```text
WorldFoam / dynamic PowerFoam: 33.83s
WorldTubes / STAR UVT:         24.50s
Base dynamic 3DGS / fast-mac:  73.30s
```

Fresh metrics:

```text
WorldFoam eval PSNR mean: 19.3554, eval L1: 0.0611
STAR UVT final PSNR:      21.8107, final L1: 0.0573, render median: 13.73ms
Dynamic 3DGS final line:  Loss 0.2509, recon 0.2459
```

This supersedes the clean-run-pending note above. Next step is now a larger
visual tier, probably 128px/16f first, not more harness plumbing.
