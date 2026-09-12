# Successful source overfits and the small multicamera negative are different experiments

The user correctly challenged an overly broad interpretation of the September
11 local playground results. Historical successful image fits exist, including
retained JSONs, previews, and videos. This session inspected artifacts and code;
it did not launch training or change accepted evidence counts.

## Historical positive evidence

- `BASELINES.md` records the May 17 first-class STAR UVT 64-frame, 256px,
  32,768-tube source overfit: 200 steps, PSNR 29.8233676, SSIM 0.8571995.
  The retained result JSON reports initial MSE 0.13296336 and final MSE
  0.00104151. Its contact sheet has target above reconstruction; the scene and
  motion are recognizable, with softened fine detail. This is successful
  source fitting, not evidence of exact convergence or novel-view quality.
- The corresponding 512px multires run achieved PSNR 29.138 and SSIM 0.8606
  after 200 coarse and 50 fine steps. Both configs and results are under
  `src/train_configs/star_uvt_highmotion_*` and `star_uvt/results/`.
- The July 8 local 128px/16-frame comparison also reports a real numerical
  advantage for STAR UVT: media PSNR 21.8072 versus dynamic 3DGS 18.6425 and
  WorldFoam 17.7772. Source:
  `outputs/benchmarks/2026-07-08_paper_quality_benchmark_table/summary.json`.
  STAR's native result is 21.7685. The comparison derives common metrics from
  encoded side-by-side videos; capacities and update counts differ (2048 tubes
  /60 steps, 4096 Gaussians/60 steps, 2048 cells/80 steps). It is a local smoke
  comparison, not a newly accepted or matched publication baseline.

## What changed in September

The older `video_fit_comparison.py` optimizes `ScreenTimeTubeModel` directly in
screen-time coordinates. Its `fit_uvt` function renders the whole clip and
takes MSE against all target frames at each update. The May 256px recipe thus
uses 12,800 frame contributions across 200 updates.

The current Coffee Martini route optimizes `WorldTubeModel`: world-space
positions and velocities shared across two calibrated training cameras. It is
not the old source overfit with a heldout evaluation switch added. The small
protocol uses 128 then 256 primitives, 80 updates with two sampled target
images each (160 total), and different data, initialization, and optimization.
Relative to the May run, its final tube count is 128 times smaller and its
training-image exposure count is 80 times smaller. These are confounds, not a
demonstration that any one factor caused the failure.

The actual September run summary gives train/heldout PSNR:

| Lane | Training views | Heldout view |
| --- | ---: | ---: |
| World Tubes | 6.486679 | 6.504495 |
| Dynamic 3DGS | 6.173027 | 5.961154 |
| WorldFoam | 6.593655 | 6.643607 |

Source: `outputs/benchmarks/2026-09-11_local_playground/coffee_martini_local_playground/seed_17/run_summary.json`.
Thus the small current run fails to fit training images too; it is not solely
a heldout generalization failure. It also does not establish a regression in
the historical source-fitting implementation. Before a broad negative claim,
rerun a known successful source recipe as a control, then isolate changes in
capacity, sampling, and world/camera representation. Do not discard the old
positive evidence or assume that extra steps alone will fix the new route.
