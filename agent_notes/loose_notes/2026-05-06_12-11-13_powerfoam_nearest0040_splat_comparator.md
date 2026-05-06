# PowerFoam Nearest0040 Splat Comparator

Date: 2026-05-06

## Goal

P0.2 asked for a same-split splat comparator for the nearest0040 PowerFoam row,
so we stop comparing raw PowerFoam against an older 3-camera gsplat baseline.

## Implemented

Added the matched config:

```text
src/train_configs/local_mac_splat_baseline_multicam_deepview_nearest0040_8cam_holdout0040_free_dynamic_3dgs_128_16f_3543splats_40step.jsonc
```

The config matches:

- DeepView `03_Dog`
- train cameras `camera_0025,camera_0039,camera_0041,camera_0012,camera_0026,camera_0023,camera_0042,camera_0038`
- heldout `camera_0040`
- anchor `camera_0025`
- 16 frames
- 128px render
- 3543 primitives
- 40 steps
- seed `23`

I first tried the copied dense renderer. It was not viable for the dev loop:

- 40-step dense run was stopped after about 15 minutes with no artifacts.
- 1-step dense smoke also ran for several minutes and was stopped.
- The slow path was largely video decoding/seek plus dense rendering, not useful
  for a fast P0 comparator.

I switched the matched config to `fast_mac` with `tile_size=16`. The first
fast_mac smoke exposed the tile-size mismatch:

```text
ValueError: RasterConfig.tile_size=8 does not match runtime shader tile size 16.
```

After changing tile size to `16`, the 1-step smoke passed and wrote:

```text
outputs/gauge_fields/multicam_deepview_nearest0040_8cam_holdout0040_40step/free_dynamic_3dgs_fastmac_1step_smoke
```

Then the 40-step run passed and wrote:

```text
outputs/gauge_fields/multicam_deepview_nearest0040_8cam_holdout0040_40step/free_dynamic_3dgs
```

I added SSIM and `train_loop_elapsed_s` to
`research_experiments/gauge_fields/common.py` and
`research_experiments/gauge_fields/train_splat_baseline.py`, then reran the
40-step row so the metrics file has the same comparison fields.

I added the comparison script:

```text
research_experiments/dynamic_foam/compare_powerfoam_to_splats_nearest0040.py
```

and wrote:

```text
outputs/comparisons/powerfoam_vs_splats_nearest0040_20260506.json
```

## Commands

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src/train WANDB_MODE=offline \
  .venv/bin/python -u research_experiments/gauge_fields/train_splat_baseline.py \
  src/train_configs/local_mac_splat_baseline_multicam_deepview_nearest0040_8cam_holdout0040_free_dynamic_3dgs_128_16f_3543splats_40step.jsonc \
  --device mps \
  --steps 40 \
  --output-dir outputs/gauge_fields/multicam_deepview_nearest0040_8cam_holdout0040_40step/free_dynamic_3dgs \
  --no-wandb
```

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_to_splats_nearest0040.py \
  --powerfoam-raw-output outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_40step_denseeval \
  --powerfoam-calibrated-output outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_evalrgbcal_128_16f_3543cells_1step_denseeval \
  --splat-output outputs/gauge_fields/multicam_deepview_nearest0040_8cam_holdout0040_40step/free_dynamic_3dgs \
  --splat-config src/train_configs/local_mac_splat_baseline_multicam_deepview_nearest0040_8cam_holdout0040_free_dynamic_3dgs_128_16f_3543splats_40step.jsonc \
  --output outputs/comparisons/powerfoam_vs_splats_nearest0040_20260506.json
```

## Metrics

Raw PowerFoam, 3543 cells, 40 steps:

```text
train/eval PSNR/SSIM/L1  13.4675 / 0.2056 / 0.1618
heldout PSNR/SSIM/L1     13.2663 / 0.1117 / 0.1691
```

Calibrated PowerFoam, 3543 cells, 1 step:

```text
heldout calibrated PSNR/SSIM/L1  14.3841 / 0.1556 / 0.1552
heldout raw PSNR/SSIM/L1         12.6907 / 0.1246 / 0.1851
```

Matched free dynamic 3DGS, 3543 splats, 40 steps:

```text
train/eval PSNR/SSIM/L1  16.2282 / 0.2875 / 0.1110
heldout PSNR/SSIM/L1     10.9809 / 0.1133 / 0.2043
train loop elapsed       5.6080 s
```

## Interpretation

Under this no-code-change same-split comparator, raw PowerFoam has much higher
heldout PSNR than free dynamic 3DGS (`13.2663` vs `10.9809`) while both have low
SSIM (`0.1117` vs `0.1133`). The calibrated PowerFoam row is still a separate
eval-semantics row and should not be used as raw representation proof.

Main caveat: this splat row uses the gauge-field trainer's pinhole camera path,
while the PowerFoam row uses OPENCV_FISHEYE. Treat this as the matched split,
scale, primitive count, and step-count comparator, not as strict projection
parity.
