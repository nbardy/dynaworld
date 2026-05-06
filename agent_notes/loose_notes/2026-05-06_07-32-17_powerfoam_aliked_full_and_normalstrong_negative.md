# PowerFoam ALIKED Full8 And Normal-Strong Negative

Date: 2026-05-06 07:32

Goal: continue the remaining PowerFoam Metal completion blocker after CUDA smoke
and core/4K gates were already green. The active failing gate is paper-scale
clean heldout quality on DeepView `camera_0040`.

## ALIKED/LightGlue full8 Modal artifact

Ran the first canonical full candidate that can feed the paper verifier's
optional ALIKED row:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_geometry.py \
  --execute \
  --run-id full8_1024_lightglue_20260506 \
  --full \
  --matcher-type aliked_lightglue \
  --max-features 12000
```

Output:

```text
outputs/powerfoam_aliked_geometry/full8_1024_lightglue_20260506/full.json
research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_aliked_n16rot_aliked_lightglue_minucam2.json
research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_aliked_n16rot_aliked_lightglue_minucam2.ply
```

Result: the artifact is valid but too sparse for the paper gate:

- `point_count=319`, below the verifier's `min_point_count=2000`.
- `database_num_verified_image_pairs=496`.
- `filtered_track_length.mean=7.0878`, `p90=8.0`, `max=13.0`.
- `filtered_unique_camera_track_length.p90=2.0`.
- `filtered_unique_frame_track_length.p90=4.0`.
- `filtered_reproj_error.median=3.2298`, `p90=5.8444`.

This rejects the full-resolution ALIKED/LightGlue branch as the next acceptance
fix. Training it would duplicate 319 points to 1024 cells and could not satisfy
the point-count acceptance check even if image metrics improved.

Implementation fix: `modal_powerfoam_aliked_geometry.py` now normalizes copied
artifact JSON `output` fields from Modal's `/root/dynaworld/...` path to repo
relative paths before writing canonical local artifacts. The already returned
ALIKED JSON was patched the same way.

## Heldout residual diagnostic on selected 2714 row

Ran:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with scipy python research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_2714cells_40step_denseeval.jsonc \
  --heldout-only --batch-size 4 --support-chunk-size 1024
```

Output:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_2714cells_40step_denseeval/heldout_error_diagnostics.json
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_2714cells_40step_denseeval/heldout_error_diagnostics_panel.png
```

Important numbers:

- Heldout alpha mean `0.9745`.
- Alpha `>0.9` covers `96.42%` of pixels.
- Worst frame is heldout frame `5`, L1 `0.1901`, alpha mean `0.9758`.
- Worst-frame sphere-support proxy hits `99.87%` of pixels.
- High-residual support-hit fraction is `99.36%`.
- The dominant residual bucket is high-alpha pixels: `95.51%` of total residual.
- High-residual high-alpha pixels have normal-distance mean `0.0964`, higher
  than the high-alpha/all normal-distance mean `0.0512`.

This confirms the failure is confidently opaque wrong rendering with support,
not blank coverage. The next useful fix should target spatial/depth/material
ordering or the objective. More sparse points, background/color affine, and
coverage-only support patches are not the right next lever.

## Normal-strong 2714 run

Added and ran:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_normalstrong_128_16f_2714cells_40step_denseeval.jsonc
```

This keeps the selected 2714-cell clean SIFT init, regular triangulation,
official-objective losses, random background, and fast warmups, but raises
`losses.normal_weight` from `0.1` to `1.0` and final multiplier from `0.1` to
`0.5`.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=offline \
  uv run --with scipy python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_normalstrong_128_16f_2714cells_40step_denseeval.jsonc
```

Output:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_normalstrong_128_16f_2714cells_40step_denseeval/
wandb/offline-run-20260506_072352-sdxdjt4s
```

Result:

- Best heldout remains step `10`: PSNR `12.6686`, SSIM `0.1000`, L1 `0.180327`.
- Final heldout: PSNR `12.6505`, SSIM `0.0997`, L1 `0.181155`.
- Source improves to final PSNR/SSIM `13.5464 / 0.1953`.
- Mean train normal-distance fell from `0.0033979` in the selected final row to
  `0.0032497`, but heldout quality did not move.

Conclusion: global stronger normal-distance regularization is a bounded
negative. The heldout high-normal-distance residual is not fixed by simply
raising the train-view normal loss.

## Current boundary

Completion audit still cannot pass because the paper-quality gate remains
below threshold. The best selected clean row is still the 2714-cell
official-objective fast-warmup run at heldout PSNR/SSIM `12.6689 / 0.1000`;
the threshold remains `13.0 / 0.15`.

## Official-code comparison after the negative

I cloned the official PowerFoam repo to `/tmp/powerfoam_official_inspect` and
checked the training path. The relevant official mechanism is stronger than our
current scalar `normal_distance` loss:

- official `train.py` requests a median-depth quantile when
  `normal_supervision` is enabled;
- it reads rendered `normal` and `depth` from `model.forward(...)`;
- it optionally compares rendered normals to Metric3D normals;
- without Metric3D it bilateral-filters the rendered median depth, derives
  normals from that depth, and adds an MSE from rendered normals to those
  estimated normals;
- `ray_gt` is also passed into the forward path for surface objective /
  per-point-error accumulation.

Our Metal extension already has a non-gradient aux path that can return
`normal`, `median_depth`, and `depth_quantile_depths`
(`rasterize_power_foam_aux`), but the training path currently exposes only
`normal_distance` as a differentiable auxiliary. The normal-strong negative
therefore does not close the official normal-supervision gap. The next
implementation mechanism, if we keep chasing paper quality locally, is a
differentiable rendered-normal output and median-depth/self-normal supervision
compatible with the height+SV raytrace path.
