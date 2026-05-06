# PowerFoam Color Affine And Regular Support Thaw

Date: 2026-05-06 04:30:04 Asia/Ho_Chi_Minh

## Goal

Close or bound two cheap hypotheses for the selected clean DeepView regular
PowerFoam row:

- remaining error is mostly background/color/exposure
- regular topology only needs a tiny support/material thaw to improve heldout
  after step 0

Selected baseline:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux
```

Step 0 / best heldout remains `12.5099` PSNR, `0.1169` SSIM.

## Color/Background Bound

Added and ran:

```text
research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal \
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with scipy python \
  research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux.jsonc
```

Output:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/color_affine_diagnostics.json
```

Results:

| Variant | Heldout PSNR | Heldout SSIM | Heldout L1 |
|---|---:|---:|---:|
| black background baseline | 12.5099 | 0.1169 | 0.1794 |
| train-fit constant background | 12.7942 | 0.1221 | 0.1723 |
| train-fit channel affine | 13.9699 | 0.1411 | 0.1655 |
| train-fit RGB affine | 14.0118 | 0.1359 | 0.1628 |
| train-fit background + train-fit channel affine | 13.9892 | 0.1416 | 0.1647 |
| heldout-oracle background + oracle RGB affine | 14.2073 | 0.1363 | 0.1578 |

Takeaway: color/exposure is a real PSNR lever, but even oracle color/background
postprocessing misses the `0.15` SSIM gate. Stop treating cheap compositing as
the primary paper blocker.

## Regular Support Thaw

Added and ran:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_supportthaw_128_16f_1024cells_12step_denseeval_noaux.jsonc
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled \
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with scipy python \
  src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_supportthaw_128_16f_1024cells_12step_denseeval_noaux.jsonc
```

Output:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_supportthaw_128_16f_1024cells_12step_denseeval_noaux
```

Result:

- source PSNR/SSIM improved monotonically from `12.7537/0.1301` to
  `12.7753/0.1313`
- heldout PSNR fell from `12.5099` to `12.5037`
- heldout SSIM moved only from `0.116919` to `0.116975`
- `best_metrics.json` still selects step 0
- mean quaternion delta reached only `0.00341`; mean texel-site delta reached
  `0.000158`

Takeaway: this closes the cheap "regular topology needs tiny support/material
thaw" hypothesis. It is still source-view overfit. The next productive lever is
stronger spatial support/geometry or a genuinely heldout-improving objective,
not another small source-only schedule tweak.

