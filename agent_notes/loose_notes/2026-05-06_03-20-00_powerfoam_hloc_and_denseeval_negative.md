# PowerFoam HLOC Backend And Dense-Eval Negative

Date: 2026-05-06

## Trigger

The broad completion audit still says PowerFoam Metal is not complete:
forward/backward, official fixture parity, saved 4K benchmarks, and 4K
optimizer-step trainability are passing, but the paper-scale clean DeepView row
still fails heldout quality and selects step 0.

This chunk tried two bounded next moves:

1. make the HLOC/ALIKED builder path honest enough to generate optional clean
   geometry without relying on pycolmap's ONNX-backed ALIKED wheel;
2. test whether the selected regular-triangulation appearance-only row was
   simply missing an early post-step heldout peak.

## HLOC Builder Fix

Patched:

```text
research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py
```

Important behavior changes:

- HLOC config mutation now uses `copy.deepcopy`.
- `--feature-type aliked_n16rot` maps to HLOC/LightGlue model
  `aliked-n16rot` instead of silently running plain `aliked-n16`.
- `--feature-type aliked_n32` maps to `aliked-n32`.
- `--max-features` is propagated to
  `hloc_feature_conf["model"]["max_num_keypoints"]`.
- HLOC always imports matches and applies known-pose
  `hloc.triangulation.geometric_verification`; it no longer writes raw
  LightGlue matches as verified rows by default.
- `CameraMode.SINGLE` is rejected when selected train views have different
  intrinsics.
- HLOC summaries now include explicit actual-backend fields:
  `hloc_feature_model_name`, `hloc_feature_max_num_keypoints`,
  `known_pose_verification_applied`, `known_pose_verification_backend`, and
  `known_pose_verification_max_error`.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py

KMP_DUPLICATE_LIB_OK=TRUE PYTHONDONTWRITEBYTECODE=1 \
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  --with 'git+https://github.com/cvg/Hierarchical-Localization.git' python - <<'PY'
from research_experiments.dynamic_foam.build_pycolmap_known_pose_point_cloud import hloc_feature_conf
conf = hloc_feature_conf('aliked_n16rot')
conf.setdefault('model', {})['max_num_keypoints'] = 500
print(conf['output'], conf['model']['model_name'], conf['model']['max_num_keypoints'])
PY
```

Output:

```text
feats-aliked-n16rot aliked-n16rot 500
```

## HLOC Smokes

Wide 2-camera DeepView smoke:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train \
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  --with pycolmap==4.0.4 \
  --with 'git+https://github.com/cvg/Hierarchical-Localization.git' \
  python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc \
  --output /tmp/deepview_hloc_aliked_lightglue_2cam_frame0_256px_smoke.ply \
  --workdir /tmp/deepview_hloc_aliked_lightglue_2cam_frame0_256px_work \
  --target-size 256 --frame-index 0 \
  --train-cameras camera_0001 camera_0015 \
  --heldout-camera camera_0040 --anchor-camera camera_0001 --condition-camera camera_0001 \
  --camera-model opencv_fisheye --camera-mode per_image \
  --feature-backend hloc --feature-type aliked_n16rot --matcher-type aliked_lightglue \
  --max-features 2000 --max-reproj-error 16.0 --verify-max-error 16.0 \
  --xy-extent 100 --z-min -100 --z-max 100 --min-unique-cameras 2 \
  --max-points 8192 --keep-workdir
```

Result: backend completed, but HLOC valid matches were `0%`; `point_count=0`.

Close-overlap 4-camera frame-0 smoke:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train \
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  --with pycolmap==4.0.4 \
  --with 'git+https://github.com/cvg/Hierarchical-Localization.git' \
  python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_closeoverlap_8cam_holdout0005_pycolmap_frames0_4_8_12_512px_true_multiframe_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc \
  --output /tmp/deepview_closeoverlap_hloc_aliked_lightglue_4cam_frame0_256px_smoke.ply \
  --workdir /tmp/deepview_closeoverlap_hloc_aliked_lightglue_4cam_frame0_256px_work \
  --target-size 256 --frame-index 0 \
  --train-cameras camera_0001 camera_0002 camera_0003 camera_0004 \
  --heldout-camera camera_0005 --anchor-camera camera_0001 --condition-camera camera_0001 \
  --camera-model opencv_fisheye --camera-mode per_image \
  --feature-backend hloc --feature-type aliked_n16rot --matcher-type aliked_lightglue \
  --max-features 2000 --max-reproj-error 16.0 --verify-max-error 16.0 \
  --xy-extent 100 --z-min -100 --z-max 100 --min-unique-cameras 2 \
  --max-points 8192 --keep-workdir
```

Result: HLOC completed with some pose-consistent matches, but only
`point_count=2`, `raw_point_count=2`, `database_num_verified_image_pairs=6`,
`database_num_keypoints=942`, reprojection median `3.16px`, and two-view
tracks only. This proves the backend path is real, but it is not a dense clean
geometry solution at this local scale.

Post-patch 2-camera close-overlap schema smoke:

```text
/tmp/deepview_closeoverlap_hloc_aliked_lightglue_2cam_frame0_128px_postpatch.json
```

This verified the new summary fields:

```text
hloc_feature_model_name=aliked-n16rot
hloc_feature_max_num_keypoints=500
known_pose_verification_applied=true
known_pose_verification_backend=hloc.geometric_verification
known_pose_verification_max_error=16.0
```

It still produced `point_count=0`.

## Dense-Eval Slow-RGB Probe

Added config:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_12step_slowrgb_denseeval_noaux.jsonc
```

Purpose: keep the selected clean artifact and regular-triangulation raytrace
path unchanged, but log validation every step and reduce `texel_sv_rgb` LR from
`0.003 -> 0.0003` to `0.00075 -> 0.000075`.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=offline \
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld --with scipy python \
  src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_12step_slowrgb_denseeval_noaux.jsonc
```

Output:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_12step_slowrgb_denseeval_noaux
wandb/offline-run-20260506_033452-dhu4j6w6
```

Result:

- `best_metrics.json` still selects step `0`.
- Heldout PSNR monotonically fell from `12.5099039` at step 0 to
  `12.5056953` at step 12.
- Heldout SSIM fell from `0.1169190` to `0.1168275`.
- Train/source PSNR rose from `12.7537` to `12.7685`.

Conclusion: the selected regular row is not missing an early heldout peak, and
the slower appearance LR does not fix the step-0 blocker. Appearance-only
source reconstruction is still overfitting heldout immediately.

## Current Read

The completion blocker did not move:

- Metal forward/backward and 4K trainability remain good enough for the saved
  verifier gates.
- The clean DeepView paper row still fails PSNR/SSIM and best-step gates.
- Local HLOC/ALIKED is now honest but too sparse to replace the SIFT artifact.
- Slower appearance-only training with dense heldout eval confirms that the
  next quality move needs better geometry/support or a heldout-improving
  objective, not another appearance-only source-view schedule tweak.
