# PowerFoam CUDA Smoke And Endpoint Gate

Date: 2026-05-06 06:16 +07

## Trigger

We wanted the official CUDA/Warp PowerFoam path to be the cheap deployment
baseline instead of treating the local Metal port as the only reference. The
desired lane was: clone pinned upstream PowerFoam, run official static CUDA on
the same tiny clip/settings, apply a minimal dynamic representation fork, run
that on the same Modal L40S smoke, save JSON, and compare to the matched Metal
smoke.

## CUDA Findings

Three read-only subagents converged on the same boundary:

- no official PowerFoam checkout is vendored in this repo; the runner clones
  `https://github.com/theialab/powerfoam` at
  `96392252ebd0059fe6ca98881b62e12295d9242f`;
- official PowerFoam is Python + Warp/CUDA, not a local `.cu` extension;
- the smallest CUDA-compatible representation fork should stay appearance-side:
  add `dynamic_texel_sv_rgb_coeff: [N, K, S, 3 * sv_dof]`, compute a Gaussian
  time basis from `camera.time_index`, and substitute
  `texel_sv_rgb + basis @ coeff` inside upstream `PowerfoamScene`;
- do not t-condition `points`, `radii`, `quaternions`, `texel_sites`,
  `texel_sv_axis`, or `texel_height` in the first fork, because geometry motion
  makes adjacency/sort/update time-dependent.

The existing CUDA lane already implements this:

```text
research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py
research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py
research_experiments/dynamic_foam/export_powerfoam_smoke_dataset.py
research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch
research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py
research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py
research_experiments/dynamic_foam/POWERFOAM_CUDA_MODAL_SMOKE.md
```

Current strict saved L40S result:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json
```

It validates as `ok`. Official static CUDA on the 4f/64px/5step/256-point
smoke is `5.5405` PSNR / `0.0218` SSIM / `0.4916` L1 with warm step
`6.93 ms`. The dynamic `texel_sv_rgb(t)` fork is `5.5487` / `0.0221` /
`0.4911` with warm step `7.17 ms`; the same-camera time probe records
`dynamic_time_rgb_delta_mean=0.00019385` and max `0.0026973`, proving the time
branch changes rendered RGB.

I did not launch a new Modal GPU run in this chunk because the strict saved run
already answers the deployment smoke and validator path. The canonical rerun is:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_micro_time_causality_rerun \
  --max-gpu-minutes 8 \
  --skip-official-fixture
```

## Test Tightening

Extended `tests/test_powerfoam_cuda_smoke.py` with a no-GPU contract test for
`compare_powerfoam_cuda_metal_smoke.py`. It now guards:

- the dry-run CUDA plan and validator shape;
- the pinned official commit and dynamic patch SHA;
- that the dynamic CUDA fork remains `configs/__init__.py` + `powerfoam/scene.py`
  only, with no raster/raytrace/Warp kernel patch;
- that the CUDA-vs-Metal comparison writer emits a schema-valid JSON with a
  fully matched smoke contract.

Verified:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tests/test_powerfoam_cuda_smoke.py \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py

PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_powerfoam_cuda_smoke.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py
```

Results: py_compile passed, CUDA smoke pytest `3 passed`, saved CUDA summary
validated `ok`, and the comparison writer refreshed:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/cuda_vs_metal_summary.json
```

## Metal Endpoint Correctness

The earlier height+SV material endpoint patch was independently verified here.
The bug: when a height surface clipped the far side of the interval, material
lookup still used the near endpoint. This could make color/texture sample a
different texel from the actual surface endpoint.

Patched tests now include:

- `third_party/powerfoam-metal/tests/raytrace_check.py` path agreement for
  stream/raytrace endpoint color;
- `third_party/powerfoam-metal/tests/linear_texture_check.py` independent
  PyTorch texture-reference check, plus the existing forward/backward cases.

Verified:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  third_party/powerfoam-metal/tests/linear_texture_check.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=third_party/powerfoam-metal uv run python \
  third_party/powerfoam-metal/tests/linear_texture_check.py
```

Result: passed. The semantic endpoint color was green-dominant
`[0.0996, 0.8904, 0.0990]`; the PyTorch reference max error on the sharp
single-pixel fixture was `0.00899`, and all broad gradient parity cases stayed
well below their tolerances.

Selected checkpoint re-render after the endpoint patch:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/heldout_error_diagnostics_materialendpoint_patch.json
```

Heldout PSNR was `12.5075`, essentially unchanged from `12.5099`. So the
endpoint issue was real correctness debt, but not the paper-quality blocker.

## Quality Probes Closed

Normal-thaw:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_normalthaw_128_16f_1024cells_12step_denseeval.jsonc
```

Normals/quaternions moved (`state_mean_normal_delta=0.00669`,
`state_mean_quaternion_delta=0.00342`), but best heldout stayed step 0 at
`12.5099/0.1169`; final step 12 was `12.5037/0.1170`.

LOTO diagnostic:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_7cam_loto0003_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_normalthaw_128_16f_1024cells_12step_denseeval.jsonc
```

Holding out `camera_0003` with the reused 8-camera PLY reached heldout
`13.2527/0.1270`. This shows `camera_0040` is a harder parallax/coverage view,
but it is not a clean acceptance row and still fails SSIM. A clean LOTO row
needs a true 7-camera PLY rebuilt from the retained source cameras.

Camera perturbation probe:

```text
research_experiments/dynamic_foam/probe_powerfoam_camera_perturbations.py
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/heldout_camera_perturbations_frames0_4_8_12.json
```

Frozen finite camera nudges on the selected checkpoint found only a small
positive range. Baseline subset was `12.4830/0.1161` PSNR/SSIM. The best PSNR
candidate was `translate_forward_+0.0500` at `12.6926/0.0900`; the best SSIM
candidate was `translate_down_-0.0250` at `12.6781/0.1205`. This says camera
correction may be a minor PSNR contributor, but it does not clear either paper
threshold and can worsen SSIM.

Official-objective diagnostic:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_128_16f_1024cells_12step_denseeval.jsonc
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_128_16f_1024cells_12step_denseeval/best_metrics.json
```

This switches the selected clean row toward the upstream objective bundle:
MSE + `0.2 * SSIM`, random background, normal-distance, contribution, and
interpenetration terms. It selected step 12 rather than step 0, with heldout
`12.5242/0.1226` PSNR/SSIM and source `12.9137/0.1478`. Geometry/material moved
slightly (`state_mean_center_delta=0.00564`, `state_mean_normal_delta=0.00650`,
`state_mean_texel_sv_rgb_delta=0.00413`). This is the first small
heldout-improving objective probe on the selected row, but it remains below
the `13.0/0.15` acceptance gate. Also note the short-run warmup caveat:
density/radii/height have fixed long warmups, so a 12-step diagnostic barely
exercises those groups.

Official-objective fast-warmup follow-up:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_1024cells_40step_denseeval.jsonc
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_1024cells_40step_denseeval/best_metrics.json
```

This added `train.lr_warmup_steps` overrides for short diagnostics and set
`radii`, `density`, and `texel_height` warmups to `10` steps. The run selected
step 20 at heldout `12.5535/0.1255`, with source `13.1633/0.1749`.
The warmup override worked mechanically: mean density delta reached `0.0770`,
center delta `0.0110`, normal delta `0.0136`, SV RGB delta `0.00887`, and
height delta `6.97e-6`. Final source kept improving to `13.3737/0.1971`, while
heldout drifted down to `12.5414/0.1249`. This closes the hypothesis that the
remaining gap was simply the long density/radii/height warmup.

All-filtered official-objective capacity follow-up:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_2714cells_40step_denseeval.jsonc
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_2714cells_40step_denseeval/best_metrics.json
```

This kept all `2714` train-visible filtered OPENCV_FISHEYE points, used
`regular_triangulation`, and enabled W&B offline (`5dj1ssze`). It selected
step 10 at heldout `12.6689/0.1000`, with source `13.2509/0.1557`; final source
rose to `13.5657/0.1958` while heldout ended `12.6498/0.0998`. The run is now
registered in `verify_powerfoam_paper_acceptance.py`, so the verifier selects
this post-initial W&B-backed row. It still fails both paper thresholds:
PSNR is below `13.0`, and SSIM is far below `0.15`.

## Current Boundary

CUDA deployment and the tiny dynamic appearance fork are no longer the blocker.
The remaining PowerFoam blocker is paper-scale heldout quality. The verifier
now selects a post-initial W&B-backed row at `12.6689/0.1000`, but the
`13.0/0.15` acceptance thresholds are still not met. Neither small
support/normal motion, finite camera nudges, the material endpoint correctness
fix, official-objective short training, compressed warmups, nor keeping all
train-visible clean points has cleared the paper-quality gate.
