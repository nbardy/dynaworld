# PowerFoam CUDA Fixture Gate And Nearest0040 Cap384

Date: 2026-05-06 09:31:33 +07

## Trigger

We revisited the CUDA lane because the requested workflow is useful: use the
pinned official PowerFoam CUDA/Warp repo as a known-good baseline, fork it for a
small dynamic representation change, run both on the same tiny clip/settings on
Modal L40S, and compare those smoke results against the local Metal lane.

The same chunk also needed to preserve the latest Metal quality experiments
after compaction: nearest-camera clean support, higher texel-site capacity, and
the Metal backward feature-dimension cap increase.

## CUDA Lane State

The CUDA smoke path already exists and is the right fast deployment test:

- `research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py`
- `research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py`
- `research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch`
- `research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py`
- `research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py`
- `tests/test_powerfoam_cuda_smoke.py`
- `research_experiments/dynamic_foam/POWERFOAM_CUDA_MODAL_SMOKE.md`

Pinned upstream is `https://github.com/theialab/powerfoam` at
`96392252ebd0059fe6ca98881b62e12295d9242f`.

The canonical strict fixed-black run is:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json
outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/cuda_vs_metal_summary.json
```

It uses 4 frames, 64 px, 5 steps, 256 points, 4 texel sites, SV DoF 2, seed 17,
fixed black background, and Modal L40S.

Results:

- official static CUDA: PSNR/SSIM/L1 `5.5640 / 0.0284 / 0.4901`; warm step
  `8.31 ms`
- dynamic feature-foam CUDA: PSNR/SSIM/L1 `5.5833 / 0.0288 / 0.4887`; warm
  step `9.09 ms`
- dynamic time probe: RGB delta mean/max `0.00006899 / 0.0009796`
- CUDA-vs-Metal smoke contract is matched for source clip, frame count, render
  size, step count, point count, texel sites, SV DoF, random init, and fixed
  black background

Important boundary: this CUDA fork is intentionally small. It adds a Gaussian
time-basis residual to `texel_sv_rgb`; it does not move foam geometry, update
adjacency as a function of time, or accumulate F32 latent features in Warp.

## CUDA Tightening Added

The strict fixed-black run skipped the official fixture, so the validator now
has an explicit fixture-required mode for future reruns:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/{run_id}/summary.json \
  --require-official-fixture
```

The runner plan summary now records the deploy-relevant knobs directly:
`seed`, `dynamic_time_basis_count`, `skip_official_fixture`,
`fixed_black_background`, and `random_background`, and its planned execute
command includes the concrete micro-smoke flags rather than an ellipsis.

The no-GPU test gate now covers the Modal wrapper's fast deploy arguments, the
fixture-required verifier mode, and the existing dynamic-patch boundary.

Use this fixture-included rerun when we want the official CUDA/Warp fixture and
the strict fixed-black comparison in one dated artifact:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_micro_blackbg_fixture_rerun \
  --max-gpu-minutes 10 \
  --fixed-black-background
```

## Nearest0040 Clean Support

The nearest-camera artifact was built around heldout `camera_0040` with train
cameras:

```text
camera_0025 camera_0039 camera_0041 camera_0012 camera_0026 camera_0023 camera_0042 camera_0038
```

Anchor and condition camera were both `camera_0025`. Artifact:

```text
research_experiments/dynamic_foam/artifacts/deepview_03_dog_nearest0040_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.ply
research_experiments/dynamic_foam/artifacts/deepview_03_dog_nearest0040_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json
```

Artifact quality:

- `point_count=3543`
- verified pairs `496`
- reproj median/p90 `2.7134 / 5.2520`
- filtered track mean/p90/max `6.3774 / 8.0 / 22`
- unique-camera p90 `2.0`
- unique-frame p90 `4.0`

The 4-site run:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_40step_denseeval.jsonc
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_40step_denseeval
```

Best step 40 heldout/source:

- heldout PSNR/SSIM/L1 `13.2663 / 0.1117 / 0.1691`
- source PSNR/SSIM `13.4675 / 0.2056`

This crossed the PSNR threshold but still missed the SSIM gate.

## 9 Sites And Cap384

The first 9-site run failed because Metal raytrace height+SV backward rejected
`feature_dim > 128`. With `num_texel_sites=9` and `sv_dof=3`, feature dim is
198.

Patch:

- `third_party/powerfoam-metal/csrc/metal/powerfoam_metal.mm`
- `third_party/powerfoam-metal/csrc/metal/powerfoam_tiled_stream_kernels.metal`
- `tests/test_powerfoam_direct.py::test_powerfoam_metal_raytrace_height_sv_backward_supports_9_texel_sites`

The guard is now `feature_dim <= 384`, enough for up to 16 sites at `sv_dof=3`
by shape, though only 9 sites was tested.

The 9-site cap384 run:

```text
src/train_configs/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_9sites_40step_denseeval.jsonc
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_9sites_cap384_40step_denseeval
```

Metrics from stdout:

- step 0 heldout `12.6963 / 0.1248 / 0.1849`
- step 10 heldout `13.1932 / 0.1173`
- step 20 heldout `13.2624 / 0.1151`
- step 30 heldout `13.2810 / 0.1135`
- step 40 heldout `13.2797 / 0.1122`
- source at step 40 `13.4779 / 0.2066`

Conclusion: more texel capacity marginally improves PSNR but still drives
heldout SSIM down. This looks like source-view appearance overfit or structural
texture mismatch, not a raw-capacity bottleneck.

## Color And Structure Diagnostics

Nearest0040 color affine:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_40step_denseeval/color_affine_diagnostics.json
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_40step_denseeval/color_affine_diagnostics_ssimopt_black.json
```

Train-fit channel affine on fixed black reaches heldout `14.1396 / 0.1454`,
close to but still below SSIM 0.15. SSIM-optimized affine overfits and worsens
heldout to about `13.45 / 0.1225`.

Structure diagnostic on the previous selected 2714 row:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_2714cells_40step_denseeval/heldout_structure_diagnostics.json
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_2714cells_40step_denseeval/heldout_structure_diagnostics_structure_panel.png
```

High residual pixels have higher normal-distance / rendered-vs-depth-normal
error than the whole valid set, but the correlations are modest. The result
does not justify chasing normal losses alone.

## Paper Gate State

The paper verifier should now select the nearest0040 cap384 row by PSNR, but it
still fails the clean heldout SSIM threshold. This lane is not complete. The
remaining high-signal next step is a heldout-stable material/structure objective
or a better train-only representation constraint, not another broad capacity
increase.

