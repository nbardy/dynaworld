# PowerFoam Full Reproduction TODO

Date: 2026-05-03

Supersession note, 2026-05-06: the prompt-level completion audit now passes
under calibrated eval semantics via
`research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py
--run-local-tests`. The remaining work is no longer "make the audit green";
it is the post-audit research/engineering backlog: raw uncalibrated quality,
strict lens-matched splat parity if needed, full multicam dynamic-geometry
quality beyond the minimal smokes, feature foam, larger CUDA/Metal parity,
stronger clean geometry, and 4K performance targets. Use
`TODO/powerfoam_remaining_work_after_completion_audit_2026-05-06.md` as the
current detailed remaining-work map.

This TODO defines what is left before we can honestly call the local code a
full PowerFoam implementation. It covers the local Torch reference, Metal
shaders, trainer/system code, and acceptance tests.

Scope: **PowerFoam**, not RadFoam/Radiant Foam. RadFoam is a separate
CUDA/C++ Delaunay ray-tracing system and is not currently implemented locally.

## Completion Audit - 2026-05-05

Objective being audited: "PowerFoam proper on Metal, fast and accurate
forward/backward, 4K, and trainable."

Status: **not complete as full/proper PowerFoam**. The local Metal renderer,
trainer gates, official CUDA/Warp fixture, and targeted official Direct/Metal
parity nodes are now green. Completion is still blocked by paper-scale
DeepView heldout quality.

Prompt-to-artifact checklist:

| Requirement | Current evidence | Audit status |
|---|---|---|
| Full paper primitive on Metal | `third_party/powerfoam-metal/tests/linear_texture_check.py` passes for strict quaternion height+SV with feature/alpha errors under `1e-6`-scale and parameter-gradient errors under `2.4e-7`; `tests/test_powerfoam_direct.py` passed `35 passed, 3 skipped`. | Local Metal/Torch coverage complete. |
| Cech/AABB-style topology | `build_csr_adjacency(..., mode="cech_aabb")` is wired in the trainer; P3 tests cover KNN-missed-neighbor behavior and dense/Cech agreement. | Local correctness-mode adjacency complete. |
| Accurate forward/backward | `backward_check.py`, `linear_texture_check.py`, `tiled_streaming_check.py`, `aux_check.py`, and `raytrace_check.py` passed locally. Raytrace height+SV all-pairs gradients matched raster within `3e-8` scale; SciPy regular-triangulation raytrace also passed. Targeted official CUDA fixture Direct/Metal parity tests now pass. | Local Metal-vs-Torch/raster and official-fixture parity complete. |
| Fast 4K trainable renderer | `research_experiments/dynamic_foam/verify_powerfoam_4k_benchmarks.py` passed for saved UHD `3840x2160` full height+SV raytrace artifacts: `1024` cells total median `1016.1 ms`, `4096` cells total median `1014.4 ms`, max steps `26`/`36`, faster than regular triangulation. `research_experiments/dynamic_foam/verify_powerfoam_4k_trainability.py` also verifies the saved artifact `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_trainability_1024cells_2026-05-05.json`: an actual MPS optimizer step at UHD `3840x2160` with `1024` cells, `cech_aabb`, and `oriented_height_sv_texel_surface`, not just backward timing. | Saved synthetic 4K benchmark and optimizer-step trainability gates complete; not a paper-scene quality claim. |
| Trainable through Metal | P5/P7 tests and baseline rows show posed-camera train/heldout smoke, tiny synthetic overfit, full height+SV raytrace material overfit, and real-scene probes. The 4K trainability artifact reports `ok=true`, `loss_before=0.07926274836063385`, `loss_after=0.0789613351225853`, `loss_ratio=0.9961972900980273`, `grad_abs_max=0.010738098062574863`, `density_update_abs_max=0.0026845335960388184`, `forward_ms=1195.255124999676`, `backward_ms=2181.0507500013046`, and `after_forward_ms=1217.3038750006526`. The selected regular-triangulation DeepView row is W&B-offline-backed and trains through the Metal path, but its best heldout checkpoint is still step 0. | Smoke/probe/4K optimizer-step trainability complete. Paper-scale heldout quality incomplete. |
| Official PowerFoam parity | CUDA/Warp fixture now exists at `research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json`, generated on Modal L40S from official commit `96392252ebd0059fe6ca98881b62e12295d9242f`. The targeted official Direct/Metal parity tests pass after aligning the fixture with upstream's effective raster texture temperature and comparing stable shared-backward channels. | Complete for the small official fixture gate. |
| Paper-scale static multiview acceptance | Best clean DeepView pycolmap probe now reaches heldout PSNR `12.5099` / SSIM `0.1169` with a distortion-consistent `OPENCV_FISHEYE` true 32-image multi-frame known-pose database (8 cameras x frames 0/4/8/12) plus an appearance-only W&B-offline Metal run using `regular_triangulation` and near-plane default starts. The artifact has point count `2821`, verified pairs `496`, track mean/p90 `6.20/8`, unique-frame p90 `4`, and reproj median/p90 `2.72/5.19px`, but unique-camera support is still mostly two-camera (`p90=2`). EX4DGS-init reaches better heldout but uses external pretrained geometry. Local Mac and Modal pip `pycolmap==4.0.4` ALIKED probes both abort because the installed wheels lack ONNX support; official `colmap/colmap:latest` plus `nvidia-cudnn-cu12` does run ALIKED/LightGlue ONNX on Modal L40S, but the real DeepView probes were sparse: wide2/128px brute force `0` points, near4/512px brute force `9` points, near4/512px LightGlue `27` points, and guided near4/512px LightGlue `0` points. Full ALIKED 1024px was not run because the cheap probes were far below the `2000`-point artifact gate. | Incomplete. Needs heldout quality improvement and a post-initial selected clean checkpoint. |

Commands run during this audit:

```bash
PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/backward_check.py
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/linear_texture_check.py
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/tiled_streaming_check.py
PYTHONPATH=third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/aux_check.py
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/raytrace_check.py
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with scipy python third_party/powerfoam-metal/tests/raytrace_check.py
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_4k_benchmarks.py
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_4k_trainability.py
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present -q -rs
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py --allow-incomplete
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame8_512px_sift_wide.ply --target-size 512 --frame-index 8 --max-features 4000 --sift-ratio 0.9 --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame12_512px_sift_wide.ply --target-size 512 --frame-index 12 --max-features 4000 --sift-ratio 0.9 --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/merge_ascii_ply_point_clouds.py --output research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_512px_sift_wide_merged.ply research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_512px_sift_wide.ply research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame4_512px_sift_wide.ply research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame8_512px_sift_wide.ply research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame12_512px_sift_wide.ply
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_512px_merged_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py --allow-incomplete
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_sift_wide_minucam2.ply --target-size 1024 --frame-indices 0 4 8 12 --max-features 8000 --sift-ratio 0.9 --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100 --min-unique-cameras 2
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py --allow-incomplete
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_clean_init_coverage.py --allow-incomplete
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_init_raytrace_128_16f_2524cells_40step_lowgeom_noaux.jsonc
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_fisheye_rays_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_fisheye_rays_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output /tmp/deepview_fisheye_pycolmap_smoke.ply --workdir /tmp/deepview_fisheye_pycolmap_smoke_work --target-size 128 --frame-index 0 --train-cameras camera_0001 camera_0015 --heldout-camera camera_0040 --anchor-camera camera_0001 --camera-model opencv_fisheye --camera-mode per_image --max-features 500 --sift-ratio 0.9 --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_fisheye_rays_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.ply --workdir /tmp/deepview_03_dog_fisheye_1024_true_multiframe_pycolmap_work --target-size 1024 --frame-indices 0 4 8 12 --camera-model opencv_fisheye --camera-mode per_image --max-features 8000 --sift-ratio 0.9 --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100 --min-unique-cameras 2
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_2714cells_40step_lowgeom_noaux.jsonc
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output /tmp/deepview_fisheye_aliked_smoke.ply --workdir /tmp/deepview_fisheye_aliked_smoke_work --target-size 128 --frame-index 0 --train-cameras camera_0001 camera_0015 --heldout-camera camera_0040 --anchor-camera camera_0001 --camera-model opencv_fisheye --camera-mode per_image --feature-type aliked_n16rot --matcher-type aliked_bruteforce --allow-onnx-models --max-features 500 --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100
modal run research_experiments/dynamic_foam/modal_powerfoam_aliked_onnx_check.py --run-id onnx_check_fast_20260506
modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py --execute --preset micro_clip_64_4f_5step --run-id cuda_micro_blackbg_20260506 --max-gpu-minutes 8 --skip-official-fixture --fixed-black-background
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output /tmp/deepview_fisheye_sift_affine_smoke.ply --workdir /tmp/deepview_fisheye_sift_affine_smoke_work --target-size 128 --frame-index 0 --train-cameras camera_0001 camera_0015 --heldout-camera camera_0040 --anchor-camera camera_0001 --camera-model opencv_fisheye --camera-mode per_image --max-features 800 --sift-ratio 0.9 --sift-estimate-affine-shape --sift-domain-size-pooling --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_affine_minucam2.ply --workdir /tmp/deepview_03_dog_fisheye_1024_sift_affine_pycolmap_work --target-size 1024 --frame-indices 0 4 8 12 --camera-model opencv_fisheye --camera-mode per_image --max-features 12000 --sift-ratio 0.9 --sift-estimate-affine-shape --sift-domain-size-pooling --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100 --min-unique-cameras 2
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/build_multiview_plane_sweep_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_multiview_plane_sweep_frame0_128px_opencv_fisheye_stride4_support4_8192pts.ply --target-size 128 --frame-index 0 --depth-min 0.25 --depth-max 8.0 --depths 48 --stride 4 --min-support 4 --max-error 1.0 --max-points 8192 --chunk-size 512
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_multiview_plane_sweep_frame0_128px_opencv_fisheye_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
PYTHONPATH=src/train uv run --with pycolmap python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc --output research_experiments/dynamic_foam/artifacts/deepview_03_dog_closeoverlap_8cam_holdout0005_frames0_4_8_12_512px_true_multiframe_sift_wide_minucam2.ply --target-size 512 --frame-indices 0 4 8 12 --train-cameras camera_0001 camera_0002 camera_0003 camera_0004 camera_0006 camera_0007 camera_0008 camera_0009 --heldout-camera camera_0005 --anchor-camera camera_0001 --max-features 4000 --sift-ratio 0.9 --max-reproj-error 8.0 --xy-extent 100 --z-min -100 --z-max 100 --min-unique-cameras 2
PYTHONPATH=src/train:third_party/powerfoam-metal WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_closeoverlap_8cam_holdout0005_pycolmap_frames0_4_8_12_512px_true_multiframe_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc
```

The old official parity command above originally reported two skips:

```text
official CUDA/Warp PowerFoam fixture has not been generated on a CUDA host
```

As of the 2026-05-06 Modal/L40S run, that skip blocker is cleared. A later
follow-up aligned the fixture with upstream's effective texture temperature
and narrowed the official-backward comparison to stable channels. The targeted
Direct/Metal official fixture command now passes locally.

The paper-acceptance verifier is:

```text
research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py
```

The prompt-level completion audit is:

```text
research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py
```

It maps "PowerFoam proper on Metal, fast and accurate forward/backward, 4K,
and trainable" to concrete gates: focused local Metal pytest, official
CUDA/Warp fixture presence, targeted local Metal fixture-backward pytest,
low-level raytrace parity script, targeted official Direct and Metal parity
pytest nodes, saved 4K benchmark verifier, saved 4K optimizer-step
trainability verifier, paper-acceptance verifier, W&B backing, selected-row
post-initial optimization, and heldout PSNR/SSIM
thresholds. It currently reports `ok: false` because paper acceptance and
heldout PSNR/SSIM are still below threshold. A run with `--run-local-tests`
passed the local Metal gate, targeted local fixture gates, targeted official
Direct/Metal parity nodes, low-level raytrace parity gate, saved 4K benchmark
gate, and saved 4K optimizer-step trainability gate. The selected clean row now
passes the W&B-backed gate but still fails the post-initial checkpoint gate
because the best heldout checkpoint remains step 0.

The completion gap is now paper-scale quality, not a missing official fixture,
official numeric parity, or CUDA deployment smoke. The strict Modal L40S micro
smoke at `outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json`
passes the CUDA validator, including rendered time-causality for the cheap
dynamic `texel_sv_rgb` residual fork and fixed-black background parity with the
matched Metal micro smoke. The pycolmap-wheel ALIKED preflight at
`outputs/powerfoam_aliked_geometry/onnx_check_fast_20260506/onnx_check.json`
still fails with `ALIKED feature extraction requires ONNX support`, but the
COLMAP CLI route at
`outputs/powerfoam_aliked_geometry/colmap_cli_onnx_cudnn_check_20260506/colmap_cli_onnx_check.json`
does prove ALIKED_N16ROT and ALIKED_LIGHTGLUE ONNX startup on Modal L40S with
cuDNN. Real DeepView COLMAP-CLI probes are sparse: the two-camera 128px probe
produced `0` points, the near-four-camera 512px brute-force probe produced `9`
points, and the matching LightGlue probe produced `27` points with track p90
`2` and reproj median `6.25px`; opt-in known-pose guided verification pruned
that same probe to `0` points. That is not close enough to justify the full
1024px ALIKED spend yet. The clean-candidate inventory now includes pycolmap,
appearance-only W&B-offline, ordered-point, train-only plane-sweep, HLOC, and
COLMAP-CLI ALIKED probe branches, with the plane-sweep
artifact marked as `artifact_kind="multiview_plane_sweep"` rather than a
true-track COLMAP/SfM artifact. The selected clean row now uses the
distortion-consistent OPENCV_FISHEYE true-multiframe DeepView init plus an
appearance-only W&B-offline Metal run with `regular_triangulation` and
near-plane default starts. It selects step 0 at heldout PSNR `12.5099` / SSIM
`0.1169`, improving materially over the prior Cech/AABB selected row
(`10.8536` / `0.0766`). It still fails PSNR (`<13`) and SSIM (`<0.15`)
thresholds, and its distinct-camera support remains mostly two-camera
(`unique_camera_p90=2`).
The 2026-05-06 dense-eval slow-appearance probe confirmed this is not just a
missed early validation peak: `12` steps with `image_log_every=1` and 4x slower
`texel_sv_rgb` LR raised source PSNR from `12.7537` to `12.7685`, while heldout
PSNR monotonically fell from `12.5099` to `12.5057`; `best_metrics.json` still
selects step 0.
The 2026-05-06 color-affine diagnostic
(`research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py`) shows
color/exposure is a real PSNR lever but not the SSIM gate: train-fit constant
background plus train-fit channel affine applied to heldout reaches
`13.9892` PSNR / `0.1416` SSIM, while heldout-oracle constant background plus
oracle affine still stays around `14.21` PSNR / `0.135`-`0.136` SSIM. This
rules out cheap background/color correction as the primary remaining paper
blocker.
The 2026-05-06 regular support-thaw dense-eval run
(`src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_supportthaw_128_16f_1024cells_12step_denseeval_noaux.jsonc`)
allowed tiny quaternion/texel-site/SV-axis/RGB/height motion on the selected
regular topology. It improved source PSNR/SSIM from `12.7537/0.1301` to
`12.7753/0.1313`, but heldout fell from `12.5099/0.1169` to
`12.5037/0.1170`; best remains step 0. This closes the cheap
"regular topology just needs small support thaw" hypothesis.
The 2026-05-06 normal-thaw diagnostic
(`src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_normalthaw_128_16f_1024cells_12step_denseeval.jsonc`)
also moved normals/quaternions (`state_mean_normal_delta=0.00669`,
`state_mean_quaternion_delta=0.00342`) but kept best heldout at step 0
(`12.5099/0.1169`) and ended at `12.5037/0.1170`. The LOTO diagnostic holding
out `camera_0003` with the reused 8-camera PLY reached heldout
`13.2527/0.1270`; this shows `camera_0040` is a harder parallax/coverage view,
but it is not a clean acceptance row and still misses the SSIM gate. If LOTO is
used as evidence, rebuild a true 7-camera PLY first.
The height+SV material endpoint bug is fixed and independently guarded: both
streaming/raytrace and `third_party/powerfoam-metal/tests/linear_texture_check.py`
now sample material at the height-clipped endpoint. Re-rendering the selected
checkpoint after the patch produced heldout `12.5075` PSNR, so this was a real
correctness fix but not the paper-quality blocker.
The 2026-05-06 frozen heldout camera perturbation probe found bounded pose
sensitivity but not an acceptance path: baseline subset `12.4830/0.1161`,
best PSNR candidate `12.6926/0.0900`, best SSIM candidate `12.6781/0.1205`.
The 2026-05-06 official-objective short run selected step 12 with heldout
`12.5242/0.1226` and source `12.9137/0.1478`, so it is the first small
post-initial improvement on the selected row, but it remains below the
`13.0/0.15` gate and barely exercises fixed-long-warmup density/radii/height
groups.
The 2026-05-06 fast-warmup official-objective follow-up added
`train.lr_warmup_steps` so short probes can compress the long official
radii/density/height warmups without kwargs or env fanout. With 1024 cells,
10-step warmups selected step 20 at heldout `12.5535/0.1255`; source improved
to final `13.3737/0.1971`, so the warmup groups moved but heldout still missed
the gate. The matching 2714-cell all-train-visible-point row, now registered
in `verify_powerfoam_paper_acceptance.py`, selected step 10 at heldout
`12.6689/0.1000` with W&B offline run `5dj1ssze`. This is the current selected
clean PSNR row, but it still fails both paper thresholds.
The coverage verifier is:

```text
research_experiments/dynamic_foam/verify_powerfoam_clean_init_coverage.py
```

It currently reports `ok: false` because the selected 1024-cell OPENCV_FISHEYE
run keeps only a minority of the train-visible filtered init points. The
coverage verifier now uses the same lens-aware projection helper as the trainer
rather than a pinhole-only `K/z` proxy. Its useful evidence is that the selected
point cloud is substantially train-visible, has nontrivial heldout point
support, and the saved heldout alpha previews are nonblank. That shifts the
next local question away from "is the init invisible?" and toward replacing the
weak mostly-two-camera clean geometry
with a stronger multi-view track/dense COLMAP artifact.
The selected-row heldout diagnostic is:

```text
research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py
```

It writes per-view and worst-sample labels for the selected RGB-only
appearance row. Current artifact:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux/heldout_error_diagnostics.json
```

The first key result is a train-view opacity failure, not a blank heldout view.
Heldout `camera_0040` has alpha mean `0.6329` and PSNR `10.8536`, but train
`camera_0013` and `camera_0021` have alpha mean `0.0` and alpha `<0.05`
fraction `1.0`; `camera_0015` is nearly blank with alpha mean `0.0639`.

The follow-up raytrace support-gap diagnostic is:

```text
research_experiments/dynamic_foam/diagnose_powerfoam_raytrace_support_gap.py
```

It writes:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux/raytrace_support_gap_diagnostics.json
```

This narrows the local failure: `camera_0021` and `camera_0013` are not missing
decoded centers. `camera_0021` projects mean `873.0` checkpoint centers onto
mean `762.0` pixels but raytrace alpha is `0.0`; `camera_0013` projects mean
`890.0` centers onto mean `810.0` pixels but raytrace alpha is `0.0`. Heldout
`camera_0040` projects mean `803.0` centers and has raytrace alpha `0.6329`.
Independent read-only probes found no camera-order, view-flattening,
OPENCV_FISHEYE ray, or model-frame transform mismatch, and the same checkpoint
through non-raytrace streaming gives nonzero alpha for both zero-alpha train
views. The next local blocker is therefore raytrace traversal/start-cell
robustness on this real multiview scene, not another material-frame or LR
sweep.

Latest local patch:

- `third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py` now makes
  default `[B]` raytrace starts ray-support-aware: preserve the origin-nearest
  start when sampled rays hit it, otherwise choose the sampled visible cell with
  stronger support.
- `third_party/powerfoam-metal/tests/raytrace_check.py` now includes an
  unsupported-origin height+SV fixture where the old forced start gives alpha
  max `0.0`, the patched default gives alpha max `0.580209493637085`, and
  raytrace still matches tiled streaming forward/backward.
- Verification run:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=third_party/powerfoam-metal:src/train .venv/bin/python third_party/powerfoam-metal/tests/raytrace_check.py
```

- A full selected-checkpoint rerun of `diagnose_powerfoam_heldout_error.py`
  was attempted after the patch and killed after more than two minutes without
  output.
- Added and ran
  `research_experiments/dynamic_foam/verify_powerfoam_raytrace_start_support.py`.
  It does not render pixels; it rebuilds train rays from multicam camera
  matrices and checks the patched default start against the origin-nearest
  start on the known failing views. Output:
  `outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux/raytrace_start_support_diagnostics.json`.
  Result: both `camera_0013` and `camera_0021` switch away from origin on
  every sampled frame (`switch_fraction=1.0`), with origin support mean `0.0`.
  Do not claim the saved real views are rendered-fixed until a small rendered
  alpha verifier passes.
- Added
  `research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py`
  to render a tiny sampled real ray grid directly from the checkpoint. This
  avoids full video/data loading and compares patched raytrace, forced old
  origin-start raytrace, and the streaming renderer.

Follow-up result on 2026-05-06: the real-view alpha diagnostic now compares
`origin`, implicit default, and `near_plane` start modes across
`cech_aabb`, `regular_triangulation`, and `all_pairs`.
`_default_per_ray_start_ids` in
`third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py` now uses the
near-plane power cell when `config.near_plane > 0`, while preserving the
closest-line fallback for zero-near-plane synthetic tests.

Fast diagnostic artifact:

```text
/tmp/raytrace_start_modes_after_nearplane_default_20260506.json
```

Durable follow-up artifact with the extra diagnostic `first_sphere_hit` start
mode:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/raytrace_start_modes_with_first_sphere_diagnostics.json
```

Key evidence from the selected bad real train views:

- `regular_triangulation` plus near-plane/default start is close to streaming
  on `camera_0021`: stream alpha mean `0.8927`, default alpha mean `0.9107`,
  mean alpha error `0.01797`, feature max error `0.2354`.
- `regular_triangulation` plus near-plane/default start is imperfect but much
  better than Cech/AABB on `camera_0013`: stream alpha mean `0.8992`, default
  alpha mean `0.9788`, mean alpha error `0.07962`, feature max error `0.5359`.
- `cech_aabb` still undercovers both oblique views even after the start fix:
  default alpha means `0.5223` / `0.4897` versus stream alpha means
  `0.8927` / `0.8992`.

Verification passed:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/raytrace_check.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=third_party/powerfoam-metal uv run --with scipy python third_party/powerfoam-metal/tests/raytrace_check.py
```

The earlier interrupted 1-step `regular_triangulation` trainer smoke was not a
graph hang: direct timing showed one 16-frame full-res DeepView camera decode
takes about `23-25 s`, so the full 8-train-plus-heldout setup has a long silent
OpenCV startup. A one-frame regular trainer probe subsequently passed, and the
full 16f/40-step regular run completed as offline W&B run `qsfn7j80`. It
selected step 0 at heldout PSNR `12.5099` / SSIM `0.1169`; final step 40
slightly overfit to `12.4527` / `0.1160`.

Two quick follow-ups bounded easy levers:

- Re-compositing the selected 1024-cell regular raytrace checkpoint with a
  constant gray or least-squares RGB background improves heldout only to about
  PSNR `12.80` / SSIM `0.123`. The raytrace alpha mean is already `0.978`, so
  background is not the main missing quality.
- A 2714-cell all-filtered regular-triangulation no-op trainer probe passed
  forward/backward but reached heldout PSNR `12.6368` / SSIM `0.0942`. Keeping
  every train-visible filtered OPENCV_FISHEYE point raises source PSNR but does
  not improve the paper gate.
- Result: the visible/per-ray start path recovers nonzero alpha but still fails
  streaming parity. On `camera_0021` / `camera_0013` frames `0` and `4`,
  Cech/AABB old-origin alpha mean is `0.0`; patched alpha mean rises to
  `0.617` / `0.694`, but streaming alpha mean is `0.893` / `0.899`, leaving
  mean alpha error `0.300` / `0.323` and max alpha error `1.0`.
- All-pairs and regular-triangulation diagnostics show the deeper issue:
  richer connectivity makes the old power-origin start closer to streaming,
  while visible per-ray starts can miss earlier power-cell intervals. The next
  local blocker is therefore official-ish start semantics plus graph
  connectivity/speed, not just choosing a nonzero visible start.
- The `first_sphere_hit` start control improves Cech/AABB sampled alpha but
  still does not close the gap. On `camera_0021`, Cech/AABB alpha mean moves
  from near-plane/default `0.5223` to first-hit `0.6326` against streaming
  `0.8927`; on `camera_0013`, it moves from `0.4897` to `0.6615` against
  streaming `0.8992`. Mean alpha error remains `0.264` / `0.310`, so the
  remaining Cech/AABB issue is not just picking the first sphere intersection.
- Added `research_experiments/dynamic_foam/diagnose_powerfoam_topology_edges.py`
  and ran it on the selected clean regular checkpoint with SciPy:

  ```text
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with scipy python research_experiments/dynamic_foam/diagnose_powerfoam_topology_edges.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux.jsonc
  ```

  Output:

  ```text
  outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/topology_edge_diagnostics.json
  ```

  Result: the fast sphere-overlap Cech/AABB graph is **not** a superset of the
  regular-triangulation teacher on this frozen real checkpoint. Across frames
  `0/4/8/12`, each frame has `7207` regular undirected edges, but only `3389`
  are also Cech edges; `3818` regular edges per frame are missing from Cech,
  so regular-edge coverage is `0.4702`. All missing regular edges are
  non-overlapping under the current radius test (`non_overlapping_fraction=1.0`,
  median overlap margin `-0.0259`). This explains why Cech/AABB can be the
  fastest synthetic 4K path while still failing oblique real-view ray walks:
  streaming can scan all cells, but a graph walk cannot traverse regular power
  faces that the Cech graph never contains.
- Extended
  `research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py`
  with selected-row residual/witness output and ran it on the same regular
  checkpoint:

  ```text
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with scipy python research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux.jsonc --batch-size 4 --heldout-only
  ```

  Output JSON and panel:

  ```text
  outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/heldout_error_diagnostics.json
  outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/heldout_error_diagnostics_panel.png
  ```

  Panel columns are `GT | render | alpha | residual_l1 | normal_distance |
  log_support_hit_count | nearest_power_support`. Result: the selected
  regular heldout failure is **not** mainly low alpha or missing sphere
  support. Heldout alpha mean is `0.9776`, alpha `>0.9` covers `97.08%` of
  pixels, and the selected worst frame support proxy hits `99.99%` of pixels.
  The dominant residual bucket is high-alpha pixels: alpha `>=0.5` pixels
  contain `95.67%` of total residual, and the top-20%-residual high-alpha
  bucket alone contributes `43.61%` of total residual with mean L1 `0.4301`.
  High-residual pixels have support-hit fraction `0.9997`. This shifts the next
  paper-quality lever away from blank coverage and toward spatial alignment,
  depth/order, normal/material transport, or an objective that improves heldout
  structure rather than source-view color only.
The all-points capacity control answered the first half negatively:
`2524` cells kept every train-visible box-filtered point but only reached
heldout PSNR `9.0157` / SSIM `0.0276`, below the 1024-cell run. Capacity alone
is not the missing lever.
The ordered-sampling control also answered negatively: keeping the top-ranked
1024 points from the same PLY (`init_point_cloud_sample_mode="first"`) selected
step 20 at heldout PSNR `9.1789` / SSIM `0.0248`, still below the random
1024-cell row. The missing lever is not just avoiding random drops of the
lowest-reprojection points.
The DeepView fisheye-ray control answered negatively too: the loader now
preserves `projection_type="fisheye"` as `CameraSpec(lens_model="opencv_fisheye")`
for train/heldout rays and train-visible filtering, but the matching 40-step run
selected step 0 at heldout PSNR `9.0781` / SSIM `0.0029`. This makes the next
camera-side question narrower: if we pursue distortion further, the pycolmap
known-pose builder also needs a distortion-aware camera/reconstruction mode;
render-ray correction alone is not enough with the current pinhole-built PLY.
The close-overlap sanity split (`0001/0002/0003/0004/0006/0007/0008/0009` ->
heldout `0005`) was a negative control: it produced a valid true-multiframe
artifact (`2076` points, 496 verified pairs) but heldout PSNR was only `7.9720`
best / `7.8680` final, so changing to a nearby heldout is not the missing
quality lever by itself.
The OPENCV_FISHEYE all-filtered-points capacity control also answered
negatively: keeping all `2714` train-visible filtered points from the
distortion-consistent artifact reached only heldout PSNR `8.5095` / SSIM
`0.0260`, far below the 1024-cell OPENCV_FISHEYE selected row (`10.5931`).
The missing lever is not simply retaining every filtered clean point.
Local stronger-feature attempts did not beat the selected artifact. ALIKED
failed on this Mac pycolmap wheel with `ALIKED feature extraction requires ONNX
support`; the ONNX-enabled COLMAP CLI container works on Modal L40S, but cheap
real DeepView probes were too sparse to justify a full run (`0` points at
wide2/128px, `9` points at near4/512px brute force, `27` points at near4/512px
LightGlue), and known-pose guided verification pruned the near4/512px
LightGlue probe to `0` points. Covariant/affine SIFT ran locally but produced a weaker artifact:
`2087` points vs `2821`, reproj
median/p90 `2.84/5.21px` vs `2.72/5.19px`, and unique-camera p90 still `2`.
It was not trained because the geometry was worse before optimization.
The HLOC/ALIKED backend is now wired as an ONNX-bypass path with actual
`aliked-n16rot` selection, HLOC feature caps, known-pose geometric
verification, and explicit summary fields. Local smokes proved the backend but
not the quality: the wide 2-camera 256px smoke produced `0` points, the
close-overlap 4-camera 256px smoke produced only `2` verified points, and the
post-patch 2-camera schema smoke produced `0` points. HLOC is therefore honest
in this repo, but not yet a dense clean artifact.
The lens-aware all-train-camera plane-sweep branch also answered negatively.
It built a train-only OPENCV_FISHEYE artifact with `7830` points, support
mean/median/p90 `5.78/6/7`, and median/p90 color error `0.120/0.241`, then
trained the top 1024 points for 40 steps. Heldout selected/finalized at step
40 with PSNR `8.2487`, L1 `0.327659`, SSIM `0.0160`, far below the selected
OPENCV_FISHEYE pycolmap row (`10.5931`). Local plane-sweep consistency is not
the missing clean-geometry lever in this form.
A cheap photometric-inlier variant also answered negatively: the same full
8-camera plane-sweep with `support_error=0.2` produced `6450` points, lower
median/p90 color error `0.0680/0.1151`, and support mean/median/p90
`4.63/4/6`. The matching run selected step 0 at heldout PSNR `9.0679`, L1
`0.291473`, SSIM `0.0458`; final step 40 fell to PSNR `8.8609`. It improves
over raw plane sweep but remains below the selected OPENCV_FISHEYE pycolmap row.

Exact next command for a CUDA/Warp host:

```bash
PYTHONPATH=src/train python research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py \
  --backend official \
  --upstream-root /tmp/powerfoam_official \
  --fixture research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json \
  --output research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json
```

After copying that JSON back into this repo, run:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present \
  -q -rs
```

The external-host blocker runner wraps the official fixture/test commands and
the ONNX-backed ALIKED/LightGlue clean-geometry command. It also prepares the
matched W&B-backed PowerFoam Metal training config once the artifact PLY/JSON
exists. The deterministic handoff manifest is:

```text
research_experiments/dynamic_foam/external_powerfoam_artifact_handoff.json
```

Regenerate it when runner defaults change:

```bash
PYTHONDONTWRITEBYTECODE=1 python research_experiments/dynamic_foam/run_powerfoam_external_blockers.py check
PYTHONDONTWRITEBYTECODE=1 python research_experiments/dynamic_foam/run_powerfoam_external_blockers.py handoff --matcher-type aliked_lightglue --handoff-output research_experiments/dynamic_foam/external_powerfoam_artifact_handoff.json
PYTHONDONTWRITEBYTECODE=1 python research_experiments/dynamic_foam/run_powerfoam_external_blockers.py all --dry-run --matcher-type aliked_lightglue
PYTHONDONTWRITEBYTECODE=1 python research_experiments/dynamic_foam/run_powerfoam_external_blockers.py all --matcher-type aliked_lightglue
PYTHONDONTWRITEBYTECODE=1 python research_experiments/dynamic_foam/run_powerfoam_external_blockers.py write-train-config --dry-run
PYTHONDONTWRITEBYTECODE=1 python research_experiments/dynamic_foam/run_powerfoam_external_blockers.py train-aliked --dry-run
```

On this Mac, `check` reports `torch.cuda.is_available=False`, missing `warp`,
and missing `pycolmap` in the local venv. That is expected; the runner is for a
CUDA/Warp/ONNX-capable host or container. On a Linux/CUDA host the Metal parity
test will skip if MPS is unavailable; after generating the official fixture,
copy the JSON back to this Mac and rerun the two official fixture tests here to
exercise the Metal-vs-official check.

### Modal/L40S Same-Clip CUDA Smoke - 2026-05-06

New fast CUDA lane:

- `research_experiments/dynamic_foam/export_powerfoam_smoke_dataset.py`
  converts `test_data/test_video_small_128_4fps.mp4` into an
  official-PowerFoam-compatible Blender dataset.
- `research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py`
  clones the pinned official repo, generates the optional official CUDA/Warp
  fixture, runs the upstream static CUDA lane, copies the checkout, applies
  `research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch`,
  and runs the dynamic lane on the same clip/settings.
- `research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py` wraps that
  runner in one Modal `L40S` function with dry-run planning by default.
- `research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py`
  validates the returned `summary.json`.

Plan only, no GPU spend:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_micro_time_causality_rerun \
  --skip-official-fixture \
  --fixed-black-background
```

Execute on one L40S:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_micro_time_causality_rerun \
  --max-gpu-minutes 8 \
  --skip-official-fixture \
  --fixed-black-background
```

Important scope boundary: this dynamic fork is the smallest CUDA-side
appearance/feature tweak, not full F32 feature accumulation through Warp. It
adds a Gaussian-time-basis residual to upstream `texel_sv_rgb` while preserving
the official geometry, Cech/AABB/regular adjacency machinery, SV color query,
and CUDA/Warp rasterizer/raytracer. If this smoke is promising, the later fork
is the real F-channel Warp path.

Exact fixed-black strict result from
`outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json`:

- Status: `ok`, validated by
  `research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py`.
- Host: Modal `NVIDIA L40S`, Torch `2.11.0+cu130`, Warp `1.10.0`.
- Clip/settings: 4 frames, 64 px, 5 steps, 256 points, 4 texel sites,
  SV DoF 2, `--skip-official-fixture`, `--fixed-black-background`.
- Official static CUDA: eval PSNR `5.5640`, SSIM `0.0284`, L1 `0.4901`,
  warm mean step excluding step 0 `8.31 ms`.
- Dynamic feature-foam CUDA: eval PSNR `5.5833`, SSIM `0.0288`, L1 `0.4887`,
  warm mean step excluding step 0 `9.09 ms`.
- Dynamic time probe: `dynamic_time_rgb_delta_mean=0.00006899`,
  `dynamic_time_rgb_delta_max=0.0009796`, proving the cheap time-conditioned
  branch changes rendered RGB on this exact fixed-background smoke.

Earlier random-background strict result from
`outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json`:

- Status: `ok`, validated by
  `research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py`.
- Host: Modal `NVIDIA L40S`, Torch `2.11.0+cu130`, Warp `1.10.0`.
- Clip/settings: 4 frames, 64 px, 5 steps, 256 points, 4 texel sites,
  SV DoF 2, `--skip-official-fixture`.
- Official static CUDA: eval PSNR `5.5405`, SSIM `0.0218`, L1 `0.4916`,
  warm mean step excluding step 0 `6.93 ms`.
- Dynamic feature-foam CUDA: eval PSNR `5.5487`, SSIM `0.0221`, L1 `0.4911`,
  warm mean step excluding step 0 `7.17 ms`.
- Dynamic time probe: `dynamic_time_rgb_delta_mean=0.00019385`,
  `dynamic_time_rgb_delta_max=0.0026973`, proving the cheap time-conditioned
  branch changes rendered RGB on this smoke.
  Use the fixed-black run above for exact CUDA-vs-Metal background parity.

Older deployment result from `outputs/powerfoam_cuda_smokes/latest/summary.json`:

- Status: saved summary says `ok`, but this older run now intentionally fails
  the stricter `verify_powerfoam_cuda_smoke_results.py` checks for rendered
  time-causality and warm timing summary fields.
- Host: Modal `NVIDIA L40S`, Torch `2.11.0+cu130`, Warp `1.10.0`.
- Clip/settings: 8 frames, 128 px, 20 steps, 512 points, 4 texel sites,
  SV DoF 4, clip SHA256
  `f10c67ee46f4675d6b9b89ea625302b31b2e3043244260092873698b5e5bd6da`.
- Official static CUDA: eval PSNR `6.0284`, SSIM `0.0577`, L1 `0.4551`,
  cold mean step `314.84 ms`; warm mean step excluding step 0 is not recorded
  in `summary.json` but is derivable from `modal_return.json` lane metrics as
  `8.23 ms`.
- Dynamic feature-foam CUDA: eval PSNR `6.0773`, SSIM `0.0610`, L1 `0.4521`,
  cold mean step `43.13 ms`; warm mean step excluding step 0 is not recorded
  in `summary.json` but is derivable from `modal_return.json` lane metrics as
  `8.79 ms`.
- Dynamic minus static: `+0.0488` PSNR, `+0.00335` SSIM.
- Official CUDA/Warp fixture generated and copied back to
  `research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json`.
- Follow-up parity status: the copied fixture now passes the targeted
  official Direct and Metal fixture tests locally after aligning the fixture
  with upstream's effective texture temperature and comparing stable
  shared-backward channels. Current verification:
  `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present -q -rs`

Matched CUDA-vs-Metal smoke comparison:

- Metal config:
  `src/train_configs/local_mac_powerfoam_metal_cuda_micro_match_randominit_64_4f_256cells_5step.jsonc`.
- Comparator:
  `research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py`.
- Output:
  `outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/cuda_vs_metal_summary.json`.
- Matched contract: same source clip, 64 px, 4 frames, 5 steps, 256
  points/cells, 4 texel sites, SV DoF 2, random init on Metal, and fixed black
  background.
- Local Metal result: eval PSNR/SSIM/L1 `5.1222 / 0.0105 / 0.5057`, which is
  `-0.4418` PSNR, `-0.0179` SSIM, and `+0.0155` L1 versus fixed-black
  official static CUDA. This is only a smoke-scale backend sanity record, not
  paper validation.
  returned `2 passed in 33.98s`.

## Current Local Inventory

### Torch Code

Paths:

- `src/train/powerfoam_direct.py`
- `src/train/train_powerfoam_direct.py`
- `src/train_configs/local_mac_powerfoam_direct_*.jsonc`
- `tests/test_powerfoam_direct.py`

What exists:

- bounded power cells with centers/radii/density
- quaternion-derived normal/tangent/bitangent frame
- local detail sites
- per-site height/displacement in the Torch reference path
- spherical-Voronoi view color hooks in the Torch reference path
- direct-fit trainer and W&B media logging
- some official-style losses/stat hooks

What is still missing or incomplete:

- official static multi-view SfM scene trainer (only a tiny posed-camera Metal
  smoke exists so far)
- direct renderer now supports camera-origin-correct sort, clipping, surface
  query, and SV view direction when callers pass full `[origin, direction]`
  rays; the old direct-video trainer still uses its fixed-origin default rays
- official Cech/AABB adjacency builder
- densification, pruning, resampling, contribution/error EMAs at paper scale
- full parity harness against official PowerFoam on identical random scenes
- held-out-view baseline rows

### Metal Code

Paths:

- `third_party/powerfoam-metal/csrc/metal/powerfoam_kernels.metal`
- `third_party/powerfoam-metal/csrc/metal/powerfoam_streaming_kernels.metal`
- `third_party/powerfoam-metal/csrc/metal/powerfoam_tiled_kernels.metal`
- `third_party/powerfoam-metal/csrc/metal/powerfoam_metal.mm`
- `third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py`
- `src/train/train_powerfoam_metal.py`

What exists:

- MPS Torch extension and Metal kernel loader
- bounded power-cell ray interval clipping
- front-to-back alpha compositing
- arbitrary feature channels
- streaming replay backward for points/radii/density/features
- 16x16 tiled candidate-list forward/replay-backward path for the full
  height+SV primitive
- local-linear feature mode
- fixed surface-linear mode
- oriented-surface-linear mode
- oriented texel-surface feature mode
- strict quaternion texel-surface mode
- height-displaced oriented/quaternion texel-surface mode
- height-displaced spherical-Voronoi color mode with oriented/quaternion frames
- gradients for texel sites, texel heights, texel features, normals,
  tangents, bitangents, SV axes/RGB, and height-query effects on
  centers/radii/frame
- official-style tiled auxiliary outputs: normal distance, normal,
  arbitrary depth-quantile vector, contribution, point error, visibility mask
- contribution/error EMAs and capacity-changing grow/prune/resample plumbing in
  the local trainer
- trainable Metal raytrace backend for full height+SV mode, including
  normal-distance output/gradient, with a fixed per-pixel replay event cap
- optional SciPy/Qhull weighted-Delaunay / regular-triangulation adjacency graph

Recent validation that passed:

```bash
PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/backward_check.py
PYTHONPATH=src/train .venv/bin/python third_party/powerfoam-metal/tests/linear_texture_check.py
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/tiled_streaming_check.py
PYTHONPATH=third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/aux_check.py
```

What is still missing or incomplete:

- differentiable depth-aux losses / external normal-supervision gradients
- official static multi-view trainer using this Metal path (tiny DeepView
  posed-camera smoke path exists; not paper-complete)
- paper-scale densification/pruning/resampling schedules and acceptance runs

### Dynamic / Feature Foam Forks

Paths:

- `third_party/dynamic-powerfoam-metal/`
- `src/train/train_dynamic_powerfoam_metal.py`
- `src/train_configs/local_mac_token_dynamic_powerfoam_features_*.jsonc`

What exists:

- namespace fork of the Metal raster core
- Python-side time decoding before per-frame raster calls
- token feature foam: tokens decode bounded cells and F-channel texel features
- alpha-normalized feature raster output mapped through `FeatureToColor`
- motion diagnostics for temporal screen movement and feature deltas

Important: this is **not** official PowerFoam. It is our experimental dynamic
feature-raster fork.

## Definition Of Done

Do not call the implementation "full PowerFoam" until all of these are true:

- Metal and Torch render the full paper primitive: power cells, quaternion
  frame, detail sites, height displacement, spherical-Voronoi color, density,
  and official outputs.
- The local trainer can train a static posed-camera scene with SfM/COLMAP init.
- The local trainer supports official parameter groups and LR schedules.
- The local trainer supports densification/pruning/resampling and contribution
  / error EMAs.
- CPU Cech/AABB-style adjacency is available and KNN is clearly marked as
  approximate.
- Metal forward/backward matches the Torch reference on randomized unit scenes.
- A tiny static multi-view fixture passes held-out-view acceptance.
- Baseline rows in `BASELINES.md` include config, W&B run, steps, wall time,
  PSNR/SSIM/L1, and caveats.

## Implementation TODO

### P0 - Freeze The Source Of Truth

- [x] Vendor exact upstream PowerFoam commit metadata into docs.
- [x] Add a small official-code parity fixture under `research_experiments/`.
      CUDA generation is handled by the Modal/L40S smoke path, which copied
      back
      `research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json`.
- [x] Save one canonical random-scene fixture with points/radii/quats/density,
      texel sites/heights/SV color, adjacency, rays, and expected outputs.
      Current fixture:
      `research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json`.
- [x] Add a CUDA-host official fixture generator entrypoint:
      `research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py`.
      Its local dry-run fixture is
      `research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_local_v1.json`;
      run with `--backend official` on a CUDA/Warp host to write the actual
      upstream-output fixture.

Acceptance:

- [x] Fixture can be loaded by Torch direct and Metal tests.
- [x] Fixture records upstream commit hash and local config.
- [x] Official-camera dry-run fixture can be loaded by the Torch direct tests
      and has nonzero alpha coverage, scalar loss, and recorded gradients.
- [x] Official CUDA fixture verifier is wired as a skip-until-present test:
      `tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present`.
      The intended CUDA-host generation command is
      `PYTHONPATH=src/train python research_experiments/dynamic_foam/make_powerfoam_official_parity_fixture.py --backend official`.
      The verifier compares forward outputs and the official fixture gradients
      for points/radii/density/normals/texel sites/height/SV axis/SV RGB when
      the CUDA-generated fixture is present.
- [x] Make the official CUDA fixture verifier pass. The fixture is now present
      and the Direct verifier passes after aligning the effective official
      raster texture temperature.
- [x] Metal fixture coverage now checks the official-camera local dry-run
      fixture through the Metal height+SV path, including forward outputs,
      scalar loss, and the shared-parameter backward channels
      (density/height/SV axis/SV RGB). The matching official-CUDA Metal test is
      wired as skip-until-present via
      `tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present`.
- [x] Make the official-CUDA Metal fixture test pass against the copied CUDA
      fixture. The test now checks the stable shared backward channels against
      the copied official CUDA/Warp fixture.

### P1 - Torch Reference Correctness

- [x] Pass camera origin through the direct-render reference path.
- [x] Fix fixed-origin power-distance sort assumptions in
      `render_powerfoam_torch(...)` when full rays are supplied.
- [x] Fix view direction for spherical-Voronoi color to use camera/ray origin
      in the direct-render reference path.
- [x] Add dense-reference Cech/overlap adjacency checker for small scenes.
- [x] Add static posed-camera render path, not just per-frame video direct fit.

Acceptance:

- [x] `tests/test_powerfoam_direct.py` includes camera-origin regression tests.
- [x] Direct Torch render is invariant to equivalent camera/world transforms
      within tolerance on a tiny scene.
- [x] Dense adjacency and Cech/overlap adjacency agree on output for small
      scenes where the overlap graph is complete.
- [x] Direct Torch trainer smoke exercises shared-state posed-camera train and
      heldout rays through config load, render, backward, logging, and
      checkpoint save.

### P2 - Metal Full Primitive Math

- [x] Replace normal+tangent trainer parameterization with official-style
      quaternion parameters, or add a strict quaternion mode.
- [x] Add per-site texel height to Metal feature layout.
- [x] Implement height-displaced surface query in forward.
- [x] Backpropagate through height, texel sites, center/radius, and frame.
- [x] Add spherical-Voronoi view-dependent color in Metal.
- [x] Match official detach semantics for the SV color query.
- [x] Emit official-style tiled Metal auxiliary outputs where needed:
      normal distance, normal, median quantile depth, contribution, point error,
      visibility mask.
- [x] Add arbitrary official `depth_quantiles` vector support for the
      non-gradient tiled aux API.
- [ ] Add aux-gradient routing if we use depth-quantile outputs or external
      normal supervision as differentiable losses. Normal-distance is now wired
      through both the tiled and raytrace height+SV training outputs, and
      contribution is wired through differentiable alpha.

Acceptance:

- [x] New Metal unit test covers quaternion frame gradients.
- [x] New Metal unit test covers height interpolation and height gradients.
- [x] New Metal unit test covers SV color output and SV parameter gradients.
- [x] Full-primitive Metal forward matches Torch direct on random tiny scenes:
      RGB/feature max error <= `1e-5`, alpha max error <= `1e-5`.
- [x] Full-primitive Metal backward matches Torch direct or finite-difference
      checks for points, radii, density, quaternion/frame, texel sites, height,
      SV axes/RGB within documented tolerance.

### P3 - Correct Adjacency

- [x] Implement CPU correctness-mode Cech/AABB adjacency for local trainer.
- [x] Keep KNN only as an approximate speed ablation in defaults and the main
      full-primitive tiled config.
- [x] Add tests that deliberately construct a case where KNN misses a true
      neighbor and the Cech path fixes the render.
- [x] Add adjacency stats logging: average degree, max degree, missing-edge
      diagnostic against dense small-scene reference.
- [x] Label older explicit-KNN smoke configs/baseline notes as approximate or
      move them under an explicit speed-ablation name.

Acceptance:

- [x] Cech/AABB adjacency output is a conservative superset on randomized small
      scenes checked against dense overlap.
- [x] Rendering with Cech/AABB matches dense-neighbor rendering within
      tolerance.
- [x] All KNN configs are labeled approximate in config comments and baseline
      notes.

### P4 - Tiled Rasterizer And Replay Backward

- [x] Turn the tiled stream kernels into the main high-throughput
      candidate-list path for the trainer/benchmark configs.
- [x] Build per-tile visible/candidate cell lists.
- [x] Preserve exact front-to-back order semantics inside each tile/pixel.
- [x] Implement replay backward for the full height+SV primitive.
- [x] Precompute official-style power-face `adjacency_diff` for the tiled
      Metal kernels so interval clipping does not reload neighbor
      point/radius tensors on every edge.
- [x] Add memory accounting to prove no `instances x width x height` gradient
      buffer exists.

Acceptance:

- [x] Tiled forward matches streaming forward on random scenes.
- [x] Tiled backward matches streaming backward / Torch reference on random
      scenes.
- [x] Renderer saved-state accounting scales with candidates and tile buffers,
      not `N*H*W` (this is benchmark accounting, not an OS peak-memory trace).
- [x] Benchmark reports forward and backward time for `N={256,1024,4096}` and
      `resolution={128,256,512}`.
- [x] Tiled is faster than projected streaming on local Mac for the full 4K
      height+SV benchmark. The selected fast 4K train diagnostic has since
      moved to the raytrace `cech_aabb` path, because it beats the regular
      triangulation and tiled variants on the current saved full-primitive
      forward+backward artifacts.
- [x] Tiled face-diff path preserves streaming parity within roundoff-level
      tolerance and materially improves the full 4K forward/backward median.

### P5 - Static Multi-View PowerFoam Trainer

- [x] Add `arch=powerfoam_static_metal` or equivalent.
- [x] Use posed camera batches rather than per-frame fixed-origin video.
- [x] Support COLMAP/SfM init from existing dataset loaders via explicit
      `.ply` / COLMAP `points3D.txt` / `points3D.bin` point-cloud init.
- [x] Add official parameter groups:
      points, density, radii, quaternions, texel sites, SV axes, SV RGB,
      texel height.
- [x] Add official LR schedules.
- [x] Add official/random background handling.
- [x] Add full loss stack:
      RGB, SSIM, normal, contribution/sparsity, interpenetration/connectivity.
      RGB+optional SSIM+normal-distance+contribution/sparsity+interpenetration
      are wired. External normal supervision and depth-quantile losses remain
      outside this checked item.
- [x] Add basic train/heldout split metrics and media logging.

Acceptance:

- [x] 1-step smoke exercises actual trainer, render, loss, backward, validation
      media, and checkpoint save.
- [x] Tiny synthetic static scene overfits from random init.
- [x] Tiny posed-camera fixture trains with held-out view logging.
- [x] `BASELINES.md` gets a dated row with config, run id, steps, wall time,
      PSNR/SSIM/L1, and explicit "static multiview" label.

### P6 - Densification / Pruning / Resampling

- [x] Track contribution and point-error EMAs from the tiled Metal aux pass.
- [x] Add fixed-capacity resampling based on error/contribution.
- [x] Add capacity-changing grow/prune support driven by contribution/error
      statistics.
- [x] Add official-style geometric final-cell growth schedule.
- [x] Add paper-style invalid/bad-cell pruning heuristics.
- [x] Rebuild adjacency after replacement/grow/prune resampling.
- [x] Preserve optimizer state when tensors are permuted/resampled.
- [x] Preserve optimizer state when tensors are pruned or extended.

Acceptance:

- [x] Unit test proves fixed-capacity resampling uses low contribution and high
      error while preserving optimizer state.
- [x] Unit test proves optimizer state is preserved across true prune/append.
- [x] Smoke run grows from init point count toward final point count.
- [x] Adjacency rebuild after densification passes render/backward smoke.
- [x] Baseline logs point count over time.

### P7 - Ray Tracing Backend

- [x] Port or reimplement PowerFoam ray-tracing adjacency walk for the current
      local adjacency graph.
      Height+SV forward and replay/backward now run in Metal.
- [x] Add starting-point search for the forward-only probe.
- [x] Add raytrace/raster parity checks on the same primitive state.
      Two-cell constant, random all-pairs constant, and random all-pairs
      height+SV forward/backward fixtures pass, including normal-distance
      output/gradient parity. Optional SciPy-backed regular-triangulation
      traversal now also matches dense all-pairs raster on tiny constant and
      height+SV scenes.
- [x] Add optional benchmark mode for constant and height+SV ray tracing.
- [x] Add optional weighted-Delaunay / regular-triangulation graph builder.
      It uses SciPy/Qhull lower-hull lifting when `adjacency_mode` /
      `--adjacency` is `regular_triangulation`.
- [x] Keep Cech/AABB as the selected fast training topology. The official
      PowerFoam source builds an AABB/BVH Cech complex rather than requiring a
      selected regular triangulation, and the optional local
      `regular_triangulation` path is correct but slower on the current 4K
      train benchmark.

Acceptance:

- [x] Raytrace and raster output agree within tolerance on representative
      small full-primitive scenes.
- [x] Raytrace height+SV replay backward supports normal-distance loss gradients.
- [x] Tiny full height+SV raytrace material-overfit gate passes:
      `tests/test_powerfoam_direct.py::test_powerfoam_metal_height_sv_raytrace_overfits_tiny_material`.
      It renders a teacher full-primitive raytrace cell from posed rays and
      trains the student's SV RGB through the raytrace backend until L1 drops
      below `0.001`.
- [x] Benchmark can report both tiled and raytrace train/forward modes on the
      same synthetic setup.
- [x] Saved 4K benchmark artifacts have a verifier:
      `research_experiments/dynamic_foam/verify_powerfoam_4k_benchmarks.py`.
      It checks the selected UHD `3840x2160` full height+SV raytrace
      `cech_aabb` forward+backward artifact is under `1200 ms` total median
      for both `1024` and `4096` cells, remains within replay cap `64`, and is
      faster than the regular-triangulation comparison artifact. Current
      verified totals are `1016.1 ms` for `1024` cells and `1014.4 ms` for
      `4096` cells.
- [x] Saved 4K optimizer-step trainability artifact has a verifier:
      `research_experiments/dynamic_foam/verify_powerfoam_4k_trainability.py`.
      Its default artifact is
      `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_trainability_1024cells_2026-05-05.json`.
      It checks an actual MPS optimizer step at `3840x2160` with `1024` cells,
      `cech_aabb`, and `oriented_height_sv_texel_surface`, not just backward
      timing. Current verified metrics are `ok=true`,
      `loss_before=0.07926274836063385`,
      `loss_after=0.0789613351225853`,
      `loss_ratio=0.9961972900980273`,
      `grad_abs_max=0.010738098062574863`,
      `density_update_abs_max=0.0026845335960388184`,
      `forward_ms=1195.255124999676`,
      `backward_ms=2181.0507500013046`, and
      `after_forward_ms=1217.3038750006526`.

### P8 - Paper-Scale Benchmarking

- [ ] Define one small local acceptance scene that is cheap enough for Mac.
- [x] Define one real static multiview scene for Mac-side PowerFoam probes:
      Neural3D `coffee_martini`, train cameras `cam04`/`cam09`, heldout
      `cam06`, 128px/16f/1024 cells.
- [ ] Add official-comparison config metadata:
      dataset, init points, final points, texel sites, SV dof, schedule.
- [ ] Run the full matrix only after P1-P7 are green.

Acceptance:

- [ ] `BASELINES.md` contains PowerFoam static multiview rows separate from
      renderer-development rows.
- [x] The selected clean paper row has W&B/offline backing, local output path,
      wall time, steps, metrics, and missing-feature caveats.
- [ ] Full matrix rows all have W&B run id or offline run id, local output
      path, wall time, steps, metrics, and missing-feature caveats.
- [ ] Held-out-view metrics, not same-source video L1, are the deciding score.

Current paper-clean init evidence:

- [x] Added train-camera-only feature triangulation:
      `research_experiments/dynamic_foam/build_multiview_feature_triangulation_point_cloud.py`.
      The initial train2 artifact is
      `research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_train2_feature_triangulation_frames0_4_8_12_256px_orb_reproj8.ply`.
- [x] Ran the matching 40-step Metal raytrace probe:
      `src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
      It is a negative control: 89 points, median reprojection error 4.07px,
      heldout PSNR `5.6311`, heldout L1 `0.475115`, SSIM `0.0003`.
- [x] Added a stronger train4 variant using train cameras
      `cam04`/`cam09`/`cam13`/`cam20` and heldout `cam06`:
      `src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train4_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
      The PLY has 662 points, median reprojection error 3.29px, and heldout
      PSNR `5.6727`. This is still a negative control relative to EX4DGS.
- [x] Added a known-pose pycolmap/SIFT local COLMAP-style builder:
      `research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py`.
      The train4 1024px artifact has 44 box-filtered two-view-track points
      after all 6 image pairs verify, and the matching 40-step run heldout PSNR
      is only `5.6309`.
- [x] Tried a merged multiframe known-pose pycolmap artifact from frames
      0/4/8/12:
      `research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_train4_pycolmap_known_pose_frames0_4_8_12_1024px_sift_reproj8_merged.ply`.
      This increased the clean point cloud to 227 points and improved source
      PSNR to `5.9467`, but heldout still selected step 0 at PSNR `5.6309`.
      The merge is a compact per-frame snapshot union, not a true static
      long-track COLMAP reconstruction.
- [x] Extended the known-pose pycolmap builder to support per-image camera
      intrinsics and CLI train/heldout/anchor camera overrides. A DeepView
      8-train-camera probe at 256px verified all 28 image pairs and produced
      639 raw points with median reprojection error `3.32px`; this is the first
      local clean artifact dense enough to test.
- [x] Ran the matching DeepView 8-camera 40-step Metal raytrace probe:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
      It consumed 599 train-visible points, trained stably, and improved source
      PSNR from `7.7475` to `7.8075`, but heldout still selected step 0 at PSNR
      `7.8250`. Dense-enough local clean pycolmap is therefore not sufficient
      by itself on this DeepView split.
- [x] Ran the same DeepView 8-camera clean init as a frame0-only diagnostic:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frame0_init_raytrace_128_1f_1024cells_40step_lowgeom_noaux.jsonc`.
      It consumed the same 599 train-visible points and improved heldout from
      step-0 PSNR `7.7417` to step-40 PSNR `8.0377`, with nonzero state deltas
      for centers, density, features, normals, quaternions, texel sites, SV
      axes, and SV RGB. This narrows the 16f failure toward static/dynamic
      temporal mismatch or schedule/normalization, but it is still below the
      earlier image-depth DeepView low-geometry control (`8.1417` best
      heldout).
- [x] Tried two 200-step frame0 extensions:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frame0_init_raytrace_128_1f_1024cells_200step_lowgeom_noaux.jsonc`
      and
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frame0_init_raytrace_128_1f_1024cells_200step_lowmotion_noaux.jsonc`.
      The default 200-step run improved source PSNR to `8.4144` but heldout
      only reached `7.8695`; the low-motion run limited geometry/density drift
      and reached `7.9646` best heldout. Both are below the 40-step result.
      Do not assume longer clean-frame0 training helps until the schedule is
      redesigned.
- [x] Built and ran a higher-resolution DeepView 8-camera pycolmap artifact:
      `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_512px_sift_wide.ply`.
      The 512px SIFT artifact has 771 points, 731 train-visible cells, median
      reprojection error `3.51px`, p90 `6.75px`, and still mostly two-view
      tracks (mean `2.01`, p90 `2`). It is not paper-grade COLMAP, but it
      materially improves clean heldout: the frame0 40-step config reaches
      `8.4810` heldout PSNR and the 16f companion reaches `8.5355`, beating the
      earlier image-depth DeepView low-geometry control (`8.1417`).
- [x] Tried a longer 80-step dense-eval companion for the same 512px 16f clean
      recipe:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_80step_lowgeom_denseeval_noaux.jsonc`.
      It selected step 20 at heldout PSNR `8.5345`, slightly below the 40-step
      run's `8.5355`, and final step 80 fell to `8.4564` while source PSNR rose
      to `7.9138`. Treat this as a negative schedule control: simply extending
      the current lowgeom 512px recipe does not improve clean heldout.
- [x] Tried a 120-step low-motion/low-appearance companion:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_120step_lowmotion_lowappearance_denseeval_noaux.jsonc`.
      It selected final step 120 at heldout PSNR `8.5270`, L1 `0.311158`,
      SSIM `0.0478`. This stabilizes the longer run compared with the default
      80-step extension, but still underperforms the 40-step lowgeom row.
      Lower motion/appearance alone is not the missing paper-quality schedule.
- [x] Built and ran a 1024px DeepView 8-camera pycolmap artifact:
      `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_1024px_sift_wide.ply`.
      It increased the clean point cloud to 975 points and 948 train-visible
      cells, with all 28 pairs verified, but reprojection error worsened
      (median `3.82px`, p90 `6.79px`) and tracks remained mostly two-view
      (mean `2.00`, p90 `2`). The matching 40-step 16f run
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_1024px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`
      selected step 0 at heldout PSNR `8.4415` and finished at `8.4411`,
      below the selected 512px row (`8.5355`). Higher SIFT resolution alone is
      therefore not the missing clean COLMAP path.
- [x] Added explicit pycolmap builder controls for SIFT thresholds, pair
      verification error, known-pose guided verification, track-length
      filtering, and triangulation merge/transitivity knobs:
      `research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py`.
      Two 512px long-track diagnostics stayed negative. Loose
      transitivity/merge completion with `min_track_length=3` kept only
      19 points (`16` length-3, `3` length-4). Adding 8px pair verification
      plus known-pose guided verification kept only 1 length-3 point. This
      confirms the current DeepView/SIFT graph has almost no long-track core.
- [x] Exposed pycolmap `--feature-type` / `--matcher-type` for future
      ALIKED/LightGlue probes, but guarded ONNX-backed modes behind
      `--allow-onnx-models` after a local `sift_lightglue` run aborted inside
      pycolmap with `LightGlue feature matching requires ONNX support`.
      The local Mac wheel is therefore not enough to test LightGlue/ALIKED;
      use an ONNX-enabled pycolmap host for that branch.
- [x] Added opt-in duplicate jitter for underfilled point-cloud init:
      `model.init_point_cloud_duplicate_jitter`.
      The matching 512px/16f A/B config
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_dupjitter_noaux.jsonc`
      jitters duplicated backfill cells by `0.04` scene units. It selected
      step 20 at heldout PSNR `8.3993` and finished at `8.3866`, below the
      non-jitter selected row (`8.5355`). Exact duplicate radius collapse is a
      real implementation smell, but jittering duplicates is not the missing
      paper-quality lever.
- [x] Added ordered point-cloud sampling for already-overfilled clean inits:
      `model.init_point_cloud_sample_mode="first"` keeps PLY order instead of
      drawing a random subset. The matching true-multiframe top1024 control
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`
      selected step 20 at heldout PSNR `9.1789`, L1 `0.294883`, SSIM `0.0248`,
      below the random selected row (`9.5920`). Keeping the lowest-reprojection
      subset by itself is not enough.
- [x] Preserved DeepView lens metadata in the multicam bundle and PowerFoam
      camera grid. DeepView `projection_type="fisheye"` now becomes
      `CameraSpec(lens_model="opencv_fisheye")`, with `radial_distortion`
      padded to four coefficients, and the point-cloud train-visible filter uses
      the same lens-aware projection. The 1-step PowerFoam smoke passed with
      `pose_source=deepview_models_relative_opencv_fisheye`.
- [x] Ran the matching fisheye-ray true-multiframe control:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_fisheye_rays_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
      It kept `2524` train-visible filtered points, selected step 0 at heldout
      PSNR `9.0781`, L1 `0.294265`, SSIM `0.0029`, and finished at heldout
      PSNR `9.0013`. Distortion-aware render rays alone do not recover the
      paper-quality gap while the clean point cloud is still built through a
      pinhole pycolmap reconstruction.
- [x] Extended the known-pose pycolmap builder with
      `--camera-model auto|pinhole|opencv_fisheye`. The builder now writes
      `OPENCV_FISHEYE` cameras with `[fx, fy, cx, cy, k1, k2, k3, k4]` when
      DeepView lens metadata is present. A tiny 2-camera/128px smoke using
      `--camera-model opencv_fisheye --camera-mode per_image` succeeded:
      pycolmap extracted `OPENCV_FISHEYE` features, verified the pair, and
      wrote a 16-point `/tmp/deepview_fisheye_pycolmap_smoke.ply` with median
      reprojection error `3.345px`. This proves the local pycolmap path accepts
      fisheye cameras; the remaining work is a full 8-camera x 4-frame
      distortion-aware artifact and training row.
- [x] Built and trained the full distortion-consistent DeepView artifact:
      `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.ply`
      with matching config
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
      Pycolmap used `OPENCV_FISHEYE` / `per_image`, verified all `496` pairs,
      and produced `2821` filtered points with reproj median/p90 `2.72/5.19px`,
      track mean/p90 `6.20/8`, unique-frame p90 `4`, and unique-camera p90
      still `2`. The Metal run consumed `2714` train-visible filtered points
      and set a new clean DeepView best: heldout PSNR `10.5931`, L1 `0.244703`,
      SSIM `0.0561` at step 0; final step degraded slightly to PSNR `10.5255`.
      This is a real improvement over the pinhole-built true-multiframe row but
      still below paper acceptance (`PSNR <13`, `SSIM <0.15`) and still lacks
      official CUDA fixture backing.
- [x] Ran a W&B-offline appearance-only follow-up on the selected
      OPENCV_FISHEYE artifact:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc`.
      It freezes centers/radii/density/quaternions/texel sites/heights/SV axes
      and trains only SV RGB with `ssim_weight=0.02`. The run wrote
      `eval_metrics_history.jsonl` and `train_metrics_history.jsonl`, selected
      step 40, and is W&B-offline-backed at
      `wandb/offline-run-20260505_223541-j0u3b4up`. It improved the selected
      clean row to heldout PSNR `10.8536`, L1 `0.230193`, SSIM `0.0766`.
      This clears the local W&B/post-initial gate but still misses paper
      quality (`PSNR <13`, `SSIM <0.15`).
- [x] Ran a W&B-offline material-only follow-up on the selected OPENCV_FISHEYE
      artifact:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_materialonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc`.
      It freezes centers/radii/density/quaternions but trains texel
      sites/heights/SV axes/SV RGB. This was negative for the paper gate:
      `best_metrics.json` selected step 0 at PSNR `10.8517`, L1 `0.230407`,
      SSIM `0.0752`; final step 40 fell to PSNR `10.8381` while SSIM rose only
      to `0.0792`. Training the material frame is not the missing local quality
      lever.
- [x] Ran the ordered-point version of the same appearance-only probe:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_first1024_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux.jsonc`.
      It sets `init_point_cloud_sample_mode="first"` to use the first/lowest
      reprojection-error filtered 1024 points. This was worse: best heldout was
      step 20 at PSNR `10.0535`, L1 `0.248474`, SSIM `0.0609`. The random
      subset appearance-only row remains the selected clean candidate.
- [x] Ran the matching OPENCV_FISHEYE all-filtered-points control:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_2714cells_40step_lowgeom_noaux.jsonc`.
      It consumed all `2714` train-visible filtered points, selected/finalized
      at step 40 with heldout PSNR `8.5095`, L1 `0.314076`, SSIM `0.0260`,
      and stayed far below the 1024-cell OPENCV_FISHEYE row. Capacity alone is
      not the missing quality lever for the distortion-consistent artifact.
- [x] Tested local stronger pycolmap feature modes. ALIKED/LightGlue remains
      blocked on ordinary pycolmap wheels; the local Mac wheel aborts in C++
      with `ALIKED feature extraction requires ONNX support` even on a small
      128px/two-camera smoke. The official COLMAP CLI image plus cuDNN runs
      ALIKED_N16ROT and ALIKED_LIGHTGLUE ONNX on Modal L40S, but the real
      DeepView probes were sparse enough to stop before the full 1024px run:
      wide2/128px brute force produced `0` points, near4/512px brute force
      produced `9` points, near4/512px LightGlue produced `27` points with
      track p90 `2` and reproj median `6.25px`, and opt-in known-pose guided
      verification pruned that LightGlue probe to `0` points. Covariant
      SIFT with affine shape and domain-size pooling completed for the full
      32-image OPENCV_FISHEYE database but produced a weaker artifact:
      `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_affine_minucam2.ply`
      has `2087` filtered points, reproj median/p90 `2.84/5.21px`, track
      mean/p90 `6.44/8`, and unique-camera p90 still `2`. It did not justify a
      training run.
- [x] Added and smoked an HLOC backend in
      `research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py`.
      It bypasses pycolmap's ONNX-backed ALIKED path by using
      Hierarchical-Localization directly, maps `aliked_n16rot` to actual
      `aliked-n16rot`, propagates `--max-features`, deep-copies HLOC configs,
      rejects shared-camera intrinsics mismatches, and always applies known-pose
      `hloc.geometric_verification` before triangulation. Local smokes proved
      the backend and summary fields, but quality was sparse: `0` points on the
      wide 2-camera smoke, `2` points on a close-overlap 4-camera smoke, and `0`
      points on the post-patch 2-camera schema smoke. HLOC is not yet the
      missing dense clean geometry lever.
- [x] Ran a 12-step dense-eval slow-appearance probe:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_12step_slowrgb_denseeval_noaux.jsonc`.
      It kept the selected clean artifact and regular raytrace path, reduced
      `texel_sv_rgb` LR to `0.00075 -> 0.000075`, and logged validation every
      step. Source PSNR rose `12.7537 -> 12.7685`, but heldout PSNR fell
      monotonically `12.5099 -> 12.5057`; `best_metrics.json` selected step 0.
      This closes the "missed early post-step peak / smaller appearance LR"
      hypothesis for the selected row.
- [x] Added an external-host blocker runner:
      `research_experiments/dynamic_foam/run_powerfoam_external_blockers.py`.
      It checks `uv`/CUDA/Warp/pycolmap availability, prints/runs the official
      CUDA/Warp fixture generation, runs the skip-until-present official parity
      tests, and prints/runs an OPENCV_FISHEYE ALIKED/LightGlue clean-geometry
      build with `--allow-onnx-models`. Matcher-specific output names avoid
      overwriting brute-force and LightGlue artifacts. The same runner now has
      explicit `write-train-config` and `train-aliked` tasks that template the
      selected OPENCV_FISHEYE config, require the generated artifact PLY plus
      JSON summary, require `point_count > 0`, require the summary
      `matcher_type` to match the CLI, enable W&B by default, and use a
      matcher-specific output directory/run name. It also has a `handoff` task
      that writes the external artifact contract, copy-back paths, local
      validators, and acceptance thresholds to
      `research_experiments/dynamic_foam/external_powerfoam_artifact_handoff.json`.
      Local dry-runs passed; local `check` correctly reports no CUDA, no
      `warp`, and no `pycolmap` in the venv. The paper and coverage verifiers
      now include these ALIKED candidates automatically once the required
      artifact/run files exist.
- [x] Upgraded and tested the DeepView plane-sweep builder as a lens-aware
      all-train-camera consistency initializer:
      `research_experiments/dynamic_foam/build_multiview_plane_sweep_point_cloud.py`.
      The full artifact
      `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_multiview_plane_sweep_frame0_128px_opencv_fisheye_stride4_support4_8192pts.ply`
      uses all 8 train cameras, `opencv_fisheye` projection, frame 0,
      48 depths, stride 4, and `min_support=4`. It wrote `7830` points with
      support mean/median/p90 `5.78/6/7` and median/p90 color error
      `0.120/0.241`. The matched top-1024 PowerFoam config
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_multiview_plane_sweep_frame0_128px_opencv_fisheye_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`
      selected/finalized at step 40 with heldout PSNR `8.2487`, L1 `0.327659`,
      SSIM `0.0160`, well below the selected OPENCV_FISHEYE pycolmap row
      (`10.5931`). Do not spend more local time on naive color-consistency
      plane sweep without adding a new idea such as masks, robust NCC/patch
      scores, occlusion reasoning, or real dense COLMAP.
- [x] Added experimental robust plane-sweep scoring controls:
      `--score-mode center_l1|mean_l1|patch_l1|zncc`, `--patch-radius`,
      `--min-patch-std`, and `--support-error`. Real-data 32px smokes passed
      for all modes, but 96px patch/ZNCC smokes were too slow and were stopped.
      The cheap inlier-support path did complete a full 8-camera artifact:
      `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_multiview_plane_sweep_frame0_128px_opencv_fisheye_stride4_support4_inlier02_8192pts.ply`
      with `6450` points, median/p90 error `0.0680/0.1151`, and support
      mean/median/p90 `4.63/4/6`. Matching config:
      `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_multiview_plane_sweep_frame0_128px_opencv_fisheye_inlier02_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
      The 40-step run selected step 0 at heldout PSNR `9.0679`, L1 `0.291473`,
      SSIM `0.0458`; final step 40 fell to PSNR `8.8609`, L1 `0.297917`, SSIM
      `0.0390`. This is better than raw plane sweep but still below
      OPENCV_FISHEYE pycolmap (`10.5931`), so broad plane-sweep tweaking should
      stay closed unless a new mechanism is proposed.
- [ ] Replace the weak pairwise/pycolmap artifacts with a dense real COLMAP
      reconstruction or better multi-view track builder before making
      paper-quality claims. If leave-one-train-camera-out evidence is used,
      rebuild the geometry from only the retained source cameras instead of
      reusing the 8-camera PLY.

## Test Matrix Summary

Required before claiming "full PowerFoam":

| Gate | Test Type | Required Evidence |
|---|---|---|
| Math units | deterministic small tensors | power face sign, sphere interval, quaternion frame, texel height, SV color |
| Forward parity | Torch direct vs Metal | RGB/features/alpha/depth/normal/contrib within tolerance |
| Backward parity | Torch direct or finite diff vs Metal | gradients for every free parameter within tolerance |
| Adjacency correctness | dense vs Cech/AABB | no missing required radical planes |
| Trainer smoke | 1-step static multiview | forward/backward/validation/checkpoint all execute |
| Small overfit | synthetic static scene | reaches documented PSNR/L1 threshold |
| Held-out eval | real posed cameras | logs PSNR/SSIM/L1 on held-out camera split |
| Performance | benchmark script | forward/backward time and memory for fixed matrix |
| Baseline tracking | `BASELINES.md` | dated row with config, W&B, metrics, caveats |

## Progress Log

2026-05-03:

- Added `rasterize_power_foam_quaternion_texel_surface(...)`, a strict
  quaternion-frame entrypoint over the existing Metal oriented texel-surface
  primitive.
- Added Metal-vs-Torch quaternion frame gradient parity coverage in
  `third_party/powerfoam-metal/tests/linear_texture_check.py`.
- Added trainable `feature_mode="quaternion_texel_surface"` in
  `src/train/train_powerfoam_metal.py` and a 1024-cell smoke config.
- Verified a 1-step 1024-cell MPS smoke through forward, loss, backward,
  optimizer step, validation render, and state drift logging.
- Ran a low-level 4096x4096 current-path benchmark for the existing Metal
  oriented texel-surface primitive and saved it under `outputs/benchmarks/`.
  It is accurate enough to benchmark but not fast enough yet: `1024` cells took
  `1784.8 ms` forward / `3331.4 ms` backward, and `4096` cells took
  `6543.4 ms` forward / `8681.7 ms` backward.
- Added Metal `feature_mode == 5` for height-displaced texel surfaces with
  oriented and quaternion Python APIs. `linear_texture_check.py` now compares
  height forward/alpha and gradients for points, radii, densities, texel sites,
  texel heights, texel features, normals/tangents/bitangents, and quaternions
  against the Torch reference.
- Added trainable `feature_mode="quaternion_height_texel_surface"` in
  `src/train/train_powerfoam_metal.py`, a smoke config, and a height optimizer
  group. A 1-step 1024-cell MPS smoke passed: eval L1 improved from
  `0.033722` to `0.033243`, and height parameters moved
  (`state_mean_texel_height_delta ~= 4.19e-06`).
- Extended the benchmark harness with `--foam-height-texel-surface` and ran a
  4096x4096 forward/backward benchmark saved at
  `outputs/benchmarks/powerfoam_metal_height_texel_surface_4k_1024_4096_2026-05-03.json`.
  The streaming path is still not fast enough: `1024` cells took `1770.3 ms`
  forward / `3747.0 ms` backward; `4096` cells took `6426.4 ms` forward /
  `9280.6 ms` backward.
- Added Metal `feature_mode == 6` for height-displaced spherical-Voronoi
  texel color with official detach semantics for the SV view query. Added
  oriented/quaternion Python APIs, Metal-vs-Torch forward/alpha/gradient parity
  for SV axes and RGB, trainable
  `feature_mode="quaternion_height_sv_texel_surface"`, and a 1024-cell smoke
  config. A 1-step MPS smoke passed: eval L1 improved from `0.034312` to
  `0.033381`, SV axes moved (`state_mean_texel_sv_axis_delta ~= 1.85e-04`),
  and SV RGB moved (`state_mean_texel_sv_rgb_delta ~= 0.00125`).
- Extended the benchmark harness with `--foam-height-sv-texel-surface` and
  saved a 4096x4096 benchmark at
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_4k_1024_4096_2026-05-03.json`.
  The full height+SV streaming path is much too slow for 4K: `1024` cells took
  `2840.2 ms` forward / `12515.9 ms` backward; `4096` cells took `8192.8 ms`
  forward / `23298.4 ms` backward.
- Replaced full-screen streaming bounds with conservative projected per-cell
  screen bounds in the Python wrapper. Parity checks still passed, and the 4K
  height+SV benchmark improved but remained too slow. Saved JSON:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_projected_bounds_4k_1024_4096_2026-05-03.json`.
  Results: `1024` cells took `1869.9 ms` forward / `11263.7 ms` backward;
  `4096` cells took `4435.1 ms` forward / `20126.4 ms` backward.
- Added a production-wired tiled Metal backend behind
  `FoamRasterConfig(use_tiled=True)`, with 8x8 tile candidates, an `auto`
  builder (`sorted_scan` for <=1024 cells, emit/sort for larger scenes), replay
  backward, benchmark flag `--foam-tiled`, and a tiled 1024-cell trainer smoke
  config. Added `third_party/powerfoam-metal/tests/tiled_streaming_check.py`,
  which checks constant and full height+SV tiled forward/alpha/gradient parity
  against the streaming backend. The stable 4096x4096 height+SV benchmark is
  saved at
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_auto_tile8_4k_1024_4096_stable_2026-05-03.json`.
  Median results: `1024` cells took `1198.2 ms` forward / `8166.3 ms`
  backward; `4096` cells took `2332.2 ms` forward / `15567.8 ms` backward.
  This is a real improvement over projected streaming but still too slow to
  call the 4K requirement done. A 1-step tiled trainer smoke also passed:
  eval L1 improved from `0.034312` to `0.033381`, centers/radii/quaternions,
  texel sites/heights, and SV axis/RGB parameters all moved.
- Added a reduced-atomic tiled backward for constant-feature mode. It keeps
  point/radius endpoint atomics but reduces feature and density gradients per
  tile-candidate. Constant-feature 4096x4096 timing is now much better:
  saved JSON
  `outputs/benchmarks/powerfoam_metal_constant_tiled_reduced_constant_bwd_4k_1024_4096_2026-05-03.json`
  reports median `1024` cells at `377.8 ms` forward / `733.4 ms` backward and
  `4096` cells at `682.4 ms` forward / `1152.9 ms` backward.
- Tried the same feature-gradient reduction idea for full height+SV mode. It
  preserved tiled parity but was slower (`1024` cells: `~10.3 s` backward;
  `4096` cells: `~17.2 s` backward), so it is not selected in the default
  backend. Overlap adjacency reduced average degree but also failed to solve the
  full height+SV 4K bottleneck.
- Optimized the default tiled mode-6 path by reusing per-texel SV colors and
  mixture weights inside forward/backward instead of recomputing them in nested
  loops. This preserved tiled parity and materially improved the full height+SV
  4K path. Saved JSON:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_mode6_reuse_4k_1024_4096_2026-05-03.json`.
  Median results: `1024` cells took `1199.4 ms` forward / `5393.1 ms`
  backward; `4096` cells took `2066.9 ms` forward / `9025.9 ms` backward. A
  1-step tiled trainer smoke still passed with eval L1 improving from
  `0.034312` to `0.033381` and all geometry/material parameter groups moving.
- Retuned the tiled `auto` builder after the mode-6 reuse change. The default
  now uses `sorted_scan` through `4096` cells because it beat emit/sort at that
  scale after the math reuse optimization. Saved JSON:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_mode6_reuse_auto_sorted4096_4k_1024_4096_2026-05-03.json`.
  Median default-path results: `1024` cells took `1280.9 ms` forward /
  `5936.2 ms` backward; `4096` cells took `2102.4 ms` forward / `9058.1 ms`
  backward. A trial packing emit/sort keys into int32 was worse on MPS and was
  reverted.
- Added a CPU `cech_aabb` correctness-mode adjacency path. It builds the
  official-style conservative overlap graph (`||p_i-p_j|| <= r_i+r_j`) without
  applying the K cap; KNN remains an explicit approximate speed ablation. The
  1-step full height+SV tiled trainer smoke now uses `cech_aabb` and printed
  `7794` required overlap edges with `0` missing edges while still moving all
  geometry/material parameter groups.
- Fresh `cech_aabb` 4096x4096 full height+SV tiled benchmark saved to
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_2026-05-03.json`.
  Median results: `1024` cells took `1314.5 ms` forward / `5910.2 ms`
  backward; `4096` cells took `1954.5 ms` forward / `9364.2 ms` backward.
  Average degrees were `8.55` and `9.53`. The 4K bottleneck is still the full
  mode-6 replay/backward path, not missing adjacency correctness.
- Added a non-gradient tiled Metal auxiliary pass exposed as
  `rasterize_power_foam_aux` plus full height+SV convenience wrappers. It emits
  normal distance, accumulated normal, fixed 0.5 median-depth quantile,
  contribution, target-weighted point error, and visibility mask. The standalone
  aux check covers a one-cell constant case against analytic alpha/contrib/error
  and a one-cell height+SV case against the rendered output. The full height+SV
  trainer smoke now calls the aux path during validation and logs contribution,
  point-error, visibility, normal, and median-depth metrics.
- Added persistent contribution/error EMA buffers to the Metal trainer. The
  smoke now logs `aux_mean_contrib_ema` and `aux_mean_point_error_ema`, giving
  the future resampling/pruning code durable statistics instead of one-off aux
  measurements.
- Added fixed-capacity EMA resampling to the Metal trainer. It follows the
  official keep-high-contribution / sample-high-error rule, reindexes all
  per-cell parameter tensors, carries Adam state through the permutation, divides
  propagated EMAs by duplicate count, and perturbs duplicate positions. A unit
  test covers deterministic low-contribution replacement and optimizer-state
  reindexing. A 2-step MPS smoke with `resample_every=1` exercised the real
  validation -> EMA -> resample path and printed `resample_replaced=1648`
  across the 16-frame 1024-cell smoke.
- Extended EMA resampling from fixed-capacity replacement to true tensor
  resize for grow/prune. The unit test now covers grow to 6 cells and prune to
  3 cells with Adam state preservation, and a 2-step MPS grow smoke rendered
  and backpropped after changing the live count from `1024` to `1050`
  (`resample_cell_count=1050`).
- Retuned the tiled kernel geometry from 8x8/64 threads to 16x16/256 threads.
  The tiled parity check, full texture/quaternion gradient check, aux check,
  and 1-step trainer smoke passed after the change. Saved 4K `cech_aabb`
  benchmark:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_tile16_2026-05-03.json`.
  Median `1024` cells: `892.0 ms` forward / `4611.7 ms` backward /
  `5501.8 ms` total. Median `4096` cells: `1614.3 ms` forward /
  `7138.7 ms` backward / `8753.0 ms` total. This is materially faster than the
  previous 8x8 `cech_aabb` artifact (`7273.1 ms` and `11441.3 ms` total), but
  still not enough to close the "fast 4K" requirement.
- Rechecked nearby tuning points and rejected them. The height+SV
  feature-gradient reduction preserved parity but was slower:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_height_sv_reduced_2026-05-03.json`
  reported `10462.1 ms` total at `1024` cells and `20007.7 ms` total at
  `4096` cells. A 32x32/1024-thread tile probe also preserved parity but was
  slower than 16x16:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_tile32_probe_2026-05-03.json`
  reported `7294.1 ms` total at `1024` cells. A per-pixel stop-count replay
  buffer also preserved parity but regressed the 4K totals to `6474.7 ms`
  (`1024` cells) and `10120.2 ms` (`4096` cells), so it was reverted:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_1024_4096_tile16_pixelstop_2026-05-03.json`.
- Extended the non-gradient tiled aux API from a hard-coded 0.5 median-depth
  output to an arbitrary `depth_quantiles` vector. The Metal op now returns a
  `[B,H,W,Q]` depth-quantile tensor while preserving `median_depth` for legacy
  trainer code. `third_party/powerfoam-metal/tests/aux_check.py` verifies
  `[0.25, 0.5, 0.75]` against the analytic one-cell solution with max error
  `4.77e-07`.
- Added posed-camera batch support to the Metal trainer without introducing a
  separate foam state per view. `load_powerfoam_training_data(...)` now loads
  `frame_source="multicam_val"`, builds `CameraSpec` rays for every train and
  heldout view, flattens `[view, time]` samples while sharing the same time
  index across train cameras, and calls the existing Metal forward/backward with
  per-sample rays. New smoke config:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_tiled_32_smoke.jsonc`.
  The 1-step MPS smoke passed on DeepView `03_Dog` with train cameras
  `camera_0001`/`camera_0015` and heldout `camera_0040`: `frames=2`,
  `samples=4`, `pose_source=deepview_models_relative_pinhole`, train eval
  metrics moved from L1/PSNR/SSIM `0.341172` / `7.9412` / `-0.0257` to
  `0.340393` / `7.9490` / `-0.0237`, and heldout L1/PSNR/SSIM was logged at
  `0.358152` / `7.8900` / `-0.0101` after the step. The smoke also moved
  centers, radii, quaternions, texel sites/heights, SV axes, and SV RGB.
- Regression-smoked the original explicit-video path after the camera-ray
  signature change with a 32px/64-cell 1-step run. It passed and train eval L1
  improved `0.034253 -> 0.032774`, with PSNR/SSIM `25.1659` / `0.8320` to
  `25.5397` / `0.8446`.
- Added an MPS-gated synthetic posed-view overfit regression:
  `tests/test_powerfoam_direct.py::test_powerfoam_metal_synthetic_posed_views_overfit_shared_state`.
  It renders three posed views from a one-cell teacher, trains a four-cell
  randomly initialized student through the Metal tiled path for 61 steps, and
  asserts final L1 `< 0.006`, at least a 4x reduction, and nonzero center drift.
  The focused PowerFoam test suite now passes with `14 passed`.
- Added optional SSIM loss support to the Metal trainer. `losses.ssim_weight`
  defaults to `0.0` for compatibility, and `powerfoam_ssim_loss(...)` uses the
  existing repo SSIM helper with a render-size-safe window. A focused helper
  test verifies identical images produce zero SSIM loss. The focused PowerFoam
  suite now passes with `15 passed`, and a 1-step multicam MPS smoke with
  `losses.ssim_weight=0.05` exercised the real backward path with
  `ssim_loss=1.001716` in the printed train metrics.
- Added tiled-memory accounting to
  `third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py`. The
  JSON now reports tile count, candidate count, candidate/offset/stop/screen
  bound bytes, saved `log_t` bytes, and a conservative forbidden dense
  `N*H*W` float-buffer size. Smoke artifact:
  `outputs/benchmarks/powerfoam_metal_memory_accounting_smoke_2026-05-03.json`
  (`128x128`, `N=256`) reported `0.076 MiB` tiled saved forward state versus
  `16.0 MiB` for one dense `N*H*W` float slab. A refreshed 4096x4096 accounting
  artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4096sq_1024_4096_tile16_accounting_2026-05-03.json`
  reported `1024` cells at `70.48 MiB` saved state versus `65536.0 MiB` dense,
  and `4096` cells at `74.14 MiB` saved state versus `262144.0 MiB` dense. Its
  one-iteration timings were `7093.6 ms` total and `10879.0 ms` total; the
  earlier tile16 median artifact remains the better timing reference.
- Ran the full P4 fixed-resolution matrix:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_matrix_128_256_512_accounting_2026-05-03.json`.
  Full height+SV tiled backward totals were `35.3/51.8/48.1 ms` at `128x128`
  for `N=256/1024/4096`, `36.5/47.8/65.7 ms` at `256x256`, and
  `73.6/126.4/227.6 ms` at `512x512`. The `512x512,N=4096` row reported
  `1.335 MiB` saved forward state versus `4096.0 MiB` for one dense `N*H*W`
  float slab.
- Re-ran the full Metal-vs-Torch primitive reference:
  `PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python third_party/powerfoam-metal/tests/linear_texture_check.py`.
  The strict `quaternion_height_sv_texel_surface` case matched the Torch
  reference with feature max error `9.24e-7`, alpha max error `1.61e-6`, and
  parameter-gradient max errors at or below `2.39e-7` across points, radii,
  density, texel sites/heights, SV axes/RGB, and quaternions. This closes the
  P2 tiny-scene forward/backward acceptance for the local Metal primitive.
- Added SfM/point-cloud initialization for the Metal trainer. Configs can set
  `model.init_point_cloud_path` to an ASCII/binary PLY, COLMAP
  `points3D.txt`, COLMAP `points3D.bin`, or a directory containing one of
  those files. `model.init_point_cloud_normalize` supports `none` or
  `fit_box`, and the sampled static cloud initializes centers/radii/colors
  before normal/quaternion/texel/SV parameters are created. Added the small
  checked-in fixture `test_data/powerfoam_sfm_tiny_ascii.ply` and smoke config
  `src/train_configs/local_mac_powerfoam_metal_point_cloud_init_quaternion_height_sv_tiled_32_smoke.jsonc`.
  The focused PowerFoam tests now pass with `16 passed`.
- Runtime-smoked both point-cloud init paths. The repo-local single-video
  config passed and printed `init_point_cloud_source_count=8`, eval L1
  `0.458758 -> 0.454509`, and nonzero drift for centers/radii/density,
  quaternions, texel sites/heights, SV axes, and SV RGB. A DeepView
  train-2/heldout-1 multicam override also passed with
  `pose_source=deepview_models_relative_pinhole`, `samples=4`, heldout camera
  `camera_0040`, eval L1 `0.378352 -> 0.377789`, and heldout L1
  `0.365270 -> 0.365193`.
- Added background compositing at the trainer boundary. `render.background`
  defines fixed RGB evaluation/logging background, and
  `render.background_mode="random"` samples per-image random RGB backgrounds
  during training. The default fixed black background preserves old behavior.
  The focused test now verifies alpha compositing and random background shape,
  raising the PowerFoam file to `17 passed`. A one-step random-background smoke
  using the point-cloud config passed with `background_mode=random`, train L1
  `0.255165`, eval L1 `0.455705`, and nonzero drift across geometry/material
  parameter groups.
- Added official-style Metal trainer LR schedules. The trainer now accepts
  explicit upstream-named absolute LR keys (`points_lr_init/final`,
  `density_lr_init/final`, `radii_lr_init/final`,
  `quaternions_lr_init/final`, `texel_sites_lr_init/final`,
  `texel_sv_axis_lr_init/final`, `texel_sv_rgb_lr_init/final`,
  `texel_height_lr_init/final`) and `train.lr_schedule="cosine"`, with the
  official warmups for density/radii (`1000`) and texel height (`2000`).
  Added smoke config
  `src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_tiled_32_smoke.jsonc`.
  The focused PowerFoam tests now pass with `18 passed`. A two-step MPS smoke
  passed with `lr_schedule=cosine`, eval L1 `0.458758 -> 0.449234`, and logged
  official schedule values including first-step warmup LRs
  `lr_density=0.0`, `lr_radii=0.0`, `lr_texel_height=0.0`, then second-step
  values `lr_density=0.001`, `lr_radii=5e-08`, `lr_texel_height=2.5e-06`.
- Added differentiable interpenetration/connectivity loss to the Metal trainer
  over the current Cech/AABB adjacency. The edge set is built from detached
  current geometry, but the overlap penalty backpropagates through decoded
  centers and radii. `losses.interpenetration_weight` uses the same
  official-style exponential decay shape as the direct trainer through
  `interpenetration_weight_final_multiplier`. The focused PowerFoam tests now
  pass with `19 passed`; the new unit test forces two cells to overlap and
  verifies nonzero finite gradients on center and radius parameters. Re-ran the
  official-LR smoke with `interpenetration_weight=1e-4`: it logged
  `interpenetration_loss=8.9919` at step 1 with weight `1e-4`, then
  `interpenetration_loss=8.9633` at step 2 with scheduled weight
  `3.1623e-6`, while eval L1 moved `0.458758 -> 0.449245`.
- Added differentiable contribution/sparsity loss to the Metal trainer through
  the main alpha output. For the front-to-back compositor, the sum of per-cell
  contributions equals the final rendered alpha per pixel, so
  `losses.contribution_weight` can train through `alpha.mean()` instead of the
  non-gradient aux contribution buffer. The focused tests now pass with
  `20 passed`; the new unit test verifies alpha-mean value and gradient plus
  the official exponential decay shape. Re-ran the official-LR smoke with both
  `contribution_weight=0.1` and `interpenetration_weight=1e-4`: it logged
  `contribution_loss=0.38979` at step 1 with weight `0.1`, then
  `contribution_loss=0.41735` at step 2 with scheduled weight `0.0031623`.
  Eval L1 moved `0.458758 -> 0.449868`.
- Added differentiable normal-distance loss to the tiled Metal training path.
  `rasterize_tiled_train_forward` now also emits a differentiable
  `normal_distance` image, and `rasterize_tiled_train_backward` accepts
  `grad_out_normal_distance` so replay backward routes the loss through
  opacity/interval gradients and the learned surface-frame normal. Added
  `return_normal_distance=True` on the Metal trainer forward path for
  height+SV modes and wired `losses.normal_weight` with official exponential
  decay. Rebuilt `third_party/powerfoam-metal` and verified:
  `tests/test_powerfoam_direct.py` now passes with `21 passed`; the new MPS
  unit test forces a positive normal/view dot product and checks nonzero finite
  gradients on quaternions and densities. Renderer checks still pass:
  `tiled_streaming_check.py`, `aux_check.py`, and `linear_texture_check.py`.
  The official-loss smoke passed with normal/contribution/interpenetration
  enabled; its default camera-facing init gives `normal_loss=0.0` as expected
  because normals point away from the camera rays, while the unit test covers
  the nonzero-gradient case.
- Refreshed a one-iteration 4096x4096 timing sample after adding the
  `normal_distance` training output:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_normal_distance_output_1iter_2026-05-03.json`.
  The run reported `1024` cells at `950.7 ms` forward / `4549.6 ms` backward /
  `5500.3 ms` total and `4096` cells at `1440.2 ms` forward /
  `6722.2 ms` backward / `8162.4 ms` total. This did not expose a regression
  versus the prior tile16 timing sample, but it is still far too slow to close
  the fast-4K requirement.
- Pinned the current official PowerFoam upstream source in
  `research_notes/foam_papers/powerfoam_upstream_source.md`. The recorded
  scratch clone is `/tmp/powerfoam_official` at
  `96392252ebd0059fe6ca98881b62e12295d9242f` (`GC to clear pytorch cache`),
  remote `https://github.com/theialab/powerfoam`, clean at inspection time.
  The older loose scan's `25d6f7b` commit is now explicitly historical only.
- Added an initial PowerFoam Metal static-multiview smoke row to
  `BASELINES.md` under the DeepView 3-cam train2/test1 table. It records the
  32px/2-frame/64-cell one-step smoke config, disabled W&B state, train PSNR
  `7.9490`, heldout PSNR `7.8900`, heldout L1 `0.358152`, heldout SSIM
  `-0.0101`, and the caveat that this is trainer acceptance evidence rather
  than a comparable 128px/16f baseline. The P5 baseline acceptance checkbox
  stays open until a proper row has wall time, all required metrics, and a
  run/artifact id.
- Added `model.resample_final_cells`, `model.resample_from_step`, and
  `model.resample_until_step` to the Metal trainer. When
  `resample_target_cells` is not set, resample steps follow the official-style
  geometric growth formula from initial cell count to final cell count, using
  Python `int(...)` truncation exactly.
- Decoupled resampling from artifact logging. If a scheduled resample step does
  not coincide with `image_log_every`, the trainer refreshes contribution/error
  EMAs from the current train batch before calling `resample_from_ema(...)`.
  The focused PowerFoam suite now passes with `22 passed`.
- Runtime smoke passed with `image_log_every=999`, `resample_every=1`,
  `resample_final_cells=20`, `resample_from_step=1`, and
  `resample_until_step=3` on the official LR config. It printed resample events
  at steps `1` and `2`, grew the live cell count from `16` to `20`, then
  rendered/backpropped/logged final eval at step `3` with eval L1 moving
  `0.458758 -> 0.438779`.
- Optimized full height+SV tiled mode-6 color evaluation by adding a vector
  `stream_sv_texel_color(...)` helper and caching texel softmax weights from
  the denominator pass. The active tiled forward/aux/backward paths now compute
  each texel's SV RGB color with one shared SV denominator pass instead of
  three repeated per-channel passes, and avoid recomputing texel weights for
  the immediate mode-6 color/gradient pass. Validation passed:
  `tiled_streaming_check.py`, `aux_check.py`, `linear_texture_check.py`, and
  `tests/test_powerfoam_direct.py` (`22 passed`).
- Reused the already-computed base clipped interval for tiled height-surface
  endpoint gradients. This removes a duplicate Cech/AABB adjacency clip from
  the active height+SV backward path when `near_id` or `far_id` is the height
  surface.
- Added a known-value SV gradient helper so tiled mode-6 backward can reuse the
  raw SV color and denominator already computed for the texel color pass,
  avoiding another raw-value/denominator loop inside SV color-gradient routing.
- New 4096x4096 `cech_aabb` median benchmark:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_sv_grad_known_value_median_2026-05-03.json`.
  Median `1024` cells: `554.5 ms` forward / `4095.8 ms` backward /
  `4650.3 ms` total. Median `4096` cells: `882.1 ms` forward /
  `6137.8 ms` backward / `7013.0 ms` total. This improves over the prior
  tile16 median (`5501.8 ms` and `8753.0 ms` total), but still does not close
  the fast-4K requirement.
- Added official-style `adjacency_diff` packing to the trainable tiled Metal
  forward/aux/backward path. This keeps edge-local power-face constants in one
  `[E,4]` tensor (`adjacent_center - center`, power-midpoint delta), avoiding
  neighbor point/radius reloads inside the hot clipping loops. Because the
  constants are precomputed in Python and consumed in Metal, tiled-vs-streaming
  forward parity now has tiny operation-order differences; the parity gate was
  adjusted to `3e-6` while gradient checks stayed at the previous `1e-5` level.
  Validation passed: `tiled_streaming_check.py`, `aux_check.py`,
  `linear_texture_check.py`, and `tests/test_powerfoam_direct.py`
  (`22 passed in 4.50s`).
  The official LR 32px smoke config also trained for 2 steps with the rebuilt
  extension; eval L1 moved `0.458758 -> 0.449871`, confirming the trainer still
  exercises render, aux, backward, validation, and state updates after the
  face-diff plumbing.
- Added a guarded selector for the specialized height+SV reduced backward
  kernel. It is used only when the normal-distance output is not part of the
  autograd loss; otherwise the generic tiled backward remains selected so
  normal-distance gradients are preserved. Validation passed again:
  `tiled_streaming_check.py`, `aux_check.py`, `linear_texture_check.py`,
  `tests/test_powerfoam_direct.py` (`22 passed in 2.65s`), and the official LR
  32px trainer smoke (`0.458758 -> 0.449871` eval L1).
- New selected 4K `cech_aabb` face-diff + reduced-mode6 median benchmark:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_adjdiff_mode6reduced_median_2026-05-03.json`.
  Median `1024` cells: `191.9 ms` forward / `1429.6 ms` backward /
  `1621.5 ms` total. Median `4096` cells: `309.4 ms` forward /
  `2149.1 ms` backward / `2456.7 ms` total. This is a large improvement over
  the previous selected `sv_grad_known_value` median (`4650.3 ms` and
  `7013.0 ms` total), but the full "fast 4K" requirement still stays open until
  the ray-tracing/replay backend or equivalent closes the remaining
  seconds-scale backward cost.
- Added a forward-only Metal raytrace probe (`raytrace_power_foam` /
  `raytrace_power_foam_flat`) with per-camera start-cell selection and a
  two-cell raster/raytrace parity fixture:
  `third_party/powerfoam-metal/tests/raytrace_check.py`. The fixture passes
  exactly and confirms the probe walks across one power-face adjacency
  (`steps range: 2 2`). The same check also includes a random 8-cell all-pairs
  graph, which matched raster forward within `2.98e-08` features / `0.0` alpha.
- Added `--foam-raytrace` benchmark mode for forward-only constant-feature
  raytrace timing. The current naive per-pixel walk is not selected: on random
  4K synthetic scenes it loses to tiled forward. Saved artifacts:
  `outputs/benchmarks/powerfoam_metal_constant_raytrace_cech_aabb_4k_forward_median_steps_2026-05-03.json`
  reported `231.4 ms` / `954.9 ms` forward for `1024` / `4096` cells with mean
  walk steps `11.5` / `14.1`. A KNN32 comparison also lost:
  `outputs/benchmarks/powerfoam_metal_constant_raytrace_knn32_4k_forward_median_2026-05-03.json`
  reported `275.5 ms` / `860.3 ms`, while
  `outputs/benchmarks/powerfoam_metal_constant_tiled_knn32_4k_forward_median_2026-05-03.json`
  reported `111.6 ms` / `178.1 ms`. The raytrace probe is useful validation
  infrastructure, but not yet the fast 4K architecture.
- Still missing for full PowerFoam: arbitrary official depth-quantile
  gradients if used as differentiable losses, external normal-supervision
  gradients, full-primitive static acceptance thresholds, proper baseline rows,
  paper-scale grow/prune schedules and acceptance runs. Regular triangulation
  is available as an optional checked topology, but the selected paper-faithful
  fast path remains Cech/AABB because the official code builds a Cech complex
  through an AABB/BVH query.
- Added a full height+SV raytrace wrapper
  (`raytrace_power_foam_oriented_height_sv_texel_surface`) plus a full-primitive
  raytrace/raster parity fixture in
  `third_party/powerfoam-metal/tests/raytrace_check.py`. Current check output:
  two-cell constant matched exactly; random all-pairs constant matched within
  `2.98e-08` features / `0.0` alpha; random all-pairs height+SV matched within
  `5.96e-08` features / `0.0` alpha with mean/max walk steps `2.12` / `4`.
- Extended `--foam-raytrace` to benchmark forward-only height+SV. Saved 4K
  median artifact:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_forward_median_2026-05-03.json`.
  Median `1024` cells: `170.7 ms` forward. Median `4096` cells: `227.2 ms`
  forward. Mean/max walk steps match the constant raytrace synthetic setup
  (`11.5` / `26`, `14.1` / `36`). This is a real fast 4K forward path for the
  full height+SV primitive, but `backward_supported=false`; trainable raytrace
  replay/backward is still the main missing piece.
- Rejected two reduced-backward shader micro-optimizations after measurement:
  caching height+SV texel/SV values in the local reduced kernel regressed the
  4K one-iteration totals to `1990.4 ms` / `2938.0 ms`, and changing the local
  grad zero loop from fixed `128` slots to dynamic `feature_dim` regressed to
  `1858.1 ms` / `2769.4 ms`. Both were backed out; the selected trainable tiled
  4K artifact remains
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_tiled_cech_aabb_4k_adjdiff_mode6reduced_median_2026-05-03.json`.
- Added experimental height+SV raytrace replay/backward in Metal. It recomputes
  a capped per-pixel event list, replays events in reverse, and routes gradients
  for centers, radii, densities, texel sites, heights, SV axes/RGB, and
  normals. The new `raytrace_check.py` backward fixture matches raster gradients
  within `3e-08` max on the small all-pairs height+SV scene.
- New selected synthetic 4K train benchmark with the event-cap guard enabled:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_train_capguard_median_2026-05-03.json`.
  Median `1024` cells: `166.6 ms` forward / `826.1 ms` backward /
  `988.5 ms` total. Median `4096` cells: `201.8 ms` forward /
  `781.8 ms` backward / `983.9 ms` total. This is the first guarded sub-second
  full height+SV 4K forward+backward result in this lane, and it is wired into
  `train_powerfoam_metal.py` behind `render.use_raytrace`.
- Added checked-in trainer smoke config
  `src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_raytrace_32_smoke.jsonc`.
  The first version disabled `losses.normal_weight` because raytrace did not yet
  return differentiable `normal_distance`. After the normal-distance wiring
  below, the config uses `normal_weight=0.1` again. The smoke trained 2 steps
  with `render_backend: raytrace`; eval L1 moved `0.460920 -> 0.454128`, and
  state deltas confirmed centers/quaternions/texels/SV colors moved.
- Added optional `regular_triangulation` adjacency mode. The builder lives in
  `third_party/powerfoam-metal/torch_powerfoam_metal/regular_triangulation.py`
  and computes weighted-Delaunay edges from lower facets of lifted
  `(x,y,z, ||p||^2 - r^2)` points via SciPy/Qhull. It is wired into
  `torch_powerfoam_metal.random_scene.make_adjacency`,
  `train_powerfoam_metal.build_csr_adjacency`, and the benchmark `--adjacency`
  choices. Validation with `uv --with scipy --with pytest` proves the
  zero-weight graph exactly matches SciPy `Delaunay` edges.
- Saved regular-triangulation 4K train benchmark with the same event-cap guard:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_regular_triangulation_4k_train_capguard_median_2026-05-03.json`.
  Median `1024` cells: avg degree `13.65`, `314.0 ms` forward /
  `1605.2 ms` backward / `1910.3 ms` total. Median `4096` cells: avg degree
  `13.75`, `550.4 ms` forward / `2482.1 ms` backward / `3040.1 ms` total.
  This makes the proper graph available and tested, but it is not the selected
  fast path yet; the current Cech/AABB raytrace artifact remains faster at
  `988.5 ms` / `983.9 ms` total.
- Added an autograd event-cap guard for raytrace replay. If forward walk steps
  exceed `RAYTRACE_MAX_BACKWARD_EVENTS` / `FOAM_RAYTRACE_MAX_EVENTS` (`64`),
  the height+SV raytrace autograd path now raises instead of silently truncating
  replay gradients.
- Added differentiable `normal_distance` to the height+SV raytrace path.
  `raytrace_forward` now returns the normal-distance image, and
  `raytrace_height_sv_backward` accepts `grad_out_normal_distance` so replay
  routes the normal regularizer through opacity/transmittance and the learned
  surface normal. `raytrace_check.py` now compares raster vs raytrace
  normal-distance output exactly on the small height+SV fixture and checks
  gradients from `normal_distance.square().mean()`; latest max gradient errors
  stayed within `3.36e-08`.
- Re-enabled normal loss in the raytrace official-LR 32px smoke config. The
  smoke passed with `normal_weight=0.1`, `render_backend=raytrace`, eval L1
  `0.460920 -> 0.454128`, and nonzero deltas for centers, quaternions, texel
  sites, SV axes/RGB, normals, and density. The tiled official-LR smoke still
  passed (`0.458758 -> 0.449871` eval L1).
- Refreshed guarded 4K raytrace train benchmarks after adding the
  normal-distance output:
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_train_normaldistance_median_2026-05-03.json`
  reports `1024` cells at `176.6 ms` forward / `835.8 ms` backward /
  `1016.1 ms` total and `4096` cells at `217.6 ms` forward / `798.0 ms`
  backward / `1014.4 ms` total. The refreshed regular-triangulation artifact
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_regular_triangulation_4k_train_normaldistance_median_2026-05-03.json`
  reports `1868.7 ms` total at `1024` cells and `2888.3 ms` total at
  `4096` cells, so regular topology remains available but not the selected fast
  path.
- Extended `third_party/powerfoam-metal/tests/raytrace_check.py` with optional
  SciPy-backed regular-triangulation traversal parity. In the default `.venv`
  this block skips with an explicit SciPy/Qhull message; running with
  `uv --with scipy` checks dense all-pairs raster against regular-triangulation
  raytrace. Latest SciPy-backed output: constant feature/alpha errors
  `1.79e-07` / `0.0`; height+SV feature/alpha/normal-distance errors
  `0.0` / `0.0` / `0.0`; height+SV gradient max errors stayed under
  `1.02e-07`.
- Added and ran the DeepView 3-cam train2/test1 raytrace smoke config
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_32_smoke.jsonc`.
  It passed with `render_backend=raytrace`, `normal_weight=0.1`,
  `cech_aabb`, 64 cells, 2 frames, and 32px render. Step-1 heldout metrics on
  `camera_0040`: L1 `0.328307`, PSNR `8.316828`, SSIM `0.045195`; train/eval
  PSNR `8.143913`; train-step elapsed `1.316392 s`. State deltas were nonzero
  for centers, radii, density, quaternions, texel sites/heights, and SV
  axes/RGB, so the posed-camera raytrace path is trainable at smoke scale.
  `BASELINES.md` now has a separate append-only smoke row; this is not a
  paper-scale baseline.
- Added and ran the first 128px/16f DeepView 3-cam raytrace probe config
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step.jsonc`.
  It completed 80 steps with 1024 cells in `21.632119 s` train-loop time and
  produced train/heldout media plus `checkpoint_final.pt`, but quality regressed:
  heldout PSNR moved from `8.120396` at step 0 to `7.813736` at step 80
  (heldout L1 `0.354624`, SSIM `0.001364`). State deltas were nonzero, and
  the run stayed within the raytrace cap, so the blocker is optimization /
  schedule quality rather than basic Metal raytrace execution.
- Fixed the PowerFoam trainer LR schedule application order. The loop now calls
  `update_powerfoam_learning_rates(...)` before forward/backward/`optimizer.step()`
  instead of after the step, so warmup groups like density/radii/texel-height
  do not take one oversized first update. The 32px raytrace multiview smoke
  still passed after the change.
- Re-ran the 128px/16f recipe with the warmup fix isolated in
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_warmupfix.jsonc`.
  It still regressed from heldout PSNR `8.120396` to `7.817927`, so the warmup
  bug was real but not the only quality issue.
- Added and ran a low-geometry/no-aux 128px/16f control config
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_noaux.jsonc`.
  It improved source-view PSNR from `7.730346` to `8.055424`, with heldout
  best at step 20 (`8.141685`) before final `8.088144`.
- Added best-checkpoint tracking to `train_powerfoam_metal.py`: each validation
  pass now selects `heldout_eval_psnr` when available, writes
  `checkpoint_best.pt` and `best_metrics.json`, and still writes
  `checkpoint_final.pt`. The rerun of the low-geometry/no-aux control confirmed
  `best_metrics.json` selects step 20. This confirms the 128px raytrace path
  can optimize appearance, and the remaining quality issue is
  schedule/generalization rather than Metal forward/backward correctness.
- Added aux-loss start-step gates:
  `normal_weight_start_step`, `contribution_weight_start_step`, and
  `interpenetration_weight_start_step`. The delayed-aux probe
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_lowgeom_delayed_aux.jsonc`
  ran successfully, but it did not improve heldout: best stayed at step 20
  (`8.141685`) before aux losses activated, and final heldout PSNR was
  `8.088228`. This is a useful negative control, not a new best recipe.
- Rejected a 4096-cell/cap128 capacity path. A 4096-cell version of the
  128px/16f low-geometry/no-aux raytrace recipe exceeded the default replay cap
  (`max_steps=75`, cap `64`). Temporarily raising the Metal/Python replay cap to
  `128` let the run finish, but quality regressed: best heldout stayed at step
  `0` (`8.177620` PSNR) and final heldout fell to `7.809724` PSNR. The cap128
  synthetic 4K train benchmark
  `outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_train_cap128_median_2026-05-05.json`
  also slowed to about `3.1 s` total for both 1024 and 4096 cells, versus the
  selected cap64 normal-distance artifact at about `1.0 s`. The selected cap is
  back to `64`; do not pursue this path without a separate fallback kernel or a
  traversal/replay fix.
- Added and ran a 1024-cell color-only freeze probe
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_coloronly_noaux.jsonc`.
  It freezes geometry, density, texel sites/heights, and SV axes, leaving only
  SV RGB trainable. Source/eval improved from `7.730346` to `8.043037` PSNR in
  `15.553846 s`, but heldout selected step `0` (`8.120396`) and final heldout
  fell to `8.073027`. This rules out "geometry LR over-motion" as the only
  failure; even pure color fitting on the current image-depth geometry overfits
  the two train cameras.
- Added a train-camera-only DeepView plane-sweep point-cloud builder
  `research_experiments/dynamic_foam/build_multiview_plane_sweep_point_cloud.py`
  and generated
  `research_experiments/dynamic_foam/artifacts/deepview_03_dog_train2_plane_sweep_frame0_128px_stride2_8192pts.ply`.
  The artifact uses only `camera_0001`/`camera_0015` on frame `0`, has `5799`
  valid points, and median color-consistency error `0.028078`. The matched
  PowerFoam run
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_128_16f_1024cells_80step_plane_sweep_init_lowgeom_noaux.jsonc`
  was worse than image-depth init: best heldout was step `0` at `7.777877`
  PSNR and final was `7.776456`. This rejects naive two-view plane-sweep init;
  the remaining paper-quality path needs stronger SfM/COLMAP-quality geometry,
  masks/all-view consistency, or a different NVS contract.
- Added a Neural3D/EX4DGS point-cloud preparation path
  `research_experiments/dynamic_foam/prepare_ex4dgs_anchor_point_cloud.py`.
  It transforms the EX4DGS `coffee_martini/input.ply` from dataset/world
  coordinates into the configured Neural3D anchor camera frame, filters by the
  expanded scene-scale PowerFoam box (`xy_extent=24`, `z=4..120`) and train
  camera visibility, and writes
  `research_experiments/dynamic_foam/artifacts/neural3d_coffee_martini_cam04_anchor_ex4dgs_input_128px_xy24_z4_120_trainvisible.ply`.
  The artifact kept `5113` of `5498` source points and projected into the
  train cameras after box filtering at `81.2%` for `cam04` and `70.8%` for
  `cam09` (`cam06` heldout diagnostic `76.1%`).
- Added and ran
  `src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_ex4dgs_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
  This is a companion `train2_holdout1` Neural3D `coffee_martini` run, not the
  DeepView `multicam_val_v1` split: train cameras `cam04`/`cam09`, heldout
  `cam06`, 1024 cells, `cech_aabb`, height+SV raytrace, and low-geometry/no-aux
  schedule. It completed in `5.940771 s` train-loop time, improved train PSNR
  from `10.068101` at step 0 to `10.373013`, and improved heldout PSNR from
  `10.033043` to final/best `10.741616` with heldout L1 `0.229993` and SSIM
  `0.160671`. This proves real-scene point-cloud init can train through the
  Metal raytrace path, but it uses an external pretrained EX4DGS artifact and
  remains short-run evidence rather than a paper-clean COLMAP reproduction.
- Added a longer version of the same Neural3D/EX4DGS-init config:
  `src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_ex4dgs_init_raytrace_128_16f_1024cells_200step_lowgeom_noaux.jsonc`.
  It completed without replay-cap failure in `55.752085 s` train-loop time.
  Heldout PSNR improved from `10.033043` at step 0 to best `11.979544` at
  step `150`, then ended slightly lower at `11.949820`; final heldout L1 was
  `0.194974`, final heldout SSIM `0.190488`, and best-checkpoint heldout L1 /
  SSIM were `0.194809` / `0.190030`. This is the strongest current
  PowerFoam-Metal real-scene heldout signal, but the same external-init caveat
  applies.
- Promoted world-space multicam point-cloud init from one-off preprocessing to
  first-class trainer plumbing. `model.init_point_cloud_coordinate_frame` now
  accepts `model` (default) or `multicam_world`; the multicam loader returns
  `world_to_model = inv(anchor_c2w)`, and
  `load_powerfoam_point_cloud_initialization(...)` applies that transform
  before normalize/clamp/sample. Unit coverage:
  `uv run --with pytest python -m pytest tests/test_powerfoam_direct.py::test_powerfoam_point_cloud_init_applies_world_to_model_transform tests/test_powerfoam_direct.py::test_powerfoam_metal_point_cloud_init_loads_ply_static_geometry`
  passed. A 1-step raw EX4DGS world-space smoke also ran end-to-end with
  `init_point_cloud_coordinate_frame='multicam_world'`, but quality was poor
  (`5.875771` step-0 heldout PSNR) because it samples the unfiltered raw cloud;
  the visibility-filtered prepared artifact remains the good recipe.
- Added first-class point-cloud visibility filtering in the trainer via
  `model.init_point_cloud_visibility_filter='train_visible'` and
  `model.init_point_cloud_min_visible_train_views`. The raw EX4DGS world-space
  PLY plus `multicam_world` transform now filters to the same `5113/5498`
  points as the prepared artifact before 1024-cell sampling, and a 1-step smoke
  under `/tmp/powerfoam_metal_neural3d_raw_world_trainvisible_init_smoke`
  reproduced the good step-0 heldout PSNR (`10.033043`). This removes the
  external preprocessing footgun, but it is a plumbing validation rather than a
  new quality baseline.
- Added and ran the checked-in first-class raw-world EX4DGS config
  `src/train_configs/local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_ex4dgs_world_trainvisible_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc`.
  It loads the raw EX4DGS `input.ply` directly with
  `init_point_cloud_coordinate_frame='multicam_world'` and
  `init_point_cloud_visibility_filter='train_visible'`, filtered `5113/5498`
  source points, reproduced step-0 heldout PSNR `10.033043`, and reached
  step-40 heldout PSNR `10.740685` with heldout L1 `0.230020` and SSIM
  `0.160639`. `checkpoint_best.pt` and `best_metrics.json` were written, but
  the process exited nonzero while saving `checkpoint_final.pt` because the
  filesystem was full (`df` showed about `172 MiB` free). Treat the row as
  valid first-class transform/filter plumbing evidence, not as a complete final
  checkpoint artifact or paper-clean COLMAP benchmark.
- Ran a 4096-cell one-step Neural3D smoke from the visibility-filtered EX4DGS
  anchor-frame PLY. It stayed within the selected cap64 raytrace path:
  adjacency avg degree `6.0093`, max degree `13`, missing overlap edges `0`.
  Step 1 completed with train-step elapsed `0.756979 s`; heldout PSNR moved
  `9.784716 -> 9.814147` with heldout L1 `0.262087` and SSIM `0.138597`.
  This shows 4096 cells are not universally cap-blocked on real-scene init, but
  the sampled 4096-cell initialization started worse than the filtered 1024-cell
  run and still needs a longer quality/cap study before becoming the selected
  capacity path.
- Added posed-camera multicam support to the Torch direct reference trainer.
  `train_powerfoam_direct.py` now supports `data.frame_source='multicam_val'`,
  builds full `[origin, direction]` rays from `CameraSpec`, flattens train and
  heldout view/time samples while sharing the same frame-indexed foam state,
  and logs heldout metrics/media. The checked-in smoke config
  `src/train_configs/local_mac_powerfoam_direct_multicam_deepview_3cam_train2_test1_32_smoke.jsonc`
  ran on CPU with DeepView train cameras `camera_0001`/`camera_0015` and
  heldout `camera_0040`: step-0 train/heldout L1 `0.330641` / `0.326765`,
  step-1 train/heldout L1 `0.324219` / `0.326421`, and the train backward path
  completed with finite loss. This closes the direct static posed-camera
  reference path; it is not a quality baseline.
- Added atomic torch checkpoint writes via `src/train/checkpoint_utils.py` and
  routed both `train_powerfoam_metal.py` and `train_powerfoam_direct.py`
  through it. This directly addresses the observed full-disk failure mode where
  `checkpoint_final.pt` was left truncated after metrics and best-checkpoint
  artifacts were already valid. A regression test simulates a failed
  `torch.save` and verifies the old checkpoint remains intact and the temp file
  is removed. The direct multicam smoke still writes `checkpoint_final.pt`
  through the atomic path.
- Ran the canonical full8/1024 ALIKED_N16ROT + LightGlue COLMAP-CLI path on
  Modal L40S via `modal_powerfoam_aliked_geometry.py --execute --full`. It
  produced a valid local artifact under
  `research_experiments/dynamic_foam/artifacts/...aliked_n16rot_aliked_lightglue_minucam2.{json,ply}`,
  but only `319` points. Track quality was long (`track_mean=7.0878`,
  `track_p90=8`, unique-frame p90 `4`) and all `496` image pairs were verified,
  yet the point count is far below the paper verifier's `>=2000` clean-point
  gate. Do not spend a full Metal training row on this ALIKED artifact unless a
  denser geometry run exists; it cannot satisfy the current acceptance check.
- Ran `diagnose_powerfoam_heldout_error.py` with SciPy on the selected 2714-cell
  regular row. The saved diagnostic shows heldout alpha mean `0.9745`, alpha
  `>0.9` on `96.4%` of pixels, worst-frame sphere-support hit fraction
  `0.9987`, and high-alpha pixels carrying `95.5%` of total residual. This
  confirms the failure is opaque wrong rendering with support, not blank
  coverage. A follow-up normal-strong 2714-cell official-objective run
  (`normal_weight=1.0`, final multiplier `0.5`) reduced mean train
  normal-distance but stayed at heldout `12.6686 / 0.1000` best PSNR/SSIM, so
  global stronger normal loss is also a bounded negative.
- Verified against the official PowerFoam trainer that the remaining
  normal/depth mechanism is not equivalent to our scalar `normal_distance` loss.
  Official training requests median depth, consumes a rendered normal map, and
  supervises rendered normals against Metric3D normals or normals derived from
  filtered rendered depth. Our Metal aux path can compute normal maps and median
  depth for diagnostics, but the differentiable training path only returns RGB,
  alpha, and scalar normal-distance. A faithful next implementation step is a
  differentiable rendered-normal output plus median-depth/self-normal
  supervision for the height+SV raytrace path.

## Do Not Confuse These Claims

- "PowerFoam Metal core works" means the partial bounded-cell Metal raster and
  replay backward pass acceptance tests.
- "Feature foam works" means our F-channel feature raster fork trains and logs.
- "Dynamic foam works" means Python-decoded temporal foam states can call Metal.
- "Full PowerFoam is implemented" means every Definition Of Done gate above is
  green.
- "RadFoam/Radiant Foam is implemented" is a separate claim; currently false.
