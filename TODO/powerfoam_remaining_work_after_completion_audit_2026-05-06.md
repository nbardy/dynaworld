# PowerFoam Remaining Work After Completion Audit

Date: 2026-05-06

This is the detailed backlog after the local completion audit for:

```text
PowerFoam proper on Metal, fast and accurate forward/backward, 4K, and trainable.
```

The audit gate is now green, but that does **not** mean the research program is
finished. It means the repo has an audited Metal implementation and smoke-scale
CUDA comparison path with concrete evidence. The remaining work below is the
work needed before we can make stronger claims such as:

- raw PowerFoam beats splats on the same heldout split
- dynamic geometry foam improves multicam/heldout quality beyond the current
  minimal mechanics smokes
- CUDA and Metal agree beyond tiny fixture/smoke cases
- feature foam is a first-class feature-splatting replacement
- the paper mechanism is competitive without heldout-blind color calibration
- 4K is fast enough for the intended production regime, not merely verified
  as runnable/trainable on a synthetic benchmark

## Current Audited State

### Gates that are considered complete

These should be kept green and treated as regression gates:

- Completion audit:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests
```

Current result: `ok: true`, `next_blockers: []`.

Saved audit artifact:

```text
outputs/powerfoam_completion_audits/p0_completion_audit_full_20260506.json
```

Strict raw-quality audit remains red by design:

```text
outputs/powerfoam_completion_audits/p0_completion_audit_require_raw_quality_20260506.json
```

- Local regression bundle inside the audit:

```text
50 passed, 1 skipped
```

- Focused target set:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_powerfoam_cuda_smoke.py \
  tests/test_powerfoam_eval_color_calibration.py \
  tests/test_powerfoam_paper_acceptance.py \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_raytrace_rendered_normal_backprops \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_raytrace_height_sv_backward_supports_9_texel_sites \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_normal_map_loss_uses_aux_median_depth_without_metric3d \
  tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present \
  -q -rs
```

Current result: `15 passed`.

### Best accepted calibrated evidence

Selected calibrated clean DeepView row:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_evalrgbcal_128_16f_3543cells_1step_denseeval
```

Best step `0` calibrated heldout:

```text
PSNR 14.384051322937012
SSIM 0.155595600605011
L1   0.15519042313098907
```

Post-initial trained row, step `1`:

```text
calibrated heldout PSNR 14.354509353637695
calibrated heldout SSIM 0.15373001992702484
calibrated heldout L1   0.15598230063915253
max state delta          0.0022691828198730946
```

Raw uncalibrated same row:

```text
step 0 raw PSNR 12.690682411193848
step 0 raw SSIM 0.12455591559410095
step 1 raw PSNR 12.788052558898926
step 1 raw SSIM 0.12439404428005219
```

This distinction is the central caveat for the remaining work. The accepted row
is heldout-blind train-fit RGB matrix calibration evidence. It is not proof
that raw PowerFoam geometry/material has solved the DeepView heldout problem.

### CUDA smoke status

The CUDA lane is a deployment and comparison smoke, not a paper-scale dynamic
representation benchmark:

- Official upstream CUDA/Warp PowerFoam runs on Modal L40S.
- The original small dynamic CUDA fork time-conditions `texel_sv_rgb` only.
- A second small dynamic-geometry CUDA fork now runs on the same smoke and
  decodes time-conditioned centers, radii, quaternions, and heights before the
  existing CUDA/Warp renderer.
- Fixed-black comparison:

```text
official CUDA PSNR/SSIM/L1 5.5640 / 0.0284 / 0.4901
dynamic CUDA  PSNR/SSIM/L1 5.5833 / 0.0288 / 0.4887
Metal micro   PSNR/SSIM/L1 5.1222 / 0.0105 / 0.5057
```

Dynamic-geometry CUDA micro artifact:

```text
outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json
```

Strict verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json \
  --require-dynamic-geometry
```

Result: `ok: true`. The geometry lane records nonzero center/radius/
quaternion/height coefficients, center delta, alpha delta, and support delta.

## Remaining Work Overview

Priority labels:

- `P0`: blocks honest next claims
- `P1`: high-value implementation/research work
- `P2`: important hardening, scaling, and ablation work
- `P3`: cleanup, packaging, and long-tail polish

Status labels:

- `not started`: no concrete branch yet
- `partial`: some code/artifacts exist
- `needs audit`: code/artifacts exist but do not yet prove the claim
- `negative evidence`: tried; should not be repeated without a new reason

## P0 - Stop Treating Calibration As Raw Quality

### P0.1 Raw PowerFoam heldout quality gate

Status: verifier implemented and tested on 2026-05-06; actual raw quality still
fails.

Current evidence:

- Calibrated row passes `13.0/0.15`.
- Raw uncalibrated row does not.
- Key learning explicitly records this caveat in `agent_notes/key_learnings.md`.
- `verify_powerfoam_paper_acceptance.py` reports both calibrated and
  uncalibrated metrics.

Implementation result:

- `verify_powerfoam_paper_acceptance.py` now reports `raw_quality_ok` and
  `calibrated_quality_ok` separately.
- Default paper/completion audit semantics remain calibrated-green.
- `--require-raw-quality` makes raw checks blocking and fails on the current
  selected calibrated row.
- `verify_powerfoam_completion_audit.py --require-raw-quality` forwards the raw
  gate and fails with explicit raw blockers.

Validated commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_powerfoam_paper_acceptance.py -q
```

Result: `7 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py
```

Result: `ok: true`, `raw_quality_ok: false`, `calibrated_quality_ok: true`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py \
  --require-raw-quality
```

Result: exit `1`, `ok: false`. Current selected row fails raw checks:

```text
calibrated PSNR/SSIM 14.384051322937012 / 0.155595600605011
raw PSNR/SSIM        12.690682411193848 / 0.12455591559410095
post-initial raw row none
```

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests
```

Result: `ok: true`, `next_blockers: []`; raw status is nonblocking and red.

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests --require-raw-quality
```

Result: exit `1`, `next_blockers: ["paper_acceptance_verifier",
"paper_acceptance_raw_quality_status"]`.

Problem:

The current audit proves a usable calibrated eval path. It does not prove that
the raw PowerFoam representation produces sufficiently good heldout structure,
depth, material, and color without a train-fit RGB matrix. Future agents could
misread the green audit as "raw PowerFoam solved."

Completed P0.1 deliverables:

- Separate raw-quality verifier mode:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py \
  --require-raw-quality
```

- The raw mode requires:

```text
uncalibrated_heldout_eval_psnr >= 13.0
uncalibrated_heldout_eval_ssim >= 0.15
step > 0 row exists OR explicit raw step-0 acceptance mode is requested
no eval_color_calibration OR calibrated metrics ignored for pass/fail
```

- Tests construct fake summaries/history with:

- calibrated pass / raw fail -> verifier fails raw mode
- raw pass / calibrated pass -> verifier passes
- raw pass only at step 0 -> verifier reports whether trainability evidence is
  absent

- Dashboard-friendly fields:

```json
"raw_quality_ok": false,
"calibrated_quality_ok": true
```

Current P0.1 acceptance:

- The normal completion audit can remain calibrated-audit green.
- The raw-quality command fails on the current selected row and explains why.
- Future final answers can say "audit green, raw gate red" without ambiguity.

Remaining quality work:

- Improve raw uncalibrated heldout quality. The gate is implemented; the raw
  representation quality itself is still below the paper acceptance threshold.

### P0.2 Fair same-split splat comparison

Status: matched fast_mac/pinhole comparator completed on 2026-05-06; exact
OPENCV_FISHEYE splat comparator remains open if we need strict projection
parity.

Implementation result:

- Added config:
  `src/train_configs/local_mac_splat_baseline_multicam_deepview_nearest0040_8cam_holdout0040_free_dynamic_3dgs_128_16f_3543splats_40step.jsonc`.
- Added SSIM and `train_loop_elapsed_s` to gauge-field splat metrics.
- Added comparison script:
  `research_experiments/dynamic_foam/compare_powerfoam_to_splats_nearest0040.py`.
- Wrote comparison artifact:
  `outputs/comparisons/powerfoam_vs_splats_nearest0040_20260506.json`.
- Added the row to `BASELINES.md`.

Measured matched rows:

```text
raw PowerFoam 3543 cells, 40 steps:
  train/eval PSNR/SSIM/L1     13.4675 / 0.2056 / 0.1618
  heldout PSNR/SSIM/L1        13.2663 / 0.1117 / 0.1691

calibrated PowerFoam 3543 cells, 1 step:
  heldout calibrated          14.3841 / 0.1556 / 0.1552
  heldout raw                 12.6907 / 0.1246 / 0.1851

matched free dynamic 3DGS, 3543 splats, 40 steps:
  train/eval PSNR/SSIM/L1     16.2282 / 0.2875 / 0.1110
  heldout PSNR/SSIM/L1        10.9809 / 0.1133 / 0.2043
  train loop elapsed          5.6080 s
```

Caveats:

- Splat row uses `fast_mac`; the older 3-cam splat baseline used the dense
  PyTorch renderer.
- Splat row uses the current gauge-field trainer's pinhole `CameraSpec`;
  PowerFoam nearest0040 uses `opencv_fisheye`.
- Do not claim a strict lens-matched win until the splat trainer can consume the
  same OPENCV_FISHEYE camera model/distortion path.

Completed P0.2 contract:

- Primary comparator is raw nearest0040 PowerFoam versus a matched free dynamic
  3DGS splat baseline.
- The calibrated PowerFoam row is included only as a separate eval-semantics row
  with raw fields disclosed; it is not counted as raw representation quality.
- The splat config matches the split/scale fields below:

```text
multicam_train_cameras camera_0025,camera_0039,camera_0041,camera_0012,camera_0026,camera_0023,camera_0042,camera_0038
heldout camera_0040
anchor camera_0025
sample id deepview_03_Dog_camera_0001_to_camera_0040
max_frames 16
render_size 128
num_splats 3543
steps 40
frames_per_step 4
seed 23
output outputs/gauge_fields/multicam_deepview_nearest0040_8cam_holdout0040_40step/free_dynamic_3dgs
```

Run command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  research_experiments/gauge_fields/train_splat_baseline.py \
  src/train_configs/local_mac_splat_baseline_multicam_deepview_nearest0040_8cam_holdout0040_free_dynamic_3dgs_128_16f_3543splats_40step.jsonc \
  --device mps \
  --steps 40 \
  --output-dir outputs/gauge_fields/multicam_deepview_nearest0040_8cam_holdout0040_40step/free_dynamic_3dgs \
  --no-wandb
```

Caveat: the current splat trainer constructs `CameraSpec(... lens_model="pinhole")`,
while the PowerFoam nearest0040 config uses `opencv_fisheye`. The smallest
no-code-change P0.2 is matched on split, frames, render size, primitive count,
steps, seed, and heldout camera, but not perfectly matched on projection.

Comparison artifact:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_to_splats_nearest0040.py \
  --powerfoam-output <dir> \
  --splat-output <dir> \
  --output outputs/comparisons/powerfoam_vs_splats_nearest0040_20260506.json
```

Current P0.2 acceptance:

- A table with raw PowerFoam, calibrated PowerFoam, and matched splat rows.
- `BASELINES.md` records the matched splat row and the PowerFoam comparison
  numbers.
- `verify_powerfoam_completion_audit.py --run-local-tests` now checks the
  comparison row labels, metric semantics, linked configs, exact nearest0040
  split, seed/sampling, output artifacts, wall-clock field, and projection
  caveat.
- Do not claim strict lens-matched superiority until the splat trainer consumes
  the same OPENCV_FISHEYE camera model/distortion path.

## P0 - Minimal Dynamic Geometry Mechanics Complete; Quality Still Open

### P0.3 Define and implement minimal dynamic geometry foam

Status: minimal Metal dynamic-geometry proof complete for explicit-video smoke
on 2026-05-06; nearest0040/multicam quality run still open.

Implementation result:

- The local Metal dynamic model already has temporal coefficients for centers,
  z, radii, densities, features, normals/tangents, and texel sites in
  `DynamicMetalPowerFoamVideo`.
- Added tests that prove:
  - zero RBF coefficients reproduce static decoded state;
  - geometry coefficients move centers/radii without repainting features;
  - feature coefficients repaint features without moving geometry;
  - on MPS, geometry coefficient motion changes rendered alpha;
  - on MPS, render-alpha loss backprops into geometry coefficients while
    features are frozen.
- Added a geometry-only 40-step explicit-video config and run summary:

```text
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke.jsonc
outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke/dynamic_geometry_summary.json
```

- The trainer now writes:
  - `train_metrics_history.jsonl`
  - `dynamic_geometry_summary.json`
  - final render/side-by-side MP4s
- Added verifier:

```text
research_experiments/dynamic_foam/verify_dynamic_powerfoam_geometry_run.py
```

Current verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_dynamic_powerfoam_metal.py -q -rs

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  .venv/bin/python -u src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke.jsonc

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_dynamic_powerfoam_geometry_run.py \
  outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke \
  --require-geometry-motion \
  --require-alpha-support-motion \
  --require-appearance-freeze-control
```

Results:

```text
pytest: 15 passed
dynamic geometry verifier: ok true
final eval_l1 0.06529680639505386
state_mean_temporal_screen_delta_px 0.44571685791015625
state_p95_temporal_screen_delta_px 1.0568746328353882
eval_mean_temporal_alpha_delta 0.008722909726202488
eval_mean_temporal_support_delta 0.0075358073227107525
state_mean_temporal_feature_abs_delta 0.0
```

Training A/B against fixed-geometry repaint control:

```text
outputs/dynamic_powerfoam_metal/motion_vs_repaint_comparison_1024_16f_40step_20260506.json
```

```text
geometry-only mean/min SNR    14.8959 / 13.5512
color-only mean/min SNR       14.6847 / 12.6700
geometry-only mean/min PSNR   19.5667 / 18.2504
color-only mean/min PSNR      19.3554 / 17.3692
geometry-only alpha/support   0.00957 / 0.00793
color-only alpha/support      0.0 / 0.0
geometry-only feature drift   0.0
color-only feature drift      0.00427
```

Nuance: the color-only fixed-geometry branch has lower average L1 (`0.06108`
vs `0.06514`), while geometry-only has better MSE, mean SNR, and worst-frame
SNR. Treat per-frame SNR/PSNR plus support motion as the stronger evidence for
motion-vs-repaint, not a single scalar loss.

High-motion YouTube A/B at 128px:

```text
rank artifacts:
outputs/dynamic_powerfoam_metal/youtube_motion_rank_segments_20260506.json
outputs/dynamic_powerfoam_metal/youtube_motion_rank_curated_raw_20260506.json

sampled clip:
data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4

configs:
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_128_16f_40step.jsonc
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_128_16f_40step.jsonc

comparison:
outputs/dynamic_powerfoam_metal/motion_vs_repaint_comparison_youtube_hlaZbH_128_16f_40step_20260506.json
```

```text
geometry-only mean/min SNR    8.3076 / -2.1480
color-only mean/min SNR       7.8124 / -2.6612
geometry-only mean/min PSNR   13.2116 / 8.0830
color-only mean/min PSNR      12.7164 / 7.5699
geometry-only L1/MSE          0.17012 / 0.05391
color-only L1/MSE             0.18416 / 0.06105
geometry-only screen delta    1.06725 mean / 2.52813 p95 px
color-only screen delta       0.0 / 0.0
geometry-only alpha/support   0.00642 / 0.00538
color-only alpha/support      0.0 / 0.0
geometry-only feature drift   0.0
color-only feature drift      0.00495
```

Nuance: this high-motion probe used a materialized 4fps/16f sampled MP4 because
the current dynamic trainer reads consecutive video frames. Future benchmark
configs need a first-class `sample_fps` or sequence-loader contract so clip
selection, ranking, and training are not coupled through a manual derived
media artifact.

512px center-crop 8fps follow-up:

```text
derived clip:
data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_16f.mp4

configs:
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_center_crop_8fps_512_16f_40step.jsonc
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_center_crop_8fps_512_16f_40step.jsonc

comparison:
outputs/dynamic_powerfoam_metal/motion_vs_repaint_comparison_youtube_hlaZbH_center_crop_8fps_512_16f_40step_20260506.json
```

```text
geometry-only mean/min SNR    10.5224 / 1.8168
color-only mean/min SNR       9.4584 / -0.2398
geometry-only mean/min PSNR   15.2546 / 10.2760
color-only mean/min PSNR      14.1907 / 8.2194
geometry-only L1/MSE          0.11574 / 0.03222
color-only L1/MSE             0.13603 / 0.04473
geometry-only screen delta    4.29663 mean / 10.31853 p95 px
color-only screen delta       0.0 / 0.0
geometry-only alpha/support   0.01701 / 0.01516
color-only alpha/support      0.0 / 0.0
geometry-only feature drift   0.0
color-only feature drift      0.00420
```

Performance caveat:

```text
512 geometry train-loop elapsed_s 7.6606; full wall time 24.95s
512 color train-loop elapsed_s    7.6853; full wall time 28.39s
```

This supports the user's hunch that fixed-cell train steps remain reasonably
cheap at 512, but the measured full process is dominated by eval/media logging.
A proper scaling benchmark should add timing fields for forward/backward/
optimizer/eval/media and run matched center-crop 128/256/512 configs.

Remaining P0.3 limitation:

- This is an explicit-video dynamic-geometry proof, not a nearest0040/multicam
  heldout-quality result. The current dynamic Metal trainer only loads a plain
  video sequence, so true nearest0040/multicam support needs a loader/camera
  contract extension before a fair heldout row can exist.

2026-05-06 subagent clarification:

- P0.3/P0.4 should not be counted as complete by adding more appearance-side
  feature/RGB residuals.
- Dynamic geometry means time changes support/alpha through geometry state.
  A passing test must show alpha/support changes when geometry coefficients
  change and color/features are frozen.
- The feature-only CUDA fork is still an RGB negative control:
  `texel_sv_rgb(t) = base + B(t) coeff`, with no center/radius/orientation/
  density/height motion.
- The CUDA dynamic-geometry fork now moves centers/radii/quaternions/heights
  before the official CUDA/Warp render call and the strict verifier requires
  alpha/support motion.

Minimal coefficient shapes:

```text
center_coeff      [B, N, 3]
radius_coeff      [B, N]
density_coeff     [B, N]
normal_coeff      [B, N, 3] or quaternion coeff
texel_site_coeff  [B, N, S, 2]
height_coeff      [B, N, S, H]
```

Implementation boundary:

- Decode dynamic state in Python before each render call.
- Keep raster kernels stateless: each call receives already-time-conditioned
  `points`, `radii`, `densities`, `texel_sites`, `features`, and `normals`.
- Rebuild adjacency per decoded frame in the first correct version.

Current implemented dynamic CUDA behavior:

```text
feature lane:  texel_sv_rgb(t) = texel_sv_rgb_base + residual(time_basis(t))
geometry lane: points/radii/quaternions/heights = base + residual(time_basis(t))
```

The feature lane is appearance/time conditioning only. The geometry lane is the
minimal scene-state dynamic fork.

Full dynamic geometry means time affects at least some of:

```text
p_i(t)        center / power site
r_i(t)        radius / support extent
q_i(t)        orientation / normal frame
sigma_i(t)    density
s_ij(t)       local detail-site position
h_ij(t)       detail-site height/displacement
adjacency(t)  true or conservative time-varying neighbor set
```

Minimal useful version:

```text
p_i(t) = p_i0 + sum_b B_b(t) dp_i_b
q_i(t) = normalize(q_i0 + small_rotation_basis(t))
h_ij(t) = h_ij0 + sum_b B_b(t) dh_ij_b
```

with temporal regularization:

```text
||d/dt p_i(t)||^2
||d2/dt2 p_i(t)||^2
ARAP over foam graph
neighbor frame consistency
```

Completed P0.3 deliverables:

1. Metal minimal dynamic geometry config:

```text
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke.jsonc
```

2. Model changes:

- optional time-basis coefficients exist for centers, z, radii, densities,
  features, normals/tangents, and texel sites in
  `DynamicMetalPowerFoamVideo`;
- base static state is preserved when all dynamic coefficients are zero.

3. Renderer boundary:

- evaluate dynamic state for `frame_indices` before render
- preserve batch shape: either per-frame state expansion or per-frame render
  loop
- avoid building a hidden `[F,N,...]` memory wall for high N unless the smoke is
  explicitly tiny

4. Losses:

- temporal L2 on center velocity
- acceleration smoothness
- optional graph ARAP:

```text
|| (p_i(t)-p_j(t)) - R_i(t)(p_i(0)-p_j(0)) ||^2
```

- appearance leakage diagnostic: freeze color and see whether motion still
  reduces loss

5. Tests:

- zero dynamic coefficients reproduce static decoded state
- nonzero center trajectory changes rendered alpha/feature image
- perturb center/radius only, freeze color/features, and assert alpha/support
  changes across time
- perturb feature/RGB only, freeze geometry, and assert RGB can change while
  alpha/support does not
- gradients flow into center/radius trajectory coefficients
- temporal regularizer returns zero for constant trajectory and positive for
  moving trajectory

6. Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  .venv/bin/python src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke.jsonc
```

Current P0.3 acceptance:

- A dynamic geometry row where nonzero temporal screen/center/radius movement is
  logged.
- A control showing appearance-only time conditioning is not the sole source of
  improvement.
- A note explicitly separating "motion" from "repaint."
- `verify_powerfoam_completion_audit.py --run-local-tests` checks the summary
  artifacts, 16-frame/40-step/1024-cell geometry-only smoke scope, state deltas,
  alpha/support deltas, and frozen dynamic-feature control.

Still open beyond P0.3:

- nearest0040/multicam dynamic-geometry heldout quality. The current dynamic
  trainer is explicit-video-only and needs a loader/camera contract extension
  before a fair multicam dynamic-geometry row can exist.

### P0.4 CUDA dynamic geometry fork

Status: minimal CUDA dynamic-geometry smoke complete on 2026-05-06.

Implemented CUDA dynamic forks:

- `dynamic_feature_foam.patch`: appearance-side time-conditioned SV RGB.
- `dynamic_geometry_foam.patch`: scene-side time-conditioned centers, radii,
  quaternions, and heights decoded before the official CUDA/Warp render call.
- does not touch CUDA/Warp raster kernels

Completed P0.4 deliverables:

1. Create a second patch:

```text
research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch
```

2. Minimal upstream state additions:

```text
dynamic_center_coeffs
dynamic_radius_coeffs
dynamic_quaternion_coeffs or dynamic_normal_frame_coeffs
dynamic_height_coeffs
```

3. Integrated at scene/state decode time first, not inside the core raster
   kernel.

4. Added Modal smoke args:

```bash
--dynamic-geometry
--dynamic-center-basis-count 4
--dynamic-height-basis-count 4
```

5. Added result fields:

```json
"dynamic_center_delta_mean": ...
"dynamic_radius_delta_mean": ...
"dynamic_height_delta_mean": ...
"time_alpha_delta_mean": ...
"time_rgb_delta_mean": ...
"same_camera_support_delta_mean": ...
```

6. Tests:

- patch stays small enough to review
- patch touches geometry state, not only RGB
- time causality probe shows alpha changes when centers/radii/heights change

Acceptance:

- Modal L40S micro run with official CUDA base, appearance dynamic fork, and
  geometry dynamic fork on the same clip/settings.
- Saved comparison JSON showing geometry fork changes alpha/support over time.
- RGB-only time causality is not enough.
- `verify_powerfoam_completion_audit.py --run-local-tests` now checks the strict
  micro contract (`L40S`, `64px`, `4f`, `5` steps, `256` points, `4` texel
  sites, `sv_dof=2`, fixed black background), same-run lane availability, and
  the RGB-only feature-lane negative control.

Implementation evidence:

- Patch:
  `research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch`
- Runner/verifier:
  `research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py`
  `research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py`
  `research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py`
- Tests:
  `tests/test_powerfoam_cuda_smoke.py`
- Modal artifact:
  `outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json`

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  tests/test_powerfoam_cuda_smoke.py

git -C /tmp/powerfoam_upstream_963 apply --check \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train uv run --with pytest python \
  -m pytest -p no:cacheprovider tests/test_powerfoam_cuda_smoke.py -q

PYTHONDONTWRITEBYTECODE=1 uv run --with modal modal run \
  research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute --preset micro_clip_64_4f_5step \
  --run-id cuda_dynamic_geometry_micro_20260506 \
  --max-gpu-minutes 8 --skip-official-fixture \
  --fixed-black-background --dynamic-geometry

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json \
  --require-dynamic-geometry
```

Results:

```text
pytest: 8 passed
strict CUDA dynamic-geometry verifier: ok true
official_static_cuda     PSNR/SSIM/L1 5.5640 / 0.0284 / 0.4901
dynamic_feature_foam_cuda PSNR/SSIM/L1 5.5833 / 0.0288 / 0.4887
dynamic_geometry_foam_cuda PSNR/SSIM/L1 5.5910 / 0.0291 / 0.4882
geometry warm step mean 11.64 ms
dynamic_center_delta_mean 0.0005558
dynamic_radius_delta_mean 0.00000796
dynamic_height_delta_mean 0.0007690
time_alpha_delta_mean 0.002022
same_camera_support_delta_mean 0.003174
```

Remaining limitation:

- This is a 64px/4-frame/5-step smoke. It proves the pinned CUDA/Warp path can
  host a scene-side dynamic-geometry fork and that the verifier rejects
  RGB-only time causality. It is not a real-quality dynamic-geometry benchmark.

## P1 - Feature Foam As A Real Feature-Splatting Replacement

### P1.1 Define "feature foam" contract

Status: partial.

Current conceptual contract:

```text
representation -> per-pixel feature tensor [B,F,H,W] + alpha -> colorizer
```

This can be implemented by:

- Gaussian feature splats
- PowerFoam feature cells
- screen disks
- oriented slabs
- gauge-field supports

The key is not ray tracing. The key is differentiable feature accumulation and
heldout feature/color quality.

Deliverables:

1. Write a short contract doc:

```text
research_notes/gauge_powerfoam/feature_foam_contract.md
```

Must specify:

```text
input state shapes
rendered feature shape
alpha semantics
background feature semantics
colorizer interface
normalization / LayerNorm obligations
train/eval metrics
browser/export implications
```

2. Define which PowerFoam modes support arbitrary F:

- constant feature cell mode
- oriented surface feature mode
- height+SV RGB paper mode
- hybrid mode: paper geometry + arbitrary feature payload

3. Decide whether feature channels are:

- attached to cell center
- attached to detail sites
- attached to SV basis entries
- produced by a local MLP from cell/material coordinates

Acceptance:

- A future agent can implement feature foam without guessing whether it should
  use raytrace, raster, SV RGB, or arbitrary F channels.

### P1.2 Metal feature foam renderer path

Status: partial / needs audit.

Known local facts:

- The Metal renderer has feature-capacity constraints in some height+SV paths.
- Feature-dim > 384 had required a backward fix for 9 sites/cap384.
- The current acceptance row is RGB/SV, not an F32 feature-foam row.

Deliverables:

1. Add or verify a feature-render path:

```python
features, alpha = render_powerfoam_features(...)
```

where `features.shape == [B,F,H,W]`.

2. Add explicit configs:

```text
src/train_configs/local_mac_feature_powerfoam_F16_nearest0040_128_16f_smoke.jsonc
src/train_configs/local_mac_feature_powerfoam_F32_nearest0040_128_16f_smoke.jsonc
```

3. Add colorizer integration:

- use existing `FeatureToColor` or equivalent
- include background feature handling
- include PCA video logging
- include LayerNorm handling if F32 feature path requires it

4. Tests:

- F=3 path matches RGB mode within tolerance where intended
- F=16 smoke runs forward/backward
- F=32 smoke runs forward/backward
- alpha output is finite and nonblank
- colorizer receives expected shape

5. Metrics:

```text
feature_l2
rgb_l1
rgb_psnr
rgb_ssim
alpha_mean
feature_pca_video_written
```

Acceptance:

- F16 and F32 feature foam smokes train without shape hacks.
- Matched feature splat baseline exists for the same data.
- A comparison note says whether feature foam improves heldout or merely works.

### P1.3 CUDA feature foam

Status: not started for true feature accumulation.

Current CUDA dynamic fork is RGB/SV residual, not arbitrary F feature
accumulation.

Deliverables:

1. Decide implementation surface:

- upstream CUDA/Warp raster returns RGB only
- extend it to return `F` feature channels
- or use feature channels in `texel_sv_rgb`-like payload chunks

2. Modal smoke:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_feature_foam_F16_micro_YYYYMMDD \
  --feature-dim 16
```

3. Result contract:

```json
"feature_dim": 16,
"feature_loss": ...,
"rgb_colorizer_loss": ...,
"feature_delta_time_mean": ...
```

4. Tests:

- CUDA summary verifier rejects claiming feature foam when feature_dim is
  absent
- patch contains actual feature-channel plumbing, not only RGB residuals

Acceptance:

- CUDA F16 smoke passes.
- CUDA-vs-Metal feature smoke comparison exists on same clip/settings.

## P1 - Raw Geometry / Material Quality

### P1.4 Diagnose why raw SSIM stays low

Status: partial.

2026-05-06 subagent refinement:

- Extend `research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py`
  instead of creating a new script.
- Add `--calibration-json eval_color_calibration_step_XXXX.json`.
- Compute raw and calibrated SSIM component maps for the worst heldout frame:
  luminance, contrast, and structure.
- Add correlations of `1 - ssim_structure` with normal distance, median depth,
  alpha, and residual L1.
- Decision rule: if calibration mostly fixes luminance/contrast, call it
  color/exposure; if calibrated structure loss remains on object edges/depth/
  normal regions, call it geometry/material failure.

Current evidence:

- Raw nearest0040 PSNR crosses roughly `13`, but SSIM stays around `0.11` to
  `0.125`.
- Higher texel-site capacity and cap384 did not clear SSIM.
- Color affine improves PSNR and some SSIM, but raw SSIM remains below gate.
- High-alpha/high-support diagnostics showed the failure is not blank coverage
  for selected regular rows.

Hypotheses:

1. Geometry is slightly shifted/blurred; color calibration hides color but not
   structure.
2. Material lookup still smears high-frequency detail across wrong local
   surface coordinates.
3. Power cells are too coarse / wrong support topology for fine silhouette
   boundaries.
4. COLMAP points/tracks are too sparse or mostly two-camera, so cell support is
   not truly multiview.
5. The objective rewards train-view repainting more than heldout-stable
   structure.
6. SSIM is punished by local contrast/texture misalignment more than PSNR.

Deliverables:

1. Add heldout structure panel:

```text
GT | raw render | calibrated render | raw 1-SSIM-structure | calibrated 1-SSIM-structure | raw edge residual | alpha | median_depth | normal_distance
```

2. Add numeric diagnostics:

```text
edge_l1
highpass_l1
foreground_l1
alpha_weighted_residual
residual_by_alpha_quantile
residual_by_normal_distance_quantile
residual_by_depth_quantile
SSIM window worst-percentile map
raw/calibrated ssim_luminance_mean
raw/calibrated ssim_contrast_mean
raw/calibrated ssim_structure_mean
worst-20% structure-loss share
correlation of 1-SSIM-structure with normal_distance / median_depth / alpha / residual L1
```

3. Add deformation/warp diagnostic:

- estimate small optical flow or local displacement between raw render and GT
- test if a tiny 2D warp lifts SSIM substantially
- if yes, geometry/silhouette misalignment is the main issue
- if no, texture/material/color frequency is the main issue

4. Add material-coordinate stability diagnostic:

- for train and heldout rays, record local `(u,v)` site coordinates
- report out-of-range fractions
- compare per-view material-coordinate distributions

Acceptance:

- A note identifies whether the raw SSIM gap is mostly geometry alignment,
  texture/material, color/exposure, or metric artifact.
- At least one diagnostic falsifies a tempting but wrong explanation.

### P1.5 Official-style normal/depth supervision

Status: partial.

Current evidence:

- Metal exposes differentiable rendered normals.
- Normal-distance scalar loss exists.
- Normal-map loss using aux median depth exists in tests.
- Official PowerFoam-style external normal/Metric3D supervision is not fully
  wired as a production training lane.

Deliverables:

1. Define normal/depth training modes:

```text
none
self_median_depth_normals
metric3d_normals
rendered_depth_filtered_normals
depth_quantile_loss
```

2. Add config keys under `losses`, normalized once in config load.

3. Add target generation/cache:

- Metric3D normal maps if available
- self-normal maps from GT/depth if no external model
- valid masks
- confidence masks

4. Add differentiable losses:

```text
normal_map_l1_or_cosine
median_depth_consistency
depth_quantile consistency if exposed
edge-aware normal weighting
```

5. Tests:

- normal target shape mismatch fails loudly
- zero valid mask returns zero loss and finite gradients
- rendered normal backward touches expected state
- config smoke with normal loss runs one step

6. Experiments:

- nearest0040 raw row + self-normal loss
- nearest0040 raw row + external normal if cached
- freeze color vs train geometry controls

Acceptance:

- Normal/depth supervision either improves raw heldout SSIM or is logged as a
  negative result with artifacts.

### P1.6 Densification, pruning, resampling, contribution/error EMA

Status: partial local resample/grow plumbing exists; paper-scale system not
complete.

Current evidence:

- Completion audit does not require paper grow/prune behavior.
- The paper uses adaptive capacity as part of its quality story.
- Our clean rows often fail from support/track/material quality rather than
  mere point count, but adaptive capacity is still part of a faithful system.

Deliverables:

1. Audit current `resample_*` keys in `train_powerfoam_metal.py`.

2. Define paper-like stats:

```text
contribution EMA
point-error EMA
visibility fraction
screen-space gradient proxy
normal/depth residual proxy
```

3. Implement operations:

```text
grow/split high-error cells
prune low-contribution cells
resample cells from high-error regions
preserve material/detail-site state during split
rebuild adjacency after topology change
reset optimizer state safely
```

4. Tests:

- split increases cell count and preserves finite render
- prune decreases cell count and preserves finite render
- optimizer state remains valid
- adjacency offsets remain consistent
- one-step smoke after grow/prune works

5. Experiments:

- nearest0040 raw row with adaptive capacity
- 4K synthetic trainability with one grow/prune event at lower resolution first
- compare to fixed 3543 cells

Acceptance:

- Adaptive capacity either improves raw heldout metrics or is documented as a
  bounded/negative lever.

## P1 - CUDA/Metal Parity Beyond Tiny Fixtures

### P1.7 Larger official fixture suite

Status: partial.

Current evidence:

- Official CUDA/Warp fixture exists for a tiny height+SV case.
- Targeted Direct/Metal official parity nodes pass.
- CUDA-vs-Metal smoke quality does not match, but it is a training/config
  smoke, not a strict numeric fixture.

Deliverables:

1. Generate official CUDA/Warp fixtures on Modal for:

```text
constant features, tiny
height+SV, tiny
height+SV, 9 texel sites
nonzero ray origin / posed camera
fisheye or generic ray bundle if upstream supports it
multiple cells with Cech false-positive edges
```

2. Fixture schema must include:

```json
{
  "official_commit": "...",
  "texture_temperature": 10.0,
  "camera_model": "...",
  "seed": ...,
  "state_tensors": "...",
  "forward_outputs": "...",
  "backward_gradients": "..."
}
```

3. Add tests:

- Direct Torch matches each official fixture
- Metal matches stable channels for each official fixture
- tests skip only when fixture absent; strict CI mode requires fixtures

4. Add verifier:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_official_fixture_suite.py \
  --require-all
```

Acceptance:

- More than one fixture.
- At least one fixture includes nonzero camera origin / posed rays.
- At least one fixture includes high-detail height+SV.

### P1.8 Training-level CUDA/Metal comparison

Status: partial smoke only.

Problem:

Even if tiny forward/backward fixtures match, training-level behavior can still
diverge due to:

- optimizer schedules
- background handling
- initialization
- adjacency construction
- precision
- random seeds
- loss definitions

Deliverables:

1. Freeze a micro contract:

```text
same clip
same frame count
same render size
same cells
same seed
same background
same init
same steps
same texel sites / SV DoF
same loss weights where possible
```

2. Run lanes:

- official static CUDA
- dynamic appearance CUDA
- dynamic geometry CUDA once available
- Metal static
- Metal dynamic geometry once available

3. Save:

```text
summary.json
cuda_vs_metal_summary.json
preview panels
train/eval metric histories
timing breakdown
```

4. Add comparison tolerances:

- exact numeric parity is not expected after independent optimization
- but forward fixture parity should be exact-ish before training
- training smoke should compare trends and timings

Acceptance:

- A single report clearly states what matches, what does not, and whether
  divergence is expected.

## P1 - 4K Performance Is Verified, Not Yet Optimized

### P1.9 Define the real 4K target

Status: needs decision.

Current evidence:

Saved synthetic 4K selected `cech_aabb` height+SV raytrace:

```text
1024 cells total median ~1016 ms
4096 cells total median ~1014 ms
```

4K trainability artifact:

```text
forward_ms       1195.255
backward_ms      2181.051
after_forward_ms 1217.304
```

Problem:

These prove that 4K forward/backward/trainability exists. They do not prove
that the renderer is fast enough for the desired use case. "Fast" needs a
target:

```text
interactive preview?
offline training?
single-image fit?
video training?
browser runtime?
comparison against 3DGS?
```

Deliverables:

1. Add a performance target doc section:

```text
TODO/powerfoam_performance_targets.md
```

or add a section to this file with agreed targets:

```text
4K forward target:       <X ms>
4K backward target:      <Y ms>
4K train step target:    <Z ms>
cell count target:       1024 / 4096 / 65536
feature dim target:      3 / 16 / 32
hardware:                M3 Max? M4? L40S? A100?
```

2. Benchmark matrix:

```text
resolution: 128, 256, 512, 1024, 2160p, 4K
cells: 1024, 4096, 16384, 65536
feature modes: RGB, F16, F32, height+SV
topology: cech_aabb, regular_triangulation
backend: tiled, raytrace, streaming
```

3. Add p50/p90/p95, not only medians.

Acceptance:

- We can say whether current 4K is "fast enough" for a concrete target, not in
  the abstract.

### P1.10 Optimize selected 4K path

Status: partial.

Known opportunities:

- raytrace replay cap and event-list memory
- selected `cech_aabb` is faster than regular topology but may be less correct
  on real views
- regular topology is correct but slower
- current 4K synthetic times are around one second total, not real-time

Deliverables:

1. Profile:

```text
adjacency build
start-cell selection
ray traversal
event replay
normal-distance output
height endpoint material sampling
SV color eval
gradient accumulation atomics
optimizer step
```

2. Candidate optimizations:

- tile-level ray packet coherence
- two-stage traversal: cheap coarse start + local walk
- compressed event replay
- split kernels for RGB-only vs full aux
- regular graph traversal cache
- mixed precision for adjacency diffs/SV weights where parity permits
- precomputed ray bundles for static cameras
- cap-specialized kernels

3. Tests:

- each optimization must run `raytrace_check.py`
- each optimization must run official fixture nodes if it touches math
- each optimization must update 4K benchmark JSON

Acceptance:

- At least one meaningful speed target improves without losing parity.

## P2 - Clean Geometry / Initialization

### P2.1 Stronger clean multiview geometry

Status: partial with many negative probes.

Current evidence:

- DeepView nearest0040 SIFT artifact has `3543` points and good long temporal
  tracks but unique camera support p90 is still `2`.
- ALIKED/LightGlue pycolmap wheel lacks ONNX.
- COLMAP CLI ONNX route starts but cheap real DeepView probes were sparse.
- Plane sweep and inlier plane sweep were not enough.
- Close-overlap heldout did not solve the quality problem.

Deliverables:

1. Build a geometry intake matrix:

```text
SIFT pycolmap known pose
COLMAP CLI ALIKED brute force
COLMAP CLI ALIKED LightGlue
HLOC ALIKED/LightGlue
dense plane sweep
monocular depth guided multiview
external EX4DGS as non-clean upper bound
```

2. Add cheap gates before expensive runs:

```text
point_count >= 2000
unique_camera_p90 >= 3
unique_frame_p90 >= 4
reproj_median <= 4px
reproj_p90 <= 8px
train_visible_fraction >= threshold
heldout_projection_support >= threshold
```

3. If a cheap probe passes, run the full 1024px/8cam/4frame build.

4. Add verifier fields:

```text
artifact_kind
camera_model
feature_type
matcher_type
unique_camera_track_mean/p90
unique_frame_track_mean/p90
point_count
train_visible_count
heldout_visible_count
```

Acceptance:

- A clean artifact with stronger distinct-camera support than current SIFT
  artifact.
- Or a documented negative result that proves DeepView camera geometry is not
  sufficient with local feature methods.

### P2.2 Dataset/source expansion

Status: not started for this PowerFoam acceptance target.

Problem:

DeepView 03 Dog / camera_0040 may be a hard scene/camera for clean local
PowerFoam, and current conclusions may be overfit to one scene.

Deliverables:

1. Pick 3 heldout-validation scenes:

- one easy static-ish scene
- one medium scene
- one hard dynamic/parallax scene

2. For each:

- build clean known-pose SIFT artifact
- run raw PowerFoam 1-step init
- run short official-objective train
- run calibrated eval
- run matched splat baseline

3. Add `BASELINES.md` section:

```text
PowerFoam cross-scene raw/calibrated/splat comparison
```

Acceptance:

- We know whether current weakness is scene-specific or representation-wide.

## P2 - Traversal / Topology Correctness

### P2.3 Regular topology speed

Status: correct but slower.

Current evidence:

- Regular triangulation raytrace parity passes on synthetic tests.
- Regular topology improved real DeepView support/quality in earlier rows.
- Regular 4K benchmark is slower than selected `cech_aabb`.

Deliverables:

1. Profile regular traversal:

```text
degree distribution
walk step count
event count
missing/extra edges vs cech
time by kernel section
```

2. Optimize:

- lower-degree regular graph representation
- edge ordering by ray direction
- start-cell heuristics for regular graph
- hybrid Cech + regular fallback

3. Real-scene test:

- selected checkpoint render with Cech
- selected checkpoint render with regular
- measure alpha/residual/support differences

Acceptance:

- A rule for when to choose `cech_aabb` vs `regular_triangulation`.
- If regular is the quality path, it needs a speed path.

### P2.4 Dynamic adjacency

Status: not started.

Problem:

Full dynamic geometry means cells move. Static adjacency can become invalid:

- missing true neighbors causes wrong cell extent
- false-positive neighbors are safe but expensive
- rebuilding every frame may be too slow

Deliverables:

1. Conservative dynamic adjacency strategy:

```text
union graph over sampled time endpoints
or inflated Cech radius
or rebuild every K steps / K frames
```

2. Tests:

- moving two-cell fixture where true neighbor relation appears/disappears
- union graph matches dense reference
- missing-edge regression fails

3. Metrics:

```text
avg_degree(t)
max_degree(t)
missing_dense_edges(t)
false_positive_edges(t)
render error vs dense
speed overhead
```

Acceptance:

- Dynamic geometry smoke has a correctness-mode adjacency path.

## P2 - Objective / Evaluation

### P2.5 Heldout-first objective design

Status: partial; many train-view overfit negatives.

Current evidence:

- Multiple runs improve source PSNR while heldout degrades.
- Raw/calibrated gap suggests color and structure are entangled.
- Existing official-objective fastwarmup improves source more than heldout.

Deliverables:

1. Add diagnostics that predict heldout degradation from train views:

```text
multi-view consistency loss
neighbor material agreement
view-cycle reconstruction
support overlap regularity
depth/normal self-consistency
```

2. Add no-heldout leak controls:

- losses must use train views only
- heldout used only for eval
- no target heldout features in source-world decode

3. Add objective ablations:

- MSE + SSIM
- edge/highpass loss
- normal/depth loss
- material smoothness
- graph ARAP/support regularization

Acceptance:

- A run where post-initial raw heldout improves, not just calibrated metrics or
  source metrics.

### P2.6 Metric suite beyond PSNR/SSIM

Status: partial.

Problem:

PSNR/SSIM alone cannot tell whether failure is:

- color/exposure
- structure
- blur
- support coverage
- depth ordering
- dynamic motion
- feature quality

Deliverables:

1. Add metrics:

```text
LPIPS if dependency acceptable
edge L1
highpass PSNR
foreground/masked metrics
alpha residual metrics
depth-normal metrics
feature PCA/video consistency
temporal flicker/motion metrics
```

2. Add artifact panels:

```text
GT | raw | calibrated | residual | alpha | depth | normal | local material coords
```

3. Add ranking table that separates:

```text
raw RGB quality
calibrated RGB quality
geometry/support quality
feature quality
temporal quality
```

Acceptance:

- The next "is it better?" answer does not rely on a single scalar.

## P2 - Paper/System Fidelity

### P2.7 Official paper training schedule parity

Status: partial.

Current evidence:

- Warmup bug fixed.
- Config-driven warmup overrides exist.
- Short probes use compressed warmups.

Deliverables:

1. Compare official schedules:

```text
points/radii/density/quaternion/texel/height/SV LR init/final
warmup steps
regularizer decay
grow/prune cadence
background policy
normal/depth supervision
```

2. Create config:

```text
src/train_configs/local_mac_powerfoam_metal_official_schedule_nearest0040_128_16f.jsonc
```

3. Run a short and long schedule:

- short smoke for stability
- longer run only if short heldout does not immediately degrade

Acceptance:

- A table of official-vs-local schedule differences.
- A run proving whether schedule fidelity helps raw heldout.

### P2.8 Paper primitive variants and ablation table

Status: partial.

Deliverables:

Rows:

```text
constant bounded cells
surface-linear
oriented surface
texel surface
height texel
height+SV
height+SV + normal/depth
height+SV + adaptive capacity
feature foam F16/F32
```

For each:

```text
train PSNR/SSIM
heldout PSNR/SSIM
raw/calibrated
speed
memory
gradient coverage
failure notes
```

Acceptance:

- A durable table in `BASELINES.md` or separate `research_notes/foam_papers`
  note.

## P2 - Browser / Runtime / Export

### P2.9 Browser/runtime story

Status: not started.

Problem:

Metal implementation is local training/backend work. If PowerFoam or feature
foam becomes the representation, browser/runtime export needs a separate plan.

Deliverables:

1. Define export state:

```text
centers
radii
density
quaternion/frame
texel sites
height
SV axes/RGB or feature payload
adjacency graph
calibration transform if used
```

2. Decide browser renderer:

- WebGPU raster approximate
- WebGPU raytrace/walk
- convert foam to splat-like proxy
- offline bake feature/rgb video

3. Add small export smoke:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/export_powerfoam_state.py \
  --checkpoint <checkpoint> \
  --output outputs/browser_powerfoam/export.json
```

Acceptance:

- A future browser task knows whether foam is intended as training-only,
  runtime representation, or source for baked splats/features.

## P3 - Documentation And Hygiene

### P3.1 Update canonical TODO/status files

Status: complete for the supersession note; ongoing for future status hygiene.

Resolved issue:

`TODO/powerfoam_full_reproduction_todo.md` contains historical audit text that
predates the final calibrated completion audit. This new file records remaining
work, but the canonical TODO should point here to avoid stale conclusions.

Completed deliverables:

1. Add a supersession note at the top of:

```text
TODO/powerfoam_full_reproduction_todo.md
```

2. It should say:

- completion audit is green under calibrated eval semantics
- raw-quality improvement and full multicam dynamic-geometry quality remain
- detailed remaining backlog is this file

Acceptance:

- A future agent opening the old file will not think the official fixture or
  completion audit is still blocked in the old way.

### P3.2 Keep notes append-only and searchable

Status: ongoing.

Deliverables:

For every major future run, add:

```text
agent_notes/loose_notes/{YYYY-MM-DD_HH-MM-SS}_{topic}.md
```

Required fields:

```text
what changed
why it was tried
commands
outputs
metrics
what failed
what was ruled out
next test
```

Only add to `agent_notes/key_learnings.md` when a result changes future
reasoning.

### P3.3 Clean dirty repo surfaces before publication

Status: needed before any commit/PR.

Current likely changed/untracked surfaces include:

```text
research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py
research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py
tests/test_powerfoam_paper_acceptance.py
agent_notes/key_learnings.md
agent_notes/loose_notes/2026-05-06_09-58-23_powerfoam_cuda_and_eval_color_calibration_gate.md
```

There are many other untracked PowerFoam files in the repo from prior work. Do
not blindly commit all untracked files.

Deliverables:

1. Run scoped status:

```bash
git status --short -- \
  TODO/powerfoam_remaining_work_after_completion_audit_2026-05-06.md \
  TODO/powerfoam_full_reproduction_todo.md \
  research_experiments/dynamic_foam \
  tests/test_powerfoam_paper_acceptance.py \
  agent_notes
```

2. Decide commit scope:

- verifier/audit/test changes
- notes/TODO docs
- CUDA smoke docs
- Metal calibration code
- unrelated older PowerFoam work

3. Avoid bundling unrelated dirty tree changes.

Acceptance:

- A clean, intentional commit set or a handoff note listing what remains
  uncommitted.

## Suggested Next Work Order

1. Improve raw uncalibrated PowerFoam quality or document the next negative
   result with SSIM/depth/normal/material diagnostics.
2. Add strict OPENCV_FISHEYE splat-camera parity only if the next claim needs a
   lens-matched splat comparison; the no-code-change same-split/pinhole
   comparator already exists.
3. Extend the dynamic Metal trainer to the multicam/nearest0040 loader and run a
   heldout-quality dynamic-geometry row.
4. Scale the CUDA dynamic-geometry fork beyond the 64px/4f/5-step smoke after
   the minimal scene-state fork remains stable.
5. Make feature foam contract explicit and run F16/F32 Metal feature-foam
   smokes.
6. Add raw SSIM failure diagnostics: edge/highpass/residual/depth/normal/local
   material coordinate panels.
7. Improve clean multiview geometry only after cheap artifact gates pass.
8. Define real 4K performance targets and optimize against them.
9. Add official-style normal/depth supervision and adaptive capacity.
10. Keep `BASELINES.md`, completion audit, loose notes, and key learnings in
    sync as new measured rows land.

## Definition Of Done For The Next Honest Claim

### Claim: "Raw PowerFoam beats splats"

Done only when:

- same split, same frames, same render size
- raw PowerFoam metrics pass without eval color calibration
- matched splat baseline is run
- metrics include PSNR, SSIM, L1, and at least one structure-sensitive metric
- artifacts are in `BASELINES.md`

### Claim: "Dynamic geometry foam works"

Done only when:

- time changes centers/radii/orientation/height, not only RGB
- zero dynamic coefficients reproduce static
- gradients flow into dynamic geometry coefficients
- alpha/support changes over time in a controlled probe
- appearance-freeze control shows motion matters

### Claim: "Feature foam replaces feature splats"

Done only when:

- F16 and F32 foam render/train paths work
- matched feature splat baseline exists
- colorizer contract is identical or differences are documented
- heldout feature/RGB metrics improve or the result is logged as negative

### Claim: "CUDA and Metal match"

Done only when:

- fixture parity covers more than one tiny case
- at least one posed/nonzero-origin fixture exists
- stable forward/backward channels match within documented tolerances
- training-level smoke comparison explains expected divergences

### Claim: "4K is fast"

Done only when:

- target latency is explicit
- benchmark matrix covers target resolution/cells/features
- p50/p95 are reported
- comparison to selected baseline exists
- quality/parity gates stay green
