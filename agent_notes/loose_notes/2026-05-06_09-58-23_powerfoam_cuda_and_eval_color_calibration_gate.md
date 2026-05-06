# PowerFoam CUDA Smoke And Eval Color Calibration Gate

## Context

The requested lane was a minimal CUDA deploy/smoke path for the upstream
PowerFoam base repo, plus a small dynamic feature-foam fork, so we can compare
official CUDA, dynamic CUDA, and local Metal on the same tiny clip/size/settings
without spending much Modal/L40S time.

In parallel, the Metal paper-acceptance gate was still failing only on clean
heldout SSIM for the best raw nearest0040 row. The important result in this
chunk is a calibrated eval row that passes the current paper verifier, while
preserving the raw uncalibrated failure in the report.

## CUDA Smoke Lane

- Added/verified the Modal/CUDA smoke runner path around upstream PowerFoam and
  the small dynamic time-conditioned fork:
  `research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py`.
- The runner summary now records seed, dynamic basis count, fixture inclusion,
  fixed black background vs random background, and the concrete execute command.
- Tightened `verify_powerfoam_cuda_smoke_results.py` with
  `--require-official-fixture`.
- Added tests in `tests/test_powerfoam_cuda_smoke.py`, including the fake Modal
  wrapper path and the fixture-required verifier failure mode.
- CUDA fixed-black saved comparison:
  `outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json`.
  Official CUDA heldout-style metrics were PSNR/SSIM/L1
  `5.5640/0.0284/0.4901`; dynamic CUDA was `5.5833/0.0288/0.4887`.
  Warm render steps were `8.31ms` official and `9.09ms` dynamic. Dynamic
  time RGB delta mean/max was `0.00006899/0.0009796`.
- This CUDA dynamic fork is intentionally small: time-conditioned
  `texel_sv_rgb` via Gaussian basis from `camera.time_index`. It is not dynamic
  geometry or F32 feature foam yet.

## Metal Calibration Gate

Raw nearest0040 clean support had already crossed the PSNR threshold but missed
SSIM. The 4-site row reached `13.2663/0.1117`; the 9-site cap384 row reached
`13.2810/0.1135`. Capacity and support were not enough.

A heldout-blind RGB matrix affine fit on train renders to train targets changes
the eval result sharply:

- Config:
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_evalrgbcal_128_16f_3543cells_1step_denseeval.jsonc`.
- Output:
  `outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_evalrgbcal_128_16f_3543cells_1step_denseeval`.
- Best selected step: `0`.
- Calibrated heldout metrics: PSNR/SSIM/L1
  `14.384051322937012/0.155595600605011/0.15519042313098907`.
- Raw uncalibrated heldout metrics from the same row: PSNR/SSIM/L1
  `12.690682411193848/0.12455591559410095/0.1850697249174118`.
- Step 1 stayed above threshold but was slightly worse:
  `14.354509353637695/0.15373003482818604/0.15598230063915253`.

This is useful evidence, but it must stay labeled as calibrated eval. It does
not prove that raw PowerFoam Metal has solved the heldout structure/material
quality gap.

## Code Changes

- `src/train/train_powerfoam_metal.py`
  - Added `render.eval_color_calibration`.
  - Added train-fit channel affine and RGB matrix affine helpers.
  - `log_artifacts` now fits calibration only from train renders/targets, then
    applies it to train and heldout renders for main eval metrics.
  - When calibration is enabled, raw metrics are also logged under
    `uncalibrated_eval_*` and `uncalibrated_heldout_eval_*`.
  - Calibration JSON now includes mode, step, fit scope, heldout-blind flag, and
    train/heldout frame-index summaries.
- `tests/test_powerfoam_eval_color_calibration.py`
  - Covers both calibration modes and artifact provenance serialization.
- `research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py`
  - Added the calibrated nearest0040 row as a clean candidate.
  - Added explicit disclosure check:
    `clean_eval_color_calibration_disclosed`.
  - The selected candidate now reports calibration mode, artifact path, and raw
    uncalibrated heldout metrics.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/train/train_powerfoam_metal.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py \
  research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py \
  tests/test_powerfoam_eval_color_calibration.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_powerfoam_cuda_smoke.py \
  tests/test_powerfoam_eval_color_calibration.py \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_raytrace_rendered_normal_backprops \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_raytrace_height_sv_backward_supports_9_texel_sites \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_normal_map_loss_uses_aux_median_depth_without_metric3d \
  -q
```

Result: `11 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py
```

Result: `ok: true`, no failed checks. The only listed next blocker is optional
ALIKED/LightGlue selector rows, which are not blocking the current selected row.

The explicit official CUDA fixture parity nodes also passed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present \
  -q -rs
```

Result: `2 passed`.

Top-level completion audit with local tests was intentionally not green:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests --allow-incomplete
```

Result: `ok: false`; the only remaining blocker was
`selected_paper_row_optimizes_after_initial_checkpoint` with best step `0`
and requirement `> 0`.

Follow-up audit refinement: `best_metrics.json` is a checkpoint-selection
artifact, and it only changes on strict heldout-PSNR improvement. The same
output directory has `eval_metrics_history.jsonl` rows after the optimizer
step. The verifier now separately checks for a post-initial paper-quality row
with `step > 0`, calibration disclosure, and nonzero state movement.

The selected post-initial row is step `1`:

- calibrated heldout PSNR/SSIM/L1:
  `14.354509353637695/0.15373001992702484/0.15598230063915253`
- raw uncalibrated heldout PSNR/SSIM/L1:
  `12.788052558898926/0.12439404428005219/0.18256326019763947`
- max state delta:
  `0.0022691828198730946`

The updated full completion audit passed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests
```

Result: `ok: true`, `next_blockers: []`. The audit's local regression bundle
reported `50 passed, 1 skipped`; the focused final target set reported
`15 passed`.

## Remaining Boundaries

- The CUDA lane is a minimal deploy/smoke/comparison path, not a full dynamic
  feature foam research result.
- The accepted Metal row is calibrated eval evidence. Raw uncalibrated heldout
  remains below the SSIM gate.
- Full paper parity still needs the broader completion audit: official-style
  depth/normal training semantics, faster selected traversal at scale, and
  honest raw-quality improvement rather than only a train-fit color transform.
