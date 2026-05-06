# PowerFoam Raw Quality Gate And Subagent Plans

Date: 2026-05-06

## Context

The completion audit was green for calibrated PowerFoam evidence, but it was
still too easy to misread that as raw PowerFoam quality. The selected
nearest0040 row passes only after heldout-blind train-fit RGB matrix eval
calibration.

## Code Changes

Implemented P0.1 raw-vs-calibrated verifier split:

- `research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py`
  now reports `raw_quality_ok` and `calibrated_quality_ok`.
- Added raw metric helpers that use `uncalibrated_heldout_eval_*` whenever eval
  color calibration is active, and fall back to `heldout_eval_*` only when
  `eval_color_calibration == "none"`.
- Added `--require-raw-quality` and `--allow-raw-step0-acceptance`.
- Added `post_initial_raw_quality_rows(...)`; default requires `step > 0`,
  state motion, calibration disclosure, and raw threshold pass.
- `research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py`
  forwards `--require-raw-quality` and exposes raw status as nonblocking in the
  default audit.
- `tests/test_powerfoam_paper_acceptance.py` now covers calibrated-pass/raw-fail,
  calibrated raw pass, no-calibration fallback, post-initial raw filtering, and
  explicit step-0-only acceptance.

## Verification

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_powerfoam_paper_acceptance.py -q
```

Result: `7 passed`.

Normal paper audit:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py
```

Result: `ok: true`, `raw_quality_ok: false`,
`calibrated_quality_ok: true`.

Strict raw paper audit:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_paper_acceptance.py \
  --require-raw-quality
```

Result: exit `1`, `ok: false`.

Selected row:

```text
calibrated PSNR/SSIM 14.384051322937012 / 0.155595600605011
raw PSNR/SSIM        12.690682411193848 / 0.12455591559410095
```

Post-initial calibrated row:

```text
step 1 calibrated PSNR/SSIM 14.354509353637695 / 0.15373001992702484
step 1 raw PSNR/SSIM        12.788052558898926 / 0.12439404428005219
max state delta             0.0022691828198730946
```

Strict raw failure reasons:

- `clean_raw_heldout_psnr_threshold`: raw `12.690682411193848 < 13.0`.
- `clean_raw_heldout_ssim_threshold`: raw `0.12455591559410095 < 0.15`.
- `clean_post_initial_raw_quality_row`: no post-initial raw row passes.

Top-level completion audit:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests
```

Result: `ok: true`, `next_blockers: []`; raw status is nonblocking and red.

Top-level strict raw audit:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests --require-raw-quality
```

Result: exit `1`, blockers `paper_acceptance_verifier` and
`paper_acceptance_raw_quality_status`.

## Subagent Findings Recorded In Backlog

Audit/verifier agent:

- Confirmed the original paper verifier selected by calibrated PSNR and used raw
  metrics only as disclosure.
- Provided the raw helper/test checklist used for this implementation.

Baseline/measurement agent:

- P0.2 should run a same-split free dynamic 3DGS baseline against the raw
  nearest0040 PowerFoam row.
- The no-code-change splat baseline can match split, frames, render size,
  primitive count, steps, seed, and heldout camera, but not lens model because
  the current splat baseline trainer constructs pinhole cameras while the
  PowerFoam row uses OPENCV_FISHEYE.

Dynamic/feature foam agent:

- P0.3/P0.4 should not be counted as complete by adding more RGB/feature
  residuals.
- Dynamic geometry must show alpha/support changes from time-conditioned
  geometry with color/features frozen.
- Current CUDA dynamic fork is appearance-only: `texel_sv_rgb(t) = base + B(t)
  coeff`.

Geometry/quality agent:

- Next raw-quality diagnostic should extend
  `diagnose_powerfoam_heldout_error.py` with `--calibration-json` and SSIM
  luminance/contrast/structure maps.
- The decision split is color/exposure versus geometry/material: if calibration
  repairs luminance/contrast but structure stays bad around normals/depth/edges,
  prioritize geometry/material rather than another color affine.

## Next Work

P0.1 verifier plumbing is done, but raw quality is not fixed. Continue with:

1. P0.2 matched nearest0040 free dynamic 3DGS baseline config/run.
2. P1.4 raw/calibrated SSIM component diagnostic.
3. P0.3 minimal Metal dynamic geometry alpha/support causality tests.
4. P0.4 CUDA geometry fork after the Metal state contract is proven.
