# PowerFoam P0 Completion Audit

Date: 2026-05-06 12:51 +07

## What Changed

- Tightened `research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py`
  so the top-level audit now covers the P0 artifacts directly:
  - P0.1 raw/calibrated quality fields are exposed at top level.
  - P0.2 same-split PowerFoam-vs-splat comparison checks row artifacts, linked
    configs, exact nearest0040 split, seed/sampling, eval/heldout metrics,
    primitive count, wall-clock field, `BASELINES.md`, and projection caveat.
  - P0.3 Metal dynamic-geometry check verifies summary artifacts, 16f/40-step/
    1024-cell geometry-only smoke scope, state deltas, alpha/support motion, and
    frozen dynamic-feature control.
  - P0.4 CUDA dynamic-geometry check verifies the strict Modal micro contract,
    same-run static/feature/geometry lanes, feature-lane RGB-only negative
    control, and geometry-lane alpha/support motion.
- Updated `TODO/powerfoam_remaining_work_after_completion_audit_2026-05-06.md`
  so P0.1-P0.4 no longer read as future work after their gates/smokes landed.
- Updated `TODO/powerfoam_full_reproduction_todo.md` supersession text so it no
  longer says the same-split splat comparison itself is missing.
- Saved final audit reports:
  - `outputs/powerfoam_completion_audits/p0_completion_audit_full_20260506.json`
  - `outputs/powerfoam_completion_audits/p0_completion_audit_require_raw_quality_20260506.json`

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py \
  --run-local-tests --require-raw-quality --allow-incomplete
```

## Results

Default calibrated/P0 audit:

```text
ok True
next_blockers []
raw_quality_ok False
calibrated_quality_ok True
```

Focused gates inside the audit:

```text
local Metal bundle: 50 passed, 1 skipped
P0.1 paper/raw gate tests: 7 passed
P0.3 dynamic Metal tests: 15 passed
P0.4 CUDA smoke tests: 8 passed
official direct parity node: 1 passed
official Metal parity node: 1 passed
raytrace parity script: passed
```

Strict raw-quality audit:

```text
ok False
next_blockers ["paper_acceptance_verifier", "paper_acceptance_raw_quality_status"]
raw_quality_ok False
calibrated_quality_ok True
```

## What This Proves

- P0.1 is complete as a gate: calibrated audit can be green while raw quality
  remains explicitly red and machine-readable.
- P0.2 has a measured same-split comparator for the no-code-change contract:
  raw PowerFoam vs free dynamic 3DGS on nearest0040 with raw/calibrated semantics
  separated and the pinhole-vs-OPENCV_FISHEYE caveat enforced.
- P0.3 has a minimal Metal geometry-only dynamic proof on explicit video:
  screen/center/radius motion changes alpha/support while dynamic features are
  frozen.
- P0.4 has a minimal CUDA scene-side dynamic-geometry fork on L40S with a
  feature-only RGB negative control and a geometry lane that changes
  alpha/support.

## What Remains Weak

- Raw PowerFoam heldout quality still fails the paper gate; the gate exists, but
  the representation quality problem is unsolved.
- The splat comparator is same split/scale but not strict lens parity: splats use
  legacy pinhole, PowerFoam uses OPENCV_FISHEYE.
- Metal dynamic geometry is explicit-video only, not nearest0040/multicam heldout
  quality.
- CUDA dynamic geometry is a 64px/4-frame/5-step smoke with the official fixture
  intentionally skipped; it proves causality/plumbing, not quality.
