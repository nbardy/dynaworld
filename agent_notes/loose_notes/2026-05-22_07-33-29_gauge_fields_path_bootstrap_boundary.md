# Gauge Fields Path Bootstrap Boundary

## Context

The Gauge Fields lane already had `research_experiments/gauge_fields/common.py`
for shared device, metrics, media, and JSON artifact helpers. Two direct
entrypoints still carried their own path preamble:

- `cheat_probe_material_gauge.py`
- `smiley_smoke.py`

Both needed the Gauge experiment directory on `sys.path` so `from train` would
resolve `research_experiments/gauge_fields/train.py`, while `common.py` also
needed `src/train` for shared losses/artifact helpers.

## Change

- Added `EXPERIMENT_DIR` and `ensure_sys_path(...)` to Gauge `common.py`.
- `common.py` now enforces path priority by removing an existing path before
  reinserting it. This matters because direct script execution already places
  the script directory on `sys.path`; blindly adding `src/train` ahead of it
  makes `from train` resolve `src/train/train.py`, which is wrong for this lane.
- Removed duplicated `EXPERIMENT_DIR`/`DYNAWORLD_ROOT`/`sys.path.insert(...)`
  preambles from:
  - `cheat_probe_material_gauge.py`
  - `smiley_smoke.py`
- Both scripts now import `DYNAWORLD_ROOT` from `common.py`.

## Validation

- First direct `--help` smoke caught the ordering bug where `src/train/train.py`
  shadowed Gauge-local `train.py`.
- After enforcing path priority:
  - `py_compile` passed for `common.py`, `cheat_probe_material_gauge.py`, and
    `smiley_smoke.py`.
  - `smiley_smoke.py --help` passed.
  - `cheat_probe_material_gauge.py --help` passed.
  - A path-order smoke confirmed the Gauge experiment dir is before
    `src/train` in `sys.path`.
  - targeted `git diff --check` passed.

## Current State

This keeps Gauge Fields separate from the main trainer surface while removing
duplicated script bootstrap. It is intentionally not a broader unification of
Gauge, PowerFoam, STAR UVT, and Token-GS optimizer/training semantics.
