# Dynamic Foam JSON Reader Wrapper Cleanup

## Context

After the Dynamic Foam report/path boundary cleanup, a few scripts still had
local JSON object reader wrappers even though `report_artifacts.load_report_json`
already owned the same contract.

## Change

- Removed the pass-through `load_json(...)` wrapper from
  `compare_powerfoam_to_splats_nearest0040.py`.
- Removed the pass-through `load_json(...)` wrapper from
  `diagnose_powerfoam_raytrace_support_gap.py`.
- Removed the duplicate `load_json_object(...)` implementation from
  `run_powerfoam_external_blockers.py`.
- Updated call sites to use `load_report_json(...)` directly.
- Moved `diagnose_powerfoam_raytrace_support_gap.py` onto the shared Dynamic
  Foam path bootstrap before importing `config_utils` and sibling verifier
  modules.

## Validation

- `py_compile` passed for the three touched scripts.
- Direct `--help` passed for:
  - `compare_powerfoam_to_splats_nearest0040.py`
  - `diagnose_powerfoam_raytrace_support_gap.py`
  - `run_powerfoam_external_blockers.py`
- The support-gap diagnostic initially failed direct `--help` because it
  imported `config_utils` before adding `src/train` to `sys.path`; the shared
  bootstrap fixed that.
- `tests/test_dynamic_foam_report_artifacts.py -q` passed: `5 passed`.
- Targeted `git diff --check` passed.

## Current State

Dynamic Foam simple report-object reading now has one owner. Tolerant readers,
JSON-list fixtures, Modal settings/config inputs, and embedded upstream runner
writes remain local because they have different semantics.
