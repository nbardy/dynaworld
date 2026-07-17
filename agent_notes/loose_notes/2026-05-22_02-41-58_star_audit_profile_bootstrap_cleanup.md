# STAR Audit/Profile Bootstrap Cleanup

## Context

The active modularization goal is still to move repeated script/trainer
boundaries into small helpers without turning the STAR UVT experiment code into
a broad framework. `report_artifacts.py` is now the local STAR feature-tube
script boundary for Dynaworld/train/STAR-UVT path setup and report-shaped
artifact writes.

This pass handled older STAR audit/profile/orchestration scripts that still
rebuilt path roots before importing train helpers.

## Changes

- `firstclass_backward_breakdown.py` now uses the shared report bootstrap and
  writes JSON/markdown via `write_report_json(...)` /
  `write_report_text(...)`.
- `star_uvt_feature1_wholegraph_profile.py` now imports `report_artifacts`
  before train-local helpers and relies on that bootstrap.
- `star_uvt_vjepa_bridge_audit.py` imports `ROOT` from `report_artifacts` as
  `DYNAWORLD_ROOT` and drops its local train-root `sys.path` setup.
- `run_alpha_background_ablation.py` imports `ROOT` from `report_artifacts` as
  `DYNAWORLD_ROOT`, drops local train-root setup, and writes its config/result
  and summary artifacts through the report wrappers.
- Kernel/profile math, config audit logic, trainer orchestration, stdout log
  capture, and row schemas remain local.

## Validation

Commands run from the Dynaworld root:

```bash
rtk env PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py
```

Result: exit 0.

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
script_dir = Path('research_experiments/star_uvt_feature_tubes').resolve()
sys.path.insert(0, str(script_dir))
import firstclass_backward_breakdown as backward
import star_uvt_feature1_wholegraph_profile as wholegraph
import star_uvt_vjepa_bridge_audit as audit
import run_alpha_background_ablation as ablation
print(backward._mean([1.0, 3.0]))
print(wholegraph._fmt(1.234, digits=2))
print(audit.SELECTED_FAST_CONFIG.endswith('.jsonc'))
print(ablation.DEFAULT_OUTPUT_ROOT.name)
PY
```

Output:

```text
2.0
1.23
True
2026-05-21_alpha_background_ablation
```

The known `uv run` parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
appeared before successful exits. An earlier smoke in this slice asked for a
nonexistent wholegraph helper and failed with `AttributeError`; the corrected
import smoke above passed.

## State

This is plumbing cleanup only. It does not rerun the ablation, update benchmark
standings, or change the STAR UVT quality interpretation. Remaining cleanup
should keep following live scans and only fold repeated boundaries that already
have a clear shared helper contract.
