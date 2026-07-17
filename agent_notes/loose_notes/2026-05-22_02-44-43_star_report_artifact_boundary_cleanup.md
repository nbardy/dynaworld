# STAR Report Artifact Boundary Cleanup

## Context

The active modularization goal is to keep moving repeated train/report script
boundaries into small reusable helpers without changing the experiment logic.
`report_artifacts.py` is the shared STAR UVT feature-tube report boundary for
root-relative paths, bootstrap, report JSON/text/CSV writes, and report-shaped
JSON reads.

This pass handled three scripts that still bypassed parts of that boundary.

## Changes

- `background_cheat_diagnostic.py` now imports `report_artifacts` before
  train-local objective modules, then writes report JSON/markdown with
  `write_report_json(...)` and `write_report_text(...)`.
- `compare_compact_visual_vjp_gate.py` now uses
  `load_report_json(...)` instead of a local JSON-open helper.
- `star_uvt_vjepa_vs_gaussian_comparison.py` now imports `ROOT` from
  `report_artifacts` instead of rebuilding it with `Path(__file__)`.
- Kept diagnostic math, comparison table construction, and summary schemas
  local.

## Validation

Commands run from the Dynaworld root:

```bash
rtk env PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/background_cheat_diagnostic.py \
  research_experiments/star_uvt_feature_tubes/compare_compact_visual_vjp_gate.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py
```

Result: exit 0.

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
script_dir = Path('research_experiments/star_uvt_feature_tubes').resolve()
sys.path.insert(0, str(script_dir))
import background_cheat_diagnostic as bg
import compare_compact_visual_vjp_gate as compact
import star_uvt_vjepa_vs_gaussian_comparison as compare
print(bg.DEFAULT_ALPHAS[-1])
print(compact._mean([1.0, 3.0]))
print(compare.ROOT.name)
PY
```

Output:

```text
1.0
2.0
dynaworld
```

The first import smoke in this slice exposed an ordering bug: the background
diagnostic imported `objective` before `report_artifacts`, so direct import
without external `PYTHONPATH` failed with `ModuleNotFoundError: No module named
'objective'`. Moving `report_artifacts` to the first local import fixed it.
The known `uv run` parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
appeared before successful exits.

## State

This is report plumbing cleanup only. It does not rerun the background
ablation, compact VJP comparison, or V-JEPA-vs-Gaussian comparison, and it does
not change their conclusions.
