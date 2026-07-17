# STAR Report Bootstrap Variant Root

## Context

`report_artifacts.py` already owned most STAR report helpers, but some directly
launched reports still rebuilt the same path bootstrap before importing train
or STAR UVT modules. `firstclass_scale_report.py` was the clean next slice: it
defined its own Dynaworld root, train root, STAR UVT variant root, `_ensure_paths`
helper, and imported `train_artifacts` directly for JSON/markdown writes.

## Changes

- Added `STAR_UVT_ROOT` to the `report_artifacts.py` import-time bootstrap.
- Simplified `firstclass_scale_report.py` to import `load_report_json`,
  `write_report_json`, and `write_report_text` from `report_artifacts`.
- Removed `firstclass_scale_report.py`'s local root constants and
  `_ensure_paths(...)` helper.
- Kept `config_utils` / `trainer_registry` imports local inside `_run_config`,
  preserving the existing lazy runtime import behavior for `--run`.
- Updated `TODO/trainer_landscape_unification.md` and `CODE_ORGANIZATION.md`.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py \
  tests/test_star_uvt_report_artifacts.py
```

Passed.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_star_uvt_report_artifacts.py -q
```

Passed: `13 passed in 0.13s`.

Import smoke:

```bash
uv run python - <<'PY'
import sys
from pathlib import Path
script_dir = Path('research_experiments/star_uvt_feature_tubes').resolve()
sys.path.insert(0, str(script_dir))
import report_artifacts
import firstclass_scale_report
print(report_artifacts.STAR_UVT_ROOT in [Path(p) for p in sys.path])
print(firstclass_scale_report.__name__)
PY
```

Passed and printed:

```text
True
firstclass_scale_report
```

The known parent-project `uv` warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
remained, but all commands exited `0`.
