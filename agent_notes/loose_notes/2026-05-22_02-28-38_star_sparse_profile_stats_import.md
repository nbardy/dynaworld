# STAR Sparse Profile Stats Import Cleanup

## Context

The earlier stats-helper cleanup moved the reusable zero-empty timing-summary
contract into `report_artifacts.summary_stats(...)`. A follow-up scan found two
profile scripts still reaching into the private
`star_uvt_feature1_wholegraph_profile._stats` helper:

- `star_uvt_sparse_forward_profile.py`
- `star_uvt_targetgrid_vjp_bridge_profile.py`

That private helper no longer exists, so this was both cleanup and a real
direct-profile import breakage.

## Changes

- Routed both scripts through `report_artifacts.summary_stats(...)`.
- Routed both scripts through the `report_artifacts` bootstrap root instead of
  rebuilding Dynaworld/train/STAR-UVT roots and mutating `sys.path` locally.
- Replaced local `DYNAWORLD_ROOT` path joins with `report_artifacts.ROOT`.
- Left each script's profile math, report schema, and markdown formatting local.
- Updated `TODO/trainer_landscape_unification.md` and `CODE_ORGANIZATION.md`.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_sparse_forward_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  tests/test_star_uvt_report_artifacts.py
```

Passed.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_star_uvt_report_artifacts.py -q
```

Passed: `13 passed in 0.14s`.

Import smoke:

```bash
uv run python - <<'PY'
import sys
from pathlib import Path
script_dir = Path('research_experiments/star_uvt_feature_tubes').resolve()
sys.path.insert(0, str(script_dir))
import star_uvt_sparse_forward_profile as sparse
import star_uvt_targetgrid_vjp_bridge_profile as bridge
print(sparse.DEFAULT_OUT_BASE.name)
print(bridge.DEFAULT_OUT_BASE.name)
print(bridge.summary_stats([4.0, 2.0])['mean'])
PY
```

Passed and printed:

```text
2026-05-19_star_uvt_sparse_forward_profile
2026-05-19_star_uvt_targetgrid_vjp_bridge_profile
3.0
```

The known `uv` parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
remained, but all commands exited `0`.
