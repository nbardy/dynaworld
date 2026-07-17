# STAR Batched Sparse Report Bootstrap Cleanup

## Context

The trainer-unification goal is still active. The useful pattern has been to
remove live duplication at shared boundaries without turning the trainer and
benchmark code into a base-class rewrite. For STAR UVT feature-tube reports,
`research_experiments/star_uvt_feature_tubes/report_artifacts.py` now owns the
local report path bootstrap, root-relative report writes, JSON loading, parsing
helpers, timing helpers, and stats helpers.

This pass handled the remaining batched sparse-forward report pair that still
rebuilt Dynaworld/train/STAR-UVT roots locally.

## Changes

- Routed `sparse_forward_batched_target_vjp_profile.py` through
  `report_artifacts.ROOT`, `write_report_json(...)`,
  `write_report_text(...)`, and `distribution_stats(...)`.
- Routed `sparse_forward_batched_step_benchmark.py` through the same shared
  bootstrap and `distribution_stats(...)`.
- Removed local `Path(__file__)` root rebuilding, local train/variant root
  constants, local `sys.path` mutation, and import-order `# noqa: E402`
  comments from that script pair.
- Kept benchmark/profile math, report schemas, and timing loops local.

## Validation

Commands run from the Dynaworld root:

```bash
rtk env PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_target_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_step_benchmark.py \
  tests/test_star_uvt_report_artifacts.py
```

Result: exit 0.

```bash
rtk env PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_star_uvt_report_artifacts.py -q
```

Result: `13 passed in 0.14s`.

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
script_dir = Path('research_experiments/star_uvt_feature_tubes').resolve()
sys.path.insert(0, str(script_dir))
import sparse_forward_batched_target_vjp_profile as target
import sparse_forward_batched_step_benchmark as step
print(target.DEFAULT_OUT_BASE.name)
print(step.DEFAULT_OUT_BASE.name)
print(step.distribution_stats([1.0, 3.0])['mean'])
PY
```

Output:

```text
2026-05-19_star_uvt_sparse_forward_batched_target_vjp_profile
2026-05-19_star_uvt_sparse_forward_batched_step_benchmark
2.0
```

The known `uv run` parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
appeared in this environment before successful exits.

## State

This reduces another live STAR report duplicate. It does not claim that STAR
UVT training quality is solved; this was a report/profile plumbing cleanup.
Next cleanup passes should keep using targeted scans for active duplication,
then prefer deleting compatibility shims only after `rg` proves no live import
path remains.
