# STAR Profile Bootstrap Cleanup

## Context

The STAR UVT feature-tube reports have been converging on
`report_artifacts.py` as the local script boundary for Dynaworld/train/STAR-UVT
path setup plus report-shaped artifacts. Several profile and diagnostic scripts
still carried the same explicit bootstrap preamble.

## Changes

- `alpha_only_visibility_profile.py` now relies on the shared report bootstrap.
- `dense_alpha_failure_diagnostic.py` now relies on the shared report bootstrap.
- `sparse_visual_loss_vjp_profile.py` imports `ROOT` from `report_artifacts` for
  its default config path and relies on the helper for path setup.
- `star_uvt_logit_handoff_rgb_vjp_profile.py` imports `ROOT` from
  `report_artifacts` for its default config/output paths and relies on the
  helper for path setup.
- Removed local `sys` imports, local Dynaworld/train/STAR-UVT root constants,
  and local `sys.path` mutation from that cluster.
- Kept profile math, checkpoint loading, colorizer/model construction, timing
  loops, and report schemas local.

## Validation

Commands run from the Dynaworld root:

```bash
rtk env PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py
```

Result: exit 0.

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
script_dir = Path('research_experiments/star_uvt_feature_tubes').resolve()
sys.path.insert(0, str(script_dir))
import alpha_only_visibility_profile as alpha
import dense_alpha_failure_diagnostic as dense
import sparse_visual_loss_vjp_profile as sparse_visual
import star_uvt_logit_handoff_rgb_vjp_profile as logit
print(alpha._mean([1.0, 3.0]))
print(dense.ALPHA_THRESHOLDS[0])
print(sparse_visual.DEFAULT_CONFIG.name)
print(logit.DEFAULT_OUT_BASE.name)
PY
```

Output:

```text
2.0
0.01
star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_manualvjp_from1500_lr001_5step_media.jsonc
2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile
```

The known `uv run` parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
appeared before successful exits. An earlier import smoke in this slice used a
nonexistent alpha-profile constant and failed with `AttributeError`; the
corrected import smoke above passed.

## State

This is another report/profile plumbing cleanup. It does not alter feature
kernel behavior, dense-alpha diagnostics, sparse-visual VJP math, or logit
handoff timing semantics. The remaining cleanup should continue from live-file
scans and avoid touching one-off WorldFoam/kernel probes unless they become
reusable trainer or benchmark entrypoints.
