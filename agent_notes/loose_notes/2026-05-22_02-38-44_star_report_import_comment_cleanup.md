# STAR Report Import Comment Cleanup

## Context

Several STAR UVT matrix/sweep scripts already used `report_artifacts.py` as the
first local import for path setup, but still carried stale `# noqa: E402`
comments left over from older local `sys.path` preambles.

## Changes

- Removed stale import-order `# noqa: E402` comments from:
  - `support_birth_split_sweep.py`
  - `targetgrid_render_mode_trainer_matrix.py`
  - `sparse_forward_scale_matrix.py`
  - `sparse_forward_timing_repeat.py`
- Kept `report_artifacts` as the first local import because it intentionally
  establishes Dynaworld/train/STAR-UVT paths before `config_utils`,
  `star_uvt_render_modes`, and report-profile imports.

## Validation

Commands run from the Dynaworld root:

```bash
rtk env PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py \
  research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_scale_matrix.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_timing_repeat.py
```

Result: exit 0.

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
script_dir = Path('research_experiments/star_uvt_feature_tubes').resolve()
sys.path.insert(0, str(script_dir))
import support_birth_split_sweep as support
import targetgrid_render_mode_trainer_matrix as render_matrix
import sparse_forward_scale_matrix as scale
import sparse_forward_timing_repeat as repeat
print(support.BASE_CONFIG.name)
print(len(render_matrix.DEFAULT_MODES))
print(scale.DEFAULT_OUT_BASE.name)
print(repeat.DEFAULT_OUT_BASE.name)
PY
```

Output:

```text
star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_uncovered_from1500_lr001_5step_media.jsonc
6
2026-05-19_star_uvt_sparse_forward_scale_128_256_512
2026-05-19_star_uvt_sparse_forward_512_repeat_timing
```

The known `uv run` parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
appeared before successful exits.

## State

This is a readability cleanup around an already-shared bootstrap boundary. It
does not alter matrix/sweep execution behavior.
