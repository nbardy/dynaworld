# STAR Sparse Hidden Bootstrap Cleanup

## Context

The active modularization goal is to keep folding repeated trainer and
benchmark boundaries into small shared helpers without collapsing the actual
experiment math into a broad framework. The STAR UVT feature-tube report helper
already owns Dynaworld/train/STAR-UVT path bootstrap for direct report and
benchmark scripts.

Two native sparse hidden kernel benchmark scripts still repeated that local
bootstrap.

## Changes

- `sparse_hidden_sigmoid_mse_kernel_benchmark.py` now imports
  `report_artifacts.write_report_json` directly and relies on the helper's
  bootstrap to expose `train_devices` and the STAR UVT variant package.
- `sparse_hidden_target_area_kernel_benchmark.py` now follows the same pattern.
- Removed local `sys` imports, local `ROOT`/`STAR_UVT_ROOT` constants, local
  `sys.path` mutation, and import-order `# noqa: E402` comments from both
  scripts.
- Left kernel parity, scene construction, timing loops, and report payloads
  local.

## Validation

Commands run from the Dynaworld root:

```bash
rtk env PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/sparse_hidden_sigmoid_mse_kernel_benchmark.py \
  research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py
```

Result: exit 0.

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
script_dir = Path('research_experiments/star_uvt_feature_tubes').resolve()
sys.path.insert(0, str(script_dir))
import sparse_hidden_sigmoid_mse_kernel_benchmark as sigmoid
import sparse_hidden_target_area_kernel_benchmark as target_area
print(sigmoid.UVTRenderConfig.__name__)
print(len(target_area.TARGET_AREA_BACKWARD_MODES))
PY
```

Output:

```text
UVTRenderConfig
15
```

The known `uv run` parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
appeared before successful exits.

## State

This is a plumbing cleanup only. It does not change STAR UVT feature-kernel
behavior or promote any benchmark row. The useful next cleanup is still
live-file-driven helper routing: only remove local benchmark helpers when they
match an established shared contract.
