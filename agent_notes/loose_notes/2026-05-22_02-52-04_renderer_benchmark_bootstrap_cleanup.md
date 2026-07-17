# Renderer Benchmark Bootstrap Cleanup

## Context

The previous benchmark cleanup added `src/benchmarks/benchmark_bootstrap.py` for
trainer/parity CLIs. The renderer benchmark CLIs still had repeated
project/train/benchmark root setup, plus local `sys.path` insert blocks for
vendored renderer packages.

This pass extends the shared helper without hiding renderer-specific choices.

## Changes

- Extended `benchmark_bootstrap.py` with:
  - `PROJECT_ROOT`
  - `BENCHMARK_DIR`
  - `ensure_sys_path(...)`
- `benchmark_bootstrap.py` now inserts both project root and `src/train`, so it
  supports scripts importing either `src.train.*` modules or train modules as
  top-level imports.
- Routed these scripts through the shared root/bootstrap helper:
  - `depth_aware_dof_demo.py`
  - `splat_renderer_benchmark.py`
  - `splat_renderer_accuracy.py`
  - `mac_renderer_stack_compare.py`
- Kept renderer-specific vendored path choices local:
  - Taichi splatting path in the splat renderer pair.
  - Fast-Mac variant and Taichi paths in the Mac stack comparison.
- Removed duplicated local project-root/train-root/benchmark-root discovery and
  raw `sys.path` mutation from that cluster.

## Validation

Commands run from the Dynaworld root:

```bash
rtk env PYTHONPATH=src/benchmarks:src/train:. uv run python -m py_compile \
  src/benchmarks/benchmark_bootstrap.py \
  src/benchmarks/depth_aware_dof_demo.py \
  src/benchmarks/splat_renderer_benchmark.py \
  src/benchmarks/splat_renderer_accuracy.py \
  src/benchmarks/mac_renderer_stack_compare.py
```

Result: exit 0.

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
bench_dir = Path('src/benchmarks').resolve()
sys.path.insert(0, str(bench_dir))
import benchmark_bootstrap
import depth_aware_dof_demo as dof
import splat_renderer_benchmark as bench
import splat_renderer_accuracy as acc
print(benchmark_bootstrap.PROJECT_ROOT.name)
print(dof.DEFAULT_OUTPUT_DIR.name)
print(bench.DEFAULT_CONFIG['device'])
print(acc.DEFAULT_CONFIG['baseline_device'])
PY
```

Output:

```text
dynaworld
depth_aware_dof_demo
auto
cpu
```

Mac stack import smoke:

```bash
rtk uv run python - <<'PY'
import sys
from pathlib import Path
bench_dir = Path('src/benchmarks').resolve()
sys.path.insert(0, str(bench_dir))
import mac_renderer_stack_compare as mac
print(mac.DEFAULT_BG)
PY
```

Output:

```text
(1.0, 1.0, 1.0)
```

The known `uv run` parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking `[project]`
appeared before successful exits.

## State

This is a bootstrap cleanup only. It does not rerun renderer benchmarks or
update performance standings. Raw-Metal and WorldFoam benchmark scripts still
own their specialized path setup because those are one-off third-party
integration probes, not generic trainer/renderer benchmark CLIs.
