# WorldFoam Owner-Run Bridge And Paper Table Update

## What changed

- Added the WorldFoam owner-run/Metal comparison row to the shared paper-runner
  table contract.
- Regenerated
  `outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.json` and
  `summary.md` after the WorldFoam comparison artifact became available.
- Updated `TODO/README.md` and `EXPERIMENTS.md` so future agents see the bridge
  row as green rather than pending.

## Evidence

- WorldFoam comparison artifact:
  `outputs/benchmarks/2026-07-08_worldfoam_owner_run_metal_comparison_report/summary.json`.
- Bridge summary:
  `owner_run_metal_comparison_rows_ok=true`,
  `bridge_scope=contract_plus_visual_capacity_smoke`, and
  `paper_ready=false`.
- Shared table summary after regeneration: seven green evidence rows, three
  representation rows, `paper_ready=false`, and only
  `paper_quality_benchmark_table` missing.
- Focused paper-table test passed:
  `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_paper_runner_table_report.py -q`
  reported `7 passed`.

## Boundary

The WorldFoam bridge is not a full optical-transfer parity proof inside the
Metal shader. It only joins the CPU optical-transfer contract with the current
WorldFoam/PowerFoam Metal visual-capacity lane.

## Remaining work

The next runner is the final paper-quality benchmark table across World Tubes,
WorldFoam, and dynamic 3DGS. That table should consume the green runner
evidence already present and add matched dataset/config/metric rows rather than
creating another local fixture-only proof.
