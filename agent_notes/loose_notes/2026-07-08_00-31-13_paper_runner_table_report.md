# Paper runner table report

## Context

The active goal is to build enough runner infrastructure to ablate and compare
World Tubes, WorldFoam, and the dynamic 3DGS baseline for the papers. The
previous chunks added:

- World Tubes decisive-demo fixture runner.
- World Tubes visibility stress runner.
- WorldFoam optical-transfer fixture runner.

This chunk added the first shared comparison/table surface.

## Implemented

Added:

- `research_experiments/paper_runner_suite/paper_runner_table_report.py`
- `tests/test_paper_runner_table_report.py`

The report consumes:

- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json`
- `outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json`
- `outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_capacity_128_clean_all_lanes.json`

It writes:

- `outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.json`
- `outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.md`

The verifier requires:

- World Tubes decisive-demo evidence row is present and `ok`.
- World Tubes visibility-stress evidence row is present and `ok`.
- WorldFoam optical-transfer evidence row is present and `ok`.
- Representation rows exist for World Tubes, WorldFoam, and dynamic 3DGS.
- The report stays `paper_ready=false` while required paper rows are missing.
- Missing rows include:
  - `world_tubes_real_video_media_rows`
  - `worldfoam_owner_run_metal_comparison_rows`
  - `paper_quality_benchmark_table`

## Verification

Passed:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_paper_runner_table_report.py -q
```

Result:

```text
7 passed in 0.04s
```

Passed:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/paper_runner_suite/paper_runner_table_report.py
```

Passed:

```bash
PYTHONPATH=src/train uv run python research_experiments/paper_runner_suite/paper_runner_table_report.py \
  --out-dir outputs/benchmarks/2026-07-08_paper_runner_table_report

PYTHONPATH=src/train uv run python research_experiments/paper_runner_suite/paper_runner_table_report.py \
  --verify-report outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.json
```

Saved summary:

- evidence rows: `6`
- green evidence rows: `6`
- representation rows: `3`
- missing rows: `3`
- `paper_ready=false`

## Important boundary

This is the table surface, not the final table. It deliberately keeps the
report incomplete until real-video/media rows, WorldFoam owner-run/Metal rows,
and final paper-quality benchmark rows exist.

## Next work

1. Extend `projective_decisive_demo_report.py` with real-video/media rows.
2. Add WorldFoam owner-run/Metal comparison rows that consume or mirror the
   optical-transfer fixture contract.
3. Fill the final paper-quality benchmark table after those rows are green.
