# World Tubes real-video media row

## Context

The shared paper-runner table still listed `world_tubes_real_video_media_rows`
as missing. This chunk extended the World Tubes decisive-demo report so it can
consume the saved 128px/16f WorldTubes/STAR UVT visual-compare artifact and
emit a verifier-backed real-video media row.

## Implemented

Updated:

- `research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py`
- `tests/test_star_uvt_projective_decisive_demo_report.py`
- `research_experiments/paper_runner_suite/paper_runner_table_report.py`
- `tests/test_paper_runner_table_report.py`

The decisive-demo runner now supports:

```bash
--include-saved-real-video
--saved-real-video-summary outputs/visual_comparisons/star_uvt_worldtubes_metal_128_16f_60step_2048tubes.json
```

It copies saved media into:

- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/contact_sheet.jpg`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/side_by_side.mp4`

It also emits report-side SVG artifacts:

- `fallback_heatmap.svg`
- `runtime_bars.svg`
- `memory_bars.svg`

The saved decisive-demo summary now has:

- `has_real_video_media_rows=true`
- `real_video_media_rows_ok=true`
- `real_video_min_psnr=21.768529415130615`
- `real_video_max_l1=0.054596319794654846`
- `real_video_min_artifact_count=5`

The paper table now removes `world_tubes_real_video_media_rows` from
`missing_ids`; remaining missing rows are:

- `worldfoam_owner_run_metal_comparison_rows`
- `paper_quality_benchmark_table`

## Verification

Passed:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_decisive_demo_report.py -q
```

Result:

```text
9 passed in 0.65s
```

Passed:

```bash
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --fixture-only \
  --include-saved-real-video \
  --out-dir outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture

PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --verify-report outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json
```

Passed:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_paper_runner_table_report.py -q
```

Result:

```text
7 passed in 0.02s
```

Passed:

```bash
PYTHONPATH=src/train uv run python research_experiments/paper_runner_suite/paper_runner_table_report.py \
  --out-dir outputs/benchmarks/2026-07-08_paper_runner_table_report

PYTHONPATH=src/train uv run python research_experiments/paper_runner_suite/paper_runner_table_report.py \
  --verify-report outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.json
```

## Important boundary

This fills a saved real-video media row for World Tubes, not a full real-video
benchmark sweep. The table is still `paper_ready=false`.

## Next work

1. Add WorldFoam owner-run/Metal comparison rows that consume or mirror the
   optical-transfer fixture contract.
2. Fill the final paper-quality benchmark table after the WorldFoam row is
   green.
