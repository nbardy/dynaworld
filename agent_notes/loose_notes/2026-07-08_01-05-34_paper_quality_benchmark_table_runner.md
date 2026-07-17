# Paper Quality Benchmark Table Runner

## What changed

- Added
  `research_experiments/paper_runner_suite/paper_quality_benchmark_table_report.py`
  and `tests/test_paper_quality_benchmark_table_report.py`.
- The runner consumes the 128px/16-frame capacity visual comparison and derives
  matched media PSNR/L1 from the saved side-by-side videos for:
  - World Tubes / STAR UVT
  - WorldFoam / dynamic PowerFoam
  - dynamic 3DGS / fast-mac
- It carries native metrics where the underlying lane emitted them:
  STAR UVT from its JSON artifact and WorldFoam from `per_frame_metrics`.
  Dynamic 3DGS currently has media-split metrics only.
- Regenerated the shared paper-runner table so it now consumes this quality
  table as an evidence row.

## Saved artifacts

- Quality table:
  `outputs/benchmarks/2026-07-08_paper_quality_benchmark_table/summary.json`
  and `summary.md`.
- Shared table:
  `outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.json`
  and `summary.md`.

## Current numbers

- Quality-table scope: `capacity_128_local_video_smoke`.
- World Tubes / STAR UVT: 2048 tubes, 60 steps, 17.077s, media PSNR 21.807,
  media L1 0.0545, native PSNR 21.769, native L1 0.0546, native SSIM 0.519.
- WorldFoam / dynamic PowerFoam: 2048 cells, 80 steps, 32.811s, media PSNR
  17.777, media L1 0.0806, native PSNR 17.745, native L1 0.0801.
- Dynamic 3DGS / fast-mac: 4096 Gaussians, 60 steps, 89.572s, media PSNR
  18.643, media L1 0.0764.
- Shared table summary: 8 green evidence rows, 3 representation rows,
  `paper_ready=true`, and no missing IDs.

## Verification

- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_paper_quality_benchmark_table_report.py -q`
  reported `7 passed`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_paper_quality_benchmark_table_report.py tests/test_paper_runner_table_report.py -q`
  reported `14 passed`.

## Boundary

This closes the runner-spine/plumbing gap, not the whole scientific paper
benchmark. The current table is a reproducible local-video capacity smoke.
Next work is a scaled paper benchmark: repeated seeds, real paper datasets,
heldout/novel-view splits, and native metric export for the dynamic 3DGS lane.
