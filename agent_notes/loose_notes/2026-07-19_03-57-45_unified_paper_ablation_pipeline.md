# Unified Paper Ablation Pipeline

## Goal

Implement one clean ablation surface for World Tubes, WorldFoam, and dynamic
3DGS using shared data order, progressive stages, cost accounting, and heldout
evaluation while preserving each representation's existing Metal kernel.

## What Changed

- Added typed paper dataset, stage, space-time batch, kernel, and cost records.
- Added exact shuffled space-time epochs with same-time/local-time grouping.
- Added rectangular multicamera loading and aspect-preserving frame/ray/K
  resizing.
- Added checked-in 300-frame Coffee Martini manifest and progressive,
  pixel-matched fixed, global-shuffle, staged-smoke, and full-temporal-smoke
  protocols.
- Integrated the protocol into PowerFoam Metal, STAR UVT World Tubes, and
  fast-mac per-frame dynamic 3DGS.
- Changed World Tubes paper batches from full-sequence rasterization followed
  by K-frame selection to selected-global-time Metal windows. Target and
  rasterized cost are now both K.
- Added the unified launcher with manifest validation, exact cost rejection,
  offline/online W&B, and one combined run summary.
- Replaced repeated per-frame MP4 seeks with sequential decode for contiguous
  or nearby frame requests; sparse requests retain seeking.
- Split PowerFoam optimizer-update elapsed time from evaluation/media wall time.

## Verification

- Unified focused gate: `77 passed, 1 skipped` before the sequential-loader
  addition; focused loader/runner gate then passed `25 passed`; focused
  PowerFoam/accounting gate passed `62 passed, 1 skipped`.
- Staged MPS summary:
  `outputs/benchmarks/2026-07-19_unified_paper_ablation_smoke_v2/coffee_martini_protocol_smoke_2step/seed_17/run_summary.json`.
  Exact shared cost is 2 steps, 4 frames, and 30,720 pixels. Optimizer-update
  elapsed: World Tubes `0.298237s`, dynamic 3DGS `0.299486s`, WorldFoam
  `0.608338s`; WorldFoam wall loop including eval/media is `1.867592s`.
- Full-temporal MPS summary:
  `outputs/benchmarks/2026-07-19_unified_paper_ablation_smoke/coffee_martini_full_300f_smoke_1step/seed_17/run_summary.json`.
  It loaded all 300 frames at 30 fps, built the 600-pair train universe,
  completed all three Metal optimizer steps, evaluated train and heldout views,
  and wrote media/offline W&B. This artifact predates the PowerFoam elapsed
  split, so use it for full-temporal software coverage, not matched timing.

## Truth Boundary

The 512-wide progressive/fixed/global protocols are implemented but have not
been trained to paper quality. No new `BASELINES.md` row is justified yet.
Native 2704x2028 is not supported by the eager target/ray cache; it requires
the streaming work in `TODO/unified_paper_ablation_pipeline.md`.
