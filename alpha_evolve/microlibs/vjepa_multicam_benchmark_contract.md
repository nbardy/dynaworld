# Microlib: V-JEPA/F32 Multicam Benchmark Contract

## Problem

Turn the current V-JEPA/F32 multicam evidence into a repeatable benchmark
contract with source/camera-disjoint manifests, leakage probes, pose-error
diagnostics, and explicit baseline rows.

## Why Now

The current goodset run is useful, but it is not enough to claim solved
novel-view behavior. The repo notes already warn that pose recovery is not
physically solved and that current prepared data is small.

## Allowed Edits

Initial evaluator surface:

- `src/dataset_configs/`
- `src/dataset_scripts/`
- `research_experiments/multicam_train2_holdout1/`
- `tests/test_multicam_video_data.py`
- a new validator under `research_experiments/` if needed

Avoid changing trainer behavior until validators exist.

## Baseline

Current useful row:

- F32 goodset alpha `1/128`, heldout PSNR `13.6248`, SSIM `0.1922`

Known caveats:

- relpose head under-recovers calibrated camera deltas
- DeepView fisheye metadata can be discarded into pinhole paths
- small sample count
- need source/camera-disjoint manifests before broad quality claims

## Evaluator Cascade

Stage 0:

- manifest schema check
- split names and camera ids present

Stage 1:

- validator proves no train camera appears as heldout target for the same
  scene/time record
- source paths and target paths are disjoint where intended
- heldout RGB features are not consumed by the encoder for heldout eval unless
  explicitly named as a query-conditioned experiment

Stage 2:

- tiny smoke on the manifest
- pose-error diagnostics emitted
- lens-model/fisheye metadata preserved or explicitly reported as dropped

Stage 3:

- promoted config run
- W&B id and `BASELINES.md` row added only after the benchmark surface is stable

## Primary Metrics

- leakage check pass
- pose-error fields present
- heldout PSNR/SSIM/L1 present
- source/camera-disjoint sample count
- lens metadata status

## Hard Rejects

- Treating goodset PSNR as physical pose recovery.
- Comparing pinhole and fisheye paths without naming the projection mismatch.
- Using target RGB/heldout features as hidden encoder input.
- Backfilling `BASELINES.md` with guessed metrics.

## Promotion Gate

Promote the evaluator first. Implementation evolution can start after a
manifest validator and smoke gate exist.
