# Dataset V1 Reflection

## Context

This reflects on the dataset push that turned the local setup from a
file-distinct debug split into:

- a scene/source-distinct YouTube training smoke set,
- an 8-pair GT multi-camera validation seed,
- rehydrate/status scripts,
- a documented camera contract,
- and a matched 256px local/V-JEPA baseline path.

Chronology is already covered in these loose notes:

- `2026-04-24_16-18-43_local_mac_30_clip_default_dataset.md`
- `2026-04-24_17-01-34_youtube_curated_span_followup.md`
- `2026-04-24_17-25-41_dataset_v1_multicamera_validation.md`
- `2026-04-24_18-44-00_multicam_val_v1_seed.md`
- `2026-04-25_11-15-09_ex4dgs_pretrained_assets.md`
- `2026-04-25_11-40-04_deepview_video_gt_intake.md`
- `2026-04-25_16-52-47_scene_distinct_256_vjepa_baseline.md`

The shorter public-facing contract is `DATASET_V1.md`.

## What We Did Well

We corrected the most important dataset mistake early: different MP4 files were
not enough. The default train/test split now treats scene/source diversity as
the invariant, not just filename non-overlap. That matters because camera views
from one scene would let the model overfit one environment while pretending to
generalize.

We separated local training smoke from GT validation. YouTube clips are good for
fast local training and qualitative coverage, but they cannot validate novel
camera accuracy. Multi-camera datasets are now a separate track with explicit
source/target camera pairs.

We kept raw validation videos native and pushed downsampling to load time. That
was the right correction after the rough preview-FPS issue: metric tensors can
stay tiny, while raw data and human previews do not have to inherit the metric
sampling rate.

The rehydrate/status layer is useful. `local_data_status.sh`, checked-in JSONC
configs, and `data/REHYDRATE.md` make it possible to delete local media and
rebuild only the subset needed for a given experiment.

We documented the renderer-facing camera contract before implementing all
adapters. That makes the missing work explicit: DeepView/ViVo/Neural3D/AIST are
not just "datasets"; each needs a verified projection convention and scale
adapter before it can drive renderer metrics.

The 256px V-JEPA setup is now a fairer A/B than the earlier single-video config.
The local encoder and frozen V-JEPA configs use the same 30-clip manifest,
16-frame window, 256px input/render size, 2048 splats, fast-mac renderer, loss,
and step count.

## What Could Have Been Smarter

We spent too long treating the validation target count as the blocker. The
better framing is: 8 diverse paired samples are enough to start; unified camera
objects and projection checks are the real blocker for meaningful novel-view
metrics.

The first local dataset materialization was only 64px. That was fine for a
smoke test, but the 256px V-JEPA baseline was predictable enough that the
dataset builder should have been parameterized or paired with a 256 variant from
the start. The current 256 clips are correct 256px tensors, but they were built
from source sections originally downloaded with the 64px config's 360px cap.

We should have classified every external asset by "raw GT video", "derived
checkpoint/eval bundle", or "qualitative clip" before downloading. Ex4DGS is
useful, but it is not raw validation video; that distinction should have been
made before it entered the validation conversation.

The first preview MP4s exposed a process miss: QA media and metric tensors have
different requirements. The preview should have been 30fps from the beginning,
while the metric sample stayed 4fps.

The V-JEPA configs lagged the dataset work. They existed as single-video overfit
smokes even after the scene-distinct manifest became the default. Baseline
configs should move with the dataset contract as soon as the contract changes.

The work landed in a very dirty tree with many adjacent dataset/model changes.
That is tolerable during exploration, but it makes later review harder. The
smarter cadence is to checkpoint coherent layers: ingest/index scripts, then
loader/status docs, then train configs, then metric runners.

## Current Confidence

High confidence:

- 20 train / 10 test scene-distinct YouTube clips exist at 64px and 256px.
- 8 paired multi-camera validation samples exist and loader-smoke at
  `(16, 3, 128, 128)`.
- The 256px local-encoder training path runs one step on MPS.
- The 256px frozen HF V-JEPA encoder/decode path runs a no-render/no-backward
  forward on MPS.

Medium confidence:

- The 256px V-JEPA training config is a good local baseline. It is wired and
  forward-smoked, but a real training run still needs metrics/media review.
- DeepView is the best first camera-adapter target because it has clean local
  `models.json` calibration, but the y-axis/projection convention still needs a
  synthetic projection sanity check.

Low confidence:

- AIST camera correctness. We have synchronized fixed-camera videos, but no
  local intrinsics/extrinsics source yet.
- ViVo compact MP4 camera correctness until rotation/crop and per-frame metadata
  are audited against the compact RGB videos.
- Neural3D camera correctness until the LLFF/poses_bounds coordinate adapter is
  implemented and tested.

## Next Work

1. Implement canonical `CameraSpec` adapters, starting with DeepView.

   Cheap test: parse `models.json`, project a few world points with the adapter,
   and compare against the dataset projection formula before touching the model
   loop.

2. Extend `load_multicam_val_sample` to return source/target frames plus
   source/target `CameraSpec` when an adapter is ready.

   Cheap test: DeepView samples should report `camera_adapter_status=ready`;
   AIST can remain `missing_calibration`.

3. Add a tiny validation metric runner.

   First version can filter to camera-ready samples only, render target-camera
   frames, and report SSIM/PSNR/LPIPS if available. It should not pretend AIST
   is camera-ready.

4. Run the 256px local/V-JEPA pair.

   Start with the new 100-step configs, inspect step-0/final media, then decide
   whether the V-JEPA feature path is worth a longer run or a precomputed
   feature cache.

5. Decide whether to refresh YouTube raw sections at 720px.

   The current 256px tensors are usable for local baselines. A 720px refresh is
   only worth it if the 256px media looks visibly source-limited or if we keep
   256px as a recurring baseline.

6. Commit in reviewable layers.

   The clean layers are: dataset intake/configs, multi-camera val + camera
   contract, 256px V-JEPA baseline configs, and docs/runbooks.

## Falsification Tests

- If DeepView projected rays do not align under the documented adapter, stop and
  fix coordinate conventions before adding more datasets.
- If 256px local encoder and V-JEPA both fail to reconstruct even same-camera
  YouTube clips, the issue is probably still baseline architecture/init/loss,
  not multi-camera validation.
- If V-JEPA is much slower but not visibly better after short runs, move to
  precomputed feature caches before running longer baselines.
- If 8 validation pairs already produce contradictory qualitative results, do
  not chase 10 samples; inspect scene/camera coverage and adapter correctness.
