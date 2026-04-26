# Video Model Bootstrap Reflection

## Context

This note consolidates the recent V-JEPA, LTX, Wan-VACE, known-camera, 256px
baseline, and init-diagnostics work. Earlier notes captured the chronology, but
the reflection was spread across several files.

Related notes:

- `2026-04-24_16-12-37_vjepa_video_encoder_fork.md`
- `2026-04-24_16-27-28_vjepa21_pretrained_smoke.md`
- `2026-04-24_17-32-50_ltx_wan_vace_bootstrap_architectures.md`
- `2026-04-25_16-52-47_scene_distinct_256_vjepa_baseline.md`
- `2026-04-25_21-41-11_init_diagnostics_rgb_uniform.md`
- `research_notes/training_queue_video_model_bootstrap.md`

## What Actually Landed

- V-JEPA is now a swappable video encoder backend for the video-token
  implicit-camera model.
- The V-JEPA 2.1 tiny TorchHub path was smoke-tested with real pretrained
  weights after patching around the upstream localhost checkpoint URL.
- A faster HF V-JEPA fpc16/256 comparison path exists for 16-frame local A/B
  tests.
- Local 16f implicit-camera, frozen V-JEPA 16f implicit-camera, and known-camera
  controls were run at 128px.
- LTX and Wan-VACE were documented as diffusion/editing hidden-state feature
  priors. Wan-VACE was added as a feature-cache extractor surface, but not
  full-run validated.
- A 256px scene-distinct 30-clip dataset/config pair was set up and smoke
  checked, but the full 256px local-vs-V-JEPA run table has not been produced.
- Init diagnostics were added after the model appeared to collapse into
  background/mean-color solutions. The strongest measured issue was weak RGB
  diversity and low same-split inter-token spread at initialization.

## What Went Well

- The architecture stayed swappable. V-JEPA, local encoder, LTX, Wan-VACE, and
  known-camera controls entered through explicit config/model boundaries rather
  than replacing the main path.
- The work moved from claims to actual smokes. The V-JEPA path was not only
  config-parsed; real pretrained model loading and a model forward were tested.
- The comparison framing improved. The initial single-video smoke was useful,
  then the mismatch became visible: 128px raster target, V-JEPA 256px feature
  input, short 250-step run, background-dominated loss, and no step-0 logging.
- The known-camera control clarified a separate axis. It showed that camera
  supervision/prebake is not the same question as pretrained video features.
- The init probe found a concrete low-level failure mode instead of only
  speculating about model capacity. RGB starts near gray and token diversity is
  weak unless explicitly initialized.

## What Could Have Been Smarter

- The first V-JEPA A/B should have been 256px end-to-end immediately. Running
  128px raster targets while V-JEPA internally saw 256px made the visual result
  harder to interpret.
- Step-0 W&B logging should have been added before running comparisons. Without
  initialization renders, it is harder to distinguish collapse, bad init, slow
  convergence, and target-data issues.
- The source-frame contract should have been fixed before expanding LTX/Wan.
  Feature extractors should read source frames at their own feature resolution,
  not resized training tensors.
- The 128px single-video comparison had too little visual diagnostic power.
  Full-frame loss on sky/grass lets missing foreground objects hide behind a
  decent scalar loss.
- The 256px scene-distinct setup is useful, but it used existing local source
  sections that were originally downloaded by the 64px/360p pass. A cleaner
  rerun should refresh source MP4s before judging high-resolution detail.
- Several pieces were added in a dirty shared worktree. That was pragmatic, but
  it increases the risk of bundling unrelated dataset, renderer, and model
  changes unless commits are staged very carefully.

## Current Belief

Confidence: medium.

The missing dog/detail problem is probably not one bug. It likely combines:

- low render resolution and low effective splat budget for detail,
- background-dominated reconstruction loss,
- no step-0 render diagnostics,
- weak RGB/diversity initialization,
- short smoke runs,
- and possibly source/video preprocessing mismatches.

V-JEPA has not yet been falsified. The 128px run was a smoke, not a fair
evidence-quality comparison.

## Next Moves

1. Add step-0 W&B render/video logging to the video-token trainer.
2. Run the RGB-uniform/strong-init ablation on the same 128px single-video
   target to see if the foreground collapse improves before changing backbones.
3. Run the 256px end-to-end local-vs-V-JEPA pair and produce a new table with
   W&B links, eval metrics, runtime, and visible media notes.
4. Refresh the 256px source MP4s from higher-quality downloads if the current
   clips look source-limited.
5. Add a 256px known-camera or MASt3R/DUSt3R control only after the camera
   contract is explicit for that data path.
6. Fix feature-frame loading and feature-bundle metadata before using LTX/Wan
   results as serious evidence.
7. Add a foreground/motion-sensitive diagnostic. Full-frame scalar loss is not
   enough when the object of interest is small relative to sky/grass.

## Falsification Tests

- If step-0 renders already show broad color/position diversity but the dog
  disappears after optimization, focus on loss weighting, sampling, and capacity.
- If RGB-uniform init improves early foreground detail, keep it as an opt-in
  baseline ablation and sweep down the stronger token/head scale.
- If 256px local and V-JEPA both miss foreground detail, the bottleneck is likely
  renderer/model/loss/data rather than pretrained encoder quality.
- If 256px local recovers detail and V-JEPA does not, inspect the V-JEPA adapter:
  feature projection, token compression, dtype/device, and whether the SSV2
  checkpoint is the wrong semantic prior.
- If known-camera 256px succeeds while implicit-camera fails, prioritize camera
  representation and camera supervision over video-feature backbone work.
