# Video-Token Overfit Next Plan

This document is the durable plan for the DynaWorld 16-frame dog-clip overfit
thread as of 2026-04-27. Raw chronology is in `agent_notes/loose_notes/`;
operational best-run commands are in `agent_notes/best_tweaks.md`.

## Working Bar

The immediate bar is not broad world-model generalization. It is a controlled
source-view debugging ladder:

1. Fit the tiny dog clip with visible object detail.
2. Animate the object with measured frame-to-frame variation near GT.
3. Preserve camera sanity instead of explaining everything with camera motion.
4. Transfer the same recipe to held-out clips and multi-camera validation.

Do not replace the older baselines yet. The current best recipe is an opt-in
research branch until held-out and novel-view evidence exists.

## Current Best Recipe

Use:

```bash
./src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh 1000
```

Config:

```text
src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc
```

Current best run:

```text
https://wandb.ai/nbardy/dynaworld/runs/mybv736f
```

Important caveat: this was a 1000-step config, but the observed strong result
was interrupted after the step-500 video checkpoint, around step 525.

Latest synced metrics:

```text
Eval/Loss                     0.0547
Eval/L1                       0.0413
Eval/SSIM                     0.7836
Eval/PSNR                     23.69
Eval/TemporalAdjacentL1Ratio  0.8009
```

## Evidence Summary

Baseline and init:

| Variant | W&B | Eval loss | L1 | SSIM | Temporal adjacent ratio | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| local video-token baseline | `h5lefdtf` | 0.1482 | 0.0969 | 0.2934 | 0.0246 | blobby, weak dog detail |
| RGB-uniform strong init | `d8i7kw6f` | 0.1444 | 0.0949 | 0.3151 | 0.0446 | better detail, still weak animation |
| clean local baseline | `bbk7maml` | 0.1491 | 0.0986 | 0.2985 | 0.0219 | step-0 logged baseline |
| clean init/cross1 | `4fnvky3n` | 0.1410 | 0.0937 | 0.3401 | 0.0398 | cleaner evidence init helps |

Temporal architecture:

| Variant | Eval loss | SSIM | Temporal adjacent ratio | Read |
| --- | ---: | ---: | ---: | --- |
| cross1 | 0.1410 | 0.3401 | 0.0398 | better init, still frozen |
| cross2 | 0.1490 | 0.3047 | 0.0252 | worse |
| cross4 | 0.1694 | 0.2511 | 0.0072 | deeper learned-time froze harder |
| sinusoidal cross4 | 0.1435 | 0.3211 | 0.2695 | motion improved, camera/runtime caveat |
| static/dynamic 96/32 | 0.1195 | 0.4287 | 0.3408 | best local temporal branch |

Feature-conditioned best:

| Variant | W&B | Eval loss | L1 | SSIM | Temporal adjacent ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| static/dynamic + V-JEPA, 250 steps | `oaor6um2` | 0.0881 | 0.0615 | 0.6109 | 0.6322 |
| static/dynamic + V-JEPA, ~525 steps | `mybv736f` | 0.0547 | 0.0413 | 0.7836 | 0.8009 |

## What We Learned

The init ablation worked, but only as an init ablation. It made the model less
blobby and improved high-frequency visual detail, but it did not solve temporal
coherence. The old "did it work?" question should now split into:

- did it improve startup diversity? yes
- did it improve scalar reconstruction? modestly
- did it improve visible detail? yes, according to previews
- did it solve dog animation? no
- did it prove video conditioning is needed? no

The strongest architecture lesson is that structured capacity allocation beat
plain depth. More learned-time cross-attention layers made the model more
frozen, while a static/dynamic token split gave the optimizer a usable dynamic
channel.

The strongest evaluation lesson is that full-frame L1/SSIM are too weak for
the user-facing question. A model can fit sky/grass/horizon and still lose the
dog. Future claims need foreground, edge, and motion-sensitive metrics.

## Remaining Issues

### Foreground And Detail Metrics

Missing metrics:

- high-pass / Laplacian L1
- predicted edge energy vs GT edge energy
- motion-masked L1 where GT adjacent difference is high
- foreground/object mask L1 from motion + darkness/color heuristics
- late-frame metrics, because the user observed the dog being lost at the end

Cheap first implementation:

```text
Eval/HighPassL1
Eval/MotionMaskCoverage
Eval/MotionMaskedL1
Eval/MotionMaskedSSIM or masked DSSIM
Eval/PredEdgeEnergy
Eval/GTEdgeEnergy
Eval/EdgeEnergyRatio
```

Use these as metrics first. Only add loss terms after the metrics show which
failure mode tracks the visual problem.

### Camera Compensation

The best V-JEPA/static-dynamic run also increased camera motion. That may be
valid, but it may also be using camera flexibility to explain appearance
change. Add a matched control with reduced camera range or higher camera
regularization before claiming the dynamic bank learned object motion.

### Generalization

All strong evidence here is source-view overfit. Move the same recipe to:

- scene-distinct local train/eval clips
- held-out windows from the same video
- multi-camera validation once unified camera loaders are ready

### Feature Cache Scale

The precomputed feature path currently bakes whole loaded `SequenceData`.
Before longer clips:

- add clip-aware cache entries, or
- add a feature temporal downsampler, or
- require explicit `data.max_frames` caps in configs.

## Next Experiment Queue

### 1. Add Metrics Before More Architecture

Patch validation to log high-pass and motion-masked metrics. Then run only:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh local
./src/train_scripts/train_video_temporal_ablation_suite.sh split4
./src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh 250
```

Expected result:

- `d8i7kw6f`-style init should improve edge/detail metrics over baseline.
- static/dynamic should improve motion-masked metrics over cross1.
- V-JEPA/static-dynamic should lead both scalar and motion/detail metrics.

If edge/motion metrics do not match visual judgment, inspect the metric masks
before changing the model.

### 2. Rerun Best Recipe Quietly

Repeat the 1000-step config when the Mac is not under unrelated load:

```bash
./src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh 1000
```

Compare step 250, 500, and 1000 media and metrics. The current `mybv736f`
result should be treated as a strong ~500-step checkpoint, not a completed
1000-step convergence result.

### 3. Camera-Constraint Control

Create a sibling config that keeps static/dynamic + V-JEPA but reduces camera
flexibility:

```text
max_rotation_degrees       lower than 5.0
max_translation_ratio      lower than 0.2
camera_motion_weight       higher than 0.01
camera_temporal_weight     higher than 0.02
```

Support for the current hypothesis:

- reconstruction stays strong
- decoded dynamic metrics stay high
- camera adjacent metrics drop

If quality collapses, the best run is relying heavily on camera-path freedom.

### 4. Scene-Distinct Transfer

Port the recipe to the 20/10 scene-distinct dataset. Start with 128px if speed
matters, then 256px when checking detail.

Minimum comparison:

```text
local cross1 baseline
static/dynamic local
static/dynamic V-JEPA
unconditioned token control
```

The unconditioned control matters because the single-clip result showed that
time-only token decoding can match video-conditioned models on an overfit task.

### 5. Multi-Camera Validation

Use multi-camera same-time GT to separate source-camera reconstruction from
world-token consistency. The same-source dog fit is useful only if it becomes a
stepping stone to this validation.

## Falsification Tests

RGB/init hypothesis:

- supported if step-0 RGB entropy/spread predicts early detail and edge metrics
- weakened if foreground/high-pass metrics do not improve over baseline

Static/dynamic hypothesis:

- supported if decoded dynamic metrics and motion-masked reconstruction improve
  without excessive camera motion
- weakened if camera-constrained control collapses and decoded dynamics drop

V-JEPA feature hypothesis:

- supported if V-JEPA/static-dynamic beats local/static-dynamic on held-out
  windows or scene-distinct eval
- weakened if source-view overfit is strong but held-out/multicam metrics do
  not improve

Loss/metric hypothesis:

- supported if full-frame metrics diverge from edge/motion/foreground metrics
  in the same direction as visual judgment
- weakened if all metrics agree but visual judgment still differs, implying the
  mask or logging media is missing the relevant failure

## Commit/Branch Policy

Keep baseline configs runnable. New ideas should enter as new configs/scripts
until they beat the current baseline on both scalar and visual/motion metrics.
When changing a baseline config intentionally, commit before and after so the
run surface is easy to roll back.
