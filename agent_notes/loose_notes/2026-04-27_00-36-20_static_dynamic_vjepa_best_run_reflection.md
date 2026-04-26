# Static/Dynamic V-JEPA Best Run Reflection

## Context

The run the user identified as visually "almost perfect" was:

```text
https://wandb.ai/nbardy/dynaworld/runs/mybv736f
```

Run name:

```text
ablate-time-static-dynamic-96-32-crossattn4-precomputed-vjepa2-1-vitb-384-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats-1000step
```

The user asked whether the improvement was just "the 1k steps" and then asked
to record notes/key learnings and collect the best tweaks in one spot.

## Observed Facts

The run used the strongest tiny local recipe:

```text
16 loaded frames
128px render/input
8192 splats
fast-mac renderer
learned implicit orbit/path camera
96 static tokens + 32 dynamic tokens
4 query cross-attention layers
RGB-uniform strong token/head init
precomputed V-JEPA 2.1 ViT-B/384 feature memory
```

The config requested 1000 steps, but the run was interrupted after the step-500
video checkpoint, around step 525. The impressive media should be attributed to
the same recipe trained past 250 steps, not to a completed 1000-step schedule.

Metrics progression:

```text
local static/dynamic:
  Eval/Loss 0.1195
  Eval/SSIM 0.4287
  TemporalAdjacentL1Ratio 0.3408

V-JEPA static/dynamic, 250 steps:
  Eval/Loss 0.0881
  Eval/SSIM 0.6109
  TemporalAdjacentL1Ratio 0.6322

V-JEPA static/dynamic, ~525 steps:
  Eval/Loss 0.0547
  Eval/SSIM 0.7836
  TemporalAdjacentL1Ratio 0.8009
```

## Current Model

Current belief:

```text
V-JEPA features + static/dynamic capacity split made the model viable.
Training longer made the viable model visually sharp.
```

Confidence: medium.

Evidence:

- The 250-step V-JEPA split already dominated the local static/dynamic split.
- The longer cached run substantially improved the same recipe.
- The temporal adjacent ratio moved toward the GT adjacent magnitude instead of
  staying frozen.

Could be wrong if:

- The visual quality is mostly camera-path compensation.
- The result collapses on held-out windows or scene-distinct clips.
- The result fails multi-camera novel-angle validation at the same time index.

## Branches

Hypothesis:
    The extra steps are the main difference.

Why it might be true:
    The recipe was unchanged from the 250-step V-JEPA split, and the step-500
    media looked much better.

What would make it false:
    A 500-step local static/dynamic run remains poor, or a 250-step V-JEPA run
    with better camera regularization reaches the same quality.

Cheap test:
    Run matched 500-step controls for local static/dynamic and V-JEPA
    static/dynamic. Compare videos and scalar metrics at equal step counts.

If supported:
    Use 500 steps as the default visual-quality diagnostic before judging this
    architecture.

If invalidated:
    Focus on representation or camera settings rather than schedule length.

Hypothesis:
    V-JEPA memory is the enabling feature.

Why it might be true:
    The local split improved over frozen baselines but remained visibly softer;
    V-JEPA gave much better reconstruction and temporal motion at 250 steps.

What would make it false:
    A local encoder with matched memory length/capacity and 500 steps closes most
    of the gap.

Cheap test:
    Increase local encoder capacity or memory tokens while keeping the same
    static/dynamic split and schedule.

If supported:
    Treat frozen strong video features as the default conditioning lane for the
    local Mac baseline.

If invalidated:
    V-JEPA was acting mostly as capacity/optimization help, not semantic prior.

Hypothesis:
    Camera motion is helping too much.

Why it might be true:
    Camera adjacent rotation rose from `0.0159 deg` in local static/dynamic to
    `0.1309 deg` at 250 V-JEPA steps and `0.1827 deg` in the longer cached run.

What would make it false:
    Holding or regularizing camera motion still preserves the visual quality and
    temporal adjacent ratio.

Cheap test:
    Repeat the best recipe with stronger camera temporal/global penalties, or a
    known/fixed camera control when available.

If supported:
    Add camera-motion ablation to the baseline report and avoid calling the
    result true scene dynamics.

If invalidated:
    Trust the static/dynamic splat bank more and prioritize scene-distinct data.

## Decision Implications

Protect this recipe as the current best local baseline:

```text
static/dynamic 96/32
precomputed V-JEPA 2.1 ViT-B/384
cross_attn_layers=4
RGB-uniform strong init
128px, 16 frames, 8192 splats
500+ steps
```

Do not promote it to a generalization claim yet. It is a same-source overfit
result that finally looks strong enough to serve as a reliable local diagnostic.

## Artifacts

Single spot for the recipe:

```text
agent_notes/best_tweaks.md
```

Best config:

```text
src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc
```

Launcher:

```text
src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh
```

Feature cache:

```text
data/feature_cache/ablate_time_static_dynamic_vjepa2_1_vitb_384/b6ba09206f179d4c2cc29d52.pt
```

## Next

1. Finish the full 1000-step run and compare step 250 / 500 / 1000 media.
2. Add a camera-motion constrained control.
3. Move this recipe onto scene-distinct local train/eval clips.
4. Test on multi-camera validation when the camera loader contract is unified.
