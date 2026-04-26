# Video-Token Overfit Init And Temporal Handoff

## Context

This note records the 16-frame dog-clip overfit thread around the
video-token implicit-camera model. The starting symptom was simple: the
`compare-local-video-encoder-16f-implicit-camera-128-fast-mac-8192splats`
baseline could optimize a scalar loss, but the render looked like a smooth
blob and did not preserve the dogs.

The target was intentionally tiny:

```text
source             test_data/test_video_small_128_4fps.mp4
loaded frames      46 for the original comparison, 16 for the capped feature run
train window       16 frames
render/loss size   128
splats             128 tokens * 64 splats = 8192
camera             implicit learned orbit/path camera
renderer           fast_mac
optimizer steps    usually 250, with one longer 1000-step config interrupted at ~525
```

The user explicitly wanted baseline protection: do not mutate the baseline as
if an ablation is proven. Add configs, diagnostics, and controls next to the
baseline so the bad case remains runnable.

## Original Goals

1. Verify whether the overfit scripts existed, were current, and used the fast
   renderer/config pattern.
2. Measure random initialization health instead of guessing from final renders.
3. Understand why inter-token and within-token variance, especially RGB, was
   weak.
4. Test whether a wider RGB/init contract improved the blobby dog render.
5. Add controls that can distinguish renderer/loss/init/camera failures from
   video-conditioning failures.
6. Record results with W&B links and keep rollback points via git.

## Subgoals That Evolved

The first theory was "RGB init is too collapsed." That was supported, but not
complete. The work then split into four additional questions:

- Is the model temporally frozen even when coarse shape improves?
- Does more cross-attention depth help or hurt?
- Does a static/dynamic token capacity split give the model a better motion
  bias than making every token time-varying?
- Does video conditioning matter on this tiny overfit, or can unconditioned
  tokens memorize the clip from time alone?

After the V-JEPA feature path started looking strong, a fifth question appeared:

- Can cached/precomputed V-JEPA features plus static/dynamic tokens become the
  current tiny source-view baseline, and what caveats does that carry?

## Code And Config Changes

Initialization and diagnostics:

- Added RGB-uniform bias initialization to `GaussianParameterHeads`.
- Threaded `rgb_init`, `rgb_init_min`, and `rgb_init_max` through token models
  and the video-token trainer config.
- Added `src/train/init_diagnostics.py` and
  `src/train/probe_init_diagnostics.py`.
- Added `tests/test_init_diagnostics.py`.
- Added the RGB-uniform strong-init ablation config:
  `src/train_configs/local_mac_ablate_init_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`.

Renderer/config correctness:

- Added `inputs_sorted_by_depth` to the fast-mac config surface and made the
  safe default `False`, because decoded Gaussians are not guaranteed sorted.

Temporal diagnostics and ablations:

- Added full-sequence temporal pixel ratios:
  `Eval/TemporalPredAdjacentL1`, `Eval/TemporalGTAdjacentL1`, and ratios.
- Added decoded Gaussian temporal metrics for xyz, scale, opacity, and RGB.
- Added camera adjacent/to-first motion metrics.
- Added configs/scripts for cross-attention depth, sinusoidal time, and
  static/dynamic 96/32 split:
  `src/train_scripts/train_video_temporal_ablation_suite.sh`
  and `src/train_scripts/collect_video_temporal_ablation_stats.py`.

No-conditioning and fitting controls:

- Added direct free-splat, linear-time free-splat, unconditioned-token, and
  residual-free-bank control variants.
- Extended
  `src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh`
  with `free`, `free-linear`, `tokens`, `controls`, `residual`, `matrix`, and
  `all` modes.

Feature-conditioned best tiny baseline:

- Added static/dynamic + precomputed V-JEPA 2.1 ViT-B/384 configs, including a
  longer 1000-step config.
- Added `src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh`.
- Made the feature cache instantiate extractors lazily and release large
  extractors after prebake/cache load where configured.

## Main Runs And Evidence

Original init comparison:

```text
baseline h5lefdtf
https://wandb.ai/nbardy/dynaworld/runs/h5lefdtf
Eval/Loss 0.1482, Eval/L1 0.0969, SSIM 0.2934, TemporalAdjacent ratio 0.0246

RGB-uniform strong init d8i7kw6f
https://wandb.ai/nbardy/dynaworld/runs/d8i7kw6f
Eval/Loss 0.1444, Eval/L1 0.0949, SSIM 0.3151, TemporalAdjacent ratio 0.0446
```

Interpretation: the init ablation is visibly and numerically better, but it is
not a solved dog reconstruction. The run starts with better detail and less
blob collapse, but animation is still weak and the final video remains too
static.

Clean rerun pair with step-0 diagnostics:

```text
baseline bbk7maml
https://wandb.ai/nbardy/dynaworld/runs/bbk7maml
Eval/Loss 0.1491, Eval/L1 0.0986, SSIM 0.2985, TemporalAdjacent ratio 0.0219

cross1 RGB/init diagnostic rerun 4fnvky3n
https://wandb.ai/nbardy/dynaworld/runs/4fnvky3n
Eval/Loss 0.1410, Eval/L1 0.0937, SSIM 0.3401, TemporalAdjacent ratio 0.0398
```

Temporal suite:

```text
cross1       Eval/Loss 0.1410, SSIM 0.3401, pred/GT adjacent 0.0398
cross2       Eval/Loss 0.1490, SSIM 0.3047, pred/GT adjacent 0.0252
cross4       Eval/Loss 0.1694, SSIM 0.2511, pred/GT adjacent 0.0072
sinusoidal4  Eval/Loss 0.1435, SSIM 0.3211, pred/GT adjacent 0.2695
static/dyn   Eval/Loss 0.1195, SSIM 0.4287, pred/GT adjacent 0.3408
```

Interpretation: more learned-time cross-attention depth made the model more
frozen and worse. Sinusoidal time improved motion but also increased camera
motion and runtime. The 96 static / 32 dynamic split was the first strong local
architecture change.

No-conditioning controls:

```text
local video encoder       bbk7maml Eval/Loss 0.1491, SSIM 0.2985
V-JEPA fpc16/256          pwvybmao Eval/Loss 0.1451, SSIM 0.3062
known-camera DUSt3R       ut58yk3z Eval/Loss 0.1560, SSIM 0.2661
free splats               kttrbewl Eval/Loss 0.2633, SSIM 0.6631, PSNR 9.66
free linear-time splats   zj2kis2e Eval/Loss 0.2391, SSIM 0.3564, PSNR 11.58
unconditioned tokens      xenc4w06 Eval/Loss 0.1439, SSIM 0.3267
```

Interpretation: this tiny overfit does not prove video conditioning is doing
useful work. The token decoder/head parameterization is a strong optimizer by
itself, while direct free splats are not yet a reliable oracle.

Best tiny source-view run so far:

```text
static/dynamic + precomputed V-JEPA 2.1 ViT-B/384, 250 steps
https://wandb.ai/nbardy/dynaworld/runs/oaor6um2
Eval/Loss 0.0881, Eval/L1 0.0615, SSIM 0.6109, TemporalAdjacent ratio 0.6322

same recipe, 1000-step config interrupted around step 525
https://wandb.ai/nbardy/dynaworld/runs/mybv736f
Eval/Loss 0.0547, Eval/L1 0.0413, SSIM 0.7836, TemporalAdjacent ratio 0.8009
```

Interpretation: this is now the best tiny same-source fit. It is not proof of
novel-view generalization.

## Issues Found

- Baseline renders could improve full-frame scalar loss while still losing the
  dogs; sky/grass/horizon dominate the objective.
- Original RGB/head initialization had weak color diversity and low same-split
  inter-token variance.
- The fast-mac wrapper had an unsafe sorted-depth default for unsorted decoded
  Gaussians, weakening the cleanliness of early run comparisons.
- The learned-time video-token path was too temporally frozen.
- More learned-time cross-attention layers were not a fix; cross4 froze harder.
- Direct free splats are not a useful upper-bound oracle yet; optimization and
  parameterization matter more than raw parameter count.
- The precomputed feature path bakes the whole loaded `SequenceData`, not just
  the sampled train window. Without `max_frames=16`, the "fast" V-JEPA path
  tried all 46 frames and used too much time/memory.
- Cache-hit feature runs were still paying extractor load cost until the cache
  boundary was made lazy/releasable.

## Issues Solved Or Improved

- Init health is now measurable and logged/probeable.
- RGB-uniform initialization is configurable without replacing baselines.
- Step-0 render/init diagnostics are part of the video-token run contract.
- Temporal pixel, decoded-splat, and camera motion metrics now separate frozen
  pixels from frozen state and camera compensation.
- Static/dynamic capacity split gives a concrete opt-in architecture path that
  improves motion and reconstruction on the tiny overfit.
- No-conditioning and fitting-only controls exist and are scriptable.
- The current best tiny source-view recipe has a single rerun command and an
  operational index in `agent_notes/best_tweaks.md`.

## Still Remaining

- Add foreground/high-frequency metrics and probably motion-masked losses.
  Current full-frame L1/SSIM are not enough to judge dog detail.
- Finish or repeat the longer best-run schedule under quiet system load.
- Separate real dynamic splat motion from camera-path compensation.
- Move the static/dynamic + V-JEPA recipe to scene-distinct and multi-camera
  validation.
- Make feature caching clip-aware before scaling to longer videos.
- Re-test at 256px source/render/loss if the goal is dog detail rather than
  tiny-run architecture debugging.
- Protect the best run as an opt-in config, not as a replacement for the old
  baseline until held-out evidence exists.

## Decision Implications

The current model of the problem is:

```text
RGB/init diversity helps startup and visual detail.
Static/dynamic capacity split helps temporal behavior.
V-JEPA features help the model read the scene.
Full-frame scalar loss is insufficient for foreground object fidelity.
Same-source overfit is not enough evidence for world-token generalization.
```

Next work should stop debating whether `d8i7kw6f` "worked" in binary terms.
It worked as an init ablation and failed as a complete temporal/detail solution.
The planning doc `research_notes/video_token_overfit_next_plan.md` is the
recommended route for the next pass.
