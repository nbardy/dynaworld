# Temporal Conditioning Ablation Suite

## Context

User asked to diagnose the video-token implicit-camera model after the
RGB-uniform strong-init run reconstructed a decent coarse shape but appeared
too temporally frozen. The specific asks were:

- add diagnostics for the frozen-output question
- try more cross-attention depth, especially 4 layers
- try sinusoidal time conditioning
- try the static/dynamic token split
- collect comparable stats

The run target was intentionally tiny and local:

```bash
./src/train_scripts/train_video_temporal_ablation_suite.sh all
./src/train_scripts/collect_video_temporal_ablation_stats.py --include-base
```

All runs used `test_data/test_video_small_128_4fps.mp4`, 16 training frames,
128px render/input, 8192 splats, fast-mac renderer, and 250 optimizer steps.
This is a single-video diagnostic suite, not evidence of scene generalization.

## New Diagnostics Added

Validation now logs temporal metrics beyond pixel L1:

- `Eval/TemporalPredAdjacentL1`, `Eval/TemporalGTAdjacentL1`, and ratio
- `Eval/TemporalPredToFirstL1` and ratio
- decoded Gaussian state movement:
  - `Eval/DecodedXYZAdjacentL2`
  - `Eval/DecodedXYZToFirstL2`
  - `Eval/DecodedScaleAdjacentL1`
  - `Eval/DecodedOpacityAdjacentL1`
  - `Eval/DecodedOpacityToFirstL1`
  - `Eval/DecodedRGBAdjacentL1`
  - `Eval/DecodedRGBToFirstL1`
- camera adjacent/to-first movement:
  - `Camera/EvalAdjacentRotationDeltaDegrees`
  - `Camera/EvalAdjacentTranslationDelta`
  - `Camera/EvalToFirstRotationDeltaDegrees`
  - `Camera/EvalToFirstTranslationDelta`

This separates four possibilities:

1. predicted pixels are frozen
2. decoded splat state is frozen
3. camera path is moving instead of scene state
4. dynamic split is actually using dynamic capacity

## Runs

| run | variant | cross-attn | split |
| --- | --- | ---: | --- |
| `ablate-time-crossattn1-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats` | learned time | 1 | none |
| `ablate-time-crossattn2-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats` | learned time | 2 | none |
| `ablate-time-crossattn4-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats` | learned time | 4 | none |
| `ablate-time-sinusoidal-crossattn4-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats` | sinusoidal time | 4 | none |
| `ablate-time-static-dynamic-96-32-crossattn4-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats` | learned time | 4 | 96 static / 32 dynamic |

The earlier manual run `ablate-init-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats`
is included as the pre-diagnostic reference, but it lacks decoded/camera
temporal metrics because those were added later.

## Summary Table

Key final metrics from local W&B summaries:

| run | eval loss | L1 | SSIM | PSNR | pred adj / GT adj | pred-to-first ratio | decoded XYZ adj | camera adj rot | dynamic motion | runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old cross1 ref | 0.1444 | 0.0949 | 0.3151 | 16.73 | 0.0446 | 0.4941 | n/a | n/a | 0 | 573s |
| cross1 + diagnostics | 0.1410 | 0.0937 | 0.3401 | 16.86 | 0.0398 | 0.4792 | 0.00831 | 0.0344 deg | 0 | 123s |
| cross2 | 0.1490 | 0.0993 | 0.3047 | 16.53 | 0.0252 | 0.3429 | 0.00956 | 0.0120 deg | 0 | 252s |
| cross4 | 0.1694 | 0.1182 | 0.2511 | 15.81 | 0.0072 | 0.0671 | 0.00290 | 0.0004 deg | 0 | 578s |
| sinusoidal cross4 | 0.1435 | 0.0945 | 0.3211 | 17.24 | 0.2695 | 0.5069 | 0.14538 | 0.2266 deg | 0 | 803s |
| static/dynamic 96/32 cross4 | 0.1195 | 0.0779 | 0.4287 | 18.42 | 0.3408 | 0.5557 | 0.04553 | 0.0159 deg | 0.0844 | 488s |

## Interpretation

The original concern was valid: the old learned-time cross1 path was far too
temporally static in pixel space. Its adjacent predicted motion was only about
4 percent of GT adjacent motion.

Plain "more layers" did not solve it. Cross-attn4 with learned time was worse
on reconstruction and even more frozen: adjacent predicted motion dropped to
about 0.7 percent of GT. This argues against simply making the per-frame token
refinement deeper as the default.

Sinusoidal time helped temporal movement a lot. It raised adjacent predicted
motion to about 27 percent of GT and strongly increased decoded XYZ/RGB/opacity
movement. But it was slow on MPS and its camera also moved much more than the
shallow learned-time runs, so it may be partly using camera/path motion to
explain change.

The 96/32 static/dynamic split was the best run in this suite. It improved
reconstruction substantially and raised adjacent predicted motion to about 34
percent of GT. It also produced nonzero dynamic BankRate metrics:

- dynamic motion: `0.0844`
- dynamic rotation: `0.0426`
- dynamic alpha-time: `0.4279`
- static alpha: `0.1057`

This suggests the capacity split is not just a semantic label; it is creating a
usable time-varying channel in this tiny diagnostic.

## Caveats

- This is a single-video fit, not a dataset-scale result.
- Final eval sequence is the same source video path as train in these local
  tiny configs, so the suite diagnoses representational capacity, not
  generalization.
- The 4-layer and sinusoidal paths are slower enough that they should not be
  treated as the default Mac baseline without further pruning.
- Better reconstruction can still come from camera/path compensation; use the
  decoded/camera temporal metrics before making architectural claims.

## Next Directions

Most promising next path:

1. Keep the diagnostics in the trainer.
2. Treat the static/dynamic split as the next opt-in research branch, not yet a
   default.
3. Run a matched 20-train/10-test scene-distinct config at 128 or 256 with:
   - cross1 learned-time baseline
   - static/dynamic split with the smallest stable split, probably 96/32
4. Add one validation run on multi-camera GT once unified camera adapters are
   ready, because the current suite cannot judge novel-view consistency.
5. Consider a dynamic-motion regularizer or sampler objective only after
   confirming that the split does not merely overfit source-view motion.

The biggest lesson: the issue was not simply "one cross-attn layer is too
small." More depth alone made temporal freezing worse. A structured capacity
split changed the optimization behavior more than adding depth.
