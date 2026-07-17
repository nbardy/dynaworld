# Microlib: STAR UVT Feature RGB-Gradient Handoff

## Problem

Generalize the positive benchmark-only RGB handoff direction into a trainable
STAR UVT feature path. The current `fused_first3_sigmoid_mse` prototype is fast
but too narrow: it only handles `alpha * sigmoid(feature[:3]) -> mean MSE`.
The real path needs learned `FeatureToColor` weights/biases and gradients for
both tube features and colorizer parameters.

## Why Now

Current repo facts:

- STAR UVT F32 feature tubes are first-class through
  `arch=star_uvt_feature_overfit`.
- `feature_direct_gradcache` is the current fastest valid feature mode but only
  trims a few percent.
- Skipping feature gradients is a diagnostic, not a trainable path.
- The narrow RGB handoff prototype has the strongest direction signal and
  should be generalized before chasing bigger quality rows.

## Allowed Edits

Initial implementation surface:

- `research_experiments/star_uvt_feature_tubes/`
- `src/train/train_star_uvt_feature_overfit.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/`

Prefer adding a new explicit mode before changing defaults.

## Baseline Rows

Compare against:

- `feature_direct_gradcache`, alpha `1/72`, cap 256, 64f/256px/32768t/F32.
- Tiny F4/F32 parity JSONs under `outputs/benchmarks/`.
- Existing `fused_first3_sigmoid_mse` synthetic benchmark as a direction signal,
  not as a trainer baseline.

## Evaluator Cascade

Stage 0:

- patch scope check
- build/import extension if needed
- no use of skip-feature-gradient for trainable mode

Stage 1:

- F4/F32 tiny parity for the new mode
- colorizer parameter gradient present
- feature gradient present
- geometry/opacity gradients present
- finite outputs

Stage 2:

- synthetic 64f/256px/32768t/F32 timing row
- target metric: lower backward ms than `feature_direct_gradcache`
- zero overflow where the baseline is zero-overflow

Stage 3:

- first-class 20-step config with alpha `1/72`, cap 256
- final loss/PSNR no worse than gradcache within tolerance
- media and JSON row written

## Primary Metrics

- `correct == true`
- `overflow_tile_count == 0`
- `backward_ms` lower than current gradcache
- `loss_final <= baseline_loss_final + tolerance`
- `raw_feature_grad_seen == true`
- colorizer grad norms are nonzero

## Hard Rejects

- Only supports feature channels 0:3 with sigmoid.
- Drops colorizer gradients.
- Zeros feature gradients.
- Changes the training target or frame count to win timing.
- Treats `feature_direct_fixedbin` as optimized if only the report label
  changes.

## Promotion Gate

Do not update `BASELINES.md` or claim a STAR feature replacement until:

- parity passes
- 20-step first-class smoke passes
- quality/timing JSON is saved
- the new mode is explicitly named in config and report fields
- the current RGB STAR path remains separate
