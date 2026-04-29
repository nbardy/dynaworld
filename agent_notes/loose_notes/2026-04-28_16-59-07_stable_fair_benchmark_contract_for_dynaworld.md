# Stable Fair Benchmark Contract for Dynaworld

Date: 2026-04-28

## Goal

Move the project from single-camera overfit toward a benchmark that measures
whether video-conditioned world tokens learn reusable 3D/dynamic structure.

The benchmark should answer three separate questions:

1. Can the model reconstruct train cameras?
2. Can the same learned world representation render a held-out camera from the
   same clip?
3. Can the model generalize to held-out clips and out-of-distribution videos?

Source-view PSNR alone answers only the first question.

## Dataset Splits

### Split A: Camera-Heldout Multicam

Use 10 three-camera clips.

For each clip:

```text
train_camera_ids = [camera_a, camera_b]
heldout_camera_ids = [camera_c]
condition_camera_id = one of the train cameras
```

The primary validation metric is performance on `camera_c`.

If possible, keep two heldout bins:

- overlap heldout: camera lies between or near the train cameras
- outside heldout: camera lies outside the train camera arc

This prevents one extreme camera from making every model look bad while still
preserving a hard generalization check.

### Split B: Sample-Heldout Multicam

Hold out complete multicamera clips by sample id.

These samples should not appear in training under any camera. They measure
whether the video feature and world-token prior generalize to new scenes.

For each sample-heldout clip, report:

- source/condition camera reconstruction
- heldout-camera rendering if three cameras exist
- temporal consistency

### Split C: Single-Camera YouTube Train Mix

Add 20 single-camera YouTube clips for feature bootstrapping and wider scene
coverage.

These clips should contribute source-view reconstruction and temporal priors,
but they should not be allowed to dominate the benchmark. They do not provide
camera-heldout evidence by themselves.

Optional but recommended:

```text
youtube_single_val = 5 heldout single-camera clips
```

This gives a cheap sample-heldout signal for OOD video conditioning.

## Batch Sampling Contract

Do not sample proportional to raw frame count. That will let the largest group
dominate training.

Recommended starting sampler:

```text
P(multicam batch) = 0.5
P(youtube single-camera batch) = 0.5

within multicam:
  sample clip uniformly
  sample condition camera from train cameras
  render train cameras for loss
  render heldout cameras for validation only

within youtube:
  sample clip uniformly
  render source camera for loss
```

This makes the YouTube mix a regularizer and feature-prior source, not the
primary objective.

## Metric Contract

Primary selector:

```text
camera_heldout/psnr
camera_heldout/ssim
camera_heldout/l1
```

Average these per sample first, then across samples, so a clip with more frames
does not dominate the score.

Secondary selectors:

```text
sample_heldout/source_psnr
sample_heldout/source_ssim
sample_heldout/camera_heldout_psnr, if multicam sample-heldout exists
foreground_or_highpass_l1
temporal_consistency
decoded_xyz_motion
camera_adjustment_magnitude
runtime_per_step
export_size
```

Report train-camera metrics, but do not use them as the main ranking signal.

## Fixed Evaluation Contract

Each contender should use the same:

- train sample ids
- camera-heldout sample ids
- sample-heldout ids
- frame indices
- train camera ids
- heldout camera ids
- condition camera policy
- render resolution
- loss resolution
- feature source resolution
- camera clamp policy
- number of optimizer steps
- W&B logging cadence

Two benchmark sizes are useful:

```text
smoke: 128px, 16 frames, 8192 splats, 250 steps
main: 256px, 16 frames, 8192-32768 splats, 1000+ steps
```

Do not compare a 128px render/loss run against a 256px render/loss run as if
they are one matrix. Feature resolution can differ, but render and loss
resolution should be explicit and matched inside a benchmark tier.

## Required Controls

Every benchmark matrix should include:

- unconditioned tokens, same static/dynamic split and strong init
- local encoder static/dynamic
- V-JEPA static/dynamic
- V-JEPA without static/dynamic split
- free splats or known-camera reference when available
- random or shuffled features if feature leakage is suspected

The unconditioned static/dynamic control is mandatory. It tells us whether the
video features are actually helping, rather than the decoder simply memorizing
the training set.

## Camera Policy

Camera motion is not inherently bad. It is bad only when it becomes the easiest
way to reduce photometric loss without learning the intended world.

Use two camera modes:

```text
camera_clamped: primary architecture comparison
camera_free: diagnostic capacity / escape-hatch comparison
```

If a model only wins in `camera_free`, treat the result as suspicious until it
also wins in `camera_clamped` or on known-camera heldout views.

## Feature Cache Contract

Feature caches must be keyed by:

```text
sample_id
camera_id
frame_indices
source_path
resize/crop policy
feature backend
feature model checkpoint
feature tensor layout
```

Do not trust feature comparisons until this metadata is present. Earlier
feature-cache paths have had enough source-frame and layout ambiguity that
"matched" comparisons can silently become unfair.

## Acceptance Criteria

A model is a meaningful improvement if:

- it beats unconditioned static/dynamic on camera-heldout metrics
- it does not rely on larger camera adjustments to win
- it improves or holds sample-heldout reconstruction
- it preserves train-camera reconstruction within a reasonable margin
- it runs within the agreed memory/time budget

A model is not a meaningful improvement if:

- only source-view PSNR improves
- heldout-camera PSNR drops
- camera motion grows enough to explain the win
- the feature cache is not source-faithful
- the model uses a different split, resolution, or loss schedule without being
  labelled as a separate benchmark tier

## First Stable Matrix

Recommended first stable matrix:

```text
Data:
  10 three-camera clips
  20 single-camera YouTube train clips
  optional 5 single-camera YouTube val clips

Model cells:
  unconditioned static/dynamic strong-init
  local static/dynamic strong-init
  V-JEPA static/dynamic strong-init
  V-JEPA no-split strong-init
  free-splat or known-camera reference

Validation:
  camera_heldout on third camera
  sample_heldout on withheld clips
  source/train camera reported separately
```

This matrix is small enough to interpret and large enough to stop rewarding
single-camera overfit.
