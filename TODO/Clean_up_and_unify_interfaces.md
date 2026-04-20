# Dynaworld Interface Cleanup And Unification

## Goal

Reduce script drift in `dynaworld/` without overbuilding a framework.

The problem is not that the codebase has "too many abstractions." The problem is the opposite: several training paths now share the same conceptual pipeline, but the contracts between data loading, model decode, rendering, and logging are mostly implicit and tuple-based.

The target is:

1. keep experiment velocity high
2. avoid copy-paste drift across baselines
3. make trainer/model/renderer boundaries explicit enough that new baselines fork cleanly

## Current State

### Training entrypoints

There are multiple training scripts with overlapping logic:

- `src/train/dynamicTokenGS.py`
- `src/train/train_camera_implicit_dynamic.py`
- `src/train/train_video_token_implicit_dynamic.py`

Only the video-token path currently has a real `Trainer` class. The others are still procedural script-local training loops.

### Model layer

There are multiple model families:

- known-camera image baseline
- implicit-camera image baseline
- implicit-camera video baseline

These share a lot of behavior, but not a stable forward/decode contract.

### Renderer layer

The renderer math is reasonably centralized:

- `src/train/renderers/common.py`
- `src/train/renderers/dense.py`
- `src/train/renderers/tiled.py`

But render dispatch is duplicated in each train script under slightly different helper names.

## Main Problems

### 1. Tuple soup

Models mostly return positional tuples like:

- `xyz`
- `scales`
- `quats`
- `opacities`
- `rgbs`
- `cameras`
- `camera_state`

This works for fast iteration, but it is easy to drift or reorder fields accidentally.

### 2. Implicit-camera logic is duplicated

The following concepts exist in both image and video implicit-camera models:

- zero-init camera heads
- SE(3) helper math
- camera composition
- base camera head
- path camera head
- canonical/world-space Gaussian decode

These should live in one shared implicit-camera module.

### 3. Render dispatch is duplicated

Three separate helpers do nearly the same job:

- `render_one_frame(...)`
- `render_implicit_frame(...)`
- `render_clip_frame(...)`

This should be one render dispatch helper with one config path.

### 4. Trainer structure is inconsistent

The video baseline now has clean method boundaries:

- sample
- forward
- build losses
- reconstruction backward
- logging
- validation

The older baselines still duplicate this logic inline.

### 5. Data payloads are untyped

Sequence data and sampled clips are mostly loose dicts with overlapping keys:

- `frames`
- `frame_times`
- `video_fps`
- `frame_source`
- optional `cameras`

This makes loaders flexible, but weakens downstream contracts.

## Cleanup Direction

Do not build a giant base-class hierarchy.

Instead, add a few small explicit interfaces and move shared math into the right layer.

## Proposed Unifications

### A. Add typed runtime payloads

Create small dataclasses for runtime payloads:

- `SequenceData`
- `ClipBatch`
- `DecodedSequence`
- optional `CameraState`

This is the highest-leverage cleanup because it removes positional coupling across trainers and models.

### B. Extract shared implicit-camera module

Create one shared module for:

- `build_zero_init_head`
- `axis_angle_to_matrix`
- `compose_camera_with_se3_delta`
- `GlobalCameraHead`
- `PathCameraHead`

Both implicit-camera models should depend on that module instead of duplicating the same logic.

### C. Unify render dispatch

Create one helper that owns dense vs tiled dispatch and takes a stable payload.

Suggested shape:

- `render_gaussian_frame(render_mode, render_cfg, camera, gaussian_frame, dense_grid)`

Where `gaussian_frame` is a typed object or a tiny struct containing:

- `xyz`
- `scales`
- `quats`
- `opacities`
- `rgbs`

### D. Promote one trainer pattern

Use the video trainer as the reference structure, then port the image baselines toward the same method split:

- `sample_batch`
- `forward_batch`
- `build_losses`
- `backward_reconstruction`
- `validation_payload`
- `run`

This does not require inheritance. Matching method boundaries is enough.

### E. Separate script concerns from library concerns

The train scripts should mostly do:

- config assembly
- trainer construction
- run

The reusable pieces should live under shared runtime/model modules.

## Recommended Order

### Phase 1: cheap and high value

1. add `DecodedSequence`
2. add one shared render dispatch helper
3. extract shared implicit-camera heads and SE(3) math

### Phase 2: trainer cleanup

4. port `train_camera_implicit_dynamic.py` to the same trainer structure as the video path
5. move common logging and validation helpers into shared runtime utilities

### Phase 3: data contract cleanup

6. replace loose sequence dicts with small dataclasses
7. make loaders return one typed payload across known-camera and implicit-camera paths

## Non-Goals

Avoid these unless they become clearly necessary:

- a giant abstract `BaseTrainer`
- a large framework-style registry
- deep inheritance across model families
- a rewrite of the renderer math layer

The cleanup should make experiments easier to fork, not harder to understand.

## Success Criteria

This cleanup is done when:

1. new baselines do not need to copy a whole train script
2. implicit-camera logic exists in one place
3. renderer dispatch exists in one place
4. model outputs are no longer positional tuple soup
5. image and video baselines feel like variants of one system rather than separate script families
