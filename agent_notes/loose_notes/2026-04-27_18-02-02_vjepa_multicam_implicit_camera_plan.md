# V-JEPA Static/Dynamic Multicamera Implicit-Camera Plan

## Decision

Make this a separate trainer class, not a hard fork of the V-JEPA/static-dynamic model.

The current V-JEPA static/dynamic recipe should stay inside the existing
`DynamicVideoTokenGSImplicitCamera` path for the first implementation. The new
surface should be a multicamera trainer that reuses:

- `PrecomputedFeatureImplicitTrainer` feature-cache behavior
- `DynamicVideoTokenGSImplicitCamera` static/dynamic Gaussian decoding
- `CameraSpec` and `render_clip_sequence` as the render boundary
- the gauge DeepView multicamera loader/calibration logic

The trainer should own the new multicamera sampler, camera rig initialization,
learnable camera-reference alignment, train-camera loss, and held-out-camera
eval.

Do not start by forking the whole model. The static/dynamic model already has
the strongest tiny baseline behavior. Forking the model first would duplicate
the V-JEPA/precomputed/static-dynamic surface before we know which camera-rig
contract works.

## Current Architecture Read

### What Is Plugable

- Feature extraction is already reasonably plugable. `PrecomputedFeatureImplicitTrainer`
  builds a `VideoFeatureCache`, prebakes `train_sequences + eval_sequences`, and
  swaps `model_input_for_clip(...)` to return cached features rather than raw
  frames.
- The renderer boundary is usable. `render_clip_sequence(...)` takes a
  decoded `GaussianSequence` plus an explicit tuple of `CameraSpec`s, so a
  trainer can ignore the model-predicted cameras and render the same decoded
  splats through calibrated or learned external cameras.
- `CameraSpec` is the right low-level carrier. It supports intrinsics,
  `camera_to_world`, lens model, and distortion.
- The gauge lane already has the needed DeepView multicamera data contract:
  `train_videos: [V,T,H,W,3]`, `train_K`, `train_w2c`, `heldout_video`,
  `heldout_K`, `heldout_w2c`, and named train/heldout cameras.

### What Is Not Plugable Enough

- `SequenceData` is single-view: `frames: [T,3,H,W]`, optional cameras length
  `T`. It has no view axis.
- `Trainer.step()` assumes one sampled sequence, one decoded clip, one
  reconstruction target, and the decoded cameras from the implicit model.
- `DynamicVideoTokenGSImplicitCamera.forward(...)` assumes feature batch size
  `1`. That is acceptable for anchor-conditioned decoding, but not for
  batching camera views as separate observations.
- The current implicit camera head has exactly one global camera token and one
  per-time path camera token. It has no per-view camera token, no camera-rig
  object, and no learnable base extrinsics per camera.
- `DynamicVideoTokenGSKnownCamera` is not currently a good target because it
  drops the implicit-camera tokens and is not wired for the static/dynamic split
  surface we want to test.

## Core Requirement

Train the V-JEPA static/dynamic world-token baseline with multiple camera views:

```
O = anchor/source camera video or features
H_train = frames from camera_0001 and camera_0015
H_val = frames from held-out camera_0040

W0 = E_vjepa_static_dynamic(O)
S_t = G(W0, t)
Ihat_{v,t} = R_fixed(S_t, C_{v,t})
```

The same decoded world state must render against two train cameras and one
held-out camera. This is the useful pressure. Treating camera_0001 and
camera_0015 as two independent single-camera sequences is not enough; that
would not force one shared world asset to satisfy both view queries.

## Camera-Reference Requirement

With more than one camera, the reference frame matters.

The old implicit-camera path could hide a lot inside "one init camera plus a
learned path from that init." For two cameras, each camera has a reference
frame, and the splat/world frame must align to the camera rig. If the rig is
wrong and frozen, the model is forced to compensate in splat geometry. If the
rig is too freely learned, the cameras can drift into a degenerate agreement
that preserves source-view fit while weakening held-out meaning.

Use a learnable rig, not fully independent free cameras.

Recommended first rig:

```
base_cam_0 = radius * [ 0, 0,-1], look_at origin
base_cam_1 = radius * [ 1, 0, 0], look_at origin

C_v = G_rig @ base_cam_v @ exp(delta_v)
```

Where:

- `G_rig` is a shared learnable global SE3 alignment from the synthetic rig
  frame to the model/world frame.
- `delta_v` is a small per-camera residual SE3.
- camera intrinsics start from config or DeepView calibration.
- residuals are bounded or regularized.

This gives the optimizer a way to align the camera-reference frame without
letting every camera become an unconstrained escape hatch.

For calibrated DeepView data, initialize `base_cam_v` from the DeepView
relative cameras. For synthetic/base test data without calibration, use the
orthogonal origin-looking pair above.

## Requirements

1. Config-owned multicamera data settings:
   - `data.frame_source = "multicam_val"` or a new explicit multicamera mode
   - `data.multicam_manifest`
   - `data.multicam_split`
   - `data.multicam_sample_id` / `data.multicam_sample_index`
   - `data.multicam_train_cameras`
   - `data.multicam_heldout_camera`
   - `data.multicam_anchor_camera`
   - `data.multicam_condition_camera`

2. Config-owned camera-rig settings:
   - `camera.rig_init = "deepview" | "orthogonal_origin"`
   - `camera.rig_radius`
   - `camera.rig_learn_global_se3`
   - `camera.rig_learn_per_camera_se3`
   - `camera.rig_anchor_policy = "soft" | "fixed_first" | "free"`
   - `camera.rig_rotation_degrees`
   - `camera.rig_translation_ratio`
   - `camera.rig_regularization_weight`

3. Multicam bundle type:
   - `condition_sequence: SequenceData`
   - `train_frames: [V,T,3,H,W]`
   - `train_cameras: tuple[V][T] CameraSpec`
   - `heldout_frames: [T,3,H,W] | None`
   - `heldout_cameras: tuple[T] | None`
   - metadata with camera names and pose source

4. One decoded world, many render queries:
   - decode once from anchor/condition features for sampled times
   - render the decoded Gaussian frame through sampled train cameras
   - accumulate photometric loss over `(view,time)` pairs
   - render held-out camera at validation

5. Camera alignment must be observable:
   - log rig global rotation/translation norms
   - log per-camera residual rotation/translation norms
   - log initial vs current camera centers/axes
   - log held-out PSNR/L1/SSIM separately from source/train PSNR

6. Preserve the V-JEPA static/dynamic baseline:
   - keep `static_tokens=96`, `dynamic_tokens=32`
   - keep `video_encoder_backend="precomputed"`
   - keep the V-JEPA feature cache path
   - keep `max_frames=16` for first Mac smoke
   - do not introduce a new model variant until the trainer contract works

## Needed Code Changes

### 1. Move or Reuse Gauge Multicamera Loading

Best first step: reuse `research_experiments/gauge_fields/data.py` logic from
the new trainer, even if it is a slightly ugly import.

Cleaner second step: promote common helpers into `src/train/multicam_video_data.py`
or extend `src/train/multicam_val_data.py` so both gauge and V-JEPA trainers
share:

- DeepView record selection
- arbitrary train camera video loading
- held-out camera video loading
- DeepView `models.json` camera extraction
- relative camera construction
- frame index selection

### 2. Add `LearnableCameraRig`

Create a small module near the trainer/model boundary, likely
`src/train/camera_rig.py`.

Responsibilities:

- accept base `CameraSpec`s for train and held-out views
- register base poses/intrinsics as buffers
- own learnable `global_se3` and optional per-camera residuals
- return `CameraSpec`s for requested view/time indices
- keep intrinsics fixed for first pass unless config explicitly enables
  learnable focal/principal-point residuals

Sketch:

```python
class LearnableCameraRig(nn.Module):
    def __init__(self, base_cameras_by_view, *, learn_global, learn_per_view, ...):
        ...

    def cameras_for(self, view_indices, frame_indices) -> tuple[CameraSpec, ...]:
        ...

    def regularization_loss(self) -> torch.Tensor:
        ...

    def metrics(self) -> dict[str, float]:
        ...
```

Composition should use camera-to-world transforms:

```
camera_to_world = global_transform @ base_camera_to_world @ local_delta
```

The local delta is camera-frame residual motion. The global transform is the
shared gauge alignment between rig coordinates and learned world coordinates.

### 3. Add A Multicam Precomputed V-JEPA Trainer

Add a new trainer class instead of mutating `Trainer.step()`:

```
class MulticamPrecomputedFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer):
    ...
```

Likely file:

```
src/train/train_multicam_precomputed_feature_implicit_dynamic.py
```

This class should:

- load a multicam bundle instead of ordinary train/eval sequences
- create a condition `SequenceData` for the anchor camera
- prebake/load V-JEPA features for that condition sequence
- build the normal `DynamicVideoTokenGSImplicitCamera`
- create `LearnableCameraRig`
- add camera rig params to the optimizer
- override `step`, `initial_step_result`, and `validation_video_payload`

### 4. Decode Once, Render Multiple Cameras

The training step should sample a time window and one or more train views:

```python
clip_indices = select_window_indices(T, train_frame_count)
features = feature_cache.load_or_bake(condition_sequence)
decoded = model(features, decode_times=clip_times)

for view in sampled_views:
    cameras = camera_rig.cameras_for(view, clip_indices)
    rendered = render_clip_sequence(decoded, cameras, ...)
    target = train_frames[view, clip_indices]
    loss += reconstruction_loss_per_image(rendered, target, ...)

loss = loss / view_count + bank_rate_loss + rig_regularization
```

The important thing is that `decoded` is shared across all views for the same
time window.

### 5. Ignore Or Demote Predicted Implicit Cameras In This Trainer

For the first pass, do not render train/held-out views using
`decoded.cameras`. Render using `LearnableCameraRig` cameras.

The existing implicit camera head can remain in the model because it is baked
into `DynamicVideoTokenGSImplicitCamera`, but in this trainer:

- set camera motion/global/temporal loss weights to `0.0`
- log predicted camera state as diagnostics only
- use external rig cameras for reconstruction and held-out eval

Later refactor:

- split `DynamicVideoTokenGSImplicitCamera` into world decode plus optional
  camera decode
- or add a `world_only`/`external_camera` variant that removes the two camera
  tokens cleanly

Do this only after the multicam trainer has a passing smoke.

### 6. Held-Out Validation

Validation should render:

- train camera 0 full sequence
- train camera 1 full sequence
- held-out camera full sequence

Log separate metric namespaces:

```
TrainView0/Eval/*
TrainView1/Eval/*
Heldout/Eval/*
Rig/*
CameraPred/*
```

Selector metric for this lane should be held-out camera quality, not source
camera fit.

### 7. Config And Launcher

Add a JSONC config derived from the current best V-JEPA static/dynamic config:

```
src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc
```

Add a shell wrapper:

```
src/train_scripts/train_multicam_static_dynamic_vjepa_features.sh
```

The config should keep:

- `arch`: a new value such as `"multicam_precomputed_feature_implicit_camera"`
- `features.extractor = "vjepa_torchhub"`
- `model.video_encoder_backend = "precomputed"`
- `model.static_tokens = 96`
- `model.dynamic_tokens = 32`
- `model.tokens = 128`
- `model.gaussians_per_token = 64`
- `render.render_size = 128`
- `data.max_frames = 16`

## Fork-vs-Plug-In Verdict

### Fork The Trainer

Yes. Fork/subclass the trainer surface.

Reasons:

- the sampler changes from single-view sequence windows to `(view,time)` query
  sets
- the loss changes from one decoded clip against one target clip to one decoded
  world against multiple camera queries
- validation needs held-out-camera-specific metrics
- camera-rig parameters must be optimized and logged
- trying to hide this behind the current `Trainer.step()` would create a
  tangled conditional path

### Do Not Fork The Model First

No hard model fork for the first pass.

Reasons:

- static/dynamic split is already implemented and working inside
  `DynamicVideoTokenGSImplicitCamera`
- feature-cache/precomputed V-JEPA path already works with that model
- rendering through external cameras can be done at the trainer boundary
- the existing implicit camera output can be ignored/logged while the
  multicam contract is tested

### Refactor The Model After Smoke

After a multicam smoke passes, carve out the cleaner model seam:

```
video/features -> world tokens -> GaussianSequence without cameras
optional camera head -> decoded source camera diagnostics
external camera rig -> render queries
```

That refactor should be evidence-driven. If the first trainer proves useful,
then removing the camera-token baggage is worth doing.

## Smoke Boundary

First smoke should not use full V-JEPA if avoidable.

1. Use `features.extractor = "rgb_pyramid"` or cached existing V-JEPA features
   to test trainer shape, rig optimization, and held-out eval wiring.
2. Run 2 frames, tiny token count, `wandb_mode=disabled`, dense or fast-mac
   renderer.
3. Confirm:
   - train view loss decreases
   - held-out render is produced
   - camera rig params are in optimizer
   - no shape mismatch between `[V,T]` targets and decoded `[T]` splats
   - outputs include train-view and held-out-view metrics
4. Then run the 16f/128px/8192-splat V-JEPA static/dynamic config.

## Failure Modes To Watch

- If both cameras are fully free, the rig can absorb geometry errors and weaken
  the world-token pressure.
- If both cameras are fully frozen with bad synthetic orthogonal init, the model
  may waste capacity twisting splats into a wrong rig.
- If the trainer conditions separately on each train camera, it becomes two
  independent source-view overfits rather than one world asset under multiple
  queries.
- If camera losses from the old implicit head stay active, they can fight the
  external rig objective.
- If feature cache keys do not include the camera/source identity, multicam
  feature cache collisions can silently poison the run.
- If held-out metrics are averaged with train metrics, source-view fit can again
  select the wrong representation.

## Implementation Order

1. Add or promote multicam loading helpers.
2. Add `LearnableCameraRig` with orthogonal-origin and DeepView init tests.
3. Add multicam trainer subclass with RGB-pyramid smoke config.
4. Add held-out validation and saved preview/video artifacts.
5. Add V-JEPA static/dynamic multicam config and launcher.
6. Run smoke, then 16f/128px V-JEPA held-out run.
7. Only then refactor model into `world_only` plus optional camera head.

## Open Design Questions

- Should the condition input be only anchor camera 0, or both train cameras as
  observations? First pass should use anchor only to avoid changing the V-JEPA
  feature encoder shape.
- Should per-camera residuals be active from step 0 or unfrozen after a short
  warmup? Conservative default: learn global rig from step 0, keep per-camera
  residuals small with regularization.
- Should intrinsics be learnable? First pass: no. DeepView fisheye/pinhole
  approximation is already a known caveat, but moving intrinsics adds another
  escape hatch.
- Should the predicted implicit camera be regularized against the anchor rig
  camera? First pass: diagnostic only. Add that regularizer only if the
  predicted camera becomes useful for browser/export alignment.
