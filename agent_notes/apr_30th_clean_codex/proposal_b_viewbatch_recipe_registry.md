# Proposal B: ViewBatch + TrainRecipe Registry

Date: 2026-04-30

Author role: Proposal Writer B

Scope: data/view batching, normalized config routing, feature-provider
description, and recipe dispatch. This proposal deliberately does not edit code.

## Executive Thesis

The current trainer stack routes by Python file, then uses inheritance overrides
to patch special cases. That is why the F=32 feature-splatting fix landed in the
base `Trainer.recon_backward()` but did not land in the multicam trainer: the
multicam subclass owns its own `render_view_clip()` and `multicam_recon_loss()`,
so it bypasses the shared render/loss path.

The replacement should be data-first:

```text
ExperimentSpec -> DataSource -> ViewBatch -> ModelProgram -> RenderObjective -> TrainRecipe
```

The train loop should not know whether a batch came from a single source camera,
precomputed V-JEPA/LTX/Wan features, or a DeepView train2/test1 held-out-camera
bundle. It should receive a `ViewBatch`, decode one `GaussianSequence`, and ask a
single `RenderObjective` to render/loss every `TargetView`.

The registry should make `arch` real. Today, `arch` is mostly a config label and
the actual route is "which Python file the shell script invoked." The proposed
`src/train/train.py` should load a normalized `ExperimentSpec`, resolve an
`arch -> TrainRecipe`, and expose `--explain-routing CONFIG` before any behavior
changes.

## Current Evidence

Important code references:

- `SequenceData`, `ClipBatch`, `GaussianSequence` already exist in
  [runtime_types.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/runtime_types.py:46).
- The base video-token trainer normalizes config in
  [train_video_token_implicit_dynamic.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_video_token_implicit_dynamic.py:174),
  builds models in
  [train_video_token_implicit_dynamic.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_video_token_implicit_dynamic.py:809),
  and owns the current random-background F=32 loss path in
  [train_video_token_implicit_dynamic.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_video_token_implicit_dynamic.py:1309).
- The precomputed-feature trainer only overrides feature cache and
  `model_input_for_clip()` in
  [train_precomputed_feature_implicit_dynamic.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_precomputed_feature_implicit_dynamic.py:55).
  That shape is close to what we want: feature conditioning should be a provider,
  not a trainer subclass.
- The multicam trainer currently overrides data, sampling, decode, render, loss,
  validation, and export in
  [train_multicam_precomputed_feature_implicit_dynamic.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_multicam_precomputed_feature_implicit_dynamic.py:73).
  It returns the new `(features, alpha)` tuple as if it were a tensor at
  [train_multicam_precomputed_feature_implicit_dynamic.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_multicam_precomputed_feature_implicit_dynamic.py:177)
  and passes that tuple to `reconstruction_loss_per_image()` at
  [train_multicam_precomputed_feature_implicit_dynamic.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/train_multicam_precomputed_feature_implicit_dynamic.py:189).
- `MulticamVideoBundle` already contains the right raw ingredients: condition
  sequence, train frames/cameras, held-out frames/cameras, pose source, and
  metadata in
  [multicam_video_data.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/multicam_video_data.py:17).
- `load_multicam_video_bundle()` already distinguishes train cameras, held-out
  cameras, condition camera, and anchor camera in
  [multicam_video_data.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/multicam_video_data.py:933).
- `VideoFeatureCache` already owns cache fingerprinting, prebake, loading, and
  channel inference in
  [video_feature_cache.py](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/video_feature_cache.py:642).

## Current Config Families And Routes

There are currently 96 `src/train_configs/*.jsonc` files. Observed `arch` counts:

| `arch` | Count | Current route | Registry target |
|---|---:|---|---|
| `tokengs_video_implicit_camera` | 35 | `src/train/train_video_token_implicit_dynamic.py` | `VIDEO_TOKEN_IMPLICIT_RECIPE` |
| `gauge_fields_material_surfel` | 31 | `research_experiments/gauge_fields/train.py` | external recipe shim, not merged into this cleanup |
| `tokengs_prebaked_camera` | 9 | `src/train/dynamicTokenGS.py` | legacy known/prebaked recipe shim |
| `precomputed_feature_implicit_camera` | 4 | `src/train/train_precomputed_feature_implicit_dynamic.py` | `VIDEO_TOKEN_PRECOMPUTED_RECIPE` |
| `multicam_precomputed_feature_implicit_camera` | 4 | `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` | `MULTICAM_PRECOMPUTED_RECIPE` using the same objective |
| `tokengs_image_implicit_camera` | 2 | `src/train/train_camera_implicit_dynamic.py` plus typo/alias shims | legacy image implicit recipe shim |
| `tokengs` | 2 | `src/train/train_video_token_implicit_dynamic.py` | alias of `VIDEO_TOKEN_IMPLICIT_RECIPE` |
| `splat_baseline_free_dynamic_3dgs` | 2 | `research_experiments/gauge_fields/train_splat_baseline.py` | external baseline recipe shim |
| `wan_vace_feature_implicit_camera` | 1 | shell script calls precomputed trainer | alias of `VIDEO_TOKEN_PRECOMPUTED_RECIPE` |
| `ltx_feature_implicit_camera` | 1 | shell script calls precomputed trainer; empty LTX subclass also exists | alias of `VIDEO_TOKEN_PRECOMPUTED_RECIPE` |
| `tokengs_video_known_camera` | 1 | base trainer file, selected by `model.variant` | `VIDEO_TOKEN_KNOWN_CAMERA_RECIPE` |
| `tokengs_prebaked_camera_tiled` | 1 | `src/train/dynamicTokenGS.py` / tiled legacy path | legacy known/prebaked recipe shim |
| `tokengs_single_image` | 1 | `src/train/tokenGS.py` | legacy single-image recipe shim |
| `tokengs_single_image_tiled` | 1 | legacy tiled single-image path | legacy single-image recipe shim |
| `splat_baseline_static_3dgs` | 1 | `research_experiments/gauge_fields/train_splat_baseline.py` | external baseline recipe shim |

Shell scripts confirm the drift: most video-token configs invoke
`train_video_token_implicit_dynamic.py`, multicam invokes
`train_multicam_precomputed_feature_implicit_dynamic.py`, LTX/Wan/V-JEPA
precomputed scripts invoke `train_precomputed_feature_implicit_dynamic.py`, and
prebaked-camera configs still invoke `dynamicTokenGS.py`.

## Design Goals

1. Make `arch` the only training route decision.
2. Keep single-cam, known-camera, precomputed-feature, and multicam batches under
   one train-loop contract.
3. Make camera roles explicit: condition camera, anchor camera, train target
   cameras, held-out target cameras.
4. Make feature caches self-describing before model construction so trainers do
   not mutate `model.video_feature_channels` after loading data.
5. Keep `**kwargs` and section dict pass-through only at validated factory
   boundaries. Inside warm code paths, use typed dataclasses/protocols.
6. Keep old entrypoints as compatibility shims until smoke parity is proven.
7. Preserve held-out-camera evaluation as a first-class output, not a logging
   afterthought.

Non-goals for this proposal:

- It does not redesign `RenderObjective` internals; proposal B only specifies the
  data/recipe interfaces it must consume.
- It does not merge gauge-field research code into the video-token trainer. It
  routes that stack as an external recipe shim.
- It does not delete legacy trainers immediately.

## Core Type Vocabulary

Use small literals for semantic roles. Avoid booleans like `is_eval` where camera
semantics matter.

```python
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

import torch

from camera import CameraSpec
from runtime_types import GaussianSequence, SequenceData

Tensor = torch.Tensor

ArchName = Literal[
    "tokengs_video_implicit_camera",
    "tokengs",
    "tokengs_video_known_camera",
    "precomputed_feature_implicit_camera",
    "ltx_feature_implicit_camera",
    "wan_vace_feature_implicit_camera",
    "multicam_precomputed_feature_implicit_camera",
    "tokengs_prebaked_camera",
    "tokengs_prebaked_camera_tiled",
    "tokengs_image_implicit_camera",
    "tokengs_single_image",
    "tokengs_single_image_tiled",
    "gauge_fields_material_surfel",
    "splat_baseline_free_dynamic_3dgs",
    "splat_baseline_static_3dgs",
]

ViewRole = Literal["source", "train", "heldout", "eval", "debug"]
CameraRole = Literal["condition", "anchor", "train_target", "heldout_target", "model_predicted"]
ConditionKind = Literal["rgb_video", "precomputed_features", "none"]
CameraOwner = Literal["model", "batch", "external_rig", "none"]
FeatureExtractorName = Literal[
    "rgb_pyramid",
    "vjepa_hf",
    "vjepa_torchhub",
    "ltx",
    "wan_vace",
]
BackgroundMode = Literal["white", "black", "random_rgb", "fixed_rgb", "none"]
TrainPhase = Literal["train", "eval", "initial", "export"]
```

## Normalized Config Schema

The normalized config should be represented once as `ExperimentSpec`. Raw JSONC
sections still pass through, but all defaults and type coercions happen during
normalization. After that, runtime code should not scatter
`cfg.get("key", default)` across samplers, factories, and training loops.

```python
@dataclass(frozen=True)
class ExperimentSpec:
    config_path: Path | None
    arch: ArchName
    raw: Mapping[str, Any]
    data: "DataSpec"
    views: "ViewsSpec"
    features: "FeatureProviderSpec | None"
    model: "ModelSpec"
    camera: "CameraSpecConfig"
    rig: "CameraRigSpec | None"
    render: "RenderSpec"
    objective: "ObjectiveSpec"
    train: "TrainSpec"
    logging: "LoggingSpec"
    export: Mapping[str, Any]


@dataclass(frozen=True)
class DataSpec:
    # Single-sequence inputs.
    sequence_dir: Path | None
    frames_dir: Path | None
    video_path: Path | None
    manifest_path: Path | None
    eval_manifest_path: Path | None
    split: str
    eval_split: str
    frame_source: str
    max_frames: int | None
    frame_indices: tuple[int, ...] | None
    camera_json: Path | None
    camera_image_size: int
    camera_focal_mode: str

    # Multicam manifest inputs.
    multicam_manifest: Path | None
    multicam_split: str | None
    multicam_sample_id: str | None
    multicam_sample_index: int | None

    # Dataset-specific scaling knobs retained as a section dict at the edge.
    dataset_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewsSpec:
    # Source/condition view. In single-cam this is the same pixels used as target.
    condition_camera: str | None
    condition_kind: ConditionKind

    # Coordinate-frame anchor. Multicam relative poses are expressed in this frame.
    anchor_camera: str | None

    # Supervised RGB target cameras.
    train_cameras: tuple[str, ...]
    heldout_cameras: tuple[str, ...]

    # Sampling.
    train_views_per_step: int
    train_frame_count: int
    frame_sampler: Literal["contiguous_window", "fixed_indices", "all_frames"]
    eval_policy: Literal["source_only", "all_train_views", "all_train_and_heldout"]

    # Semantics for metrics/logging keys.
    primary_metric_role: Literal["source", "heldout", "train_mean"]


@dataclass(frozen=True)
class FeatureProviderSpec:
    enabled: bool
    extractor: FeatureExtractorName
    model_id: str | None
    layers: tuple[str, ...] | None
    cache_dir: Path
    sample_cache_key: str
    cache_version: int
    force_rebake: bool
    keep_in_memory: bool
    release_extractor_after_prebake: bool
    save_dtype: torch.dtype
    runtime_dtype: torch.dtype
    extractor_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticDynamicSpec:
    static_tokens: int | None
    dynamic_tokens: int | None
    dynamic_time_basis_count: int
    dynamic_time_max_frequency: float
    dynamic_motion_extent: float
    dynamic_rotation_degrees: float
    dynamic_alpha_logit_extent: float
    dynamic_coeff_output_init_std: float


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    image_size: int
    train_frame_count: int
    tokens: int
    gaussians_per_token: int
    feature_dim: int
    video_encoder_backend: str
    video_feature_layers: tuple[str, ...] | None
    video_feature_channels: Mapping[str, int] | None
    gaussian_head: Mapping[str, Any]
    transformer: Mapping[str, Any]
    init: Mapping[str, Any]
    static_dynamic: StaticDynamicSpec


@dataclass(frozen=True)
class CameraSpecConfig:
    global_head: str
    lens_model: str
    base_fov_degrees: float
    base_radius: float
    max_fov_delta_degrees: float
    max_radius_scale: float
    max_rotation_degrees: float
    max_translation_ratio: float
    camera_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraRigSpec:
    init: Literal["deepview", "aist", "neural_3d_video", "vivo", "orthogonal_origin"]
    radius: float
    learn_global_se3: bool
    learn_per_camera_se3: bool
    anchor_policy: Literal["fixed", "soft", "learned"]
    rotation_degrees: float
    translation_ratio: float
    regularization_weight: float
    dataset_scale_options: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderSpec:
    renderer: str
    input_size: int
    render_size: int
    tile_size: int
    camera_projection: str
    fast_mac: Mapping[str, Any]


@dataclass(frozen=True)
class BackgroundSpec:
    # The RGB-space composition background used after feature colorization.
    train_mode: BackgroundMode
    eval_mode: BackgroundMode
    fixed_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    random_per: Literal["step", "view", "frame", "pixel"] = "step"


@dataclass(frozen=True)
class ColorizeSpec:
    hidden_dim: int | None
    activation: str
    pre_norm: bool
    weight_init: str
    weight_init_gain: float
    view_condition: str
    detach_view_condition: bool


@dataclass(frozen=True)
class ObjectiveSpec:
    reconstruction: Mapping[str, Any]
    background: BackgroundSpec
    colorize: ColorizeSpec | None
    feature_pca_log: bool
    regularizers: Mapping[str, float]


@dataclass(frozen=True)
class TrainSpec:
    steps: int
    lr: float
    camera_rig_lr: float | None
    amp: bool
    recon_backward_strategy: Literal["batched", "microbatch", "framewise"]
    temporal_microbatch_size: int
    seed: int | None
    optimizer: Mapping[str, Any]


@dataclass(frozen=True)
class LoggingSpec:
    wandb_project: str
    wandb_run_name: str | None
    wandb_tags: tuple[str, ...]
    log_every: int
    image_log_every: int
    video_log_every: int
    always_log_last_step: bool
    media: Mapping[str, bool]
```

### Raw JSONC To Normalized Mapping

Current keys should map into the normalized schema like this:

| Current key | New location | Notes |
|---|---|---|
| top-level `arch` | `ExperimentSpec.arch` | Becomes authoritative route key. |
| `data.sequence_dir`, `manifest_path`, `frame_source` | `DataSpec` | Single-cam source sequence. |
| `data.multicam_*` | `ViewsSpec` plus `DataSpec.multicam_*` | Stop burying camera roles in generic `data`. |
| `features.*` | `FeatureProviderSpec` | Feature cache owns channel description. |
| `model.static_tokens`, `dynamic_tokens`, dynamic params | `ModelSpec.static_dynamic` | Validated once; model factory receives typed static/dynamic spec. |
| `model.video_feature_channels = null` | `FeatureProvider.describe()` result | Trainer should not mutate `model_cfg` after feature cache warmup. |
| top-level `colorize` | `ObjectiveSpec.colorize` | Required when `feature_dim != 3`. |
| `render.fast_mac.background` | `RenderSpec.fast_mac` | Rasterizer RGB background for F=3 legacy path only. |
| `render.fast_mac.feature_background` | `RenderSpec.fast_mac` | Feature-space background before colorize for F!=3. |
| hardcoded random background in trainer | `ObjectiveSpec.background.train_mode = "random_rgb"` | This is run provenance and must be in W&B config. |
| `camera.rig_*` | `CameraRigSpec` | Only valid for external-rig recipes. |
| `train.train_views_per_step` | `ViewsSpec.train_views_per_step` | It is a view sampler property, not optimizer state. |

## Data And View Types

The central new type is `ViewBatch`. It contains one conditioning input and N
target views. Single-cam training is just a one-target `ViewBatch`; multicam is a
multi-target `ViewBatch`; held-out validation is the same structure with
`TargetView.role == "heldout"`.

```python
@dataclass(frozen=True)
class ConditioningInput:
    sample_id: str
    scene_id: str | None
    kind: ConditionKind

    # Raw RGB condition video. Present for local encoder and unconditioned smokes
    # that still want source pixels in the batch object. Shape [1, T, 3, H, W].
    frames: Tensor | None

    # Precomputed features. Shape is extractor-dependent; examples:
    # V-JEPA tokens: {"vjepa_tokens": [1, T', N, C] or [N, C]}
    # LTX/Wan blocks: {"block_12": [1, C, T', H', W'] or compatible adapter shape}
    features: Mapping[str, Tensor] | None

    # Time/index metadata. decode_times shape [1, K] or [K, 1] after adapter.
    frame_indices: Tensor              # [K], long, native selected frame ids
    frame_times: Tensor                # [K, 1], float normalized 0..1
    video_fps: float

    # The camera attached to conditioning pixels, if known. For implicit-camera
    # source-video training this can be None. For multicam it is the selected
    # condition camera, often also the anchor camera.
    camera_name: str | None
    cameras: tuple[CameraSpec, ...] | None  # len K or None

    source_path: Path | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetView:
    view_id: str
    role: ViewRole
    camera_role: CameraRole

    # Supervised RGB target. Shape [K, 3, H_in, W_in] before render-size resize.
    frames: Tensor
    frame_indices: Tensor              # [K], long, same timeline as decode
    frame_times: Tensor                # [K, 1]
    video_fps: float

    # Cameras used for rendering this target. None means render through
    # model-predicted cameras already attached to GaussianSequence.
    camera_name: str | None
    cameras: tuple[CameraSpec, ...] | None  # len K or None
    camera_owner: CameraOwner

    # Training/eval controls.
    loss_weight: float = 1.0
    metrics_prefix: str | None = None
    log_media: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewBatch:
    batch_id: str
    sample_id: str
    scene_id: str | None
    phase: TrainPhase
    device: torch.device

    conditioning: ConditioningInput
    targets: tuple[TargetView, ...]

    # Decode request to the model. Shape [K, 1] is preferred because current
    # SequenceData uses [T, 1], but adapters may expose [1, K] to old models.
    decode_times: Tensor               # [K, 1]
    frame_indices: Tensor              # [K]

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int: ...

    @property
    def target_count(self) -> int: ...
```

### Camera Role Invariants

Camera roles should be enforced by the data builder before a model ever runs.

```python
def validate_view_batch(batch: ViewBatch) -> None:
    """Raise on shape/role contradictions before train_step."""
```

Required invariants:

- `batch.decode_times.shape[0] == batch.frame_indices.shape[0]`.
- Every target has the same `frame_indices` and `frame_times` as the batch unless
  the sampler explicitly declares cross-time supervision.
- `TargetView.camera_owner == "model"` implies `TargetView.cameras is None`.
- `TargetView.camera_owner in {"batch", "external_rig"}` implies
  `len(TargetView.cameras) == batch.frame_count`.
- For multicam recipes, `ViewsSpec.condition_camera` and
  `ViewsSpec.anchor_camera` must both exist.
- `anchor_camera` is a coordinate-frame role. It need not be a loss target in
  future experiments, but the current DeepView path requires it to be in
  `train_cameras`.
- `condition_camera` is a conditioning-data role. It may equal the anchor camera,
  but code should not assume equality.
- Held-out cameras can never participate in train loss unless the view role is
  intentionally migrated from `heldout` to `train`.

## DataSource And ViewSampler Protocols

Instead of trainer subclasses owning `load_train_sequences()`, `sample_clip()`,
`sample_views()`, and multicam frame slicing, use composable data sources and
samplers.

```python
@dataclass(frozen=True)
class DataBundle:
    sample_id: str
    scene_id: str | None
    source_sequences: tuple[SequenceData, ...]

    # Multicam stores all target views here. Single-cam stores one source target.
    train_views: tuple["ViewRecord", ...]
    heldout_views: tuple["ViewRecord", ...]
    eval_views: tuple["ViewRecord", ...]

    condition_view_id: str | None
    anchor_view_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewRecord:
    view_id: str
    role: ViewRole
    camera_role: CameraRole
    camera_name: str | None
    frames: Tensor                     # [T, 3, H, W]
    frame_times: Tensor                # [T, 1]
    video_fps: float
    cameras: tuple[CameraSpec, ...] | None
    source_path: Path | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DataSource(Protocol):
    def load(self, spec: ExperimentSpec, *, device: torch.device) -> DataBundle:
        """Load frames/cameras/metadata but do not sample a train window."""


class ViewSampler(Protocol):
    def sample(
        self,
        bundle: DataBundle,
        spec: ExperimentSpec,
        *,
        phase: TrainPhase,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> ViewBatch:
        """Select frames and target views for one train/eval operation."""
```

### Single-Cam Source-View Training

Single-cam source-view training becomes:

```text
DataBundle
  condition_view_id = "source"
  anchor_view_id = None
  train_views = (ViewRecord(view_id="source", role="train", camera_owner="model"),)
  heldout_views = ()

ViewBatch.targets = (
  TargetView(
    view_id="source",
    role="train",
    camera_role="model_predicted",
    camera_owner="model",
    cameras=None,
  ),
)
```

The model predicts cameras inside `GaussianSequence.cameras`. The objective
renders target views with `target.cameras or decoded.cameras`.

Canonical sampler signature:

```python
def sample_single_cam_source_batch(
    bundle: DataBundle,
    spec: ExperimentSpec,
    *,
    phase: TrainPhase,
    step: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> ViewBatch: ...
```

### Known-Camera Single-Cam Training

Known-camera training is the same `DataBundle`, but the target view owns cameras:

```text
TargetView.camera_owner = "batch"
TargetView.cameras = tuple(CameraSpec for selected frames)
```

The model program receives target cameras in `ModelInput.render_cameras`; the
objective renders with `TargetView.cameras`.

Canonical sampler signature:

```python
def sample_known_camera_source_batch(
    bundle: DataBundle,
    spec: ExperimentSpec,
    *,
    phase: TrainPhase,
    step: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> ViewBatch: ...
```

### Multicam DeepView Train/Heldout Training

The current `MulticamVideoBundle` should be converted once into `DataBundle`.

```text
condition camera: camera_0001
anchor camera:    camera_0001
train targets:    camera_0001, camera_0015
heldout targets:  camera_0040
```

The important distinction:

- The condition camera supplies model input pixels/features.
- The anchor camera defines the coordinate frame used to express external poses.
- Train cameras contribute reconstruction loss.
- Held-out cameras contribute validation metrics and media only.

Canonical builder signatures:

```python
def load_multicam_bundle_as_data_bundle(
    spec: ExperimentSpec,
    *,
    device: torch.device,
) -> DataBundle: ...


def sample_multicam_batch(
    bundle: DataBundle,
    spec: ExperimentSpec,
    *,
    phase: TrainPhase,
    step: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> ViewBatch: ...
```

Training phase:

```text
targets = selected train views
role = "train"
camera_owner = "external_rig"
```

Validation phase:

```text
targets = all train views + all heldout views
role = "train" or "heldout"
camera_owner = "external_rig"
```

The held-out target should be the default selector for multicam baselines. Source
or train-view PSNR can still be logged, but it should not be the headline metric.

## FeatureProvider Interface

The feature cache should become a provider with a description step. This removes
the current mutation where the precomputed trainer fills
`model.video_feature_channels` after `VideoFeatureCache.infer_channels()`.

```python
@dataclass(frozen=True)
class FeatureLayerDescription:
    name: str
    channels: int
    layout: Literal["tokens", "cthw", "bcthw", "thwc", "unknown"]
    dtype: torch.dtype
    spatial_size: tuple[int, int] | None
    temporal_size: int | None


@dataclass(frozen=True)
class FeatureDescription:
    provider_name: str
    cache_key_namespace: str
    layers: tuple[FeatureLayerDescription, ...]
    sample_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def channel_map(self) -> dict[str, int]:
        return {layer.name: layer.channels for layer in self.layers}


class FeatureProvider(Protocol):
    def prebake(self, bundle: DataBundle) -> None:
        """Warm caches for all source/eval sequences needed by this run."""

    def describe(self, bundle: DataBundle) -> FeatureDescription:
        """Return layer names/channels/layout before model construction."""

    def features_for(self, conditioning: ConditioningInput) -> Mapping[str, Tensor]:
        """Return feature tensors on the active device for the batch condition."""

    def release(self) -> None:
        """Optionally release heavy extractor weights after successful prebake."""
```

Concrete constructors:

```python
def build_feature_provider(
    spec: ExperimentSpec,
    *,
    device: torch.device,
) -> FeatureProvider | None: ...


def build_video_feature_cache_provider(
    spec: ExperimentSpec,
    *,
    device: torch.device,
    **validated_feature_kwargs: Any,
) -> FeatureProvider: ...
```

Supported precomputed paths:

- V-JEPA HF / TorchHub: `features.extractor in {"vjepa_hf", "vjepa_torchhub"}`.
- LTX: `features.extractor == "ltx"`.
- Wan VACE: `features.extractor in {"wan_vace", "wan2_1_vace", "vace_wan"}`.
- RGB pyramid: lightweight smoke provider.

The provider should include frame selection and source-path fingerprinting in its
cache key, as `VideoFeatureCache.sample_cache_key()` does today. It should also
include `condition_camera` for multicam, because the same scene and frame indices
with a different conditioning camera are different model inputs.

## ModelProgram And Factory Boundary

`build_model_from_config()` currently unrolls a very large `model_kwargs` dict.
Keep dict pass-through, but only at a validated boundary:

```python
@dataclass(frozen=True)
class ModelInput:
    condition_frames: Tensor | None              # [1, K, 3, H, W]
    condition_features: Mapping[str, Tensor] | None
    input_times: Tensor | None                   # [1, K] or [K, 1], adapter-owned
    decode_times: Tensor                         # [K, 1]
    render_cameras: tuple[CameraSpec, ...] | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutput:
    sequence: GaussianSequence
    camera_owner: CameraOwner
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class ModelProgram(Protocol):
    @property
    def feature_dim(self) -> int: ...

    @property
    def camera_owner(self) -> CameraOwner: ...

    def make_input(self, batch: ViewBatch, provider: FeatureProvider | None) -> ModelInput: ...

    def decode(self, model_input: ModelInput) -> ModelOutput: ...

    def parameters(self) -> Iterable[torch.nn.Parameter]: ...


@dataclass(frozen=True)
class ModelFactoryPlan:
    model_class_name: str
    model_kwargs: Mapping[str, Any]
    camera_owner: CameraOwner
    requires_feature_provider: bool
    requires_target_cameras: bool
```

Factory signatures:

```python
def plan_model_factory(
    spec: ExperimentSpec,
    feature_description: FeatureDescription | None,
) -> ModelFactoryPlan: ...


def build_model_program(
    spec: ExperimentSpec,
    feature_description: FeatureDescription | None,
    *,
    device: torch.device,
    **validated_model_kwargs: Any,
) -> ModelProgram: ...
```

Rules:

- `feature_dim != 3` requires `ObjectiveSpec.colorize is not None`.
- `video_encoder_backend == "precomputed"` requires `FeatureProvider is not None`.
- `feature_description.channel_map` fills `ModelSpec.video_feature_channels`
  before `build_model_program()`.
- `static_tokens + dynamic_tokens == tokens` when either split count is set.
- Known-camera variants set `camera_owner = "batch"` and require target cameras.
- Implicit-camera variants set `camera_owner = "model"` and must return
  `GaussianSequence.cameras`.
- External-rig multicam uses an implicit-camera model for splat geometry, but
  target rendering uses `TargetView.cameras`; therefore the objective should
  prefer target cameras for external-rig views and only use decoded cameras when
  `target.camera_owner == "model"`.

## TrainRecipe Registry

The registry should replace trainer-file routing while keeping legacy entrypoints
as thin shims.

```python
@dataclass(frozen=True)
class TrainRecipe:
    name: str
    arch_aliases: tuple[ArchName, ...]
    normalize: Callable[[Mapping[str, Any], Path | None], ExperimentSpec]
    build_data_source: Callable[[ExperimentSpec], DataSource]
    build_view_sampler: Callable[[ExperimentSpec], ViewSampler]
    build_feature_provider: Callable[[ExperimentSpec, torch.device], FeatureProvider | None]
    build_model_program: Callable[
        [ExperimentSpec, FeatureDescription | None, torch.device],
        ModelProgram,
    ]
    build_objective: Callable[[ExperimentSpec, torch.device], "RenderObjective"]
    build_regularizers: Callable[[ExperimentSpec, ModelProgram], tuple["Regularizer", ...]]
    build_optimizer: Callable[[ExperimentSpec, Iterable[torch.nn.Parameter]], torch.optim.Optimizer]
    run: Callable[["TrainState"], None]


ARCH_REGISTRY: dict[ArchName, TrainRecipe] = {
    "tokengs_video_implicit_camera": VIDEO_TOKEN_IMPLICIT_RECIPE,
    "tokengs": VIDEO_TOKEN_IMPLICIT_RECIPE,
    "tokengs_video_known_camera": VIDEO_TOKEN_KNOWN_CAMERA_RECIPE,
    "precomputed_feature_implicit_camera": VIDEO_TOKEN_PRECOMPUTED_RECIPE,
    "ltx_feature_implicit_camera": VIDEO_TOKEN_PRECOMPUTED_RECIPE,
    "wan_vace_feature_implicit_camera": VIDEO_TOKEN_PRECOMPUTED_RECIPE,
    "multicam_precomputed_feature_implicit_camera": MULTICAM_PRECOMPUTED_RECIPE,
    "tokengs_prebaked_camera": LEGACY_PREBAKED_CAMERA_RECIPE,
    "tokengs_prebaked_camera_tiled": LEGACY_PREBAKED_CAMERA_RECIPE,
    "tokengs_image_implicit_camera": LEGACY_IMAGE_IMPLICIT_RECIPE,
    "tokengs_single_image": LEGACY_SINGLE_IMAGE_RECIPE,
    "tokengs_single_image_tiled": LEGACY_SINGLE_IMAGE_RECIPE,
    "gauge_fields_material_surfel": EXTERNAL_GAUGE_FIELD_RECIPE,
    "splat_baseline_free_dynamic_3dgs": EXTERNAL_SPLAT_BASELINE_RECIPE,
    "splat_baseline_static_3dgs": EXTERNAL_SPLAT_BASELINE_RECIPE,
}
```

The first wave should implement native recipes for:

1. `VIDEO_TOKEN_IMPLICIT_RECIPE`
2. `VIDEO_TOKEN_KNOWN_CAMERA_RECIPE`
3. `VIDEO_TOKEN_PRECOMPUTED_RECIPE`
4. `MULTICAM_PRECOMPUTED_RECIPE`

Legacy/gauge recipes can be compatibility wrappers initially:

```python
def run_legacy_entrypoint(spec: ExperimentSpec, module_name: str) -> None:
    """Call the current legacy main(config) with normalized-but-compatible raw config."""
```

## TrainState And Loop

The loop becomes small because recipes provide all moving parts.

```python
@dataclass
class TrainState:
    spec: ExperimentSpec
    recipe: TrainRecipe
    device: torch.device
    data: DataBundle
    sampler: ViewSampler
    feature_provider: FeatureProvider | None
    feature_description: FeatureDescription | None
    model: ModelProgram
    objective: "RenderObjective"
    regularizers: tuple["Regularizer", ...]
    optimizer: torch.optim.Optimizer
    dense_grid: Tensor
    step: int = 0


@dataclass(frozen=True)
class StepResult:
    batch: ViewBatch
    decoded: GaussianSequence
    total_loss: Tensor
    recon_loss: Tensor
    regularizer_losses: Mapping[str, Tensor]
    previews: Mapping[str, Tensor]
    metrics: Mapping[str, float]


def build_train_state(spec: ExperimentSpec, recipe: TrainRecipe) -> TrainState: ...


def train_step(
    state: TrainState,
    *,
    keep_preview: bool,
    generator: torch.Generator | None = None,
) -> StepResult: ...


@torch.no_grad()
def initial_step_result(state: TrainState) -> StepResult: ...


@torch.no_grad()
def validation_payload(state: TrainState) -> dict[str, Any]: ...


def run_train_loop(state: TrainState) -> None: ...
```

`train_step()` should not branch on multicam vs single-cam. Pseudocode:

```python
def train_step(state: TrainState, *, keep_preview: bool, generator=None) -> StepResult:
    state.optimizer.zero_grad(set_to_none=True)

    batch = state.sampler.sample(
        state.data,
        state.spec,
        phase="train",
        step=state.step,
        device=state.device,
        generator=generator,
    )
    validate_view_batch(batch)

    model_input = state.model.make_input(batch, state.feature_provider)
    model_output = state.model.decode(model_input)

    objective_result = state.objective.loss(
        decoded=model_output.sequence,
        batch=batch,
        phase="train",
        keep_preview=keep_preview,
    )
    regularizer_result = evaluate_regularizers(state.regularizers, model_output, batch)

    total = objective_result.loss + regularizer_result.total
    total.backward()
    state.optimizer.step()

    return StepResult(...)
```

The objective is the only place where `(features, alpha)` becomes final RGB. This
is the key invariant that prevents a future multicam drift.

## Routing API

Create one user-facing entrypoint:

```python
def load_experiment_spec(config_path: str | Path) -> ExperimentSpec: ...


def normalize_experiment_config(
    raw: Mapping[str, Any],
    *,
    config_path: Path | None,
) -> ExperimentSpec: ...


def resolve_recipe(spec: ExperimentSpec) -> TrainRecipe: ...


def run_config(config_path: str | Path) -> None: ...


def explain_routing(config_path: str | Path) -> "RoutingReport": ...
```

CLI:

```bash
PYTHONPATH=src/train uv run python src/train/train.py run \
  src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc

PYTHONPATH=src/train uv run python src/train/train.py --explain-routing \
  src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc
```

`--explain-routing` should be implemented before migration because it is the
cheap safety tool. It should print:

```python
@dataclass(frozen=True)
class RoutingReport:
    config_path: Path
    arch: ArchName
    recipe_name: str
    old_entrypoint: str
    model_variant: str
    model_class_name: str
    condition_kind: ConditionKind
    feature_provider: str | None
    feature_layers: Mapping[str, int] | None
    target_views: tuple[str, ...]
    heldout_views: tuple[str, ...]
    camera_owner: CameraOwner
    objective_summary: Mapping[str, Any]
    warnings: tuple[str, ...]
```

Example report for the current ultimate config should state:

```text
arch: multicam_precomputed_feature_implicit_camera
recipe: MULTICAM_PRECOMPUTED_RECIPE
old_entrypoint: src/train/train_multicam_precomputed_feature_implicit_dynamic.py
condition_kind: precomputed_features
condition_camera: camera_0001
anchor_camera: camera_0001
train_targets: camera_0001, camera_0015
heldout_targets: camera_0040
feature_provider: VideoFeatureCache(vjepa_torchhub)
model: DynamicVideoTokenGSImplicitCamera
feature_dim: 32
objective.background.train_mode: random_rgb
warnings:
  - old multicam entrypoint bypasses base feature-splatting objective
```

## Compatibility Shims

Do not break existing shell commands during cleanup. Old files should become
thin wrappers only after the registry is green.

```python
# src/train/train_video_token_implicit_dynamic.py
def main(config: dict[str, Any] | str | Path) -> None:
    from train import run_compat
    run_compat(config, expected_arches={"tokengs_video_implicit_camera", "tokengs", "tokengs_video_known_camera"})
```

```python
# src/train/train_precomputed_feature_implicit_dynamic.py
def main(config: dict[str, Any] | str | Path) -> None:
    from train import run_compat
    run_compat(config, expected_arches={"precomputed_feature_implicit_camera", "ltx_feature_implicit_camera", "wan_vace_feature_implicit_camera"})
```

```python
# src/train/train_multicam_precomputed_feature_implicit_dynamic.py
def main(config: dict[str, Any] | str | Path) -> None:
    from train import run_compat
    run_compat(config, expected_arches={"multicam_precomputed_feature_implicit_camera"})
```

Compatibility helper:

```python
def run_compat(
    config: Mapping[str, Any] | str | Path,
    *,
    expected_arches: set[str],
) -> None:
    raw, config_path = load_raw_config(config)
    arch = str(raw.get("arch", ""))
    if arch not in expected_arches:
        raise ValueError(
            f"This compatibility entrypoint expected {sorted(expected_arches)}, got arch={arch!r}. "
            "Use `src/train/train.py --explain-routing CONFIG` to see the correct route."
        )
    run_config(config_path or raw)
```

The compatibility phase gives us strict route validation without making every
script change on day one.

## Proposed Module Layout

```text
src/train/
  train.py                         # CLI, registry, explain-routing
  train_registry.py                # ARCH_REGISTRY and TrainRecipe definitions
  experiment_spec.py               # normalized config dataclasses/loaders
  data_views.py                    # DataBundle, ViewRecord, ViewBatch, samplers
  feature_provider.py              # FeatureProvider protocol + VideoFeatureCache adapter
  model_programs.py                # ModelProgram protocol + model factory plans
  objectives.py                    # RenderObjective interface consumed here
  regularizers.py                  # camera/bank/rate regularizers as functions
  loops.py                         # run_train_loop/train_step/validation_payload
  compat.py                        # old entrypoint shims
```

Existing modules can be reused:

- `runtime_types.py`: keep `SequenceData`, `GaussianSequence`, maybe move
  `ClipBatch` toward `ViewBatch` after parity.
- `multicam_video_data.py`: keep dataset adapters and camera math; return
  `DataBundle` eventually.
- `video_feature_cache.py`: keep extractors; wrap as `FeatureProvider`.
- `train_video_token_implicit_dynamic.py`: donate model factory fragments and
  regularizers, then shrink to a shim.

## Migration Plan

### Phase 0: Freeze Current Facts

Write down the current run commands and smoke configs before touching code.

Required current facts:

- Single-cam F=3 smoke command.
- Single-cam F=32 feature-splatting smoke command.
- Precomputed V-JEPA single-cam smoke command.
- Multicam RGB-pyramid smoke command.
- Multicam F=32 ultimate smoke command, expected to fail before migration.

### Phase 1: Add Read-Only Routing

Implement:

```python
load_experiment_spec()
resolve_recipe()
explain_routing()
```

No training behavior changes. Add `src/train/train.py --explain-routing CONFIG`
and run it across all 96 configs. Failures here are schema/routing problems, not
training problems.

Acceptance:

- Every config prints one recipe or a deliberate "external legacy" route.
- `ltx_feature_implicit_camera` and `wan_vace_feature_implicit_camera` resolve to
  the precomputed recipe.
- `multicam_precomputed_feature_implicit_camera` resolves to the new multicam
  recipe, not to the old subclass file.

### Phase 2: Introduce DataBundle/ViewBatch Without Changing Objective

Build adapters:

```python
SingleSequenceDataSource -> DataBundle
MulticamDataSource -> DataBundle
ContiguousWindowSampler -> ViewBatch
MulticamViewSampler -> ViewBatch
```

At this stage, old trainer methods can still call into the adapters. The goal is
to prove the batch object can represent every current case.

Acceptance:

- Single-cam source batch shape: `condition.frames [1,K,3,H,W]`, one target.
- Known-camera batch has target cameras.
- Multicam batch has condition, anchor, train, and held-out roles preserved.
- Fixed `frame_indices` configs keep native frame IDs in the batch metadata.

### Phase 3: Wrap VideoFeatureCache As FeatureProvider

Implement `VideoFeatureCacheProvider.describe()` and use it before model
construction. Stop mutating `model_cfg["video_feature_channels"]` in trainer
state.

Acceptance:

- V-JEPA, LTX, Wan, and RGB-pyramid providers all return `FeatureDescription`.
- Cache key includes frame source and condition camera for multicam.
- Model factory receives non-null `video_feature_channels` for precomputed
  backends.

### Phase 4: Route Single-Cam Through TrainRecipe

Move the base trainer path into:

```text
VIDEO_TOKEN_IMPLICIT_RECIPE
VIDEO_TOKEN_KNOWN_CAMERA_RECIPE
VIDEO_TOKEN_PRECOMPUTED_RECIPE
```

Do not delete old files yet. Old files call `run_compat()`.

Acceptance:

- F=3 smoke passes.
- F=32 smoke passes and logs alpha/PCA/composite videos.
- Precomputed V-JEPA smoke passes.
- Known-camera initial validation no longer hits stale tuple handling.

### Phase 5: Route Multicam Through The Same Objective

Replace multicam-specific render/loss with `ViewBatch -> RenderObjective`.

Acceptance:

- Multicam RGB-pyramid smoke passes.
- Multicam F=32 ultimate smoke passes.
- Held-out target uses external rig cameras.
- `Alpha_Mask_Video`, `Feature_PCA_Video`, and composite video exist for at
  least primary train view and primary held-out view.
- Train loss averages across selected train target views with explicit
  `TargetView.loss_weight`.

### Phase 6: Deprecate And Delete

Only after shims have been green across the smoke matrix:

- Delete typo shim `train_camera_implict_dynamic.py`.
- Redirect or delete `train_image_encoder_implicit_camera_baseline.py`.
- Delete empty `train_ltx_feature_implicit_dynamic.py`.
- Keep `dynamicTokenGS.py` until prebaked-camera configs are either routed or
  retired and shared utilities are moved out.

## Smoke Matrix

Every signature/config/dataclass migration must run runtime smokes, not just
`py_compile`.

| Smoke | Config | What it proves |
|---|---|---|
| F=3 single-cam source | fast `tokengs_video_implicit_camera` config patched to `train.steps=1` | Base source-view route, implicit cameras, legacy RGB splatting. |
| F=32 single-cam feature | `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc` patched to 1 step | FeatureProvider not needed, `feature_dim=32`, colorize, alpha, random-bg provenance, PCA/composite logging. |
| Known-camera source | `local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc` patched to 1 step | Target cameras flow through `ViewBatch`; stale tuple handling is gone. |
| Precomputed V-JEPA source | `precomputed_feature_implicit_camera` config patched to 1 step | FeatureProvider prebake/describe/model factory path. |
| LTX/Wan feature source | LTX and Wan one-off configs patched to 1 step | Alias arch routes to precomputed recipe without empty subclasses. |
| Multicam RGB-pyramid | `local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc` | Multicam without huge extractor; train/heldout roles and rig cameras. |
| Multicam F=32 ultimate | `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc` patched to 1 step | The real target: F32 feature splatting, precomputed V-JEPA, multicam train targets, held-out view, shared objective. |
| Explain all configs | all 96 JSONC configs | Registry coverage and legacy/external route clarity. |

Canonical smoke command shape:

```bash
PYTHONPATH=src/train WANDB_MODE=offline \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python \
  src/train/train.py run /tmp/smoke.jsonc
```

`--explain-routing` smoke:

```bash
for config in src/train_configs/*.jsonc; do
  PYTHONPATH=src/train uv run python src/train/train.py --explain-routing "$config" >/tmp/routing.txt
done
```

## Why This Fixes The Multicam Feature-Splatting Bug

Current bug shape:

```text
Base Trainer:
  render_clip_sequence -> (features, alpha)
  colorize(features)
  alpha-compose with random background
  reconstruction loss

Multicam subclass:
  render_clip_sequence -> (features, alpha)
  treats tuple as rendered tensor
  no colorize
  no alpha composition
  no random background
  no alpha/PCA/composite media
```

Proposed shape:

```text
Single-cam ViewBatch ----\
Known-camera ViewBatch ---+--> ModelProgram.decode() -> GaussianSequence
Multicam ViewBatch -------/

GaussianSequence + TargetView(s) -> RenderObjective.loss()
```

There is one conversion from rasterized feature tensors to RGB loss tensors. Any
future change to alpha composition, random backgrounds, colorize init, PCA media,
or feature renderer return signatures lands in one module.

## Factory Boundary Rules For `**kwargs`

The user preference is correct: we should not manually unroll 80 config keys at
every callsite, but we also should not leak unvalidated dicts everywhere.

Allowed:

```python
def build_dynamic_video_token_model(
    *,
    model: ModelSpec,
    camera: CameraSpecConfig,
    feature_description: FeatureDescription | None,
    **validated_overrides: Any,
) -> torch.nn.Module: ...
```

Not allowed:

```python
value = cfg.get("some_new_knob", magic_default)
```

inside render loops, samplers, logging, or objectives.

Boundary rule:

1. Normalize raw config once.
2. Validate unknown keys at the section boundary.
3. Convert to dataclasses.
4. Build one `model_kwargs` dict close to the constructor.
5. From that point on, pass typed objects.

## Open Questions

1. Should `condition_camera` be allowed to be held-out for self-supervised
   source-free experiments? Current multicam code requires it to be in train
   cameras; keep that invariant for migration, but do not bake it into the type.
2. Should `anchor_camera` be train-only? Current relative-pose math assumes yes.
   Future rigs might anchor to a virtual canonical pose.
3. Should feature cache keys include `render_size` or `model.size`? Today the
   feature provider sees source frame size; the cache must avoid collisions
   between 128px and 256px source frames.
4. Should held-out videos log full media every validation step or only at final
   step? The type supports both; W&B bandwidth may argue for primary held-out
   media only.
5. Should gauge-field experiments eventually consume `ViewBatch` too? Probably
   yes for held-out-camera comparability, but not in the first cleanup wave.

## Decision

The highest-leverage cleanup is not deleting files first. It is making
`ViewBatch` and `TrainRecipe` real, then forcing every train route through the
same `RenderObjective`. Once `arch` dispatch is centralized and
`--explain-routing` covers all configs, the duplicate/typo trainer files become
obvious shims instead of live architecture.
