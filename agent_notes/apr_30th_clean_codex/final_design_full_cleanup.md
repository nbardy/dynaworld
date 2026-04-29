# Final Design: Dynaworld Trainer Cleanup

Date: 2026-04-30
Status: design document only
Scope: `src/train` trainer, config, data, model-adapter, feature-provider,
render/objective, validation, smoke, and compatibility routing cleanup

This is the concrete implementation design distilled from the April 30
investigator/proposal/reviewer pass. It is intentionally specific: target file
tree, public types, function signatures, ownership boundaries, migration order,
and expected code size change.

## One Sentence Design

Replace trainer subclass behavior with typed dataflow:

```text
JSONC config
  -> ExperimentSpec
  -> TrainRecipe
  -> DataSource + ViewSampler
  -> ViewBatch
  -> FeatureProvider
  -> ModelProgram
  -> GaussianSequence
  -> RenderObjective
  -> ObjectiveLoss + RenderedView artifacts
  -> TrainLoop + Validation
```

The key invariant is that `RenderObjective` is the only place where rasterized
`(features, alpha)` becomes RGB and reaches reconstruction loss. This is what
prevents single-cam, known-camera, precomputed-feature, and multicam routes from
silently drifting.

## Current Files To Collapse

These are the current files whose responsibilities are tangled or duplicated:

```text
src/train/train_video_token_implicit_dynamic.py              2072 lines
src/train/train_precomputed_feature_implicit_dynamic.py       165 lines
src/train/train_multicam_precomputed_feature_implicit_dynamic.py 386 lines
src/train/train_ltx_feature_implicit_dynamic.py                32 lines
src/train/train_camera_implicit_dynamic.py                    417 lines
src/train/train_camera_implict_dynamic.py                       4 lines
src/train/train_image_encoder_implicit_camera_baseline.py        4 lines
src/train/dynamicTokenGS.py                                   731 lines
src/train/tokenGS.py                                          145 lines
src/train/rendering.py                                        446 lines
src/train/renderers/fast_mac.py                               428 lines
src/train/colorize.py                                         200 lines
src/train/video_feature_cache.py                              730 lines
src/train/multicam_video_data.py                             1095 lines
src/train/camera_rig.py                                       281 lines
src/train/runtime_types.py                                    253 lines
src/train/config_utils.py                                     107 lines
```

Measured subtotal for these active surfaces: 7496 lines.

Measured total Python under `src/train`: 17366 lines.

The first cleanup should not rewrite low-level renderers or model internals. It
should split and route the trainer surface around them.

## Target File Tree

Proposed final tree. Some names are new; some existing files stay where they
are as low-level implementations.

```text
src/train/
  train.py
  routing.py
  config_schema.py
  train_state.py
  loop.py
  validation.py
  media.py
  smoke.py
  provenance.py
  compat.py

  data/
    __init__.py
    types.py
    source.py
    sampler.py
    single_video.py
    known_camera_video.py
    multicam.py

  features/
    __init__.py
    provider.py
    video_feature_provider.py
    cache_keys.py

  models/
    __init__.py
    programs.py
    factory.py
    video_token_program.py
    known_camera_program.py
    unconditioned_program.py
    legacy_programs.py

  objective/
    __init__.py
    types.py
    background.py
    rasterize.py
    colorize_adapter.py
    loss.py
    objective.py

  regularizers/
    __init__.py
    bank_rate.py
    camera_rig.py

  recipes/
    __init__.py
    base.py
    video_token.py
    precomputed_feature.py
    multicam_precomputed.py
    known_camera.py
    legacy_image.py
    legacy_dynamic_token.py
    external_gauge.py

  legacy/
    __init__.py
    dynamic_token_utils.py
    image_implicit_adapter.py

  gs_models/
    ... keep existing model classes ...

  renderers/
    ... keep existing low-level renderer backends ...

  rendering.py
  colorize.py
  runtime_types.py
  sequence_data.py
  video_feature_cache.py
  multicam_video_data.py
  camera_rig.py
```

Compatibility shims remain during migration:

```text
src/train/train_video_token_implicit_dynamic.py
src/train/train_precomputed_feature_implicit_dynamic.py
src/train/train_multicam_precomputed_feature_implicit_dynamic.py
src/train/train_camera_implicit_dynamic.py
src/train/dynamicTokenGS.py
src/train/tokenGS.py
```

Delete after shim period:

```text
src/train/train_camera_implict_dynamic.py
src/train/train_image_encoder_implicit_camera_baseline.py
src/train/train_ltx_feature_implicit_dynamic.py
```

Do not delete in this cleanup:

```text
src/train/dynamicTokenGS.py
src/train/tokenGS.py
src/train/*tiled*.py
research_experiments/gauge_fields/*
```

Extract shared utilities out of `dynamicTokenGS.py` before any later retirement:

```text
pick_device
configure_fast_attn
fast_attn_context
optimizer / LR helpers still imported by active routes
```

## Config And Routing

File: `src/train/config_schema.py`

### Literals

```python
ArchName = str
ViewRole = Literal["source", "train", "heldout", "debug"]
CameraRole = Literal[
    "condition",
    "anchor",
    "train_target",
    "heldout_target",
    "model_predicted",
]
CameraOwner = Literal["model", "batch", "external_rig", "none"]
ConditionKind = Literal["rgb_video", "precomputed_features", "none"]
RunPhase = Literal["train", "eval", "preview", "export"]
BackgroundMode = Literal["white", "black", "fixed_rgb", "random_rgb", "none"]
BackgroundSampleScope = Literal["step", "view", "frame"]
RecipeStatus = Literal["active", "blocked", "compat", "legacy", "external"]
```

`ArchName` stays `str`. A hand-written `Literal[...]` for all arch names will
drift.

### Config Specs

```python
@dataclass(frozen=True)
class CompatibilitySpec:
    old_entrypoint: str | None = None
    old_arch: str | None = None
    compat_shim: bool = False
    migrated_from: str | None = None
    route_status: RecipeStatus = "active"
    block_reason: str | None = None


@dataclass(frozen=True)
class DataSpec:
    root: Path | None
    frames_dir: Path | None
    source_path: Path | None
    eval_source_path: Path | None
    frame_indices: tuple[int, ...] | None
    max_frames: int | None
    input_size: int
    fps: float
    dataset_kind: Literal["single_video", "known_camera_video", "multicam", "legacy"]
    multicam_manifest: Path | None = None
    multicam_split: str | None = None
    multicam_sample_id: str | None = None
    multicam_sample_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewsSpec:
    condition_camera: str | None = None
    anchor_camera: str | None = None
    train_cameras: tuple[str, ...] = ()
    heldout_cameras: tuple[str, ...] = ()
    train_views_per_step: int = 1
    render_all_train_views_for_eval: bool = True
    render_all_heldout_views_for_eval: bool = True
    primary_train_view: str | None = None
    primary_heldout_view: str | None = None


@dataclass(frozen=True)
class FeatureProviderSpec:
    name: str
    enabled: bool
    cache_dir: Path | None
    model_id: str | None
    layers: tuple[str, ...]
    input_size: int | None
    dtype: str
    layout: str
    sample_cache_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    size: int
    feature_dim: int
    num_tokens: int
    gaussians_per_token: int
    static_tokens: int | None = None
    dynamic_tokens: int | None = None
    use_static_dynamic_split: bool = False
    video_feature_channels: Mapping[str, int] = field(default_factory=dict)
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraSpecConfig:
    owner: CameraOwner
    projection: str | None
    fov_degrees: float | None
    radius: float | None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraRigSpec:
    enabled: bool
    learn_global: bool = True
    learn_per_view: bool = False
    regularization_weight: float = 0.0
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FastMacSpec:
    background: tuple[float, float, float] | None = None
    feature_background: float = 0.0
    alpha_threshold: float = 0.0
    near_plane: float = 0.01
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderSpec:
    renderer: str
    input_size: int
    render_size: int
    tile_size: int
    bound_scale: float
    camera_projection: str | None
    fast_mac: FastMacSpec


@dataclass(frozen=True)
class BackgroundSpec:
    train_mode: BackgroundMode
    eval_mode: BackgroundMode
    fixed_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_scope: BackgroundSampleScope = "step"
    apply_when_alpha_missing: bool = False


@dataclass(frozen=True)
class ColorizeSpec:
    enabled: bool
    feature_dim: int
    hidden_dim: int | None = None
    activation: str = "sigmoid"
    pre_norm: bool = False
    weight_init: str = "kaiming"
    weight_init_gain: float = 1.0
    view_condition: str = "none"


@dataclass(frozen=True)
class ReconstructionLossSpec:
    kind: str
    l1_weight: float
    ssim_weight: float
    mse_weight: float = 0.0
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveSpec:
    version: str
    reconstruction: ReconstructionLossSpec
    background: BackgroundSpec
    colorize: ColorizeSpec
    retain_train_artifacts: bool = False


@dataclass(frozen=True)
class TrainSpec:
    steps: int
    lr: float
    seed: int
    clip_frames: int
    batch_size: int = 1
    render_chunk_size: int | None = None
    optimizer: str = "adam"
    log_every: int = 10
    video_log_every: int = 100
    always_log_last_step: bool = True
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoggingSpec:
    project: str
    run_name: str | None
    mode: str
    video_fps: float
    log_alpha: bool
    log_feature_pca: bool
    log_composite: bool
    log_heldout_media: bool
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    config_path: Path | None
    arch: ArchName
    raw: Mapping[str, Any]
    data: DataSpec
    views: ViewsSpec
    features: FeatureProviderSpec | None
    model: ModelSpec
    camera: CameraSpecConfig
    rig: CameraRigSpec | None
    render: RenderSpec
    objective: ObjectiveSpec
    train: TrainSpec
    logging: LoggingSpec
    export: Mapping[str, Any]
    compatibility: CompatibilitySpec
```

### Config Functions

```python
def load_raw_config(path: str | Path) -> dict[str, Any]: ...


def normalize_experiment_config(
    raw: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> ExperimentSpec: ...


def normalize_legacy_losses(raw: Mapping[str, Any]) -> ObjectiveSpec: ...


def normalize_legacy_views(raw: Mapping[str, Any]) -> ViewsSpec: ...


def normalize_legacy_features(raw: Mapping[str, Any]) -> FeatureProviderSpec | None: ...


def validate_experiment_spec(spec: ExperimentSpec) -> None: ...


def validate_feature_splatting_spec(spec: ExperimentSpec) -> None: ...


def validate_background_spec(spec: ExperimentSpec) -> None: ...


def config_to_wandb_dict(spec: ExperimentSpec) -> dict[str, Any]: ...
```

Required validations:

```text
feature_dim != 3 -> colorize.enabled must be true
feature_dim != 3 -> objective.background.train_mode must be explicit
multicam + feature_dim != 3 -> route blocked until shared objective is enabled
render.fast_mac.feature_background != objective.background
ViewRole never includes "eval"; eval is RunPhase
```

File: `src/train/routing.py`

```python
@dataclass(frozen=True)
class RoutingReport:
    config_path: Path
    arch: ArchName
    recipe_name: str
    route_status: RecipeStatus
    old_entrypoint: str | None
    new_entrypoint: str
    compat_shim: bool
    model_variant: str
    feature_dim: int
    objective_version: str
    background_train_mode: BackgroundMode
    background_eval_mode: BackgroundMode
    data_kind: str
    feature_provider: str | None
    warnings: tuple[str, ...]
    block_reason: str | None = None
    next_required_phase: str | None = None


def load_experiment_spec(config_path: str | Path) -> ExperimentSpec: ...


def resolve_recipe(spec: ExperimentSpec) -> "TrainRecipe": ...


def explain_routing(config_path: str | Path) -> RoutingReport: ...


def format_routing_report(report: RoutingReport) -> str: ...


def assert_route_can_run(spec: ExperimentSpec, recipe: "TrainRecipe") -> None: ...


def run_config(config_path: str | Path) -> None: ...
```

Blocked route example:

```text
route_status: blocked
reason: feature_dim=32 requires shared feature_alpha objective for multicam
old_command: PYTHONPATH=src/train uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py ...
next_required_phase: migrate_multicam_objective
```

File: `src/train/train.py`

```python
def main(argv: Sequence[str] | None = None) -> int: ...


def main_run(config_path: str | Path) -> None: ...


def main_explain(config_path: str | Path) -> None: ...


def main_smoke(config_path: str | Path, *, steps: int = 1, offline: bool = True) -> None: ...
```

CLI:

```bash
PYTHONPATH=src/train uv run python src/train/train.py explain <config.jsonc>
PYTHONPATH=src/train uv run python src/train/train.py run <config.jsonc>
PYTHONPATH=src/train uv run python src/train/train.py smoke <config.jsonc> --steps 1 --offline
```

## Data And View Batching

File: `src/train/data/types.py`

```python
@dataclass(frozen=True)
class ViewRecord:
    view_id: str
    role: ViewRole
    camera_role: CameraRole
    camera_name: str | None
    frames: torch.Tensor                         # [T, 3, H, W]
    frame_times: torch.Tensor                    # [T, 1]
    video_fps: float
    cameras: tuple[CameraSpec, ...] | None
    source_path: Path | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataBundle:
    sample_id: str
    scene_id: str | None
    source_sequences: tuple[SequenceData, ...]
    train_views: tuple[ViewRecord, ...]
    heldout_views: tuple[ViewRecord, ...]
    eval_views: tuple[ViewRecord, ...]
    condition_view_id: str | None
    anchor_view_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditioningInput:
    sample_id: str
    scene_id: str | None
    kind: ConditionKind
    frames: torch.Tensor | None                  # [1, K, 3, H, W] or None
    features: Mapping[str, torch.Tensor] | None
    frame_indices: torch.Tensor                  # [K]
    frame_times: torch.Tensor                    # [K, 1]
    video_fps: float
    camera_name: str | None
    cameras: tuple[CameraSpec, ...] | None
    source_path: Path | None
    feature_cache_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetView:
    view_id: str
    role: ViewRole
    camera_role: CameraRole
    camera_owner: CameraOwner
    frames: torch.Tensor                         # [K, 3, H_in, W_in]
    frame_indices: torch.Tensor                  # [K]
    frame_times: torch.Tensor                    # [K, 1]
    video_fps: float
    camera_name: str | None
    cameras: tuple[CameraSpec, ...] | None
    loss_weight: float = 1.0
    metrics_prefix: str | None = None
    log_media: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewBatch:
    batch_id: str
    sample_id: str
    scene_id: str | None
    phase: RunPhase
    device: torch.device
    conditioning: ConditioningInput
    targets: tuple[TargetView, ...]
    decode_times: torch.Tensor                   # [K, 1]
    frame_indices: torch.Tensor                  # [K]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int: ...

    @property
    def target_count(self) -> int: ...
```

### Data Protocols

File: `src/train/data/source.py`

```python
class DataSource(Protocol):
    def load(
        self,
        spec: ExperimentSpec,
        *,
        device: torch.device,
    ) -> DataBundle: ...


def build_data_source(spec: ExperimentSpec) -> DataSource: ...
```

File: `src/train/data/sampler.py`

```python
class ViewSampler(Protocol):
    def sample(
        self,
        bundle: DataBundle,
        spec: ExperimentSpec,
        *,
        phase: RunPhase,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> ViewBatch: ...


def validate_view_batch(batch: ViewBatch) -> None: ...


def select_frame_window(
    frame_count: int,
    clip_frames: int,
    *,
    step: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor: ...


def build_conditioning_input(
    bundle: DataBundle,
    view: ViewRecord,
    frame_indices: torch.Tensor,
    *,
    kind: ConditionKind,
    device: torch.device,
) -> ConditioningInput: ...


def build_target_view(
    record: ViewRecord,
    frame_indices: torch.Tensor,
    *,
    role: ViewRole,
    camera_role: CameraRole,
    camera_owner: CameraOwner,
    loss_weight: float,
    log_media: bool,
    device: torch.device,
) -> TargetView: ...
```

File: `src/train/data/single_video.py`

```python
class SingleVideoDataSource:
    def load(self, spec: ExperimentSpec, *, device: torch.device) -> DataBundle: ...


class SingleVideoViewSampler:
    def sample(
        self,
        bundle: DataBundle,
        spec: ExperimentSpec,
        *,
        phase: RunPhase,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> ViewBatch: ...


def load_single_video_bundle(spec: ExperimentSpec, *, device: torch.device) -> DataBundle: ...


def sample_single_video_batch(
    bundle: DataBundle,
    spec: ExperimentSpec,
    *,
    phase: RunPhase,
    step: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> ViewBatch: ...
```

File: `src/train/data/known_camera_video.py`

```python
class KnownCameraVideoDataSource:
    def load(self, spec: ExperimentSpec, *, device: torch.device) -> DataBundle: ...


class KnownCameraViewSampler:
    def sample(
        self,
        bundle: DataBundle,
        spec: ExperimentSpec,
        *,
        phase: RunPhase,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> ViewBatch: ...
```

File: `src/train/data/multicam.py`

```python
class MulticamDataSource:
    def load(self, spec: ExperimentSpec, *, device: torch.device) -> DataBundle: ...


class MulticamViewSampler:
    def sample(
        self,
        bundle: DataBundle,
        spec: ExperimentSpec,
        *,
        phase: RunPhase,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> ViewBatch: ...


def load_multicam_bundle_as_data_bundle(
    spec: ExperimentSpec,
    *,
    device: torch.device,
) -> DataBundle: ...


def resolve_multicam_condition_view(bundle: DataBundle, spec: ExperimentSpec) -> ViewRecord: ...


def resolve_multicam_anchor_view(bundle: DataBundle, spec: ExperimentSpec) -> ViewRecord: ...


def select_multicam_train_views(
    bundle: DataBundle,
    spec: ExperimentSpec,
    *,
    step: int,
    generator: torch.Generator | None = None,
) -> tuple[ViewRecord, ...]: ...


def select_multicam_heldout_views(
    bundle: DataBundle,
    spec: ExperimentSpec,
    *,
    phase: RunPhase,
) -> tuple[ViewRecord, ...]: ...
```

Multicam rule: `condition_camera`, `anchor_camera`, train target cameras, and
held-out cameras are separate roles. Do not collapse them into list indices.

## Feature Provider

File: `src/train/features/provider.py`

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
    def channel_map(self) -> dict[str, int]: ...


class FeatureProvider(Protocol):
    def describe_config(
        self,
        spec: FeatureProviderSpec,
    ) -> FeatureDescription | None: ...

    def describe_data(
        self,
        bundle: DataBundle,
    ) -> FeatureDescription: ...

    def prebake(
        self,
        bundle: DataBundle,
    ) -> None: ...

    def load(
        self,
        conditioning: ConditioningInput,
    ) -> Mapping[str, torch.Tensor]: ...

    def release(self) -> None: ...


def build_feature_provider(
    spec: ExperimentSpec,
    *,
    device: torch.device,
) -> FeatureProvider | None: ...
```

`describe_config()` must not bake features or mutate cache. It is safe for
`train.py explain`.

`describe_data()` may inspect the loaded bundle and may check cache metadata. It
belongs in run/smoke, before model construction.

File: `src/train/features/cache_keys.py`

```python
@dataclass(frozen=True)
class FeatureCacheIdentity:
    extractor_name: str
    extractor_version_or_model_id: str
    feature_layer_names: tuple[str, ...]
    source_path: Path | None
    source_content_hash: str | None
    native_frame_ids: tuple[int, ...]
    frame_times_hash: str | None
    condition_camera_id: str | None
    input_resize_policy: str
    input_crop_policy: str
    dtype: str
    layout: str


def build_feature_cache_identity(
    spec: ExperimentSpec,
    bundle: DataBundle,
    conditioning: ConditioningInput,
) -> FeatureCacheIdentity: ...


def feature_cache_key(identity: FeatureCacheIdentity) -> str: ...
```

File: `src/train/features/video_feature_provider.py`

```python
class VideoFeatureProvider:
    def __init__(
        self,
        spec: FeatureProviderSpec,
        *,
        device: torch.device,
    ) -> None: ...

    def describe_config(self, spec: FeatureProviderSpec) -> FeatureDescription | None: ...

    def describe_data(self, bundle: DataBundle) -> FeatureDescription: ...

    def prebake(self, bundle: DataBundle) -> None: ...

    def load(self, conditioning: ConditioningInput) -> Mapping[str, torch.Tensor]: ...

    def release(self) -> None: ...
```

The current `video_feature_cache.py` can stay as the implementation backend.
This provider wraps it and removes trainer mutation of
`model.video_feature_channels`.

## Model Programs

File: `src/train/models/programs.py`

```python
@dataclass(frozen=True)
class ModelInput:
    condition_frames: torch.Tensor | None
    condition_features: Mapping[str, torch.Tensor] | None
    input_times: torch.Tensor | None
    decode_times: torch.Tensor
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

    def make_input(
        self,
        batch: ViewBatch,
        provider: FeatureProvider | None,
    ) -> ModelInput: ...

    def decode(
        self,
        model_input: ModelInput,
    ) -> ModelOutput: ...

    def parameters(self) -> Iterable[torch.nn.Parameter]: ...

    def train(self, mode: bool = True) -> "ModelProgram": ...

    def eval(self) -> "ModelProgram": ...
```

Do not let `ModelProgram.decode()` accept `ViewBatch`. That would hide adapter
logic inside model code and recreate subclass drift.

File: `src/train/models/factory.py`

```python
def build_model_program(
    spec: ExperimentSpec,
    feature_description: FeatureDescription | None,
    *,
    device: torch.device,
) -> ModelProgram: ...


def build_model_module(
    model_spec: ModelSpec,
    camera_spec: CameraSpecConfig,
    feature_description: FeatureDescription | None,
    *,
    device: torch.device,
) -> torch.nn.Module: ...


def validated_model_kwargs(
    model_spec: ModelSpec,
    camera_spec: CameraSpecConfig,
    feature_description: FeatureDescription | None,
) -> dict[str, Any]: ...


def reject_unknown_kwargs(
    kwargs: Mapping[str, Any],
    *,
    allowed: set[str],
    context: str,
) -> None: ...
```

`**kwargs` are allowed only inside validated factory helpers. Constructors
should not silently swallow typos.

File: `src/train/models/video_token_program.py`

```python
class VideoTokenImplicitCameraProgram:
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        feature_dim: int,
        camera_owner: CameraOwner = "model",
    ) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    @property
    def camera_owner(self) -> CameraOwner: ...

    def make_input(self, batch: ViewBatch, provider: FeatureProvider | None) -> ModelInput: ...

    def decode(self, model_input: ModelInput) -> ModelOutput: ...

    def parameters(self) -> Iterable[torch.nn.Parameter]: ...
```

File: `src/train/models/known_camera_program.py`

```python
class KnownCameraVideoTokenProgram:
    def __init__(self, module: torch.nn.Module, *, feature_dim: int) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    @property
    def camera_owner(self) -> CameraOwner: ...

    def make_input(self, batch: ViewBatch, provider: FeatureProvider | None) -> ModelInput: ...

    def decode(self, model_input: ModelInput) -> ModelOutput: ...

    def parameters(self) -> Iterable[torch.nn.Parameter]: ...
```

File: `src/train/models/unconditioned_program.py`

```python
class UnconditionedProgram:
    def __init__(self, module: torch.nn.Module, *, feature_dim: int) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    @property
    def camera_owner(self) -> CameraOwner: ...

    def make_input(self, batch: ViewBatch, provider: FeatureProvider | None) -> ModelInput: ...

    def decode(self, model_input: ModelInput) -> ModelOutput: ...

    def parameters(self) -> Iterable[torch.nn.Parameter]: ...
```

File: `src/train/models/legacy_programs.py`

```python
class LegacyImageProgram: ...


class LegacyDynamicTokenProgram: ...


def build_legacy_program(spec: ExperimentSpec, *, device: torch.device) -> ModelProgram: ...
```

## Render Objective

File: `src/train/objective/types.py`

```python
@dataclass(frozen=True)
class RasterizedView:
    view: TargetView
    features: torch.Tensor                       # [K, F, H, W]
    alpha: torch.Tensor | None                   # [K, H, W]
    cameras: tuple[CameraSpec, ...]
    feature_dim: int
    render_size: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ColorizedView:
    splat_rgb: torch.Tensor                      # [K, 3, H, W]
    logits: torch.Tensor | None = None
    view_dirs: torch.Tensor | None = None


@dataclass(frozen=True)
class BackgroundSample:
    rgb: torch.Tensor | None                     # [1|K, 3, 1, 1]
    mode: BackgroundMode
    phase: RunPhase
    scope: BackgroundSampleScope
    seed: int | None = None
    step: int | None = None


@dataclass(frozen=True)
class RenderedView:
    view: TargetView
    rgb: torch.Tensor                            # [K, 3, H, W]
    target_rgb: torch.Tensor | None              # [K, 3, H, W]
    rasterized: RasterizedView
    colorized: ColorizedView | None
    background: BackgroundSample
    phase: RunPhase
    metrics_prefix: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewLoss:
    view_id: str
    role: ViewRole
    total: torch.Tensor
    per_image: torch.Tensor                      # [K]
    weight: float
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    view_losses: tuple[ViewLoss, ...]
    rendered_views: tuple[RenderedView, ...]
```

File: `src/train/objective/background.py`

```python
class BackgroundPolicyProtocol(Protocol):
    def sample(
        self,
        *,
        phase: RunPhase,
        like: torch.Tensor,
        view_count: int,
        frame_count: int,
        generator: torch.Generator | None = None,
        step: int | None = None,
    ) -> BackgroundSample: ...


class BackgroundPolicy:
    def __init__(self, spec: BackgroundSpec) -> None: ...

    def sample(
        self,
        *,
        phase: RunPhase,
        like: torch.Tensor,
        view_count: int,
        frame_count: int,
        generator: torch.Generator | None = None,
        step: int | None = None,
    ) -> BackgroundSample: ...


def background_mode_for_phase(spec: BackgroundSpec, phase: RunPhase) -> BackgroundMode: ...


def sample_background_rgb(
    mode: BackgroundMode,
    *,
    fixed_rgb: tuple[float, float, float],
    scope: BackgroundSampleScope,
    like: torch.Tensor,
    view_count: int,
    frame_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None: ...
```

File: `src/train/objective/rasterize.py`

```python
class RasterizerProtocol(Protocol):
    def rasterize(
        self,
        decoded: GaussianSequence,
        cameras: tuple[CameraSpec, ...],
        *,
        render_spec: RenderSpec,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class GaussianRasterizer:
    def __init__(self, render_spec: RenderSpec) -> None: ...

    def rasterize(
        self,
        decoded: GaussianSequence,
        cameras: tuple[CameraSpec, ...],
        *,
        render_spec: RenderSpec | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


def cameras_for_target(
    decoded: GaussianSequence,
    target: TargetView,
) -> tuple[CameraSpec, ...]: ...


def rasterize_target_view(
    decoded: GaussianSequence,
    target: TargetView,
    *,
    render_spec: RenderSpec,
    rasterizer: RasterizerProtocol,
) -> RasterizedView: ...
```

Critical implementation:

```python
def cameras_for_target(decoded: GaussianSequence, target: TargetView) -> tuple[CameraSpec, ...]:
    if target.camera_owner == "model":
        if decoded.cameras is None:
            raise ValueError("target requires model cameras, but decoded.cameras is None")
        return decoded.cameras
    if target.cameras is None:
        raise ValueError("target requires explicit cameras, but target.cameras is None")
    return target.cameras
```

This prevents multicam from accidentally using model-predicted cameras instead
of external rig cameras.

File: `src/train/objective/colorize_adapter.py`

```python
class ColorizerProtocol(Protocol):
    feature_dim: int

    def __call__(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


def build_colorizer(spec: ExperimentSpec, *, device: torch.device) -> ColorizerProtocol | None: ...


def colorize_rasterized_view(
    rasterized: RasterizedView,
    *,
    colorizer: ColorizerProtocol | None,
) -> ColorizedView | None: ...


def maybe_compute_view_dirs(
    rasterized: RasterizedView,
    colorize_spec: ColorizeSpec,
) -> torch.Tensor | None: ...
```

File: `src/train/objective/loss.py`

```python
def resize_target_for_render(
    target: TargetView,
    *,
    render_size: int,
) -> torch.Tensor: ...


def reconstruction_loss_for_rendered_view(
    rendered: RenderedView,
    loss_spec: ReconstructionLossSpec,
) -> ViewLoss: ...


def aggregate_view_losses(
    view_losses: Sequence[ViewLoss],
) -> torch.Tensor: ...


def metrics_for_rendered_view(
    rendered: RenderedView,
    loss_spec: ReconstructionLossSpec,
) -> Mapping[str, float]: ...
```

Loss aggregation rule:

```text
total = sum(view_loss.total * view_loss.weight for view_loss in view_losses) /
        sum(view_loss.weight for view_loss in view_losses)
```

Chunking rule: if temporal chunks are used, preserve per-frame weighting:

```text
sum(chunk_loss * chunk_frame_count) / total_frame_count
```

Do not accidentally reweight temporal chunks by number of chunks.

File: `src/train/objective/objective.py`

```python
@dataclass
class RenderObjective:
    render_spec: RenderSpec
    objective_spec: ObjectiveSpec
    background_policy: BackgroundPolicyProtocol
    colorizer: ColorizerProtocol | None = None
    rasterizer: RasterizerProtocol | None = None

    def rasterize_view(
        self,
        decoded: GaussianSequence,
        target: TargetView,
    ) -> RasterizedView: ...

    def colorize_view(
        self,
        rasterized: RasterizedView,
    ) -> ColorizedView | None: ...

    def compose_view(
        self,
        rasterized: RasterizedView,
        colorized: ColorizedView | None,
        *,
        target_rgb: torch.Tensor | None,
        background: BackgroundSample,
        phase: RunPhase,
    ) -> RenderedView: ...

    def render_view(
        self,
        decoded: GaussianSequence,
        target: TargetView,
        *,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
        retain_target: bool = True,
    ) -> RenderedView: ...

    def render_view_batch(
        self,
        decoded: GaussianSequence,
        targets: Sequence[TargetView],
        *,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
        retain_targets: bool = True,
    ) -> tuple[RenderedView, ...]: ...

    def loss_for_view(
        self,
        rendered: RenderedView,
        *,
        weight: float,
    ) -> ViewLoss: ...

    def loss_for_batch(
        self,
        decoded: GaussianSequence,
        batch: ViewBatch,
        *,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
        keep_preview: bool = False,
    ) -> ObjectiveLoss: ...


def compose_rgb(
    *,
    rasterized: RasterizedView,
    colorized: ColorizedView | None,
    background: BackgroundSample,
) -> torch.Tensor: ...


def validate_rendered_rgb_shape(rendered: RenderedView) -> None: ...
```

Composition rules:

```text
F == 3 and alpha is None:
  rgb = rasterized.features

F > 3 and alpha is not None:
  splat_rgb = colorizer(features)
  rgb = alpha[:, None] * splat_rgb + (1 - alpha[:, None]) * background.rgb

F > 3 and alpha is None:
  only allowed if objective_spec.background.apply_when_alpha_missing is false
  rgb = colorizer(features)

raw F-channel features never go directly to RGB reconstruction loss
```

## Regularizers

File: `src/train/regularizers/bank_rate.py`

```python
@dataclass(frozen=True)
class RegularizerLoss:
    total: torch.Tensor
    terms: Mapping[str, torch.Tensor]


class Regularizer(Protocol):
    def loss(
        self,
        model_output: ModelOutput,
        batch: ViewBatch,
        *,
        step: int,
    ) -> RegularizerLoss: ...


class BankRateRegularizer:
    def __init__(self, spec: ExperimentSpec) -> None: ...

    def loss(
        self,
        model_output: ModelOutput,
        batch: ViewBatch,
        *,
        step: int,
    ) -> RegularizerLoss: ...
```

File: `src/train/regularizers/camera_rig.py`

```python
class CameraRigRegularizer:
    def __init__(self, spec: CameraRigSpec) -> None: ...

    def loss(
        self,
        model_output: ModelOutput,
        batch: ViewBatch,
        *,
        step: int,
    ) -> RegularizerLoss: ...
```

`RenderObjective` must not own bank-rate loss or rig regularization.

## Train State And Loop

File: `src/train/train_state.py`

```python
@dataclass
class TrainState:
    spec: ExperimentSpec
    recipe: "TrainRecipe"
    device: torch.device
    data: DataBundle
    sampler: ViewSampler
    feature_provider: FeatureProvider | None
    feature_description: FeatureDescription | None
    model: ModelProgram
    objective: RenderObjective
    regularizers: tuple[Regularizer, ...]
    optimizer: torch.optim.Optimizer
    generator: torch.Generator
    step: int = 0


@dataclass(frozen=True)
class StepResult:
    step: int
    batch: ViewBatch
    model_output: ModelOutput
    objective: ObjectiveLoss
    regularizers: RegularizerLoss
    total_loss: torch.Tensor
    previews: Mapping[str, RenderedView]
    scalars: Mapping[str, float]


def build_train_state(
    spec: ExperimentSpec,
    recipe: "TrainRecipe",
    *,
    device: torch.device,
) -> TrainState: ...


def collect_trainable_parameters(state: TrainState) -> Iterable[torch.nn.Parameter]: ...
```

File: `src/train/loop.py`

```python
def train_step(
    state: TrainState,
    *,
    keep_preview: bool = False,
) -> StepResult: ...


def run_training_loop(state: TrainState) -> None: ...


def backward_step(
    state: TrainState,
    total_loss: torch.Tensor,
) -> None: ...


def compute_regularizer_loss(
    regularizers: Sequence[Regularizer],
    model_output: ModelOutput,
    batch: ViewBatch,
    *,
    step: int,
) -> RegularizerLoss: ...


def combine_losses(
    objective: ObjectiveLoss,
    regularizers: RegularizerLoss,
) -> torch.Tensor: ...


def scalar_payload_from_step(result: StepResult) -> dict[str, float]: ...
```

Generic train step pseudocode:

```python
batch = state.sampler.sample(...)
model_input = state.model.make_input(batch, state.feature_provider)
model_output = state.model.decode(model_input)
objective_loss = state.objective.loss_for_batch(
    model_output.sequence,
    batch,
    phase="train",
    generator=state.generator,
    step=state.step,
    keep_preview=keep_preview,
)
regularizer_loss = compute_regularizer_loss(...)
total_loss = combine_losses(objective_loss, regularizer_loss)
backward_step(state, total_loss)
```

There is no `if multicam` in the loop.

## Validation And Media

File: `src/train/validation.py`

```python
@dataclass(frozen=True)
class ValidationRender:
    view: TargetView
    rendered: RenderedView
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ValidationResult:
    step: int
    renders: tuple[ValidationRender, ...]
    scalars: Mapping[str, float]
    decoded_metrics: Mapping[str, float]


def sample_validation_batch(
    state: TrainState,
    *,
    phase: Literal["eval", "preview"] = "eval",
) -> ViewBatch: ...


def render_validation_views(
    state: TrainState,
    *,
    phase: Literal["eval", "preview"] = "eval",
) -> ValidationResult: ...


def metrics_for_validation_render(render: ValidationRender) -> Mapping[str, float]: ...


def validation_scalar_payload(result: ValidationResult) -> dict[str, float]: ...
```

File: `src/train/media.py`

```python
@dataclass(frozen=True)
class ValidationPayload:
    scalars: Mapping[str, float]
    videos: Mapping[str, Any]
    images: Mapping[str, Any]
    decoded_metrics: Mapping[str, float]


def build_validation_payload(
    result: ValidationResult,
    *,
    video_fps: float,
    log_gt: bool,
    include_alpha: bool,
    include_feature_pca: bool,
    include_composite: bool,
) -> ValidationPayload: ...


def rendered_view_to_video(rendered: RenderedView) -> Any: ...


def alpha_to_video(rendered: RenderedView) -> Any | None: ...


def feature_pca_to_video(rendered: RenderedView) -> Any | None: ...


def composite_to_video(rendered: RenderedView) -> Any | None: ...


def media_key_for_view(view: TargetView, suffix: str) -> str: ...
```

Required media for F32:

```text
Alpha_Mask_Video
Feature_PCA_Video
Render_Composite_Video
Heldout*_Alpha_Mask_Video
Heldout*_Feature_PCA_Video
Heldout*_Render_Composite_Video
```

`RenderObjective` returns tensors and metadata. `media.py` converts those into
W&B objects. This keeps objective testable.

## Recipes

File: `src/train/recipes/base.py`

```python
@dataclass(frozen=True)
class TrainRecipe:
    name: str
    arch_aliases: tuple[ArchName, ...]
    status: RecipeStatus
    normalize: Callable[[Mapping[str, Any], Path | None], ExperimentSpec]
    build_data_source: Callable[[ExperimentSpec], DataSource]
    build_view_sampler: Callable[[ExperimentSpec], ViewSampler]
    build_feature_provider: Callable[[ExperimentSpec, torch.device], FeatureProvider | None]
    build_model_program: Callable[
        [ExperimentSpec, FeatureDescription | None, torch.device],
        ModelProgram,
    ]
    build_objective: Callable[[ExperimentSpec, torch.device], RenderObjective]
    build_regularizers: Callable[[ExperimentSpec, ModelProgram], tuple[Regularizer, ...]]
    build_optimizer: Callable[
        [ExperimentSpec, Iterable[torch.nn.Parameter]],
        torch.optim.Optimizer,
    ]
    run: Callable[[TrainState], None]
    expected_smokes: tuple[str, ...]
    legacy_entrypoint: str | None = None


def recipe_for_arch(arch: ArchName) -> TrainRecipe: ...


def all_recipes() -> tuple[TrainRecipe, ...]: ...
```

File: `src/train/recipes/video_token.py`

```python
VIDEO_TOKEN_RECIPE: TrainRecipe


def normalize_video_token_config(raw: Mapping[str, Any], path: Path | None) -> ExperimentSpec: ...


def build_video_token_data_source(spec: ExperimentSpec) -> DataSource: ...


def build_video_token_view_sampler(spec: ExperimentSpec) -> ViewSampler: ...


def build_video_token_model_program(
    spec: ExperimentSpec,
    feature_description: FeatureDescription | None,
    device: torch.device,
) -> ModelProgram: ...
```

File: `src/train/recipes/precomputed_feature.py`

```python
PRECOMPUTED_FEATURE_RECIPE: TrainRecipe


def normalize_precomputed_feature_config(raw: Mapping[str, Any], path: Path | None) -> ExperimentSpec: ...


def build_precomputed_feature_provider(
    spec: ExperimentSpec,
    device: torch.device,
) -> FeatureProvider: ...
```

File: `src/train/recipes/multicam_precomputed.py`

```python
MULTICAM_PRECOMPUTED_RECIPE: TrainRecipe


def normalize_multicam_precomputed_config(raw: Mapping[str, Any], path: Path | None) -> ExperimentSpec: ...


def build_multicam_data_source(spec: ExperimentSpec) -> DataSource: ...


def build_multicam_view_sampler(spec: ExperimentSpec) -> ViewSampler: ...


def build_multicam_regularizers(
    spec: ExperimentSpec,
    model: ModelProgram,
) -> tuple[Regularizer, ...]: ...
```

File: `src/train/recipes/known_camera.py`

```python
KNOWN_CAMERA_RECIPE: TrainRecipe


def normalize_known_camera_config(raw: Mapping[str, Any], path: Path | None) -> ExperimentSpec: ...
```

File: `src/train/recipes/external_gauge.py`

```python
@dataclass(frozen=True)
class ExternalRecipe:
    name: str
    arch_aliases: tuple[ArchName, ...]
    explain: Callable[[ExperimentSpec], RoutingReport]
    run_legacy: Callable[[Path], int]


GAUGE_FIELD_RECIPE: TrainRecipe


def explain_gauge_route(spec: ExperimentSpec) -> RoutingReport: ...


def run_gauge_legacy(config_path: Path) -> int: ...
```

Gauge-field internals stay isolated. The central router can point to them; it
does not absorb their representation stack.

## Compatibility Shims

File: `src/train/compat.py`

```python
def run_compat(
    config_path: str | Path,
    *,
    expected_arches: set[str],
    old_entrypoint: str,
) -> None: ...


def warn_deprecated_entrypoint(
    *,
    old_entrypoint: str,
    new_command: str,
    removal_phase: str | None = None,
) -> None: ...


def assert_expected_arch(spec: ExperimentSpec, expected_arches: set[str]) -> None: ...
```

Old files become tiny shims after each recipe is green:

```python
# src/train/train_multicam_precomputed_feature_implicit_dynamic.py
def main() -> None:
    run_compat(
        sys.argv[1],
        expected_arches={"multicam_precomputed_feature_implicit_camera"},
        old_entrypoint=__file__,
    )
```

Do not redirect an old entrypoint to `train.py run` until its recipe smoke is
green.

## Smoke And Provenance

File: `src/train/smoke.py`

```python
@dataclass(frozen=True)
class SmokeSpec:
    name: str
    config_path: Path
    steps: int = 1
    offline: bool = True
    expected_status: Literal["pass", "fail", "blocked"] = "pass"
    expected_media: tuple[str, ...] = ()
    patch_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmokeResult:
    name: str
    config_path: Path
    exit_code: int
    status: Literal["pass", "fail", "blocked"]
    expected_status: Literal["pass", "fail", "blocked"]
    run_id: str | None
    route_report: RoutingReport
    train_loss_logged: bool
    validation_payload_logged: bool
    expected_media_logged: bool
    background_mode_logged: bool
    objective_version_logged: bool
    details: Mapping[str, Any] = field(default_factory=dict)


def make_smoke_config(
    base_config_path: Path,
    *,
    steps: int = 1,
    patch: Mapping[str, Any] | None = None,
) -> Path: ...


def run_smoke(smoke: SmokeSpec) -> SmokeResult: ...


def run_smoke_matrix(smokes: Sequence[SmokeSpec]) -> list[SmokeResult]: ...


def assert_smoke_artifacts(result: SmokeResult) -> None: ...


def write_smoke_report(results: Sequence[SmokeResult], path: Path) -> None: ...
```

Required smoke matrix:

```text
Layer 0: pre-migration freeze
  F=3 single-cam 1-step: pass
  F=32 single-cam alpha/random-bg 1-step: pass
  known-camera 1-step with validation: may fail today, document
  precomputed V-JEPA single-cam 1-step: pass or documented cache issue
  multicam F32 ultimate 1-step: expected fail or blocked
  explain all configs: pass after explain lands

Layer 1: per-phase smokes
  train.py explain across all configs
  F32 background config visibility
  single-cam F=3/F=32 through RenderObjective
  known-camera validation through RenderObjective
  precomputed V-JEPA through FeatureProvider
  multicam RGB/F3 through ViewBatch and objective
  multicam F32 V-JEPA ultimate through ViewBatch, FeatureProvider, objective
  script shims through router
  legacy adapters and gauge external delegates
```

F32 artifact assertions:

```text
feature_dim = 32
colorize_enabled = true
alpha_available = true
no_raw_feature_loss = true
Alpha_Mask_Video logged
Feature_PCA_Video logged
Render_Composite_Video logged
background mode logged
objective version logged
```

Multicam artifact assertions:

```text
train-view metrics logged
held-out metrics logged
held-out render video logged
held-out alpha video logged if F32
condition camera logged
anchor camera logged
held-out camera names logged
```

File: `src/train/provenance.py`

```python
def route_provenance_payload(
    spec: ExperimentSpec,
    recipe: TrainRecipe,
    report: RoutingReport,
) -> dict[str, Any]: ...


def objective_provenance_payload(spec: ExperimentSpec) -> dict[str, Any]: ...


def feature_splatting_provenance_payload(
    spec: ExperimentSpec,
    rendered: RenderedView | None,
) -> dict[str, Any]: ...


def wandb_config_payload(
    spec: ExperimentSpec,
    recipe: TrainRecipe,
    report: RoutingReport,
) -> dict[str, Any]: ...
```

Minimum W&B fields:

```python
{
    "Route/OldEntrypoint": "...",
    "Route/NewEntrypoint": "src/train/train.py",
    "Route/CompatShim": True,
    "Route/Arch": "...",
    "Route/Recipe": "...",
    "Objective/Version": "...",
    "LossBackground/TrainMode": "...",
    "LossBackground/EvalMode": "...",
    "FeatureSplatting/FeatureDim": 32,
    "FeatureSplatting/AlphaAvailable": True,
}
```

## Migration Order

### Phase 0: Freeze Current Behavior

1. Record current commands and W&B run IDs.
2. Preserve the F=32 single-cam random-bg run:

```text
W&B: https://wandb.ai/nbardy/dynaworld/runs/9gr2dm3v
Config: src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc
Entrypoint: src/train/train_video_token_implicit_dynamic.py
Final: Loss 0.0665 / recon 0.0660
Caveat: random background was hardcoded, not config-visible.
```

3. Smoke or explicitly block multicam F32 ultimate.
4. Do not update `BASELINES.md` from smokes.

### Phase 1: Config Visibility And Explain Routing

1. Add `config_schema.py`, `routing.py`, and `train.py explain`.
2. Add `ObjectiveSpec.background`.
3. Migrate intentional F32 configs to explicit random background:

```jsonc
"losses": {
  "background": {
    "train_mode": "random_rgb",
    "eval_mode": "white",
    "sample_scope": "step"
  }
}
```

4. Run `train.py explain` over all configs.

### Phase 2: Shared Objective For Single-Cam And Known-Camera

1. Add `objective/*`.
2. Route single-cam F=3 through `RenderObjective`.
3. Route single-cam F=32 through `RenderObjective`.
4. Fix known-camera initial/final validation by using `RenderObjective`.
5. Smoke F=3, F=32, known-camera.

### Phase 3: FeatureProvider

1. Add `features/*`.
2. Wrap `video_feature_cache.py`.
3. Ensure `FeatureProvider.describe_data()` runs before model construction.
4. Smoke precomputed V-JEPA single-cam.

### Phase 4: ViewBatch And Multicam

1. Add `data/*`.
2. Convert single-cam sampler to `ViewBatch`.
3. Convert multicam sampler to `ViewBatch`.
4. Remove private multicam render/loss.
5. Route all multicam targets through `RenderObjective`.
6. Smoke multicam RGB/F3 first, then multicam F32 V-JEPA ultimate.

### Phase 5: Router Run And Shims

1. Enable `train.py run` for green recipes only.
2. Convert old entrypoints to `run_compat()`.
3. Update scripts to call `train.py run`.
4. Run script smoke suite.

### Phase 6: Deletion

After one shim period and command-reference audits:

```bash
rg "train_camera_implict_dynamic|train_image_encoder_implicit_camera_baseline|train_ltx_feature_implicit_dynamic" .
rg "dynamicTokenGS|tokenGS|train_camera_implicit_dynamic" src train_scripts research_experiments
```

Delete only:

```text
src/train/train_camera_implict_dynamic.py
src/train/train_image_encoder_implicit_camera_baseline.py
src/train/train_ltx_feature_implicit_dynamic.py
```

Do not delete legacy baselines until separate archival decisions exist.

## Code Size Shrinkage Estimate

### Baseline Counts

Measured now:

```text
selected active trainer/data/render/config surface: 7496 lines
all Python under src/train:                         17366 lines
```

The selected active surface includes current trainer entrypoints, key data
helpers, render helpers, feature cache, multicam data, camera rig, runtime
types, and config utils. It is the right denominator for this cleanup because
the low-level renderer and model code mostly stays.

### Temporary Migration Cost

During Phases 1-4, code size will grow because new modules and old entrypoints
coexist.

Expected temporary addition:

```text
config/routing/train.py:          +500 to +800 lines
data/view batch modules:          +700 to +1000 lines
features/provider wrapper:        +250 to +450 lines
model program adapters:           +500 to +800 lines
objective modules:                +900 to +1300 lines
validation/media/smoke/provenance:+500 to +800 lines
recipes/compat:                   +400 to +700 lines
```

Temporary gross addition: about +3750 to +5850 lines.

This is acceptable only while old files are still live. The migration should be
kept phase-gated so this temporary bulk does not become permanent duplication.

### Final Active Surface After Shims

Expected final target for the selected active surface:

```text
config/routing/state/loop:         900 to 1200 lines
data modules:                     1100 to 1500 lines
features provider wrapper:         350 to 550 lines
model programs/factory:            700 to 1000 lines
objective modules:                1000 to 1400 lines
validation/media/smoke/provenance: 700 to 1000 lines
recipes/compat:                    500 to 800 lines
remaining low-level kept files:   1800 to 2300 lines
```

Expected final active surface: about 5050 to 5750 lines.

Shrinkage versus selected 7496-line active surface:

```text
best case:  7496 -> 5050  = -2446 lines, about -33%
middle:     7496 -> 5400  = -2096 lines, about -28%
conservative: 7496 -> 5750 = -1746 lines, about -23%
```

### Full `src/train` Shrinkage

Because `src/train` includes many model, renderer, probe, and legacy files not
deleted in this cleanup, full-tree shrinkage is smaller.

Expected after this cleanup:

```text
17366 total Python lines -> roughly 15100 to 15800 lines
net shrink: about -1600 to -2300 lines
percent shrink: about -9% to -13%
```

If a later separate archival pass retires old image/token/tiled baselines after
replacement adapters are proven, full-tree shrinkage could reach:

```text
additional -800 to -1400 lines
total full-tree shrink: about -14% to -21%
```

Do not count that later archival shrinkage as part of this cleanup unless those
baselines are explicitly retired and smoke/archival notes are written.

## Final Implementation Checklist

Before writing code:

- [ ] Add explicit route guards for F32 multicam until shared objective is wired.
- [ ] Make random background config-visible.
- [ ] Freeze current W&B baseline provenance.
- [ ] Decide exact `objective.version` strings.
- [ ] Decide which F32 configs inherit random background.

Before enabling `train.py run`:

- [ ] `train.py explain` works across all configs.
- [ ] F=3 single-cam smoke passes.
- [ ] F=32 single-cam smoke passes and logs alpha/PCA/composite.
- [ ] Known-camera validation smoke passes.
- [ ] Precomputed V-JEPA smoke passes.
- [ ] Multicam RGB/F3 smoke passes.
- [ ] Multicam F32 ultimate smoke passes or stays blocked loudly.

Before deleting files:

- [ ] Old entrypoints have been shims for at least one compatibility commit.
- [ ] `rg` command-reference audit is clean.
- [ ] Script smoke suite passes.
- [ ] `BASELINES.md` has append-only rows only for intentional benchmark runs.
- [ ] Smoke results live in migration notes, not standings.

## Bottom Line

The cleanup should be judged by one property:

```text
No trainer route can compute RGB reconstruction loss without passing through
RenderedView from RenderObjective.
```

If a patch still lets single-cam, known-camera, precomputed-feature, or multicam
construct RGB and loss privately, the architecture has not actually been
cleaned up.
