# Review 1: Interface Consistency And Implementability

Reviewer: interface consistency / implementability
Date: 2026-04-30
Scope:

- `proposal_a_shared_objective_pipeline.md`
- `proposal_b_viewbatch_recipe_registry.md`
- `proposal_c_cleanup_deletion_migration_plan.md`

This review only checks the proposed public API surface. It does not judge
implementation effort except where a signature is too abstract, contradictory,
or likely to recreate the current inheritance drift.

## Executive Verdict

The three proposals are directionally aligned:

```text
raw JSONC
  -> normalized ExperimentSpec
  -> TrainRecipe
  -> DataSource + ViewSampler
  -> ViewBatch
  -> ModelProgram
  -> GaussianSequence
  -> RenderObjective
  -> rendered views + reconstruction loss
  -> TrainLoop / validation artifacts
```

That is the right spine. It removes the exact bug class we just hit: a subclass
continued to treat `render_clip_sequence` as a tensor-returning helper after the
base path changed it into `(features, alpha)`.

But the proposals are not yet one coherent interface. The largest contradictions
are:

1. `RenderedView` vs `RenderedClip` vs `RenderOutput`.
2. `ObjectiveLoss` vs `LossBundle` vs `StepResult.losses`.
3. `TargetView.cameras` required in proposal A, optional in B/C.
4. `TargetRole` / `ViewRole` / `TrainPhase` / `ObjectivePhase` literals disagree.
5. `BackgroundSpec` has three different shapes.
6. `FeatureProvider` and `ModelProgram` signatures disagree enough that a direct
   implementation would need adapters immediately.
7. Proposal C's `TrainComponents` drops `ViewSampler`, which means it cannot
   actually replace the current sampling subclasses without hidden state.
8. Proposal A lets `RenderObjective.validation_payload()` build media, while C
   moves media into `validation.py`. Keep the latter; objective should not own
   W&B artifact construction.

Recommended resolution: preserve proposal B's `ExperimentSpec`, `ViewBatch`,
`TargetView`, `FeatureProvider`, `ModelProgram`, and `TrainRecipe` as the
router/data/model spine; preserve proposal A's richer `RenderObjective`,
`RasterizedView`, `ColorizedView`, `BackgroundSample`, and `RenderedView` as the
render/loss spine; preserve proposal C's compatibility, shim, smoke, and
baseline-provenance machinery. Do not preserve C's simplified `RenderedClip`
or mapping-heavy `ExperimentSpec` as the final active API.

## Blocking Interface Mismatches

### 1. Target camera ownership is under-specified in A and C

Proposal A:

```python
class TargetView:
    cameras: tuple["CameraSpec", ...]      # length T
```

Proposal B:

```python
class TargetView:
    cameras: tuple[CameraSpec, ...] | None
    camera_owner: CameraOwner
```

Proposal C:

```python
class TargetView:
    cameras: tuple[CameraSpec, ...] | None
```

The proposal B version is the only one that is implementable without
reintroducing path-specific branches. A model-predicted implicit-camera target
must be able to say `camera_owner == "model"` and `cameras is None`. A known
camera or multicam rig target must say `camera_owner in {"batch",
"external_rig"}` and provide cameras.

Keep both:

- `camera_role`: semantic role in the dataset/rig, e.g. condition, anchor, train
  target, heldout target.
- `camera_owner`: source of the render cameras, e.g. model, batch, external rig.

Without `camera_owner`, `RenderObjective.rasterize_view()` has to infer camera
source from role names or trainer class, which is exactly the inheritance leak
we are trying to remove.

### 2. Phase and role literals disagree

Proposal A:

```python
ObjectivePhase = Literal["train", "eval", "preview"]
TargetRole = Literal["train", "heldout", "source", "debug"]
```

Proposal B:

```python
ViewRole = Literal["source", "train", "heldout", "eval", "debug"]
TrainPhase = Literal["train", "eval", "initial", "export"]
```

Proposal C:

```python
phase: Literal["train", "eval"]
TargetView.role: Literal["train", "eval", "heldout", "source"]
```

Use one role type and one phase type:

```python
ViewRole = Literal["source", "train", "heldout", "debug"]
RunPhase = Literal["train", "eval", "preview", "export"]
```

`eval` should be a phase, not a view role. A held-out camera is still held-out
whether rendered during validation, export, or a debug preview. `initial` should
not be a phase unless it changes objective behavior; it is an eval/preview event
with step metadata.

### 3. BackgroundSpec must not have both `mode` and train/eval mode

Proposal A:

```python
class BackgroundSpec:
    train_mode: BackgroundMode = "white"
    eval_mode: BackgroundMode = "white"
    fixed_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_scope: BackgroundSampleScope = "step"
    apply_when_alpha_missing: bool = False
```

Proposal B:

```python
class BackgroundSpec:
    train_mode: BackgroundMode
    eval_mode: BackgroundMode
    fixed_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    random_per: Literal["step", "view", "frame", "pixel"] = "step"
```

Proposal C:

```python
class BackgroundSpec:
    mode: Literal["white", "black", "random_rgb", "fixed_rgb"]
    rgb: tuple[float, float, float] | None = None
    train_mode: Literal["white", "black", "random_rgb", "fixed_rgb"] | None = None
    eval_mode: Literal["white", "black", "fixed_rgb"] = "white"
    sample_scope: Literal["step", "view", "frame"] = "step"
```

Proposal C's `mode` plus `train_mode` is ambiguous. It invites bugs like:

```text
mode=random_rgb, train_mode=None, eval_mode=white
```

where the implementation has to guess whether `mode` is a default or a concrete
train setting.

Recommended:

```python
BackgroundMode = Literal["white", "black", "fixed_rgb", "random_rgb", "none"]
BackgroundSampleScope = Literal["step", "view", "frame"]

@dataclass(frozen=True)
class BackgroundSpec:
    train_mode: BackgroundMode
    eval_mode: BackgroundMode
    fixed_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_scope: BackgroundSampleScope = "step"
    apply_when_alpha_missing: bool = False
```

Initial implementation should accept only `sample_scope == "step"` for
`random_rgb`. Keep the enum value for future work, but validate unsupported
values loudly.

### 4. `RenderedView` should win over `RenderedClip` / `RenderOutput`

Proposal A's `RenderedView` preserves the most information:

```python
class RenderedView:
    view_id: str
    role: TargetRole
    rgb: Tensor
    target_rgb: Tensor | None
    rasterized: RasterizedView
    colorized: ColorizedView | None
    background: BackgroundSample
    phase: ObjectivePhase
```

Proposal C's `RenderedClip` flattens too much:

```python
class RenderedClip:
    rgb: torch.Tensor
    features: torch.Tensor
    alpha: torch.Tensor | None
    splat_rgb: torch.Tensor | None
    background_rgb: torch.Tensor | None
    view: TargetView
```

Flattening is tempting, but it loses:

- whether `rgb` came from F=3 direct raster or F>3 colorization,
- the exact background policy metadata,
- logits/view-direction diagnostics,
- rasterizer feature dimension,
- phase-specific behavior.

Use `RenderedView`. If memory becomes a problem, add a `retain_artifacts` or
`keep_preview` switch to drop `target_rgb`, `colorized.logits`, or PCA inputs
after loss. Do not collapse the type before the interface is stable.

### 5. Objective should not own regularizers or W&B video construction

Proposal A puts `validation_payload()` on `RenderObjective`. Proposal C moves
media to a validation module. Proposal C is cleaner.

`RenderObjective` should own:

- rasterization,
- F=3 vs F>3 colorization,
- alpha-aware composition,
- background policy,
- RGB reconstruction loss and per-view aggregation.

It should not own:

- camera rig regularization,
- bank-rate loss,
- optimizer,
- W&B run state,
- video encoding objects,
- export bundle generation.

The objective may return rich `RenderedView` objects that validation code turns
into `Alpha_Mask_Video`, `Feature_PCA_Video`, and `Render_Composite_Video`.

### 6. `TrainRecipe` should include both data source and sampler

Proposal B:

```python
build_data_source: Callable[[ExperimentSpec], DataSource]
build_view_sampler: Callable[[ExperimentSpec], ViewSampler]
```

Proposal C:

```python
build_data: Callable[[ExperimentSpec, torch.device], "DataSource"]
build_loop: Callable[[ExperimentSpec, "TrainComponents"], "TrainLoop"]
```

Proposal C's `TrainComponents` has `data_source` but no sampler. That cannot
replace:

- single-cam contiguous clip sampling,
- known-camera clip sampling,
- multicam train view selection,
- held-out view validation selection,
- future scene-distinct samplers.

Keep B's split: `DataSource.load()` loads stable data; `ViewSampler.sample()`
chooses the current train/eval batch.

### 7. `FeatureProvider` should be bundle-aware

Proposal B:

```python
class FeatureProvider(Protocol):
    def prebake(self, bundle: DataBundle) -> None: ...
    def describe(self, bundle: DataBundle) -> FeatureDescription: ...
    def features_for(self, conditioning: ConditioningInput) -> Mapping[str, Tensor]: ...
```

Proposal C:

```python
class FeatureProvider(Protocol):
    def prebake(self, sequences: Sequence[SequenceData]) -> None: ...
    def describe(self, sequence: SequenceData) -> FeatureDescription: ...
    def load(self, conditioning: ConditioningInput) -> Mapping[str, torch.Tensor] | torch.Tensor: ...
```

Keep B. Multicam feature cache keys need bundle-level facts: condition camera,
anchor camera, sample id, selected frame ids, source path, and feature extractor
layout. A single `SequenceData` is not enough.

Also keep `Mapping[str, Tensor]` as the provider output. If a provider only has
one tensor, wrap it under a stable key like `"features"` or `"vjepa_tokens"`.
Allowing either `Tensor` or `Mapping` pushes shape dispatch into model adapters.

### 8. `ModelProgram` should have explicit input construction

Proposal B has the best shape:

```python
class ModelProgram(Protocol):
    @property
    def feature_dim(self) -> int: ...

    @property
    def camera_owner(self) -> CameraOwner: ...

    def make_input(self, batch: ViewBatch, provider: FeatureProvider | None) -> ModelInput: ...
    def decode(self, model_input: ModelInput) -> ModelOutput: ...
    def parameters(self) -> Iterable[torch.nn.Parameter]: ...
```

Proposal C's example snippet says:

```python
model_output = model_program.decode(batch)
```

That is a leak. The whole point of `ModelProgram` is to adapt `ViewBatch` into
the weird signatures current models already have:

- `(video, decode_times, input_times=None)`,
- `(video, decode_times, cameras, input_times=None)`,
- precomputed feature mappings,
- unconditioned `None`/dummy video paths.

Keep `make_input()` as the only place where adapter-specific tensor orientation
is allowed.

### 9. Tensor time shapes must be canonical at the new boundary

The docs mix `[T, 1]`, `[K, 1]`, and `[1, K]`. Current old models may accept
different orientations, but the new types should not.

Recommended invariant:

```text
ConditioningInput.frame_times: [K, 1]
TargetView.frame_times:       [K, 1]
ViewBatch.decode_times:       [K, 1]
ViewBatch.frame_indices:      [K]
```

Legacy adapters may transpose inside `ModelProgram.make_input()`. Do not allow
both shapes in `ViewBatch`; otherwise every downstream helper needs defensive
branches.

## Recommended Coherent Interface Set

This is the strongest interface set to preserve across the three docs.

### Core literals

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
```

Use `ArchName = str` in code unless there is a generated registry-derived
literal. A hand-written `Literal[...]` for 15 arch names will drift.

### Normalized config

Use proposal B's typed spec plus proposal C's compatibility fields:

```python
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

`ExperimentSpec` should not use C's raw `Mapping[str, Any]` for every hot
section. The whole cleanup target is to stop scattering `cfg.get(...)` through
warm paths. Typed sections are worth the up-front normalization cost.

Raw config compatibility:

```text
existing top-level losses        -> spec.objective.reconstruction
existing top-level colorize      -> spec.objective.colorize
existing losses.background       -> spec.objective.background
render.fast_mac.feature_background stays renderer-specific
```

Do not merge rasterizer background and RGB loss-composition background.

### View data

```python
@dataclass(frozen=True)
class ConditioningInput:
    sample_id: str
    scene_id: str | None
    kind: ConditionKind
    frames: Tensor | None                         # [1, K, 3, H, W] or None
    features: Mapping[str, Tensor] | None
    frame_indices: Tensor                         # [K]
    frame_times: Tensor                           # [K, 1]
    video_fps: float
    camera_name: str | None
    cameras: tuple[CameraSpec, ...] | None        # len K or None
    source_path: Path | None
    feature_cache_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetView:
    view_id: str
    role: ViewRole
    camera_role: CameraRole
    camera_owner: CameraOwner
    frames: Tensor                                # [K, 3, H_in, W_in]
    frame_indices: Tensor                         # [K]
    frame_times: Tensor                           # [K, 1]
    video_fps: float
    camera_name: str | None
    cameras: tuple[CameraSpec, ...] | None        # len K unless camera_owner == "model"
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
    decode_times: Tensor                          # [K, 1]
    frame_indices: Tensor                         # [K]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int: ...

    @property
    def target_count(self) -> int: ...
```

Required validator:

```python
def validate_view_batch(batch: ViewBatch) -> None: ...
```

Minimum checks:

```text
batch.decode_times.shape == [K, 1]
batch.frame_indices.shape == [K]
target.frame_indices == batch.frame_indices for normal same-time supervision
target.frame_times.shape == [K, 1]
target.frames.shape == [K, 3, H, W]
target.camera_owner == "model" -> target.cameras is None
target.camera_owner in {"batch", "external_rig"} -> len(target.cameras) == K
heldout targets never appear in phase="train" unless explicitly allowed
```

### Data source and sampler

```python
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
class ViewRecord:
    view_id: str
    role: ViewRole
    camera_role: CameraRole
    camera_name: str | None
    frames: Tensor
    frame_times: Tensor
    video_fps: float
    cameras: tuple[CameraSpec, ...] | None
    source_path: Path | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DataSource(Protocol):
    def load(self, spec: ExperimentSpec, *, device: torch.device) -> DataBundle: ...


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
```

### Feature provider

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
    def prebake(self, bundle: DataBundle) -> None: ...
    def describe(self, bundle: DataBundle) -> FeatureDescription: ...
    def features_for(self, conditioning: ConditioningInput) -> Mapping[str, Tensor]: ...
    def release(self) -> None: ...
```

### Model program

```python
@dataclass(frozen=True)
class ModelInput:
    condition_frames: Tensor | None
    condition_features: Mapping[str, Tensor] | None
    input_times: Tensor | None
    decode_times: Tensor
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
```

Do not let `ModelProgram.decode()` accept `ViewBatch` directly. That hides the
adapter boundary.

### Render objective

Use proposal A's object model, but move media construction out:

```python
@dataclass(frozen=True)
class RenderSpec:
    renderer: str
    input_size: int
    render_size: int
    tile_size: int
    bound_scale: float
    alpha_threshold: float
    near_plane: float
    camera_projection: str | None
    fast_mac: Mapping[str, Any]
    dense_grid: Tensor | None = None


@dataclass(frozen=True)
class BackgroundSpec:
    train_mode: BackgroundMode
    eval_mode: BackgroundMode
    fixed_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_scope: BackgroundSampleScope = "step"
    apply_when_alpha_missing: bool = False


@dataclass(frozen=True)
class RasterizedView:
    view: TargetView
    features: Tensor                            # [K, F, H, W]
    alpha: Tensor | None                        # [K, H, W]
    cameras: tuple[CameraSpec, ...]
    feature_dim: int
    render_size: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ColorizedView:
    splat_rgb: Tensor                           # [K, 3, H, W]
    logits: Tensor | None = None
    view_dirs: Tensor | None = None


@dataclass(frozen=True)
class BackgroundSample:
    rgb: Tensor | None                          # None means no post-raster compose
    mode: BackgroundMode
    phase: RunPhase
    scope: BackgroundSampleScope


@dataclass(frozen=True)
class RenderedView:
    view: TargetView
    rgb: Tensor                                 # [K, 3, H, W]
    target_rgb: Tensor | None                   # [K, 3, H, W]
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
    total: Tensor
    per_image: Tensor                           # [K]
    weight: float
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveLoss:
    total: Tensor
    reconstruction: Tensor
    view_losses: tuple[ViewLoss, ...]
    rendered_views: tuple[RenderedView, ...]
```

Protocols:

```python
class RasterizerProtocol(Protocol):
    def rasterize(
        self,
        decoded: GaussianSequence,
        cameras: tuple[CameraSpec, ...],
        *,
        render_spec: RenderSpec,
    ) -> tuple[Tensor, Tensor | None]: ...


class ColorizerProtocol(Protocol):
    feature_dim: int

    def forward(self, features: Tensor, view_dirs: Tensor | None = None) -> Tensor: ...


class BackgroundPolicyProtocol(Protocol):
    def sample(
        self,
        *,
        phase: RunPhase,
        like: Tensor,
        view_count: int,
        frame_count: int,
        generator: torch.Generator | None = None,
    ) -> BackgroundSample: ...
```

Objective:

```python
@dataclass
class RenderObjective:
    render_spec: RenderSpec
    loss_cfg: Mapping[str, Any]
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
        target_rgb: Tensor | None,
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
        keep_preview: bool = False,
    ) -> ObjectiveLoss: ...
```

Key implementation rule:

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

This is the line that prevents multicam from drifting back into a private render
path.

### Validation and media

Keep validation outside objective:

```python
@dataclass(frozen=True)
class ValidationRender:
    view: TargetView
    rendered: RenderedView
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ValidationPayload:
    scalars: Mapping[str, float]
    videos: Mapping[str, Any]
    images: Mapping[str, Any]
    decoded_metrics: Mapping[str, float]


def render_validation_views(
    state: TrainState,
    *,
    phase: Literal["eval", "preview"] = "eval",
) -> tuple[ValidationRender, ...]: ...


def build_validation_payload(
    renders: Sequence[ValidationRender],
    *,
    log_gt: bool,
    video_fps: float,
    include_alpha: bool,
    include_feature_pca: bool,
    include_composite: bool,
) -> ValidationPayload: ...
```

### Train recipe and loop

Keep proposal B's recipe, but add proposal C's status/compat fields:

```python
@dataclass(frozen=True)
class TrainRecipe:
    name: str
    arch_aliases: tuple[ArchName, ...]
    status: Literal["active", "compat", "legacy", "external"]
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
    build_optimizer: Callable[[ExperimentSpec, Iterable[torch.nn.Parameter]], torch.optim.Optimizer]
    run: Callable[[TrainState], None]
    legacy_entrypoint: str | None = None
```

If `**validated_kwargs` are needed for model construction, put them behind
concrete factory functions, not in the registry callable type.

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
    objective: RenderObjective
    regularizers: tuple[Regularizer, ...]
    optimizer: torch.optim.Optimizer
    step: int = 0


@dataclass(frozen=True)
class RegularizerLoss:
    total: Tensor
    terms: Mapping[str, Tensor]


class Regularizer(Protocol):
    def loss(
        self,
        model_output: ModelOutput,
        batch: ViewBatch,
        *,
        step: int,
    ) -> RegularizerLoss: ...


@dataclass(frozen=True)
class StepResult:
    step: int
    batch: ViewBatch
    model_output: ModelOutput
    objective: ObjectiveLoss
    regularizers: RegularizerLoss
    total_loss: Tensor
    previews: Mapping[str, RenderedView]
    metrics: Mapping[str, float]
```

Training step:

```python
def train_step(
    state: TrainState,
    *,
    keep_preview: bool,
    generator: torch.Generator | None = None,
) -> StepResult:
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

    objective = state.objective.loss_for_batch(
        model_output.sequence,
        batch,
        phase="train",
        generator=generator,
        keep_preview=keep_preview,
    )
    regularizers = evaluate_regularizers(
        state.regularizers,
        model_output,
        batch,
        step=state.step,
    )
    total = objective.total + regularizers.total
    return StepResult(...)
```

Note the property name: proposal B's pseudocode uses `objective_result.loss`,
but its dataclass defines `ObjectiveLoss.total`. Use `total`.

## Specific Proposal Edits I Would Request

These are not edits I made; they are the reviewer's requested reconciliations.

### Proposal A

Keep:

- the central `RenderObjective` thesis,
- `RasterizedView`, `ColorizedView`, `BackgroundSample`, `RenderedView`,
- explicit shape checks before reconstruction loss,
- step-scoped random background as a policy,
- held-out multicam as a first-class target role.

Change before implementation:

- Make `TargetView.cameras` optional and add `camera_owner`.
- Replace `ObjectivePhase` with shared `RunPhase`.
- Remove `RenderObjective.validation_payload()` or mark it as a separate
  validation helper.
- Use `renderer` not `mode` in `RenderSpec` to match existing config vocabulary.
- Document the camera selection helper; it is the critical no-drift boundary.

### Proposal B

Keep:

- typed `ExperimentSpec`,
- `ViewsSpec`,
- `DataBundle` + `ViewSampler`,
- `FeatureProvider.describe(bundle)`,
- `ModelProgram.make_input()` / `decode()` split,
- `TrainRecipe` registry,
- compatibility shims as thin wrappers.

Change before implementation:

- Remove `ViewRole == "eval"`; use phase for eval.
- Normalize `TrainPhase` into `RunPhase`.
- Decide whether `ConditioningInput.features` can ever be a raw tensor; prefer
  no.
- Fix pseudocode field name `objective_result.loss` -> `objective_result.total`.
- Make `TrainRecipe` callable signatures match the concrete factory signatures;
  avoid mixing `Callable[[...], X]` with keyword-only `device` examples.

### Proposal C

Keep:

- inventory/deletion discipline,
- compatibility/deprecation records,
- smoke harness,
- baseline and W&B provenance plan,
- route explain/audit commands,
- legacy adapter plan for `dynamicTokenGS.py`, `tokenGS.py`, and gauge-field
  experiments.

Change before implementation:

- Do not use mapping-heavy `ExperimentSpec` for active recipes.
- Do not use `RenderedClip` as the active render result type.
- Add `ViewSampler` to `TrainComponents`.
- Remove `mode` from `BackgroundSpec`; keep only train/eval modes.
- Keep regularizers outside `RenderObjective`.
- Do not let `ModelProgram.decode()` accept `ViewBatch` directly in examples.

## Implementability Risks That Remain

### Chunked backward and step-scoped background

The F32 path currently chunks reconstruction over time. The API must allow a
single `BackgroundSample` to be reused across chunks within a step. Proposal A
handles this with an optional `background` argument. Keep it.

Do not sample random background inside each chunk unless the config explicitly
sets `sample_scope="chunk"` someday. That would make the objective differ from
the documented 3DGS-style per-step random background.

### Multi-view loss scaling

All proposals mention view weights, but the exact aggregation must be locked:

```python
weighted_mean = sum(view_loss.total * target.loss_weight) / sum(target.loss_weight)
```

Do not divide by selected view count in one path and by weight sum in another.
That will make multicam comparisons unreadable.

### Validation memory

`RenderedView` is intentionally rich. For train steps with many target views,
holding every feature map/logit/target can be too much. Solve that with
`keep_preview` / `retain_artifacts` flags, not by making the public type too
thin.

### Literal arch names will drift

The `ArchName = Literal[...]` block in proposal B is useful as documentation but
will become stale. Runtime code should derive route coverage from
`ARCH_REGISTRY` and have a test:

```python
def test_arch_registry_covers_all_train_configs() -> None: ...
```

The type alias can be `ArchName = str` unless generated.

### Gauge-field stack should remain external initially

All proposals agree enough here: do not force gauge-field configs through the
new objective until there is a real held-out-camera parity contract. Keep it as
an external/legacy recipe in the router.

## Highest-Value First Implementation Slice

If the next agent implements only one slice, make it this:

1. Add `view_batch.py` with `ConditioningInput`, `TargetView`, `ViewBatch`, and
   `validate_view_batch()`.
2. Add `objective.py` with `BackgroundSpec`, `BackgroundSample`,
   `RasterizedView`, `ColorizedView`, `RenderedView`, `ObjectiveLoss`, and
   `RenderObjective`.
3. Wrap existing `render_clip_sequence()` in a `RasterizerProtocol`
   implementation.
4. Convert only the current single-cam F32 path to call `RenderObjective`, with
   no behavior change except moving random background into config-visible
   `BackgroundSpec`.
5. Convert known-camera initial validation through the same objective.
6. Convert multicam render/loss through `ViewBatch + RenderObjective`.

Do not start with the full router. The router is important, but it does not fix
the broken F32 multicam path unless the shared objective exists first.

## Final Recommendation

Preserve one invariant above every naming preference:

```text
Every route that computes RGB reconstruction loss must pass through:

ViewBatch.targets
  -> RenderObjective.render_view_batch()
  -> RenderedView.rgb
  -> reconstruction loss
```

The winning interface is:

- proposal B for normalized config, view/data/model/recipe routing,
- proposal A for render/objective dataclasses and alpha-aware composition,
- proposal C for compatibility, deletion, smoke, and provenance infrastructure.

Any final design that keeps subclass-owned render/loss methods, lets target
cameras be inferred from trainer type, or lets F32 feature maps reach RGB loss
without a `RenderedView` object has not actually cleaned up the bug class.
