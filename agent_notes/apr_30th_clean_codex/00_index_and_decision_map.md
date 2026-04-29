# April 30 Cleanup Codex: Index And Decision Map

Date: 2026-04-30
Scope: trainer/model/config cleanup redesign for Dynaworld
Status: documentation only; no code, config, or `BASELINES.md` changes in this pass

This folder is the deep design pass requested after the F=32 feature-splatting
and random-background fix exposed the larger trainer architecture problem. The
core bug was not only one stale tuple unpack. The deeper problem is that
training routes are selected by Python entrypoint and inheritance overrides,
while the render/loss objective lives in one subclass path. Multicam and
known-camera code can bypass the fixed feature-splatting path.

The goal of this folder is to turn that lesson into implementation-ready
interfaces before the next cleanup patch starts.

## Files

Read these in order:

0. `final_design_full_cleanup.md`
   - Final consolidated implementation design.
   - Includes target file tree, public dataclasses/protocols, function
     signatures, migration order, smoke gates, provenance fields, and code size
     shrinkage estimate.

1. `proposal_a_shared_objective_pipeline.md`
   - Author: Proposal Writer A
   - Focus: shared `RenderObjective` / `render_view_batch`
   - Key invariant: only final RGB may reach reconstruction loss; every route
     must use the same F=3, F>3, colorize, alpha-compose, and background policy.

2. `proposal_b_viewbatch_recipe_registry.md`
   - Author: Proposal Writer B
   - Focus: `ExperimentSpec`, `ViewBatch`, `TargetView`, `FeatureProvider`,
     `ModelProgram`, `TrainRecipe`, and central arch routing.
   - Key invariant: single-cam, known-camera, precomputed-feature, and multicam
     training should all be "one conditioning input, one or more target views."

3. `proposal_c_cleanup_deletion_migration_plan.md`
   - Author: Proposal Writer C
   - Focus: cleanup sequence, deletion discipline, shims, smoke matrix, W&B and
     `BASELINES.md` provenance.
   - Key invariant: deletion is the reward for proven routing, not the tool that
     forces cleanup.

4. `review_1_interface_consistency.md`
   - Reviewer: interface consistency and implementability.
   - Key result: preserve proposal B's config/data/model spine, proposal A's
     rich render/objective spine, and proposal C's compatibility/provenance
     machinery.

5. `review_2_migration_risk.md`
   - Reviewer: migration risk, smoke coverage, deletion safety, and provenance.
   - Key result: add route guards, make random background config-visible, freeze
     baseline provenance, extract `FeatureProvider` before the multicam V-JEPA
     ultimate run, and keep legacy entrypoints as shims until smokes pass.

## Immediate Answers Captured By The Docs

### Multicam vs single-cam

Yes, the repo has a separate multicam trainer path:

```text
single-cam main:
  src/train/train_video_token_implicit_dynamic.py

single-cam precomputed features:
  src/train/train_precomputed_feature_implicit_dynamic.py

multicam precomputed features:
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py
```

The multicam path is the one we want for the bigger V-JEPA 256x256, multiple
training angles, held-out novel-view baseline. It uses DeepView-style train
cameras plus held-out cameras. But it currently bypasses the fixed shared F=32
feature-splatting path, so the ultimate config must not be treated as ready
until multicam is routed through the shared objective.

### Old small baseline with random background

The single-cam F=32 random-background run did complete:

```text
W&B: https://wandb.ai/nbardy/dynaworld/runs/9gr2dm3v
Config: src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc
Entrypoint: src/train/train_video_token_implicit_dynamic.py
Final: Loss 0.0665 / recon 0.0660
Logged: Alpha_Mask_Video, Feature_PCA_Video, Render_Composite_Video
```

Caveat: random background was hardcoded in the trainer at that moment, not a
config-visible `losses.background` field. That run is real evidence, but W&B
config alone cannot prove the background policy. The cleanup must make this
explicit before any new baseline claim.

## The Four Irreducible Runtime Abstractions

The reviewers converge on four core runtime abstractions. Supporting protocols
like `FeatureProvider` and `TrainRecipe` matter, but these four are the shape of
the cleaned trainer.

### 1. `ExperimentSpec`

Normalized config. It replaces scattered `.get(..., default)` calls and
file-based trainer routing.

```python
@dataclass(frozen=True)
class ExperimentSpec:
    config_path: Path | None
    arch: str
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

Rules:

- Normalize old config keys once.
- Validate required keys once.
- Keep hot paths on typed sections, not raw dict spelunking.
- Keep renderer background separate from RGB loss-composition background.

### 2. `ViewBatch`

The common batch contract. Single-cam is one target view. Multicam is multiple
train target views plus held-out target views. Precomputed V-JEPA/LTX/Wan is a
conditioning payload variant, not a new trainer class.

```python
ViewRole = Literal["source", "train", "heldout", "debug"]
CameraRole = Literal[
    "condition",
    "anchor",
    "train_target",
    "heldout_target",
    "model_predicted",
]
CameraOwner = Literal["model", "batch", "external_rig", "none"]
RunPhase = Literal["train", "eval", "preview", "export"]

@dataclass(frozen=True)
class ConditioningInput:
    sample_id: str
    scene_id: str | None
    kind: Literal["rgb_video", "precomputed_features", "none"]
    frames: torch.Tensor | None                  # [1, K, 3, H, W] or None
    features: Mapping[str, torch.Tensor] | None
    frame_indices: torch.Tensor                  # [K]
    frame_times: torch.Tensor                    # [K, 1]
    video_fps: float
    camera_name: str | None
    cameras: tuple[CameraSpec, ...] | None       # len K or None
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
    cameras: tuple[CameraSpec, ...] | None       # len K unless owner == "model"
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

Rules:

- `eval` is a phase, not a view role.
- Keep `camera_role` and `camera_owner` separate.
- For multicam, target cameras from the external rig must control rendering;
  model-predicted cameras are diagnostics unless the target declares
  `camera_owner == "model"`.
- Canonical time shape at the boundary is `[K, 1]`; legacy adapters transpose
  inside `ModelProgram.make_input()`.

### 3. `ModelProgram`

The adapter from clean batch data into the awkward signatures of existing
models. The model still returns `GaussianSequence`; the trainer no longer cares
whether the conditioning was RGB video, precomputed features, or none.

```python
@dataclass(frozen=True)
class ModelInput:
    condition: torch.Tensor | Mapping[str, torch.Tensor] | None
    input_times: torch.Tensor | None             # [K, 1] or adapter-specific
    decode_times: torch.Tensor                   # [K, 1] or adapter-specific
    render_cameras: tuple[CameraSpec, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ModelOutput:
    sequence: GaussianSequence
    camera_owner: CameraOwner
    diagnostics: Mapping[str, Any]

class ModelProgram(Protocol):
    @property
    def feature_dim(self) -> int: ...

    @property
    def camera_owner(self) -> CameraOwner: ...

    def make_input(
        self,
        batch: ViewBatch,
        provider: "FeatureProvider | None",
    ) -> ModelInput: ...

    def decode(self, model_input: ModelInput) -> ModelOutput: ...

    def parameters(self) -> Iterable[torch.nn.Parameter]: ...
```

Rules:

- Do not let `decode()` accept `ViewBatch` directly.
- `make_input()` is where legacy tensor orientation and model-specific argument
  shapes live.
- `GaussianSequence.rgbs` should not be renamed during this cleanup even though
  it semantically holds F-channel splat features.
- Precomputed feature description must happen before model construction.

### 4. `RenderObjective`

The only place where rasterized `(features, alpha)` becomes final RGB and then
loss. This is the boundary that would have prevented the multicam bug.

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

@dataclass(frozen=True)
class RasterizedView:
    view_id: str
    role: ViewRole
    features: torch.Tensor                       # [K, F, H, W]
    alpha: torch.Tensor | None                   # [K, H, W]
    cameras: tuple[CameraSpec, ...] | None
    feature_dim: int
    renderer_mode: str

@dataclass(frozen=True)
class ColorizedView:
    rgb: torch.Tensor                            # [K, 3, H, W]
    logits: torch.Tensor | None
    view_condition: str

@dataclass(frozen=True)
class BackgroundSample:
    rgb: torch.Tensor | None                     # [1|K, 3, 1, 1]
    mode: BackgroundMode
    sample_scope: BackgroundSampleScope
    phase: RunPhase
    seed: int | None = None
    step: int | None = None

@dataclass(frozen=True)
class RenderedView:
    view: TargetView
    phase: RunPhase
    rgb: torch.Tensor                            # [K, 3, H, W]
    target_rgb: torch.Tensor | None              # [K, 3, H, W]
    rasterized: RasterizedView
    colorized: ColorizedView | None
    background: BackgroundSample

@dataclass(frozen=True)
class ObjectiveLoss:
    total: torch.Tensor
    recon: torch.Tensor
    per_view: Mapping[str, torch.Tensor]
    metrics: Mapping[str, float]
    rendered_views: tuple[RenderedView, ...]

class RenderObjective:
    def render_view(
        self,
        decoded: GaussianSequence,
        target: TargetView,
        *,
        phase: RunPhase,
        background: BackgroundSample | None = None,
    ) -> RenderedView: ...

    def render_batch(
        self,
        decoded: GaussianSequence,
        batch: ViewBatch,
        *,
        phase: RunPhase,
    ) -> tuple[RenderedView, ...]: ...

    def loss(
        self,
        decoded: GaussianSequence,
        batch: ViewBatch,
        *,
        phase: RunPhase,
    ) -> ObjectiveLoss: ...
```

Rules:

- F=3 legacy RGB can skip colorize and alpha composition unless alpha becomes
  available for that path.
- F>3 requires colorize before RGB loss.
- If F>3 and alpha exists, compose with `alpha * splat_rgb + (1-alpha) * bg`.
- Random background is sampled once per step and shared across chunks unless the
  config explicitly asks for a narrower scope.
- Objective returns tensors and metadata; validation/logging converts them to
  W&B media.
- Objective does not own bank-rate loss, rig regularization, optimizer state,
  feature cache lifecycle, or export bundle generation.

## Supporting Protocols

### `FeatureProvider`

The reviewers recommend splitting explain-safe metadata from cache-warming data
description.

```python
class FeatureProvider(Protocol):
    def describe_config(
        self,
        spec: FeatureProviderSpec,
    ) -> FeatureDescription | None: ...

    def describe_data(
        self,
        bundle: DataBundle,
    ) -> FeatureDescription: ...

    def load(
        self,
        conditioning: ConditioningInput,
    ) -> Mapping[str, torch.Tensor]: ...
```

Minimum cache key fields:

```text
extractor_name
extractor_version_or_model_id
feature_layer_names
source_path or content hash
native frame ids
frame_times if used
condition camera id/name
input resize/crop policy
feature dtype/layout
```

### `TrainRecipe`

```python
@dataclass(frozen=True)
class TrainRecipe:
    name: str
    status: Literal["green", "blocked", "legacy", "external"]
    accepted_arches: frozenset[str]
    normalize: Callable[[dict[str, Any], Path], ExperimentSpec]
    build_data_source: Callable[[ExperimentSpec], DataSource]
    build_view_sampler: Callable[[ExperimentSpec], ViewSampler]
    build_feature_provider: Callable[
        [ExperimentSpec, torch.device],
        FeatureProvider | None,
    ]
    build_model_program: Callable[[ExperimentSpec], ModelProgram]
    build_objective: Callable[[ExperimentSpec, torch.device], RenderObjective]
    build_train_loop: Callable[[ExperimentSpec, "TrainComponents"], "TrainLoop"]
    expected_smokes: tuple[str, ...]
    old_entrypoints: tuple[Path, ...]
```

### Router

```python
def load_experiment_spec(config_path: Path) -> ExperimentSpec: ...
def resolve_recipe(spec: ExperimentSpec) -> TrainRecipe: ...
def explain_routing(config_path: Path) -> RoutingReport: ...
def run_config(config_path: Path) -> None: ...
```

`train.py explain CONFIG` can land first. `train.py run CONFIG` should only
redirect old entrypoints after route smokes are green.

Blocked routes must be explicit:

```text
route_status: blocked
reason: feature_dim=32 requires shared feature_alpha objective for multicam
old_command: PYTHONPATH=src/train uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py ...
next_required_phase: migrate_multicam_objective
```

## Implementation Order Recommended By The Reviewers

### Phase 0: Freeze Current Behavior

Do this before refactor patches:

1. Record current commands and W&B run IDs for active baselines.
2. Add or write down expected-fail route guards.
3. Confirm the single-cam F32 random-bg run `9gr2dm3v` and its caveat.
4. Smoke the known broken paths where useful, especially multicam F32 ultimate,
   and document expected failure rather than trying to make a local tuple patch.

### Phase 1: Config Visibility And Route Explain

1. Add normalized `ObjectiveSpec.background`.
2. Migrate current intentional F32 feature-splatting configs to:

```jsonc
"losses": {
  "background": {
    "train_mode": "random_rgb",
    "eval_mode": "white",
    "sample_scope": "step"
  }
}
```

3. Add `train.py explain CONFIG`.
4. Run explain across all 96 configs without baking features or mutating caches.

### Phase 2: Shared Objective First

1. Add `objective.py` with `BackgroundSpec`, `BackgroundSample`,
   `RasterizedView`, `ColorizedView`, `RenderedView`, `ObjectiveLoss`, and
   `RenderObjective`.
2. Port single-cam F=3 and F=32 through it.
3. Port known-camera initial/final validation through it.
4. Keep old entrypoints running until smokes pass.

This should come before the full router. It directly fixes the bug class.

### Phase 3: FeatureProvider Before Ultimate Multicam

1. Extract V-JEPA/LTX/Wan feature cache handling into `FeatureProvider`.
2. Make `describe_config()` cheap and explain-safe.
3. Make `describe_data()` run only in train/smoke mode.
4. Confirm model construction gets feature channel metadata before instantiation.

This must happen before the V-JEPA multicam ultimate smoke.

### Phase 4: ViewBatch And Multicam

1. Convert single-cam and precomputed feature routes to emit `ViewBatch`.
2. Convert multicam sampler to emit `ViewBatch` with condition, train, anchor,
   and held-out camera roles.
3. Remove multicam-specific render/loss logic and call `RenderObjective.loss()`.
4. Add held-out alpha/PCA/composite videos for F32.
5. Smoke multicam RGB/F3, then multicam F32 ultimate.

### Phase 5: Compatibility Shims

1. Old trainer entrypoints call the router only after their recipe is green.
2. Shims log old entrypoint, new entrypoint, recipe, objective version, and
   background policy.
3. Scripts keep working during the transition.

### Phase 6: Delete Only After Audits

Immediate delete candidates after a shim period:

```text
src/train/train_camera_implict_dynamic.py
src/train/train_image_encoder_implicit_camera_baseline.py
src/train/train_ltx_feature_implicit_dynamic.py
```

Do not delete during this refactor:

```text
src/train/dynamicTokenGS.py
src/train/tokenGS.py
src/train/*tiled*.py
research_experiments/gauge_fields/*
```

Extract first from `dynamicTokenGS.py`:

```text
pick_device
configure_fast_attn
fast_attn_context
shared optimizer/LR helpers if still imported
```

## Required Smoke Matrix

Layer 0: pre-migration freeze

| Route | Expected result | Purpose |
|---|---|---|
| F=3 single-cam 1-step | pass | Preserve legacy RGB route |
| F=32 single-cam alpha/random-bg 1-step | pass | Preserve fixed feature-splatting path |
| Known-camera 1-step with validation | may fail today | Capture stale tuple issue |
| Precomputed V-JEPA single-cam 1-step | pass or documented cache issue | Protect feature cache |
| Multicam F32 ultimate 1-step | expected fail or blocked | Prevent silent broken training |
| Explain all configs | pass after explain lands | Prove routing coverage |

Layer 1: per-phase smokes

```text
1. train.py explain across all configs
2. F32 background config visibility
3. single-cam F=3/F=32 through RenderObjective
4. known-camera validation through RenderObjective
5. precomputed V-JEPA through FeatureProvider
6. multicam RGB/F3 through ViewBatch and objective
7. multicam F32 V-JEPA ultimate through ViewBatch, FeatureProvider, objective
8. script shims through router
9. legacy adapters and gauge external delegates
```

Layer 2: artifact assertions

F32 smokes must verify:

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

Multicam smokes must verify:

```text
train-view metrics logged
held-out metrics logged
held-out render video logged
held-out alpha video logged if F32
condition camera logged
anchor camera logged
held-out camera names logged
```

## Baseline And W&B Rules

1. `BASELINES.md` remains append-only.
2. Smoke results go in migration notes, not the standings table.
3. Intentional benchmark runs add dated rows with config, entrypoint, W&B ID,
   objective version, background policy, split, step count, wall time, and
   metrics.
4. Multicam or novel-view claims must use held-out metrics as the selector.
   Source/train-view metrics are diagnostics.
5. W&B should log at least:

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

## What Not To Do

- Do not "fix" multicam F32 with only tuple unpacking. That would still compare
  raw features to RGB or bypass alpha/random-bg.
- Do not launch the ultimate multicam F32 baseline until the route is either
  blocked loudly or migrated through `ViewBatch + FeatureProvider +
  RenderObjective`.
- Do not let `RenderObjective` build W&B objects.
- Do not rank multicam baselines by train/source-view loss.
- Do not delete legacy trainer files until command-reference audits and shim
  smokes pass.
- Do not merge gauge-field internals into the video-token cleanup. Route them as
  external recipes for now.

## First Implementation Patch To Write Next

The next coding patch should not start by deleting files or building the full
router. Start with the boundary that caused the bug:

```text
src/train/objective.py
src/train/config_schema.py or config normalization helpers
targeted edits in train_video_token_implicit_dynamic.py
targeted known-camera validation fix
one-step F=3 and F=32 smokes
```

Then move to `FeatureProvider`, then `ViewBatch`, then multicam. This order
fixes the active bug class while keeping the larger cleanup disciplined.
