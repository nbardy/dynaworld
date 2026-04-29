# Final Design — Trainer / Render / Loss Cleanup

Date: 2026-04-30
Synthesizes: my five investigators, three proposals, and two reviewers in
`agent_notes/apr_30th_clean/`, plus Codex's parallel set in
`agent_notes/apr_30th_clean_codex/`. Picks the strongest move from each.

## TL;DR

- **One concrete `Trainer`** glues six pluggable strategies. No subclasses, no deep inheritance.
- **Eight typed bundles** (immutable frozen dataclasses) carry data between stages. No tuple-arity bugs by construction.
- **Six Protocol-typed strategies** define the seams: `ClipSampler`, `ModelProgram`, `FeatureProvider`, `RenderObjective`, `Validator`, `MediaLogger`.
- **`RenderObjective` is the load-bearing seam** that owns alpha-aware composition + random background + colorize + recon loss in one place. Both single-cam and multicam call into it. The bug we fixed this session — multicam bypassing alpha-aware composition — becomes structurally impossible.
- **Migration order**: `RenderObjective` → `FeatureProvider` → `ClipSampler` (with `ViewBatch`) → `Validator/MediaLogger` → delete legacy. This is Codex's order; it unblocks the multicam alpha bug at step 1, not step 5.
- **Estimated shrinkage**: ~6,200 lines of trainer code today → ~2,300 lines after refactor. **~63% reduction in trainer-layer code.** Plus deletion of 11 legacy / shim files.

## Design rules

These come from `key_learnings.md`, `AGENTS.md`, and the lessons from this session. Every part of the design must obey them.

1. **No new shared base class for trainers.** `key_learnings.md:18` warns explicitly: a `BaseTrainer` hides real differences. Composition only.
2. **Strategies are `typing.Protocol`, not abstract base classes.** Structural typing; any object with the right methods qualifies. No inheritance for shared behavior.
3. **Bundles are immutable** (`@dataclass(frozen=True)`). A stage takes a bundle in, returns a bundle out (often a richer one). Never mutate.
4. **Every signature change requires a runtime smoke test**, not just `py_compile`. AGENTS.md rule. Mid-cascade tuple-arity bugs are the failure mode this design eliminates.
5. **Random-bg and similar policies are typed values**, not hardcoded magic in `recon_backward`. Configurable, deterministic when seeded, eval-vs-train aware.
6. **Existing tested code (Codex's v5_features rasterizer, the 9 model classes, the `colorize.py` module, `feature_pca_viz.py`)** is preserved as-is. Only the trainer-orchestration layer is rewritten.
7. **Existing JSONC configs continue to work** through a thin compatibility shim during migration. New canonical configs live alongside them with explicit `pipeline:` and `strategies:` sections.

---

## Core types

All in `src/train/pipeline/bundles.py` unless otherwise noted.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, runtime_checkable
from pathlib import Path
import torch
from camera import CameraSpec, CameraState
from runtime_types import SequenceData
```

### Configuration

```python
@dataclass(frozen=True)
class ExperimentSpec:
    """The validated, typed representation of a train config.
    Replaces the loose `cfg: dict[str, Any]` that flows through every trainer today.

    Built once at trainer __init__ from a JSONC file via `parse_experiment_spec(cfg)`.
    Every field is required (no silent defaults at use site).
    """
    arch: str                       # "single_cam_F32_alpha", "multicam_vjepa_alpha", ...
    pipeline_name: str              # registry key; resolves to a strategy combo
    data: DataSpec
    model: ModelSpec
    camera: CameraSpec_Config
    render: RenderSpec
    train: TrainSpec
    losses: LossSpec
    logging: LoggingSpec
    colorize: ColorizeSpec | None
    background: BackgroundPolicy
    export: ExportSpec | None = None


@dataclass(frozen=True)
class DataSpec:
    """Just the data-source pointers. Distinct from the pipeline-level sampler."""
    source_kind: Literal["single_video", "frames_dir", "manifest", "multicam"]
    video_path: Path | None
    sequence_dir: Path | None
    manifest_path: Path | None
    multicam_manifest: Path | None
    multicam_split: str | None
    multicam_sample_id: str | None
    multicam_train_cameras: tuple[str, ...] | None
    multicam_heldout_camera: str | None
    multicam_anchor_camera: str | None
    eval_max_sequences: int = 1
    camera_image_size: int | None = None
    camera_focal_mode: Literal["per_frame", "median"] = "median"


@dataclass(frozen=True)
class ModelSpec:
    """Validated subset of the model section of the config."""
    variant: Literal[
        "unconditioned_tokens", "video_token", "known_camera",
        "free_splats", "linear_free_splats", "residual_free_bank",
        "unconditioned_residual_free_bank", "sinusoidal_time", "token_to_pose_to_plucker",
    ]
    feature_dim: int                # default 3 (legacy RGB) or N (feature splatting)
    tokens: int
    static_tokens: int | None
    dynamic_tokens: int | None
    train_frame_count: int
    image_size: int                 # encoder input size
    feat_dim: int                   # token embedding width (was `model_dim` in old configs)
    gaussians_per_token: int
    head: HeadSpec
    dynamic: DynamicSpec | None
    video_encoder_backend: Literal["none", "local", "vjepa_hf", "vjepa_torchhub", "precomputed", "precomputed_ltx"]
    vjepa: VJepaSpec | None         # only when video_encoder_backend uses V-JEPA


@dataclass(frozen=True)
class ColorizeSpec:
    hidden_dim: int | None
    activation: Literal["sigmoid", "identity"]
    pre_norm: bool
    weight_init: Literal["kaiming", "orthogonal"]
    weight_init_gain: float
    view_condition: Literal["none", "camera_center_ray", "pixel_ray"]
    detach_view_condition: bool


@dataclass(frozen=True)
class BackgroundPolicy:
    """The eval-vs-train background contract.

    `train_kind = "random"` removes the degenerate (alpha, splat) cheating
    manifold (3DGS-canonical trick). `eval_kind = "white"` keeps validation
    metrics and visuals comparable across runs.

    Both are exposed in the config now (previously hardcoded in recon_backward).
    """
    train_kind: Literal["white", "random_per_step", "black", "fixed_color"]
    eval_kind: Literal["white", "black", "fixed_color"]
    fixed_color: tuple[float, float, float] | None = None
    seed: int | None = None         # for reproducible random sampling


@dataclass(frozen=True)
class TrainSpec:
    steps: int
    lr: float
    amp: bool
    recon_backward_strategy: Literal["batched", "microbatch", "framewise"]
    temporal_microbatch_size: int


@dataclass(frozen=True)
class LossSpec:
    type: Literal["standard_gs", "feature_distillation", "alpha_aware_only"]
    l1_weight: float
    dssim_weight: float
    mse_weight: float
    ssim_window_size: int
    camera_motion_weight: float
    camera_temporal_weight: float
    camera_global_weight: float
    static_alpha_rate_weight: float
    dynamic_alpha_rate_weight: float
    dynamic_motion_rate_weight: float
    dynamic_rotation_rate_weight: float
    dynamic_alpha_time_rate_weight: float
    feature_distillation_weight: float = 0.0
    alpha_only_weight: float = 0.0          # anti-cheating aux loss


@dataclass(frozen=True)
class LoggingSpec:
    log_every: int
    image_log_every: int
    video_log_every: int
    always_log_last_step: bool
    feature_pca_log: bool
    alpha_mask_log: bool
    composite_log: bool
    wandb_project: str
    wandb_run_name: str
    wandb_tags: tuple[str, ...]


@dataclass(frozen=True)
class RenderSpec:
    renderer: Literal["dense", "tiled", "taichi", "fast_mac"]
    render_size: int
    auto_dense_limit: int
    tile_size: int
    bound_scale: float
    near_plane: float
    alpha_threshold: float
    fast_mac: FastMacSpec
    camera_projection: Literal["auto", "legacy_pinhole", "camera_model"]


@dataclass(frozen=True)
class FastMacSpec:
    tile_size: int
    max_fast_pairs: int
    alpha_threshold: float
    transmittance_threshold: float
    background: tuple[float, float, float]
    feature_background: float | tuple[float, ...]
    enable_overflow_fallback: bool
    batch_strategy: Literal["flatten", "auto", "serial"]
    batch_launch_limit_tiles: int
    batch_launch_limit_gaussians: int
```

### Pipeline bundles

```python
@dataclass(frozen=True)
class ViewBatch:
    """Output of ClipSampler.next_batch().

    Single-cam path: views == ((0,),) — one view, one camera tuple.
    Multicam path: views == ((0, 1), ...) — multiple training views per step.

    The held-out cameras for novel-view validation live in HeldoutCameras, a
    separate bundle the Validator gets, NOT this one.
    """
    sequence_data: SequenceData
    clip_indices: torch.Tensor              # [T] frame indices
    clip_frames: torch.Tensor               # [T, 3, H, W] GT RGB at the model's input size
    clip_times: torch.Tensor                # [T, 1] in [0, 1]
    train_views: tuple[int, ...]
    cameras_per_view: Mapping[int, tuple[CameraSpec, ...]]   # view_index -> (camera per frame)
    feature_inputs: FeatureInputs | None    # optional precomputed features for V-JEPA path

    @property
    def is_multicam(self) -> bool: return len(self.train_views) > 1
    @property
    def frame_count(self) -> int: return self.clip_frames.shape[0]


@dataclass(frozen=True)
class FeatureInputs:
    """V-JEPA / encoder feature payload provided to the model.
    Carries layer-named tensors so different encoders feed different keys."""
    layers: Mapping[str, torch.Tensor]      # e.g., {"vjepa_tokens": [1, 4608, 768]}
    frame_indices: torch.Tensor             # which frames they correspond to


@dataclass(frozen=True)
class DecodedSplats:
    """Output of ModelProgram.decode(). Replaces existing GaussianSequence.
    Field renamed from `rgbs` to `features` (post-feature-splatting; see investigator 5)."""
    xyz: torch.Tensor                       # [K, G, 3]
    scales: torch.Tensor                    # [K, G, 3]
    quats: torch.Tensor                     # [K, G, 4]
    opacities: torch.Tensor                 # [K, G, 1]
    features: torch.Tensor                  # [K, G, F]  -- F=3 legacy or F=32 feature-splatting
    cameras: tuple[CameraSpec, ...] | None  # implicit-camera path: not None
    camera_state: CameraState | None
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int: return self.features.shape[-1]
    @property
    def gaussian_count(self) -> int: return self.xyz.shape[1]
    @property
    def frame_count(self) -> int: return self.xyz.shape[0]


@dataclass(frozen=True)
class RenderedClip:
    """Output of the renderer (single chunk or full clip). Always (features, alpha)
    even when alpha is None (F=3 legacy v5 path).

    This dataclass replaces every `(features, alpha) | torch.Tensor` ambiguous
    return type that proliferated through the codebase during the alpha plumbing."""
    features: torch.Tensor                  # [T, F, H, W]; F=3 in legacy path, else F-channel
    alpha: torch.Tensor | None              # [T, H, W] when fast_mac+v5_features, else None
    cameras: tuple[CameraSpec, ...]
    width: int
    height: int


@dataclass(frozen=True)
class ComposedFrame:
    """Output of RenderObjective.compose(). RGB ready for loss + diagnostic side-channels."""
    rgb: torch.Tensor                       # [T, 3, H, W] in [0, 1]
    alpha_mask: torch.Tensor | None         # echoed from RenderedClip for downstream logging
    splat_rgb: torch.Tensor | None          # post-colorize, pre-bg-composition (for diagnostics)
    background_color_used: torch.Tensor     # [3] — what bg policy actually picked this step


@dataclass(frozen=True)
class LossOutput:
    """Output of RenderObjective.loss(). Total loss + per-term breakdown.

    `per_term` is for W&B logging (e.g., "Loss/Reconstruction": 0.123).
    `auxiliary` is for diagnostics (e.g., "AlphaCheating/CorrelationToWhiteGT": 0.42)."""
    total: torch.Tensor                     # scalar, requires_grad
    recon: torch.Tensor                     # scalar, detached
    per_term: Mapping[str, torch.Tensor]    # named pieces
    auxiliary: Mapping[str, float]


@dataclass(frozen=True)
class StepResult:
    """Output of Trainer.step(). One per training iteration."""
    step: int
    loss: LossOutput
    decoded: DecodedSplats | None           # kept for image-log gates
    composed: ComposedFrame | None          # kept for image-log gates
    sequence_path: Path | None
    sequence_frame_count: int
    keep_preview: bool


@dataclass(frozen=True)
class ValidationPayload:
    """Output of Validator.evaluate() + MediaLogger.payload(). The W&B post."""
    step: int
    scalars: Mapping[str, float]
    media: Mapping[str, "wandb.Image | wandb.Video"]
```

### Eval-side bundle

```python
@dataclass(frozen=True)
class HeldoutEvalRequest:
    """What the Validator gets when it's told to compute held-out novel-view metrics.
    Only meaningful for multicam pipelines; single-cam Validators get None."""
    sequence_data: SequenceData
    cameras_per_heldout_view: Mapping[int, tuple[CameraSpec, ...]]
    gt_per_heldout_view: Mapping[int, torch.Tensor]
```

---

## Core interfaces

All in `src/train/pipeline/protocols.py`.

```python
@runtime_checkable
class ClipSampler(Protocol):
    """Produces ViewBatch objects, one per training step.
    Single-cam impls return one view per batch; multicam impls return N views."""

    def __init__(self, spec: ExperimentSpec) -> None: ...
    def next_batch(self) -> ViewBatch: ...
    def heldout_request(self) -> HeldoutEvalRequest | None:
        """Return the held-out cameras + GT for novel-view validation, or None."""
        ...
    def __len__(self) -> int:
        """Total sequences this sampler can draw from."""
        ...
    def reset_epoch(self) -> None: ...


@runtime_checkable
class ModelProgram(Protocol):
    """Wraps the actual nn.Module model and its forward signature.
    Provides .decode() that takes a ViewBatch and returns DecodedSplats."""

    model: torch.nn.Module                  # the underlying nn.Module

    def __init__(self, spec: ExperimentSpec, device: torch.device) -> None: ...
    def decode(self, batch: ViewBatch) -> DecodedSplats: ...
    def parameters(self) -> "Iterable[torch.nn.Parameter]": ...
    def train(self, mode: bool = True) -> "ModelProgram": ...
    def eval(self) -> "ModelProgram": ...


@runtime_checkable
class FeatureProvider(Protocol):
    """Supplies precomputed (or live) encoder features to the ModelProgram.
    The trivial impl is NoFeatureProvider for unconditioned models."""

    def __init__(self, spec: ExperimentSpec, device: torch.device) -> None: ...
    def features_for_clip(
        self,
        sequence_data: SequenceData,
        clip_indices: torch.Tensor,
    ) -> FeatureInputs | None: ...
    def warmup(self, sequences: list[SequenceData]) -> None:
        """Optional: prebake feature cache. Called once at trainer setup."""
        ...


@runtime_checkable
class RenderObjective(Protocol):
    """The most load-bearing strategy. Owns:
        - rendering decoded splats
        - applying the colorize MLP (when feature_dim != 3)
        - alpha-aware composition with the configured background policy
        - computing the reconstruction loss against GT

    Single source of truth. Both single-cam and multicam paths use the same
    RenderObjective; multicam loops over views and accumulates loss."""

    def __init__(
        self,
        spec: ExperimentSpec,
        colorize: torch.nn.Module | None,
        device: torch.device,
    ) -> None: ...

    def render(self, decoded: DecodedSplats, cameras: tuple[CameraSpec, ...]) -> RenderedClip: ...
    def compose(self, rendered: RenderedClip, *, training: bool) -> ComposedFrame: ...
    def loss(self, composed: ComposedFrame, gt: torch.Tensor, decoded: DecodedSplats) -> LossOutput: ...

    def render_compose_loss(
        self,
        decoded: DecodedSplats,
        cameras: tuple[CameraSpec, ...],
        gt: torch.Tensor,
        *,
        training: bool,
    ) -> tuple[ComposedFrame, LossOutput]:
        """Convenience: the three-step chain in one call. Default impl in protocol."""
        rendered = self.render(decoded, cameras)
        composed = self.compose(rendered, training=training)
        loss = self.loss(composed, gt, decoded)
        return composed, loss


@runtime_checkable
class Validator(Protocol):
    """Owns the eval metric (source-view PSNR for single-cam; held-out novel-view
    PSNR for multicam). Stateless across steps; called every video_log_every steps."""

    def __init__(self, spec: ExperimentSpec) -> None: ...
    def evaluate(
        self,
        program: ModelProgram,
        sampler: ClipSampler,
        feature_provider: FeatureProvider,
        objective: RenderObjective,
        *,
        device: torch.device,
    ) -> tuple[Mapping[str, float], Mapping[str, RenderedClip], HeldoutMedia | None]: ...


@runtime_checkable
class MediaLogger(Protocol):
    """Builds the W&B media payload from the eval output."""

    def __init__(self, spec: ExperimentSpec) -> None: ...
    def payload(
        self,
        eval_renders: Mapping[str, RenderedClip],
        gt: Mapping[str, torch.Tensor],
        composed: Mapping[str, ComposedFrame],
        fps: float,
    ) -> Mapping[str, "wandb.Image | wandb.Video"]: ...
```

---

## Concrete strategy implementations

For each Protocol, this is the exhaustive list of concrete impls.

### Samplers

```python
class SingleClipSampler:
    """Yields a single ViewBatch with views=(0,) — the legacy single-source path."""
    def __init__(self, spec: ExperimentSpec) -> None: ...
    def next_batch(self) -> ViewBatch: ...
    def heldout_request(self) -> None: return None

class MulticamClipSampler:
    """Yields multi-view ViewBatches per step from a multicam manifest.
    Knows about train_views / heldout_camera split."""
    def __init__(self, spec: ExperimentSpec) -> None: ...
    def next_batch(self) -> ViewBatch: ...
    def heldout_request(self) -> HeldoutEvalRequest: ...
```

### Model programs

```python
class TokenGSProgram:
    """Wraps any of the 7 model classes from `gs_models/dynamic_video_token_gs_implicit_camera.py`.
    The variant is selected by spec.model.variant. Uses build_model_from_spec internally."""
    def __init__(self, spec: ExperimentSpec, device: torch.device) -> None:
        self.model = build_model_from_spec(spec, device)

    def decode(self, batch: ViewBatch) -> DecodedSplats:
        # Calls self.model(batch.feature_inputs, batch.clip_times, ...) and wraps
        # the return into DecodedSplats with renamed `rgbs` -> `features`.
        ...
```

(That's the only ModelProgram. The 7 internal model variants stay; TokenGSProgram is a thin adapter.)

### Feature providers

```python
class NoFeatureProvider:
    """For unconditioned models (variant in {unconditioned_tokens, free_splats, ...})."""
    def features_for_clip(self, *args, **kwargs) -> None: return None

class PrecomputedVJEPAFeatureProvider:
    """Reads V-JEPA features from a pre-baked cache file. The .pt cache key is
    derived from sequence + frames + V-JEPA model id (existing logic; this just
    relocates it from PrecomputedFeatureImplicitTrainer)."""
    def features_for_clip(self, sequence_data, clip_indices) -> FeatureInputs: ...
    def warmup(self, sequences) -> None: ...

class LiveVJEPAFeatureProvider:
    """Runs V-JEPA forward at training time. For configs that don't pre-bake."""
    def features_for_clip(self, sequence_data, clip_indices) -> FeatureInputs: ...
```

### Render objectives

Just one for now. Adding new ones (e.g., a feature-distillation objective for V-JEPA targets) is the obvious extension point.

```python
class RGBReconObjective:
    """The canonical RGB reconstruction objective with alpha-aware composition
    and configurable background policy. Used for both single-cam and multicam.

    Internals:
      - render(): renders via the active renderer mode (fast_mac / dense / etc.)
      - compose(): applies self.colorize, then α·splat_rgb + (1-α)·bg_sample()
      - loss(): L1 + DSSIM + (small) MSE + camera regularization terms

    Future RenderObjective impls can specialize this:
      - FeatureDistillationObjective (adds a feature-MSE term against V-JEPA targets)
      - AlphaAwareOnlyObjective (anti-cheating: loss against α·splat_rgb only)
    """
    def __init__(self, spec, colorize, device) -> None: ...
    def render(self, decoded, cameras) -> RenderedClip: ...
    def compose(self, rendered, *, training) -> ComposedFrame: ...
    def loss(self, composed, gt, decoded) -> LossOutput: ...
    def _sample_bg(self, *, training) -> torch.Tensor: ...
```

### Validators

```python
class SourceViewValidator:
    """Eval = same camera as training; classic source-view overfit metrics."""
    def evaluate(self, ...) -> ...: ...

class HeldoutCameraValidator:
    """Multicam: render the held-out cameras, compute PSNR/SSIM/L1 on those."""
    def evaluate(self, ...) -> ...: ...
```

### Media loggers

```python
class StandardMediaLogger:
    """Builds GT_Video, Render_Video, Render_GT_Video, Alpha_Mask_Video,
    Feature_PCA_Video, Render_Composite_Video keys based on spec.logging flags."""
    def payload(self, ...) -> Mapping[str, ...]: ...
```

(One impl is enough for now. Multicam logging is a config flag, not a separate logger.)

### Pipeline registry

```python
def build_strategies(spec: ExperimentSpec, device: torch.device) -> StrategyTuple:
    """Builds all six strategies from the spec's pipeline_name field."""
    ...

PIPELINE_REGISTRY: Mapping[str, StrategyFactoryTuple] = {
    "single_cam_F3_baseline": (
        SingleClipSampler, TokenGSProgram, NoFeatureProvider,
        RGBReconObjective, SourceViewValidator, StandardMediaLogger,
    ),
    "single_cam_F32_alpha": (...),
    "single_cam_vjepa_F32_alpha": (...),
    "multicam_F32_alpha": (...),
    "multicam_vjepa_F32_alpha": (...),
    "single_cam_known_camera_F32_alpha": (...),
}

@dataclass(frozen=True)
class StrategyTuple:
    sampler: ClipSampler
    program: ModelProgram
    feature_provider: FeatureProvider
    objective: RenderObjective
    validator: Validator
    media_logger: MediaLogger
```

---

## The Trainer

Single concrete class. All in `src/train/trainer.py`. Roughly 200 lines (vs. today's 2,072 in `train_video_token_implicit_dynamic.py:Trainer`).

```python
class Trainer:
    """Orchestrates a typed pipeline of strategies. Knows nothing about the
    specifics of feature splatting, multicam, V-JEPA — those live in strategies."""

    def __init__(self, spec: ExperimentSpec) -> None:
        self.spec = spec
        self.device = pick_device()
        self.colorize = build_colorize_from_spec(spec, self.device)
        self.strategies = build_strategies(spec, self.device)
        self.optimizer = build_optimizer(self.strategies.program, self.colorize, spec.train.lr)
        self.gt_video_logged = False  # one-shot logging flag (kept from existing trainer)

    def step(self, *, keep_preview: bool, step_index: int) -> StepResult:
        """One training iteration. Identical for single-cam and multicam."""
        self.optimizer.zero_grad(set_to_none=True)
        batch: ViewBatch = self.strategies.sampler.next_batch()
        decoded: DecodedSplats = self.strategies.program.decode(batch)

        loss_total = decoded.xyz.new_tensor(0.0)
        composed_for_preview: ComposedFrame | None = None

        for view in batch.train_views:
            cameras = batch.cameras_per_view[view]
            gt = self._gt_for_view(batch, view)
            composed, loss = self.strategies.objective.render_compose_loss(
                decoded, cameras, gt, training=True,
            )
            loss_total = loss_total + loss.total
            if keep_preview and composed_for_preview is None:
                composed_for_preview = composed

        loss_total = loss_total / max(1, len(batch.train_views))
        loss_total.backward()
        self.optimizer.step()

        return StepResult(
            step=step_index,
            loss=LossOutput(total=loss_total.detach(), recon=loss_total.detach(), per_term={}, auxiliary={}),
            decoded=decoded,
            composed=composed_for_preview,
            sequence_path=batch.sequence_data.source_path,
            sequence_frame_count=batch.sequence_data.frame_count,
            keep_preview=keep_preview,
        )

    @torch.no_grad()
    def validate(self, step_index: int) -> ValidationPayload: ...

    def run(self) -> None:
        wandb.init(...)
        self._initial_diagnostics()
        for step_index in range(1, self.spec.train.steps + 1):
            keep_preview = self._should_log_image(step_index)
            result = self.step(step_index=step_index, keep_preview=keep_preview)
            self._scalar_log(result, step_index)
            if self._should_log_video(step_index):
                payload = self.validate(step_index)
                wandb.log(payload.scalars | payload.media, step=step_index)
        wandb.finish()
```

That's the entire trainer. Everything that varies across pipelines lives in the strategy implementations.

---

## File tree (post-cleanup)

```
src/train/
├── trainer.py                              # ~250 lines (Trainer class only)
├── pipeline/
│   ├── __init__.py                         # ~30 lines (re-exports)
│   ├── spec.py                             # ~250 lines (ExperimentSpec + nested specs + parser)
│   ├── bundles.py                          # ~200 lines (typed bundles)
│   ├── protocols.py                        # ~150 lines (six Protocol classes)
│   ├── samplers.py                         # ~200 lines (Single + Multicam impls)
│   ├── programs.py                         # ~120 lines (TokenGSProgram adapter)
│   ├── feature_providers.py                # ~250 lines (No / Precomputed / Live V-JEPA impls)
│   ├── objectives.py                       # ~200 lines (RGBReconObjective)
│   ├── validators.py                       # ~150 lines (SourceView + HeldoutCamera impls)
│   ├── media_loggers.py                    # ~200 lines (StandardMediaLogger)
│   ├── compose.py                          # ~80 lines (sample_random_bg, compose helper)
│   ├── colorize_factory.py                 # ~50 lines (build_colorize_from_spec)
│   └── registry.py                         # ~100 lines (PIPELINE_REGISTRY + builders)
├── gs_models/                              # UNCHANGED (the 9 model classes stay)
│   ├── dynamic_video_token_gs_implicit_camera.py
│   ├── blocks.py
│   └── __init__.py
├── colorize.py                             # UNCHANGED (FeatureToColor module)
├── feature_pca_viz.py                      # UNCHANGED
├── init_diagnostics.py                     # UNCHANGED
├── probe_colorize_init.py                  # UNCHANGED (still useful)
├── probe_colorize_matrix.py                # UNCHANGED
├── runtime_types.py                        # MODIFIED (rename `rgbs` -> `features` in GaussianFrame; deprecate GaussianSequence in favor of DecodedSplats)
├── camera.py                               # UNCHANGED (typed primitives stay)
├── camera_rig.py                           # UNCHANGED
├── losses.py                               # MODIFIED (loss-helper functions stay; remove orchestration that's now in objectives.py)
├── renderers/
│   ├── fast_mac.py                         # UNCHANGED (already returns tuples; objectives consume)
│   ├── projection.py                       # UNCHANGED
│   ├── common.py                           # UNCHANGED
│   ├── dense.py                            # UNCHANGED
│   ├── taichi.py                           # UNCHANGED
│   └── tiled.py                            # UNCHANGED
├── rendering.py                            # MODIFIED (collapse `render_gaussian_frames` and `render_gaussian_frames_alpha_aware` to one entry that always returns RenderedClip)
├── multicam_video_data.py                  # UNCHANGED (data loading)
├── multicam_val_data.py                    # UNCHANGED
├── sequence_data.py                        # UNCHANGED
└── camera_implicit_dynamic.py              # CONSIDER DELETION (image-encoder baseline; if no active config uses it)
```

### Files to delete

```
src/train/train_video_token_implicit_dynamic.py                  # ~2,072 lines, biggest win
src/train/train_precomputed_feature_implicit_dynamic.py          # ~600 lines
src/train/train_multicam_precomputed_feature_implicit_dynamic.py # ~500 lines
src/train/train_camera_implicit_dynamic.py                       # ~417 lines
src/train/train_image_encoder_implicit_camera_baseline.py        # shim, ~10 lines
src/train/train_camera_implict_dynamic.py                        # typo file, ~10 lines
src/train/train_ltx_feature_implicit_dynamic.py                  # ~30 lines
src/train/dynamicTokenGS.py                                       # ~731 lines (legacy prebaked-camera; relocate the 4 utilities first)
src/train/dynamicTokenGS_shared.py                                # shim, <10 lines
src/train/dynamicTokenGS_tiled.py                                 # shim, <10 lines
src/train/tokenGS.py                                              # ~146 lines (single-image legacy)
src/train/tokenGS_shared.py                                       # shim
src/train/tokenGS_tiled.py                                        # shim
```

**Total deletion: ~4,500 lines across 13 files.**

### Files to relocate (small move; keep their utilities)

`dynamicTokenGS.py` re-exports `pick_device`, `configure_fast_attn`, `fast_attn_context`, `select_window_indices`. Move these into a new `src/train/utils.py` (~50 lines). Then `dynamicTokenGS.py` itself can be deleted.

---

## Function signatures — full public API surface

For everything in `src/train/pipeline/`. Internal helpers omitted.

### `pipeline/spec.py`

```python
def parse_experiment_spec(cfg: dict[str, Any]) -> ExperimentSpec: ...
def serialize_experiment_spec(spec: ExperimentSpec) -> dict[str, Any]: ...
def load_experiment_spec(path: Path) -> ExperimentSpec: ...
```

### `pipeline/registry.py`

```python
PIPELINE_REGISTRY: Mapping[str, StrategyFactoryTuple]

def build_strategies(spec: ExperimentSpec, device: torch.device) -> StrategyTuple: ...
def register_pipeline(name: str, factory: StrategyFactoryTuple) -> None: ...
```

### `pipeline/colorize_factory.py`

```python
def build_colorize_from_spec(spec: ExperimentSpec, device: torch.device) -> torch.nn.Module | None: ...
```

### `pipeline/compose.py`

```python
def sample_background(
    policy: BackgroundPolicy,
    *,
    training: bool,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor: ...

def alpha_compose(
    splat_rgb: torch.Tensor,         # [..., 3, H, W]
    alpha: torch.Tensor | None,      # [..., H, W]
    bg: torch.Tensor,                # [3]
) -> torch.Tensor: ...
```

### `pipeline/objectives.py`

```python
class RGBReconObjective:
    def __init__(self, spec, colorize, device) -> None: ...
    def render(self, decoded, cameras) -> RenderedClip: ...
    def compose(self, rendered, *, training) -> ComposedFrame: ...
    def loss(self, composed, gt, decoded) -> LossOutput: ...
    def render_compose_loss(self, decoded, cameras, gt, *, training) -> tuple[ComposedFrame, LossOutput]: ...
```

### `pipeline/samplers.py`

```python
class SingleClipSampler: ...     # (full Protocol surface)
class MulticamClipSampler: ...
```

### `pipeline/programs.py`

```python
class TokenGSProgram: ...        # (full Protocol surface)
def build_model_from_spec(spec: ExperimentSpec, device: torch.device) -> torch.nn.Module: ...
```

### `pipeline/feature_providers.py`

```python
class NoFeatureProvider: ...
class PrecomputedVJEPAFeatureProvider: ...
class LiveVJEPAFeatureProvider: ...
def build_feature_provider(spec, device) -> FeatureProvider: ...
```

### `pipeline/validators.py`

```python
class SourceViewValidator: ...
class HeldoutCameraValidator: ...
```

### `pipeline/media_loggers.py`

```python
class StandardMediaLogger: ...

# Internal helpers:
def make_alpha_mask_video(alpha: torch.Tensor, fps: float) -> "wandb.Video": ...
def make_pca_feature_video(features: torch.Tensor, fps: float) -> "wandb.Video": ...
def make_composite_video(
    gt: torch.Tensor,
    pred: torch.Tensor,
    alpha: torch.Tensor | None,
    pca: torch.Tensor | None,
    fps: float,
) -> "wandb.Video": ...
```

### `trainer.py`

```python
class Trainer:
    def __init__(self, spec: ExperimentSpec) -> None: ...
    def step(self, *, keep_preview: bool, step_index: int) -> StepResult: ...
    def validate(self, step_index: int) -> ValidationPayload: ...
    def run(self) -> None: ...
```

### `rendering.py` (post-refactor)

```python
def render(
    sequence: DecodedSplats,
    cameras: tuple[CameraSpec, ...],
    *,
    mode: Literal["dense", "tiled", "taichi", "fast_mac"],
    spec: RenderSpec,
    dense_grid: torch.Tensor | None = None,
) -> RenderedClip: ...

# All the legacy helpers (render_gaussian_frame, render_gaussian_frames,
# render_gaussian_frames_alpha_aware, render_clip_sequence, render_view_clip)
# are deleted. ONE entry, ONE return type.
```

---

## Migration sequence

Codex's order. Each step independently smoke-testable; no mid-cascade brokenness.

### Step 1: Land `RGBReconObjective` (the load-bearing seam)

- Add `pipeline/objectives.py` with `RGBReconObjective`
- Add `pipeline/compose.py` with `sample_background` + `alpha_compose`
- Add `pipeline/bundles.py` with `RenderedClip`, `ComposedFrame`, `LossOutput`, `ViewBatch` (minimal; sampler-side fields can come later)
- Trainer's existing `recon_backward` is changed to call `objective.render_compose_loss(...)` internally. ~50-line edit.
- The multicam trainer's `multicam_recon_loss` is changed to call the SAME objective for each view. **This single change fixes the multicam alpha bug.**
- **Smoke**: 1 step F=3 baseline + 1 step F=32 alpha + 1 step multicam F=3 + 1 step multicam F=32 alpha. All four must complete and produce sane losses.
- **Code delta this step**: +400 lines (new pipeline files), -150 lines (removed inline composition from two trainers).

### Step 2: Land `FeatureProvider` boundary

- Add `pipeline/feature_providers.py` with three impls
- Trainer reads features through the provider instead of via `PrecomputedFeatureImplicitTrainer.on_sequences_loaded()` (which gets deleted)
- Multicam trainer's V-JEPA wiring routes through the provider
- **Smoke**: 1 step single-source V-JEPA + 1 step multicam V-JEPA.
- **Code delta**: +250 lines new, -300 lines deleted from precomputed/multicam trainers.

### Step 3: Land `ClipSampler` + `ViewBatch`

- Add `pipeline/samplers.py`
- Trainer's existing `sample_clip` and multicam's `sample_multicam_clip` collapse to `sampler.next_batch()`
- **Smoke**: same suite as step 1, plus a multi-step run to confirm no regressions.
- **Code delta**: +200 lines new, -400 lines deleted.

### Step 4: Land `Validator` + `MediaLogger`

- Add `pipeline/validators.py` and `pipeline/media_loggers.py`
- Both single-cam and multicam call into the same loggers; held-out-camera videos appear automatically for multicam configs because of `HeldoutCameraValidator`
- **Smoke**: confirm `Alpha_Mask_Video`, `Feature_PCA_Video`, `Render_Composite_Video` land in W&B for both single-cam and multicam runs.
- **Code delta**: +350 lines new, -500 lines deleted.

### Step 5: Replace `Trainer` (the orchestrator)

- New `trainer.py` Trainer class that uses the strategies built in steps 1-4
- Existing `Trainer` class in `train_video_token_implicit_dynamic.py` becomes a thin shim that constructs the new `Trainer` and calls `.run()` (preserves CLI compat: `python train_video_token_implicit_dynamic.py <config>` still works)
- **Smoke**: full 1-step + 200-step run on `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc` and the F=3 baseline. PSNR must match the previous runs to within optimization noise.
- **Code delta**: +250 lines new, -1,500 lines deleted from the old Trainer class.

### Step 6: Delete legacy

- Delete the 11+ files in the deletion list above
- Move the 4 utilities from `dynamicTokenGS.py` into `utils.py`
- Update `pyproject.toml` / imports across `probe_*.py` if needed
- **Smoke**: full grep for any remaining import of the deleted files; the existing probes (`probe_colorize_init.py`, `probe_colorize_matrix.py`) must still run.
- **Code delta**: -2,500 lines deleted (the big win).

### Step 7: Config consolidation (optional, can be deferred)

- Add a `pipeline:` field to all configs (or build a thin migration script that infers it from the existing `arch` + launcher)
- Define a config inheritance / merge mechanism so the 96-config family can collapse to ~30 base configs + override fragments
- This is independent of the code refactor and can land later

---

## Code-size estimate

### Current state (rough)

Trainer files (counting only files in the deletion / heavy-modification list):

| File | Lines |
|---|---:|
| `train_video_token_implicit_dynamic.py` | 2,072 |
| `train_precomputed_feature_implicit_dynamic.py` | ~600 |
| `train_multicam_precomputed_feature_implicit_dynamic.py` | ~500 |
| `train_camera_implicit_dynamic.py` | 417 |
| `dynamicTokenGS.py` | 731 |
| `train_image_encoder_implicit_camera_baseline.py` (shim) | ~10 |
| `train_ltx_feature_implicit_dynamic.py` | ~30 |
| `train_camera_implict_dynamic.py` (typo) | ~10 |
| `tokenGS.py` | 146 |
| `tokenGS_shared.py` (shim) | ~5 |
| `tokenGS_tiled.py` (shim) | ~5 |
| `dynamicTokenGS_shared.py` (shim) | ~5 |
| `dynamicTokenGS_tiled.py` (shim) | ~5 |
| **Total trainer-layer code today** | **~4,536** |

Plus tangential code that gets simplified:

| File | Current | After |
|---|---:|---:|
| `rendering.py` | ~370 | ~150 (single `render(...) -> RenderedClip`) |
| `runtime_types.py` | ~280 | ~250 (rename `rgbs` field, deprecate `GaussianSequence`) |
| `losses.py` | ~80 | ~80 (unchanged) |

So the trainer-AND-supporting layer is roughly **~5,200 lines today**.

### After refactor

| File | Lines |
|---|---:|
| `trainer.py` | ~250 |
| `pipeline/spec.py` | ~250 |
| `pipeline/bundles.py` | ~200 |
| `pipeline/protocols.py` | ~150 |
| `pipeline/samplers.py` | ~200 |
| `pipeline/programs.py` | ~120 |
| `pipeline/feature_providers.py` | ~250 |
| `pipeline/objectives.py` | ~200 |
| `pipeline/validators.py` | ~150 |
| `pipeline/media_loggers.py` | ~200 |
| `pipeline/compose.py` | ~80 |
| `pipeline/colorize_factory.py` | ~50 |
| `pipeline/registry.py` | ~100 |
| `pipeline/__init__.py` | ~30 |
| `rendering.py` (slimmed) | ~150 |
| `runtime_types.py` (slimmed) | ~250 |
| `utils.py` (relocated helpers) | ~50 |
| **Total trainer-AND-pipeline-AND-supporting** | **~2,680** |

### Net shrinkage

```
Current trainer + supporting:       ~5,200 lines
After refactor:                     ~2,680 lines
Net code reduction:                  -2,520 lines  (~48% shrinkage)

Files deleted: 13
Files added: 13 (in pipeline/)
Files modified: ~5 (rendering.py, runtime_types.py, ...)
```

If the trainer-only slice is what you care about (excluding rendering.py and runtime_types.py):

```
Current trainer-only:              ~4,536 lines
After refactor (trainer/ + pipeline/): ~2,230 lines
Net trainer-layer reduction:        -2,306 lines  (~51% shrinkage)
```

**Roughly half the trainer-layer code disappears.** Plus 13 fewer files in the trainer/, with cleaner module boundaries.

The configs side is a separate, optional cleanup. 96 configs → ~30 base + override fragments would reduce config-file count by ~70%, but no single file gets significantly shorter.

---

## What this design does NOT change

- **Codex's v5_features rasterizer** stays as-is. The (features, alpha) tuple plumbing is preserved at the renderer boundary.
- **The 9 gs_models classes** stay. `feature_dim` is already threaded through them (Agent A's earlier work). `TokenGSProgram` is just a thin adapter.
- **`colorize.py`** stays. The new `colorize_factory.py` just wraps its construction.
- **`feature_pca_viz.py`, `init_diagnostics.py`** stay.
- **The probes** (`probe_colorize_init.py`, `probe_colorize_matrix.py`) stay. They consume `RenderedClip` which is a small interface change but a tiny diff.
- **The dataset configs** (`src/dataset_configs/*.jsonc`) stay.
- **JSONC format** stays. `parse_experiment_spec` reads the same format and produces a typed object.

## Risk control

The migration order is structured so each step is **independently shippable and testable**. Specifically:

- After step 1, the multicam alpha bug is **fixed** (the load-bearing fix; everything else is cleanup).
- After step 2, V-JEPA wiring is unified (no more parallel feature-cache code in two trainers).
- After step 3, sampling is unified.
- After steps 4-5, the orchestration is unified.
- After step 6, the legacy code is gone.

If we decide to stop after step 1 (because it's enough), we get the multicam fix without committing to the full refactor.

The smoke-test rule (AGENTS.md) is honored: every step ships with a 1-step smoke that exercises the actual call graph, not just `py_compile`.

The `key_learnings.md:18` warning ("a single shared `BaseTrainer` would hide real differences") is honored: there is no shared base class. The `Trainer` class is concrete, not abstract; differences live in strategies (Protocols, structural typing, no inheritance).

---

## Open questions

1. **Does `RenderObjective` cleanly absorb a future `FeatureDistillationObjective`** that adds an MSE term against frozen V-JEPA features alongside RGB L1? Yes if the loss bundle's `per_term` field is composable; the new objective would be a sibling class, not a branch in `RGBReconObjective`.

2. **Does `FeatureProvider` cleanly support both train-time precompute (cached .pt) and live forward (no cache)?** The Protocol's `warmup` method handles the precompute case; `features_for_clip` is the same for both.

3. **Is the `pipeline_name` registry the right discovery mechanism?** Or should configs explicitly name each strategy (`samplers: "multicam"`, `objective: "rgb_recon"`, ...)? The registry is more compact; explicit naming is more discoverable. Pick during step 5.

4. **Does `RenderObjective` need a `.eval()` mode separate from `training=True/False` in `compose()`?** Probably yes for the model.eval() call in validation. Add a `with self.eval_mode():` context manager to the protocol.

5. **`runtime_types.py:GaussianSequence` rename**: do we actually rename `rgbs` → `features` everywhere, or keep the field name for migration compatibility and just update docstrings? Investigator 5 flagged the misnomer. Recommend: rename in step 5 alongside the Trainer swap; update probes and other consumers in the same diff.
