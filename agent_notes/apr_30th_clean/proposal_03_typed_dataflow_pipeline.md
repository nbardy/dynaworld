# Proposal 03 — Typed Dataflow Pipeline

> Author: Proposer 3 of three. Design philosophy: every training step is a
> sequence of typed stages, each consuming and producing immutable typed
> bundles. The schema of the data flowing between stages is the central
> artifact. Once the bundles are right, the implementation is mostly
> mechanical.
>
> This is the most invasive of the three proposals. I commit to that cost
> up front: I claim the bug surface (silent tuple-vs-tensor, missing alpha
> composition in multicam, drift between train and eval bgs) is rooted in
> the dataflow being implicit, and that an explicit dataflow makes the
> bugs structural rather than nominal — once the bundle types compile, the
> classes of bug that the Apr 29 multicam fix discovered cannot exist.

## TL;DR

- A `TrainStep` is a `Pipeline` — a list of typed `Stage` callables. Each
  stage takes one immutable `Bundle` in and returns another out. No stage
  reaches into another's internals. No stage owns mutable state except
  the long-lived `PipelineContext` (model, optimizer, RNG, config).
- The pipeline is **data, not code**. `MULTICAM_PIPELINE` and
  `SINGLE_CAM_PIPELINE` differ only in their stage list — not in trainer
  classes. The shared compose / loss / backward / optimize stages mean
  the multicam path inherits alpha-aware composition automatically.
- Bundles are frozen dataclasses with explicit fields. Every field has a
  type. `Literal[...]` for enums. No `dict[str, Any]` for live data.
  Optionality (e.g. `alpha`) is meaningful, not accidental.
- The validation pipeline is a parallel `EvalPipeline` that shares the
  bundle vocabulary (Sample, Forward, Render, Compose) and substitutes
  `MetricStage` + `MediaPayloadStage` for `LossStage` + `BackwardStage` +
  `OptimizeStage`.
- Migration is staged: define the bundle module first, prove single-cam
  F=3 produces equivalent training, then port single-cam F=32 alpha,
  then multicam, then delete old trainers.

## Module layout

```
src/train/
├── pipeline/
│   ├── __init__.py          # public API: Pipeline, EvalPipeline, build_pipeline
│   ├── bundles.py           # all Bundle dataclasses (the central artifact)
│   ├── context.py           # PipelineContext (model, optimizer, RNG, config, mode)
│   ├── protocol.py          # Stage protocol + StageFactory typedef
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── sample.py        # SampleStage, MulticamSampleStage
│   │   ├── forward.py       # ForwardStage (model decode)
│   │   ├── render.py        # RenderStage, MultiViewRenderStage
│   │   ├── compose.py       # ComposeStage (colorize + alpha + bg) — load-bearing
│   │   ├── loss.py          # LossStage (recon + camera + bank-rate + rig)
│   │   ├── backward.py      # BackwardStage (chunked or single-shot)
│   │   ├── optimize.py      # OptimizeStage
│   │   ├── metric.py        # MetricStage (eval-only)
│   │   └── media.py         # MediaPayloadStage (eval-only, W&B)
│   ├── pipeline.py          # Pipeline driver, PipelineContext.run_step
│   ├── eval_pipeline.py     # EvalPipeline driver
│   └── registry.py          # name -> stage list (the wiring layer)
└── trainer.py               # thin bootstrap: build context, build pipelines, run steps
```

The old trainer files (`train_video_token_implicit_dynamic.py`,
`train_precomputed_feature_implicit_dynamic.py`,
`train_multicam_precomputed_feature_implicit_dynamic.py`,
`train_camera_implicit_dynamic.py`) are deleted in the final migration
step. Stages re-use `losses.py`, `colorize.py`, `rendering.py`, the
`gs_models/` model classes, the dataset loaders, and `camera_rig.py`
without modification — those are pure helpers, not stages.

## Bundles — the central artifact

This is where the design lives. Every bundle is a frozen dataclass.
Optional fields have `None` only when absence is the meaning, not as a
"haven't decided yet" placeholder. Where alpha is `None`, that means the
renderer cannot produce alpha (legacy F=3 v5 path); where `views` is
`None`, that means single-cam mode.

### `NullBundle`

```python
@dataclass(frozen=True)
class NullBundle:
    """Initial bundle entering an empty pipeline. The Sample stage produces
    real bundle content from this seed."""
    step: int
    is_eval: bool
```

The `step` and `is_eval` flag are the only context the very first stage
needs to seed downstream behavior (e.g. recon-bg policy).

### `ClipBundle`

Output of `SampleStage`. The single-cam shape.

```python
@dataclass(frozen=True)
class ClipBundle:
    """Single-cam clip sampled from one sequence. Output of SampleStage.

    Invariants:
      - clip_frames.shape == (T, 3, H, W) where T == clip_indices.shape[0]
      - clip_times.shape == (T, 1) and values in [0, 1]
      - clip_frames.dtype is the model's training dtype (float32 or float16)
    """
    step: int
    is_eval: bool
    sequence_data: SequenceData       # opaque pointer; trainer side
    sequence_index: int               # which sequence in the train set
    clip_indices: torch.Tensor        # [T] int64, frame indices into source video
    clip_frames: torch.Tensor         # [T, 3, H, W] GT RGB, in [0, 1]
    clip_times: torch.Tensor          # [T, 1] normalized time
```

### `MulticamClipBundle`

Output of `MulticamSampleStage`. Distinct type so downstream stages can
do typed dispatch (`MultiViewRenderStage` accepts only this).

```python
@dataclass(frozen=True)
class MulticamClipBundle:
    """Multi-view clip sampled from a multicam bundle.

    Invariants:
      - len(views) >= 1
      - clip_frames_per_view[v].shape == (T, 3, H, W) for every v in views
      - clip_indices identical across views (same temporal sample)
      - cameras_per_view[v] is a tuple of length T
    """
    step: int
    is_eval: bool
    multicam_bundle: MulticamVideoBundle
    clip_indices: torch.Tensor                                  # [T] int64
    clip_times: torch.Tensor                                    # [T, 1]
    views: tuple[int, ...]                                      # which view ids this step
    clip_frames_per_view: dict[int, torch.Tensor]               # view_id -> [T, 3, H, W]
    cameras_per_view: dict[int, tuple[CameraSpec, ...]]         # view_id -> (T cameras)
    heldout_views: tuple[int, ...]                              # eval-only; train sets to ()
```

### `ModelInputBundle`

Output of an explicit `ModelInputStage` if precomputed features are in
play, otherwise the `ForwardStage` reads `clip_frames` directly. This
existing as its own bundle is what makes the precomputed-feature
"trainer" go away — it's just a different `ModelInputStage`.

```python
@dataclass(frozen=True)
class ModelInputBundle:
    """The tensor or feature dict that the model's forward takes as input.

    For live-encoder paths, `as_tensor` is the GT clip_frames passthrough.
    For precomputed-feature paths, `as_tensor` is the cached feature
    tensor for this clip. The model's `decode_times=clip_times` arg is
    carried alongside.
    """
    step: int
    is_eval: bool
    as_tensor: torch.Tensor              # what the model.forward sees
    clip_times: torch.Tensor             # passed to model as decode_times
    parent: ClipBundle | MulticamClipBundle   # full upstream bundle, for downstream stages
```

The `parent` field is how a typed-dataflow pipeline does threading: the
ClipBundle is not destroyed, just nested. Stages that want
`clip_frames` reach through `bundle.parent.clip_frames`. This is
explicit; no global context needed.

### `DecodedBundle`

Output of `ForwardStage`. Wraps `GaussianSequence` plus auxiliary fields
for the loss stage.

```python
@dataclass(frozen=True)
class DecodedBundle:
    """Model decode output. The GaussianSequence is what the renderer
    consumes; the auxiliary dict carries the per-step diagnostic tensors
    (camera_state, dynamic_A_mu, dynamic_A_rot, etc.) that LossStage
    reads for camera regularization and bank-rate losses.

    Invariants:
      - decoded.rgbs.shape[-1] == ctx.cfg.model.feature_dim
      - decoded.cameras is non-None for implicit-camera variants;
        None for known-camera variants (cameras come from clip).
    """
    step: int
    is_eval: bool
    decoded: GaussianSequence
    parent: ClipBundle | MulticamClipBundle
    auxiliary: AuxiliaryDecode    # typed wrapper around the dynamic split bookkeeping
```

```python
@dataclass(frozen=True)
class AuxiliaryDecode:
    """Per-step diagnostic tensors emitted by the model for loss + logging.
    All fields are None when the model does not emit them (e.g.
    non-static/dynamic-split variants emit no bank-rate inputs)."""
    camera_state: CameraState | None
    static_opacities: torch.Tensor | None
    dynamic_opacities: torch.Tensor | None
    dynamic_A_mu: torch.Tensor | None
    dynamic_A_rot: torch.Tensor | None
    dynamic_A_alpha: torch.Tensor | None
```

### `RenderedBundle`

Output of `RenderStage`. The (features, alpha) tuple now becomes a
proper field pair on a typed bundle — and the alpha-vs-tensor confusion
that caused the Apr 29 bug becomes a typecheck error.

```python
@dataclass(frozen=True)
class RenderedBundle:
    """Rasterizer output: F-channel features + optional alpha.

    Invariants:
      - features.shape == (T, F, H, W) where F == ctx.cfg.model.feature_dim
      - alpha is None iff the active renderer does not surface alpha
        (i.e. fast_mac F=3, dense, taichi, tiled). For fast_mac F!=3
        alpha is never None.
      - rendered_cameras has length T and matches the cameras used
        during rasterization (after viewport scaling).
    """
    step: int
    is_eval: bool
    features: torch.Tensor                              # [T, F, H, W]
    alpha: torch.Tensor | None                          # [T, H, W] or None
    rendered_cameras: tuple[CameraSpec, ...]            # post-viewport cameras
    parent: DecodedBundle
```

### `MultiViewRenderedBundle`

Multi-view variant. Holds one `RenderedBundle` per view rather than
trying to compress into a four-tensor stack. The compose stage iterates
views; this is the right shape because compose is also iterating views.

```python
@dataclass(frozen=True)
class MultiViewRenderedBundle:
    """Per-view RenderedBundle. The compose / loss stages iterate
    .renders.items().

    Invariants:
      - set(renders.keys()) == set(parent.parent.views) for the train pass
      - all renders[v].features have identical shape (T, F, H, W)
      - all renders[v].alpha are uniformly None or uniformly tensors
    """
    step: int
    is_eval: bool
    renders: dict[int, RenderedBundle]                  # view_id -> per-view output
    parent: DecodedBundle
```

### `ComposedBundle`

Output of `ComposeStage`. This is the bundle that fixes the Apr 29 bug:
it has `final_rgb` and `background` as named fields. Loss reads from
`final_rgb`; logging reads `background` for the Render_Composite_Video.

```python
@dataclass(frozen=True)
class ComposedBundle:
    """Final RGB after colorize + alpha-aware composition. This is what
    the loss stage compares against GT.

    Invariants:
      - final_rgb.shape == (T, 3, H, W) for single-cam,
        or {view: (T, 3, H, W)} for multicam
      - final_rgb values in [0, 1] when the colorize activation is sigmoid
      - background is the actual background tensor used (per-step random
        for training, scalar 1.0 broadcast for eval)
      - alpha_used preserves the rasterizer alpha so logging can mask
        the GT-vs-render diff
    """
    step: int
    is_eval: bool
    final_rgb: torch.Tensor | dict[int, torch.Tensor]   # union by mode
    background: torch.Tensor                            # [1, 3, 1, 1] or scalar
    alpha_used: torch.Tensor | dict[int, torch.Tensor] | None
    rendered_features: torch.Tensor | dict[int, torch.Tensor]   # pre-colorize, for PCA log
    parent: RenderedBundle | MultiViewRenderedBundle
```

The `final_rgb: Tensor | dict[int, Tensor]` union is the single
type-level branch in this proposal. Downstream stages dispatch on
`isinstance(bundle.final_rgb, dict)`. Two ways to dispatch are possible:
(a) keep the union and dispatch in `LossStage`, (b) split into
`SingleCamComposedBundle` / `MultiCamComposedBundle` and dispatch via
the type. I propose (b): explicit types, mechanical dispatch.

```python
@dataclass(frozen=True)
class SingleCamComposedBundle:
    step: int
    is_eval: bool
    final_rgb: torch.Tensor                  # [T, 3, H, W]
    background: torch.Tensor
    alpha_used: torch.Tensor | None
    rendered_features: torch.Tensor
    parent: RenderedBundle

@dataclass(frozen=True)
class MultiCamComposedBundle:
    step: int
    is_eval: bool
    final_rgb_per_view: dict[int, torch.Tensor]
    background: torch.Tensor
    alpha_used_per_view: dict[int, torch.Tensor | None]
    rendered_features_per_view: dict[int, torch.Tensor]
    parent: MultiViewRenderedBundle
```

`ComposedBundle = SingleCamComposedBundle | MultiCamComposedBundle` as a
type alias for documentation. LossStage type-dispatches on it.

### `LossBundle`

Output of `LossStage`. Carries the scalar tensor that `.backward()` will
be called on, plus a per-term breakdown for logging.

```python
@dataclass(frozen=True)
class LossBundle:
    """Total loss + breakdown. The total is what backward() consumes.

    Invariants:
      - total is a 0-d tensor on the model device with requires_grad=True
        (during training)
      - terms maps every loss-term name to its 0-d scalar value
        (already-weighted; sum of terms.values() == total when there are
        no chunked-backward subtleties)
      - extras carries non-loss diagnostic scalars for W&B logging
    """
    step: int
    is_eval: bool
    total: torch.Tensor                       # the scalar passed to .backward()
    terms: dict[str, torch.Tensor]            # "recon", "camera_motion", ...
    extras: dict[str, torch.Tensor]           # scalar metrics that aren't part of total
    parent: SingleCamComposedBundle | MultiCamComposedBundle
```

### `BackwardBundle`

Output of `BackwardStage`. Represents that gradients have been computed.
Carries forward enough that `OptimizeStage` and `StepResult` collection
can finish without reaching back into the loss bundle.

```python
@dataclass(frozen=True)
class BackwardBundle:
    """Post-backward state. Gradients are populated on ctx.model and
    ctx.optimizer; this bundle just acknowledges that fact and carries
    the loss values forward for logging.

    Invariants:
      - All parameters in ctx.optimizer.param_groups have .grad set or
        are None (untouched by this step).
      - chunked_loss_values[i] is the per-chunk recon loss for chunk i
        in the chunked-backward strategy; empty for batched.
    """
    step: int
    is_eval: bool
    total_loss_value: torch.Tensor            # detached scalar
    terms: dict[str, torch.Tensor]            # detached
    extras: dict[str, torch.Tensor]
    chunked_loss_values: tuple[torch.Tensor, ...]    # () for non-chunked
    grad_norm_pre_clip: torch.Tensor | None
```

### `OptimizerBundle`

Output of `OptimizeStage`. Acknowledges optimizer.step has run.

```python
@dataclass(frozen=True)
class OptimizerBundle:
    """Post-optimizer state. The model has been updated.

    Invariants:
      - lr is the learning rate(s) actually used this step (may differ
        per param group in the multicam rig case)
      - grad_norm is post-clip (or pre-clip if clip is disabled)
    """
    step: int
    is_eval: bool
    total_loss_value: torch.Tensor
    terms: dict[str, torch.Tensor]
    extras: dict[str, torch.Tensor]
    lr: dict[str, float]                      # param-group-name -> lr
    grad_norm: torch.Tensor | None
```

### `StepResult`

Final summary returned by `Pipeline.run_step`. This is what the trainer
loop accumulates into the W&B scalar and image payloads.

```python
@dataclass(frozen=True)
class StepResult:
    """The pipeline driver collects this from the final OptimizerBundle.
    The trainer-loop main file consumes it for logging.

    Invariants:
      - terms always contains "recon" key
      - extras may contain logging-only diagnostics
        (camera_motion_norm, alpha_mean, etc.)
    """
    step: int
    is_eval: bool
    total_loss: torch.Tensor                  # detached scalar
    terms: dict[str, torch.Tensor]
    extras: dict[str, torch.Tensor]
    lr: dict[str, float]
    grad_norm: torch.Tensor | None
    keep_preview: bool                        # was this step asked to keep visuals?
    preview: PreviewBundle | None             # populated only on log_images steps
```

### `PreviewBundle`

The image-tier log payload. Built by an optional `PreviewStage` near the
end of the train pipeline (only when `should_log_images(step)`). This
keeps the preview path off the hot training loop.

```python
@dataclass(frozen=True)
class PreviewBundle:
    """Per-step preview image for W&B image logging.

    Invariants:
      - clip_preview is one [3, H, 2W] image (GT | render concatenated)
      - alpha_mask_preview is None when alpha is unavailable
    """
    step: int
    clip_preview: torch.Tensor                # [3, H, 2*W]
    alpha_mask_preview: torch.Tensor | None
    feature_pca_preview: torch.Tensor | None
    caption: str
```

### `ValidationBundle`

Eval-time analog of `LossBundle`. Carries metric values rather than a
backward-target loss.

```python
@dataclass(frozen=True)
class ValidationBundle:
    """Per-sequence validation metrics. The eval pipeline emits one of
    these per (sequence, view) combination; the EvalPipeline accumulator
    averages across them.

    Invariants:
      - metrics always contains 'L1', 'MSE', 'SSIM', 'DSSIM', 'Loss', 'PSNR'
      - prefix is empty for single-cam, 'TrainView{i}/' or
        'Heldout{i}_{name}/' for multicam
    """
    step: int
    sequence_index: int
    view_id: int | None                       # None for single-cam
    prefix: str
    metrics: dict[str, torch.Tensor]
    parent: SingleCamComposedBundle | MultiCamComposedBundle
```

### `VideoLogBundle`

W&B media payload for validation video logging. Output of
`MediaPayloadStage`.

```python
@dataclass(frozen=True)
class VideoLogBundle:
    """Final W&B media payload. The eval-pipeline driver emits one per
    validation pass.

    Invariants:
      - video_panels keys are the W&B panel names ('Render_GT_Video',
        'Render_Composite_Video', 'Alpha_Mask_Video', 'Feature_PCA_Video',
        'TrainView{i}', 'Heldout{i}_{name}')
      - scalars keys are 'Eval/L1', 'Eval/MSE', 'Eval/SSIM', etc.
    """
    step: int
    video_panels: dict[str, torch.Tensor]     # name -> [T, 3, H, W]
    scalars: dict[str, float]
    extras: dict[str, Any]                    # caption strings, etc.
```

### Bundle vocabulary summary

| Bundle | Producer stage | Consumed by |
|---|---|---|
| `NullBundle` | (pipeline driver seed) | `SampleStage` |
| `ClipBundle` | `SampleStage` | `ModelInputStage`, eval stages |
| `MulticamClipBundle` | `MulticamSampleStage` | `ModelInputStage`, multicam render |
| `ModelInputBundle` | `ModelInputStage` | `ForwardStage` |
| `DecodedBundle` | `ForwardStage` | `RenderStage`, `LossStage` |
| `RenderedBundle` | `RenderStage` (single-cam) | `ComposeStage` |
| `MultiViewRenderedBundle` | `MultiViewRenderStage` | `MultiViewComposeStage` |
| `SingleCamComposedBundle` | `ComposeStage` | `LossStage`, `MetricStage`, `PreviewStage` |
| `MultiCamComposedBundle` | `MultiViewComposeStage` | same |
| `LossBundle` | `LossStage` | `BackwardStage` |
| `BackwardBundle` | `BackwardStage` | `OptimizeStage` |
| `OptimizerBundle` | `OptimizeStage` | `Pipeline.collect_step_result` |
| `StepResult` | (pipeline driver) | trainer main loop |
| `PreviewBundle` | `PreviewStage` (optional) | nested in StepResult |
| `ValidationBundle` | eval `MetricStage` | `MediaPayloadStage` |
| `VideoLogBundle` | `MediaPayloadStage` | trainer main loop |

## Stage Protocol

```python
# src/train/pipeline/protocol.py

from typing import Protocol, runtime_checkable, TypeVar, Generic

BundleIn = TypeVar("BundleIn")
BundleOut = TypeVar("BundleOut")

@runtime_checkable
class Stage(Protocol, Generic[BundleIn, BundleOut]):
    """A typed dataflow stage. Stateless except for its constructor-time
    config; mutable state lives on PipelineContext.

    The `name` attribute is a string identifier used by the registry
    and by error messages. It is not the dispatch key — Stage instances
    are picked by the registry, not looked up by name.
    """
    name: str

    def __call__(self, bundle_in: BundleIn, *, ctx: "PipelineContext") -> BundleOut:
        ...

StageFactory = Callable[["PipelineContext"], Stage]
```

## PipelineContext

The cross-stage state holder. Carries everything that does not flow
through bundles: the model itself, optimizer, RNG, configuration,
device, dtype, and a few helpers.

```python
# src/train/pipeline/context.py

@dataclass
class PipelineContext:
    """Cross-stage mutable state. Stages reach in for model + optimizer
    only; everything else they receive comes through bundles.

    Not frozen: model parameters mutate; optimizer state mutates; RNG
    advances. But each individual access is local and explicit.
    """
    model: nn.Module
    colorize: FeatureToColor | None
    optimizer: torch.optim.Optimizer
    device: torch.device
    dtype: torch.dtype
    rng: torch.Generator                      # for reproducible random_bg
    cfg: ResolvedConfig                       # frozen, validated config
    is_eval: bool                             # set by Pipeline vs EvalPipeline
    feature_cache: VideoFeatureCache | None   # precomputed-feature path; None otherwise
    multicam_bundle: MulticamVideoBundle | None
    camera_rig: LearnableCameraRig | None
    sequences: tuple[SequenceData, ...]       # train sequences
    eval_sequences: tuple[SequenceData, ...]
```

Stages access `ctx` only for the things that genuinely cannot live in
bundles: `ctx.model(model_input)`, `ctx.optimizer.step()`,
`torch.rand(..., generator=ctx.rng)`, `ctx.cfg.losses`. They never
mutate `ctx` other than `ctx.optimizer.step()` (which mutates model
parameters, not the context).

## Stages — full specs

### `SampleStage`

```python
# src/train/pipeline/stages/sample.py

class SampleStage:
    """Single-cam clip sampler. Picks one sequence, one clip window,
    materializes GT frames + clip times. Replaces Trainer.sample_clip."""
    name = "sample"

    def __init__(self, sampler: ClipSampler):
        self.sampler = sampler

    def __call__(self, bundle_in: NullBundle, *, ctx: PipelineContext) -> ClipBundle:
        sequence_index, sequence_data = self.sampler.pick_sequence(ctx)
        clip_indices = self.sampler.pick_window(sequence_data, ctx)
        clip_frames, clip_times = self.sampler.prepare_clip(
            sequence_data, clip_indices, ctx
        )
        return ClipBundle(
            step=bundle_in.step,
            is_eval=bundle_in.is_eval,
            sequence_data=sequence_data,
            sequence_index=sequence_index,
            clip_indices=clip_indices,
            clip_frames=clip_frames,
            clip_times=clip_times,
        )
```

### `MulticamSampleStage`

```python
class MulticamSampleStage:
    """Multi-view clip sampler. One temporal window, multiple views,
    optional heldout views for eval."""
    name = "multicam_sample"

    def __init__(self, sampler: MulticamClipSampler):
        self.sampler = sampler

    def __call__(self, bundle_in: NullBundle, *, ctx: PipelineContext) -> MulticamClipBundle:
        clip_indices, clip_times = self.sampler.pick_window(ctx)
        views = self.sampler.pick_views(ctx)
        heldout_views = (
            self.sampler.heldout_views(ctx) if bundle_in.is_eval else ()
        )
        cameras_per_view = {
            v: ctx.camera_rig.cameras_for_view(v, clip_indices) for v in views + heldout_views
        }
        clip_frames_per_view = {
            v: self.sampler.gt_frames_for_view(ctx.multicam_bundle, v, clip_indices)
            for v in views + heldout_views
        }
        return MulticamClipBundle(
            step=bundle_in.step,
            is_eval=bundle_in.is_eval,
            multicam_bundle=ctx.multicam_bundle,
            clip_indices=clip_indices,
            clip_times=clip_times,
            views=views,
            clip_frames_per_view=clip_frames_per_view,
            cameras_per_view=cameras_per_view,
            heldout_views=heldout_views,
        )
```

### `ModelInputStage`

The split that absorbs the precomputed-feature subclass. Two
implementations, one per data source, picked at registry time.

```python
# src/train/pipeline/stages/forward.py

class LiveEncoderModelInputStage:
    """Uses the GT clip frames as the model input. The model runs its
    own video encoder forward."""
    name = "model_input.live"

    def __call__(
        self, bundle_in: ClipBundle | MulticamClipBundle, *, ctx: PipelineContext,
    ) -> ModelInputBundle:
        # For multicam, condition view's clip_frames is the model input.
        # The condition view selection lives in the resolver, not here.
        as_tensor = _select_condition_frames(bundle_in, ctx.cfg)
        return ModelInputBundle(
            step=bundle_in.step,
            is_eval=bundle_in.is_eval,
            as_tensor=as_tensor,
            clip_times=bundle_in.clip_times,
            parent=bundle_in,
        )

class PrecomputedFeatureModelInputStage:
    """Returns the precomputed feature tensor for the clip from the
    feature cache."""
    name = "model_input.precomputed"

    def __call__(
        self, bundle_in: ClipBundle | MulticamClipBundle, *, ctx: PipelineContext,
    ) -> ModelInputBundle:
        as_tensor = ctx.feature_cache.get_clip_features(
            bundle_in.sequence_data, bundle_in.clip_indices,
        ) if isinstance(bundle_in, ClipBundle) else (
            ctx.feature_cache.get_multicam_clip_features(
                bundle_in.multicam_bundle, bundle_in.clip_indices, ctx.cfg,
            )
        )
        return ModelInputBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            as_tensor=as_tensor,
            clip_times=bundle_in.clip_times,
            parent=bundle_in,
        )
```

### `ForwardStage`

```python
class ForwardStage:
    """Calls ctx.model. Returns DecodedBundle with the GaussianSequence
    and any auxiliary diagnostics."""
    name = "forward"

    def __call__(
        self, bundle_in: ModelInputBundle, *, ctx: PipelineContext,
    ) -> DecodedBundle:
        with fast_attn_context(ctx.device), ctx.autocast_context():
            decoded = ctx.model(bundle_in.as_tensor, decode_times=bundle_in.clip_times)
        auxiliary = AuxiliaryDecode(
            camera_state=getattr(decoded, "camera_state", None),
            static_opacities=decoded.auxiliary.get("static_opacities"),
            dynamic_opacities=decoded.auxiliary.get("dynamic_opacities"),
            dynamic_A_mu=decoded.auxiliary.get("dynamic_A_mu"),
            dynamic_A_rot=decoded.auxiliary.get("dynamic_A_rot"),
            dynamic_A_alpha=decoded.auxiliary.get("dynamic_A_alpha"),
        )
        return DecodedBundle(
            step=bundle_in.step,
            is_eval=bundle_in.is_eval,
            decoded=decoded,
            parent=bundle_in.parent,
            auxiliary=auxiliary,
        )
```

### `KnownCameraForwardStage`

Variant for known-camera configs: passes precomputed cameras to the
model. Distinct stage class because the model API differs.

```python
class KnownCameraForwardStage:
    name = "forward.known_camera"

    def __call__(
        self, bundle_in: ModelInputBundle, *, ctx: PipelineContext,
    ) -> DecodedBundle:
        clip_cameras = bundle_in.parent.sequence_data.cameras_for_indices(
            bundle_in.parent.clip_indices
        )
        with fast_attn_context(ctx.device), ctx.autocast_context():
            decoded = ctx.model(
                bundle_in.as_tensor,
                decode_times=bundle_in.clip_times,
                cameras=clip_cameras,
            )
        return DecodedBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            decoded=decoded,
            parent=bundle_in.parent,
            auxiliary=AuxiliaryDecode(camera_state=None, ...),
        )
```

### `RenderStage`

```python
# src/train/pipeline/stages/render.py

class RenderStage:
    """Single-cam rasterizer. Returns features + alpha as a typed bundle.
    The (features, alpha) tuple goes away here — there is no tuple to
    misinterpret."""
    name = "render"

    def __init__(self, mode: Literal["fast_mac", "dense", "tiled", "taichi"]):
        self.mode = mode

    def __call__(
        self, bundle_in: DecodedBundle, *, ctx: PipelineContext,
    ) -> RenderedBundle:
        cameras = bundle_in.decoded.cameras  # implicit-camera path
        rendered_cameras = viewport_cameras(cameras, ctx.cfg.model.size, ctx.cfg.render.render_size)
        features, alpha = render_clip_sequence_alpha_aware(
            bundle_in.decoded, rendered_cameras, mode=self.mode, render_cfg=ctx.cfg.render,
        )
        return RenderedBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            features=features, alpha=alpha,
            rendered_cameras=rendered_cameras,
            parent=bundle_in,
        )
```

### `MultiViewRenderStage`

```python
class MultiViewRenderStage:
    """One RenderedBundle per view, sharing the decoded GaussianSequence."""
    name = "multiview_render"

    def __init__(self, mode: Literal["fast_mac", "dense", "tiled", "taichi"]):
        self.mode = mode

    def __call__(
        self, bundle_in: DecodedBundle, *, ctx: PipelineContext,
    ) -> MultiViewRenderedBundle:
        clip = bundle_in.parent  # MulticamClipBundle
        renders: dict[int, RenderedBundle] = {}
        for view in clip.views:
            view_cameras = clip.cameras_per_view[view]
            rendered_cameras = viewport_cameras(view_cameras, ctx.cfg.model.size, ctx.cfg.render.render_size)
            features, alpha = render_clip_sequence_alpha_aware(
                bundle_in.decoded, rendered_cameras, mode=self.mode, render_cfg=ctx.cfg.render,
            )
            renders[view] = RenderedBundle(
                step=bundle_in.step, is_eval=bundle_in.is_eval,
                features=features, alpha=alpha,
                rendered_cameras=rendered_cameras,
                parent=bundle_in,
            )
        return MultiViewRenderedBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            renders=renders,
            parent=bundle_in,
        )
```

### `ComposeStage`

This is the load-bearing stage. It is shared between single-cam and
multicam by virtue of dispatching on the input bundle type. The Apr 29
bug becomes impossible: there is no path through `ComposeStage` that
does not consume `alpha`.

```python
# src/train/pipeline/stages/compose.py

class ComposeStage:
    """Applies colorize MLP and alpha-aware composition. Picks per-step
    random_bg in train mode; uses fixed white in eval mode.

    The single load-bearing stage. The Apr 29 bug class — "multicam
    bypassed alpha-aware composition" — is impossible here because the
    multicam path goes through this same stage; the only difference is
    the input bundle type."""
    name = "compose"

    def __init__(
        self,
        background_policy: Literal["random_per_step", "white", "config"],
        view_condition: Literal["none", "camera_center_ray", "pixel_ray"],
        detach_view_condition: bool,
    ):
        self.background_policy = background_policy
        self.view_condition = view_condition
        self.detach_view_condition = detach_view_condition

    def __call__(
        self, bundle_in: RenderedBundle | MultiViewRenderedBundle, *, ctx: PipelineContext,
    ) -> SingleCamComposedBundle | MultiCamComposedBundle:
        background = self._sample_background(ctx, bundle_in)
        if isinstance(bundle_in, RenderedBundle):
            return self._compose_single(bundle_in, background, ctx)
        return self._compose_multi(bundle_in, background, ctx)

    def _sample_background(
        self, ctx: PipelineContext, bundle_in: RenderedBundle | MultiViewRenderedBundle,
    ) -> torch.Tensor:
        if self.background_policy == "white":
            return torch.tensor(1.0, device=ctx.device, dtype=ctx.dtype)
        if self.background_policy == "random_per_step":
            return torch.rand(
                3, device=ctx.device, dtype=ctx.dtype, generator=ctx.rng,
            ).view(1, 3, 1, 1)
        # "config" — read from cfg.render.fast_mac.background
        return _bg_tensor_from_config(ctx.cfg.render, ctx.device, ctx.dtype)

    def _compose_single(
        self, bundle_in: RenderedBundle, background: torch.Tensor, ctx: PipelineContext,
    ) -> SingleCamComposedBundle:
        final_rgb = self._compose_one(
            bundle_in.features, bundle_in.alpha, bundle_in.rendered_cameras,
            background, ctx,
        )
        return SingleCamComposedBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            final_rgb=final_rgb,
            background=background,
            alpha_used=bundle_in.alpha,
            rendered_features=bundle_in.features,
            parent=bundle_in,
        )

    def _compose_multi(
        self, bundle_in: MultiViewRenderedBundle, background: torch.Tensor, ctx: PipelineContext,
    ) -> MultiCamComposedBundle:
        final_rgb_per_view: dict[int, torch.Tensor] = {}
        alpha_used_per_view: dict[int, torch.Tensor | None] = {}
        rendered_features_per_view: dict[int, torch.Tensor] = {}
        for view, rendered in bundle_in.renders.items():
            final_rgb_per_view[view] = self._compose_one(
                rendered.features, rendered.alpha, rendered.rendered_cameras,
                background, ctx,
            )
            alpha_used_per_view[view] = rendered.alpha
            rendered_features_per_view[view] = rendered.features
        return MultiCamComposedBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            final_rgb_per_view=final_rgb_per_view,
            background=background,
            alpha_used_per_view=alpha_used_per_view,
            rendered_features_per_view=rendered_features_per_view,
            parent=bundle_in,
        )

    def _compose_one(
        self,
        features: torch.Tensor,
        alpha: torch.Tensor | None,
        cameras: tuple[CameraSpec, ...],
        background: torch.Tensor,
        ctx: PipelineContext,
    ) -> torch.Tensor:
        if ctx.colorize is not None:
            view_dirs = colorize_view_dirs_for_features(
                features, cameras,
                view_condition=self.view_condition,
                input_size=ctx.cfg.model.size,
                render_size=ctx.cfg.render.render_size,
                detach=self.detach_view_condition,
            )
            splat_rgb = ctx.colorize(features, view_dirs=view_dirs)
            if alpha is not None:
                alpha_expanded = alpha.unsqueeze(1)
                return alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * background
            return splat_rgb
        # legacy F=3 path; renderer's own bg already composited
        return features
```

### `LossStage`

```python
# src/train/pipeline/stages/loss.py

class LossStage:
    """Builds the total loss + per-term breakdown. Reads recon target
    from bundle.parent (the rendered bundle's parent's parent); reads
    auxiliary diagnostics from DecodedBundle.auxiliary; reads cfg.losses
    for weights."""
    name = "loss"

    def __init__(self, loss_cfg: LossConfig):
        self.loss_cfg = loss_cfg

    def __call__(
        self, bundle_in: SingleCamComposedBundle | MultiCamComposedBundle, *, ctx: PipelineContext,
    ) -> LossBundle:
        if isinstance(bundle_in, SingleCamComposedBundle):
            recon = self._single_cam_recon(bundle_in)
        else:
            recon = self._multicam_recon(bundle_in)

        decoded_bundle = self._find_decoded(bundle_in)
        camera_terms = self._camera_loss(decoded_bundle, ctx)
        bank_rate_terms = self._bank_rate_loss(decoded_bundle, ctx)
        rig_term = self._rig_loss(ctx)

        terms: dict[str, torch.Tensor] = {"recon": recon, **camera_terms, **bank_rate_terms}
        if rig_term is not None:
            terms["rig"] = rig_term

        total = sum(terms.values())
        extras = self._build_extras(decoded_bundle, bundle_in, ctx)
        return LossBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            total=total, terms=terms, extras=extras,
            parent=bundle_in,
        )

    def _single_cam_recon(self, bundle_in: SingleCamComposedBundle) -> torch.Tensor:
        # walk: SingleCamComposed -> Rendered -> Decoded -> ModelInput.parent (ClipBundle)
        clip = bundle_in.parent.parent.parent  # ClipBundle
        target = resize_images(clip.clip_frames, ctx.cfg.render.render_size)
        return reconstruction_loss_per_image(bundle_in.final_rgb, target, self.loss_cfg).mean()

    def _multicam_recon(self, bundle_in: MultiCamComposedBundle) -> torch.Tensor:
        clip = bundle_in.parent.parent.parent  # MulticamClipBundle
        recon = torch.zeros((), device=ctx.device, dtype=ctx.dtype)
        for view, final_rgb in bundle_in.final_rgb_per_view.items():
            target = resize_images(clip.clip_frames_per_view[view], ctx.cfg.render.render_size)
            recon = recon + reconstruction_loss_per_image(final_rgb, target, self.loss_cfg).mean()
        return recon / len(bundle_in.final_rgb_per_view)
```

The bundle parent chain is the only "context" the loss stage needs. No
`self.target_frames`, no `self.recon_size`. Everything that contributes
to the loss is in the bundle.

### `BackwardStage`

```python
# src/train/pipeline/stages/backward.py

class BackwardStage:
    """Single-shot backward. The chunked-backward strategy is a different
    stage class (ChunkedBackwardStage) — picked at registry time."""
    name = "backward"

    def __call__(
        self, bundle_in: LossBundle, *, ctx: PipelineContext,
    ) -> BackwardBundle:
        bundle_in.total.backward()
        grad_norm = self._compute_grad_norm(ctx)
        return BackwardBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            total_loss_value=bundle_in.total.detach(),
            terms={k: v.detach() for k, v in bundle_in.terms.items()},
            extras=bundle_in.extras,
            chunked_loss_values=(),
            grad_norm_pre_clip=grad_norm,
        )


class ChunkedBackwardStage:
    """Per-chunk backward with retain_graph=True except on last chunk.
    Used for memory-bounded long-clip training. Replaces
    Trainer.recon_backward without re-running the rasterizer in chunks
    — instead, the upstream pipeline runs the chunked render+compose
    via a different stage list. See registry note below."""
    name = "backward.chunked"

    def __init__(self, strategy: Literal["batched", "microbatch", "framewise"], microbatch_size: int):
        self.strategy = strategy
        self.microbatch_size = microbatch_size

    def __call__(
        self, bundle_in: LossBundle, *, ctx: PipelineContext,
    ) -> BackwardBundle:
        # In chunked mode, the upstream Compose+Loss stages have already
        # produced N pre-chunked LossBundles via a ChunkedComposePipeline
        # nested inside this driver. The migration plan addresses how
        # to handle this — see the migration section.
        ...
```

The chunked backward is the one place where the typed-dataflow model is
genuinely awkward. See "Risk analysis" — the cleanest answer is to make
the chunked path a nested `ChunkedRecoBackwardPipeline` that produces
N `BackwardBundle`s and merges them. The current code does
`backward_loss.backward(retain_graph=not is_last_chunk)` inside a loop,
which is naturally a sub-pipeline.

### `OptimizeStage`

```python
# src/train/pipeline/stages/optimize.py

class OptimizeStage:
    """Calls optimizer.step + zero_grad, optional grad clip, returns lr
    bookkeeping."""
    name = "optimize"

    def __init__(self, clip_grad_norm: float | None = None):
        self.clip_grad_norm = clip_grad_norm

    def __call__(
        self, bundle_in: BackwardBundle, *, ctx: PipelineContext,
    ) -> OptimizerBundle:
        if self.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for g in ctx.optimizer.param_groups for p in g["params"]],
                self.clip_grad_norm,
            )
        else:
            grad_norm = bundle_in.grad_norm_pre_clip
        ctx.optimizer.step()
        ctx.optimizer.zero_grad(set_to_none=True)
        lr = {g.get("name", str(i)): g["lr"] for i, g in enumerate(ctx.optimizer.param_groups)}
        return OptimizerBundle(
            step=bundle_in.step, is_eval=bundle_in.is_eval,
            total_loss_value=bundle_in.total_loss_value,
            terms=bundle_in.terms,
            extras=bundle_in.extras,
            lr=lr,
            grad_norm=grad_norm,
        )
```

### Eval-only stages

`MetricStage`, `MediaPayloadStage`, and `EvalForwardStage` /
`EvalRenderStage`. These reuse the same bundle vocabulary; the only
differences are no_grad context and a fixed white bg from the
`ComposeStage` registry config.

```python
# src/train/pipeline/stages/metric.py

class MetricStage:
    """Eval-only. Computes per-sequence metrics from a ComposedBundle.
    Emits one ValidationBundle per (sequence, view)."""
    name = "metric"

    def __call__(
        self, bundle_in: SingleCamComposedBundle | MultiCamComposedBundle, *, ctx: PipelineContext,
    ) -> ValidationBundle | tuple[ValidationBundle, ...]:
        ...

# src/train/pipeline/stages/media.py

class MediaPayloadStage:
    """Eval-only. Builds the W&B media payload from accumulated
    ValidationBundles + the alpha + feature-PCA buffers held in the
    ComposedBundle."""
    name = "media_payload"

    def __call__(
        self, bundle_in: ValidationBundle | tuple[ValidationBundle, ...],
        *, ctx: PipelineContext,
    ) -> VideoLogBundle:
        ...
```

## The `Pipeline` driver

```python
# src/train/pipeline/pipeline.py

class Pipeline:
    """Trains. Runs each stage in order; bundle threads through. Stage
    list comes from the registry."""

    def __init__(self, stages: list[Stage], ctx: PipelineContext):
        self.stages = stages
        self.ctx = ctx

    def run_step(self, step: int) -> StepResult:
        bundle = NullBundle(step=step, is_eval=False)
        for stage in self.stages:
            bundle = stage(bundle, ctx=self.ctx)
        # The final stage (OptimizeStage or PreviewStage if added) returns
        # an OptimizerBundle or a (OptimizerBundle, PreviewBundle) tuple.
        return self._collect_step_result(bundle, step)

    def _collect_step_result(
        self, terminal: OptimizerBundle, step: int,
    ) -> StepResult:
        return StepResult(
            step=step,
            is_eval=False,
            total_loss=terminal.total_loss_value,
            terms=terminal.terms,
            extras=terminal.extras,
            lr=terminal.lr,
            grad_norm=terminal.grad_norm,
            keep_preview=False,
            preview=None,
        )


class EvalPipeline:
    """Validates. Runs eval stages in order over a list of sequences;
    accumulates ValidationBundles into a single VideoLogBundle. Same
    bundle vocabulary, different stage list."""

    def __init__(self, stages: list[Stage], ctx: PipelineContext):
        self.stages = stages
        self.ctx = ctx

    @torch.no_grad()
    def run_validation(self, step: int) -> VideoLogBundle:
        accumulated: list[ValidationBundle] = []
        for sequence in self.ctx.eval_sequences:
            bundle = NullBundle(step=step, is_eval=True)
            for stage in self.stages:
                bundle = stage(bundle, ctx=self.ctx)
                if isinstance(bundle, ValidationBundle):
                    accumulated.append(bundle)
                    break
        # The final stage is MediaPayloadStage; it consumes the list.
        return self.stages[-1](tuple(accumulated), ctx=self.ctx)
```

## Pipeline configs

Each trainer "kind" today becomes a stage list. Most stages are shared.

### Single-cam baseline (F=3, RGB)

```python
SINGLE_CAM_F3_BASELINE: list[StageFactory] = [
    lambda ctx: SampleStage(SingleClipSampler(ctx.cfg.data)),
    lambda ctx: LiveEncoderModelInputStage(),
    lambda ctx: ForwardStage(),
    lambda ctx: RenderStage(mode=ctx.cfg.render.renderer),
    lambda ctx: ComposeStage(
        background_policy="random_per_step",
        view_condition=ctx.cfg.colorize.view_condition,
        detach_view_condition=ctx.cfg.colorize.detach_view_condition,
    ),
    lambda ctx: LossStage(loss_cfg=ctx.cfg.losses),
    lambda ctx: BackwardStage(),
    lambda ctx: OptimizeStage(),
]
```

### Single-cam F=32 alpha (the current canonical alpha-aware path)

```python
SINGLE_CAM_F32_ALPHA: list[StageFactory] = [
    lambda ctx: SampleStage(SingleClipSampler(ctx.cfg.data)),
    lambda ctx: LiveEncoderModelInputStage(),
    lambda ctx: ForwardStage(),
    lambda ctx: RenderStage(mode="fast_mac"),                  # forces v5_features path
    lambda ctx: ComposeStage(
        background_policy="random_per_step",
        view_condition=ctx.cfg.colorize.view_condition,
        detach_view_condition=ctx.cfg.colorize.detach_view_condition,
    ),
    lambda ctx: LossStage(loss_cfg=ctx.cfg.losses),
    lambda ctx: BackwardStage(),
    lambda ctx: OptimizeStage(),
]
```

The only difference from F=3 baseline is `RenderStage(mode="fast_mac")`
+ `feature_dim=32` in the model config. The compose stage, loss stage,
backward stage, and optimize stage are bit-identical.

### Multicam VJEPA alpha

```python
MULTICAM_VJEPA_ALPHA: list[StageFactory] = [
    lambda ctx: MulticamSampleStage(MulticamClipSampler(ctx.cfg.data)),
    lambda ctx: PrecomputedFeatureModelInputStage(),
    lambda ctx: ForwardStage(),
    lambda ctx: MultiViewRenderStage(mode="fast_mac"),
    lambda ctx: ComposeStage(                                  # SAME stage class as single-cam
        background_policy="random_per_step",
        view_condition=ctx.cfg.colorize.view_condition,
        detach_view_condition=ctx.cfg.colorize.detach_view_condition,
    ),
    lambda ctx: LossStage(loss_cfg=ctx.cfg.losses),            # SAME stage class
    lambda ctx: BackwardStage(),                               # SAME stage class
    lambda ctx: OptimizeStage(),                               # SAME stage class
]
```

The whole point: `ComposeStage`, `LossStage`, `BackwardStage`,
`OptimizeStage` are literally the same class instances. The multicam
path inherits alpha-aware composition + per-step random bg + the right
loss + the right backward not because we duplicated the code, but
because the multicam pipeline shares the post-render half of the
single-cam pipeline. The Apr 29 bug becomes a class of bug that cannot
exist: there is no longer a multicam-specific compose path to forget to
update.

### Single-cam known-camera

```python
SINGLE_CAM_KNOWN_CAMERA: list[StageFactory] = [
    lambda ctx: SampleStage(SingleClipKnownCameraSampler(ctx.cfg.data)),
    lambda ctx: LiveEncoderModelInputStage(),
    lambda ctx: KnownCameraForwardStage(),                     # different forward
    lambda ctx: RenderStage(mode=ctx.cfg.render.renderer),
    lambda ctx: ComposeStage(background_policy="random_per_step", ...),
    lambda ctx: LossStage(loss_cfg=ctx.cfg.losses.with_camera_weights_zero()),
    lambda ctx: BackwardStage(),
    lambda ctx: OptimizeStage(),
]
```

The known-camera path differs only in the forward stage and the loss
config (camera regularization weights are forced to zero — the model
isn't learning cameras). Same compose / backward / optimize.

### Eval pipeline

```python
EVAL_SINGLE_CAM: list[StageFactory] = [
    lambda ctx: SampleStage(EvalClipSampler(ctx.cfg.data)),
    lambda ctx: LiveEncoderModelInputStage(),
    lambda ctx: ForwardStage(),
    lambda ctx: RenderStage(mode=ctx.cfg.render.renderer),
    lambda ctx: ComposeStage(background_policy="white", ...),  # only difference
    lambda ctx: MetricStage(),
    lambda ctx: MediaPayloadStage(),
]

EVAL_MULTICAM: list[StageFactory] = [
    lambda ctx: MulticamSampleStage(MulticamEvalSampler(ctx.cfg.data)),  # samples heldout views
    lambda ctx: PrecomputedFeatureModelInputStage(),
    lambda ctx: ForwardStage(),
    lambda ctx: MultiViewRenderStage(mode="fast_mac"),
    lambda ctx: ComposeStage(background_policy="white", ...),  # white for eval reproducibility
    lambda ctx: MetricStage(),
    lambda ctx: MediaPayloadStage(),
]
```

The training pipelines and eval pipelines share `RenderStage` /
`ComposeStage` / `LossStage`-or-`MetricStage`. The asymmetry "training:
random; eval: white" is a single-flag difference at the registry layer.

## Configuration schema

A config picks a pipeline by name. All other fields stay; only the
trainer-class field goes away (replaced by `pipeline:`).

```jsonc
// src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc
{
  "pipeline": "multicam_vjepa_alpha",   // <-- new field; replaces arch + launcher script choice
  "data": { ... },
  "model": { ... },
  "render": { ... },
  "compose": {                          // <-- new section; was scattered between trainer and render.fast_mac
    "background_policy": "random_per_step",
    "eval_background_policy": "white",
    "view_condition": "camera_center_ray",
    "detach_view_condition": true
  },
  "losses": { ... },
  "logging": { ... }
}
```

Resolved by `pipeline/registry.py`:

```python
PIPELINE_REGISTRY: dict[str, list[StageFactory]] = {
    "single_cam_F3_baseline": SINGLE_CAM_F3_BASELINE,
    "single_cam_F32_alpha": SINGLE_CAM_F32_ALPHA,
    "single_cam_known_camera": SINGLE_CAM_KNOWN_CAMERA,
    "single_cam_precomputed_F32_alpha": SINGLE_CAM_PRECOMPUTED_F32_ALPHA,
    "multicam_vjepa_alpha": MULTICAM_VJEPA_ALPHA,
    "multicam_vjepa_alpha_eval": MULTICAM_VJEPA_ALPHA_EVAL,
}

EVAL_REGISTRY: dict[str, list[StageFactory]] = {
    "single_cam_F3_baseline": EVAL_SINGLE_CAM,
    "single_cam_F32_alpha": EVAL_SINGLE_CAM,
    "single_cam_known_camera": EVAL_SINGLE_CAM_KNOWN_CAMERA,
    "single_cam_precomputed_F32_alpha": EVAL_SINGLE_CAM_PRECOMPUTED,
    "multicam_vjepa_alpha": EVAL_MULTICAM,
}

def build_pipeline(cfg: ResolvedConfig, ctx: PipelineContext) -> Pipeline:
    factories = PIPELINE_REGISTRY[cfg.pipeline]
    return Pipeline([f(ctx) for f in factories], ctx)

def build_eval_pipeline(cfg: ResolvedConfig, ctx: PipelineContext) -> EvalPipeline:
    factories = EVAL_REGISTRY[cfg.pipeline]
    return EvalPipeline([f(ctx) for f in factories], ctx)
```

The launcher script becomes one bash script:

```bash
# src/train_scripts/train.sh
#!/usr/bin/env bash
exec uv run python -m src.train.trainer "$@"
```

`src/train/trainer.py` is ~80 lines: parse config path, resolve config,
build context, build pipeline, build eval pipeline, run train loop.

## Migration plan

The five-step plan. Each step is committable in isolation; each
includes a smoke test.

### Step 1: Add the bundle module + stage protocol (no behavior change)

- Create `src/train/pipeline/bundles.py` with all the dataclasses above.
- Create `src/train/pipeline/protocol.py` with the Stage protocol.
- Create `src/train/pipeline/context.py` with PipelineContext.
- Add unit tests that construct each bundle with valid inputs and verify
  the invariants in the docstrings (shape checks, alpha-vs-feature_dim
  consistency, parent-pointer integrity).
- **Files changed**: 4 new files. Zero existing files changed.
- **Smoke test**: `pytest tests/test_pipeline_bundles.py`. No training
  smoke needed.
- **Safe in isolation**: yes. Old trainers still run.

### Step 2: Implement single-cam F=3 pipeline + driver, run as a side-by-side comparison

- Implement `SampleStage`, `LiveEncoderModelInputStage`, `ForwardStage`,
  `RenderStage`, `ComposeStage`, `LossStage`, `BackwardStage`,
  `OptimizeStage`.
- Implement `Pipeline.run_step`.
- Implement `pipeline/registry.py` with `SINGLE_CAM_F3_BASELINE`.
- Add `src/train/trainer.py` as a thin entry point. It accepts a
  `--use-pipeline` flag; when off, defers to the existing
  `Trainer.run`. When on, builds the pipeline.
- Run `local_mac_overfit_video_token_smoke.jsonc` with both paths;
  diff the StepResult fields after step 0, 1, 100. Same loss curve,
  same scalar payload.
- **Files changed**: ~12 new files (bundles + stages + driver + registry +
  trainer entry). One existing file (`train_video_token_implicit_dynamic.py`)
  gets a small `if cfg.use_pipeline` shim at the top of `Trainer.run`.
- **Smoke test**: 100-step overfit, both paths, diff scalars. Plus W&B
  log compare on a longer run.
- **Safe in isolation**: yes. The default path is unchanged.

### Step 3: Port single-cam F=32 alpha + verify the alpha-aware composition matches

- The `RenderStage(mode="fast_mac")` + `ComposeStage(random_per_step)`
  combo becomes the F=32 path.
- Run `local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc`
  with `--use-pipeline` and compare the recon loss curve, alpha-mean
  diagnostic, and the W&B Render_Composite_Video panel against the old
  trainer.
- **Files changed**: register `SINGLE_CAM_F32_ALPHA` in the registry.
  Verify `feature_dim=32`'s flow through `RenderedBundle.features` and
  `ComposeStage._compose_one` with `colorize`. Add unit tests for
  `RenderedBundle` invariants (alpha is a `[T, H, W]` tensor when
  fast_mac F!=3; alpha is `None` when fast_mac F=3; alpha is `None` for
  dense/tiled/taichi).
- **Smoke test**: 400-step F=32 alpha run. Loss curve must match within
  1% of the old trainer's (random_bg seeding makes exact match hard;
  a fixed-seed test is a separate check).
- **Safe in isolation**: yes. Old trainer still default.

### Step 4: Port multicam — fixes the Apr 29 bug as a side effect

- Implement `MulticamSampleStage`, `MultiViewRenderStage`,
  `MultiCamComposedBundle`. The `ComposeStage` is the same instance as
  in single-cam.
- Run `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc`
  with `--use-pipeline`. Verify it does not crash on the
  tuple-vs-tensor bug (it can't — there is no tuple). Verify recon
  loss is finite. Verify the Render_Composite_Video panel exists per
  view (the multicam trainer didn't have this panel; this is a feature
  add).
- **Files changed**: 2 new stage classes; register
  `MULTICAM_VJEPA_ALPHA` in the registry. Add multicam-specific
  bundle invariant tests.
- **Smoke test**: a 16-step multicam smoke that exercises both train
  and eval pipelines. Asserts: the bundle is `MultiCamComposedBundle`
  after compose; `final_rgb_per_view` is non-empty;
  `reconstruction_loss_per_image` was called per view; the W&B media
  payload includes Heldout panels.
- **Safe in isolation**: yes. The multicam trainer's bug is currently
  blocking F!=3 multicam regardless; this step is the fix.

### Step 5: Port remaining trainers; delete old files

- Port: known-camera, precomputed-feature, prebaked-camera, single-image
  overfit, image-encoder baseline.
- For each: register a pipeline name; smoke-run the canonical config;
  diff against the old path.
- Delete: `train_video_token_implicit_dynamic.py`,
  `train_precomputed_feature_implicit_dynamic.py`,
  `train_multicam_precomputed_feature_implicit_dynamic.py`,
  `train_camera_implicit_dynamic.py`,
  `train_ltx_feature_implicit_dynamic.py`,
  `train_image_encoder_implicit_camera_baseline.py`,
  `train_camera_implict_dynamic.py` (typo file),
  `dynamicTokenGS.py` after migrating its four shared helpers
  (`pick_device`, `fast_attn_context`, `configure_fast_attn`,
  `select_window_indices`) into `src/train/runtime.py`.
- Delete shim files: `dynamicTokenGS_shared.py`,
  `dynamicTokenGS_tiled.py`, `tokenGS_shared.py`, `tokenGS_tiled.py`.
- Update launcher scripts to call `train.sh` with a config path; delete
  per-trainer launcher scripts that no longer apply.

## What gets deleted

| File | Replaced by |
|---|---|
| `src/train/train_video_token_implicit_dynamic.py` (~2072 lines) | `src/train/pipeline/{bundles, stages/*, pipeline}.py` (~800 lines total estimated) + `src/train/trainer.py` (~80 lines) |
| `src/train/train_precomputed_feature_implicit_dynamic.py` (~165 lines) | `PrecomputedFeatureModelInputStage` (~60 lines) |
| `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` (~386 lines) | `MulticamSampleStage` + `MultiViewRenderStage` (~150 lines total) |
| `src/train/train_ltx_feature_implicit_dynamic.py` (32 lines, empty subclass) | (deleted; no replacement needed) |
| `src/train/train_camera_implicit_dynamic.py` (~417 lines) | image-encoder baseline pipeline registry entry |
| `src/train/train_image_encoder_implicit_camera_baseline.py` | (deleted; no replacement needed) |
| `src/train/train_camera_implict_dynamic.py` (typo) | (deleted) |
| `src/train/dynamicTokenGS.py` (~731 lines) | a small `src/train/runtime.py` (~100 lines for the four shared helpers) + a registry entry for the prebaked-camera pipeline |
| `src/train/dynamicTokenGS_shared.py`, `dynamicTokenGS_tiled.py` | (deleted) |
| `src/train/tokenGS_shared.py`, `tokenGS_tiled.py` | (deleted) |
| `src/train/tokenGS.py` (~146 lines) | a single-image-overfit registry entry; `SingleImagePipeline` (~80 lines) |

Net: roughly 4,400 lines of trainer code deleted, replaced by roughly
1,200 lines of bundle + stage + pipeline code. The shared model heads,
camera, render, loss, dataset, and colorize modules are untouched.

## Test surface

This refactor enables a class of test that the current monolithic
trainers cannot support: per-stage tests on synthetic bundles. The
test_surface should grow significantly.

### Bundle invariant tests (`tests/pipeline/test_bundle_invariants.py`)

- `RenderedBundle` with `alpha=None` and a fast_mac F!=3 mode must raise
  in the constructor (we make this explicit via a `__post_init__`).
- `RenderedBundle` with `alpha.shape != features.shape[2:]` raises.
- `MultiCamComposedBundle` with different shapes per view raises.
- `LossBundle.total` requires_grad must be True when `is_eval=False`.

### Stage independence tests (`tests/pipeline/test_stage_*.py`)

Each stage test constructs a synthetic input bundle (small fake tensors,
~4 frames, 32x32 resolution, 16 splats) on CPU and asserts the output
bundle's invariants hold. For example, `test_compose_stage`:

- input: `RenderedBundle(features=[T=4, F=32, 32, 32], alpha=[T=4, 32, 32])`
- with `ComposeStage(background_policy="white")`
- output assertion: `final_rgb in [0, 1]`, `background.item() == 1.0`,
  `final_rgb.shape == (4, 3, 32, 32)`.

This was structurally impossible with the monolithic trainer because
"compose" was a 25-line block inside `recon_backward`.

### Pipeline integration test (`tests/pipeline/test_pipeline_smoke.py`)

- A 5-step run of `SINGLE_CAM_F3_BASELINE` on a 4-frame fixture
  (resused from `tests/fixtures/lalaland_short.npz`); asserts:
  - loss strictly decreases over the 5 steps
  - `StepResult.terms["recon"]` is finite at every step
  - `ctx.optimizer.state` accumulates Adam momentum
- A 5-step run of `MULTICAM_VJEPA_ALPHA` on a tiny synthetic multicam
  fixture; asserts the same — and asserts that the
  `MultiCamComposedBundle` was constructed at every step (regression
  guard against re-introducing the Apr 29 silent-tuple bug).

### Architectural invariant tests

- A `mock.patch("src.train.rendering.render_gaussian_frames",
  side_effect=AssertionError)` test that asserts the new pipeline never
  calls the legacy alpha-stripping render entry point on the F=32 alpha
  path. This is the regression guard for "trainer bypassed
  alpha-aware composition" in stage form.

## Risk analysis

Honest tradeoffs:

1. **Largest refactor of the three.** ~4,400 lines deleted, ~1,200 lines
   added. Every active config has to be re-run for parity. The migration
   is staged precisely because the all-at-once flag-day is too risky.
2. **Bundle proliferation.** ~10 frozen dataclass types, plus
   `SingleCamComposedBundle` / `MultiCamComposedBundle` split, plus
   `AuxiliaryDecode`. New contributors will spend their first hour
   reading `bundles.py`. Mitigation: aggressive docstrings on each
   bundle (already drafted in this proposal); a `bundles.md` cheat
   sheet in the same dir.
3. **Stages-as-classes lose IDE refactor support across stage
   boundaries.** Renaming a field on `RenderedBundle` is a one-spot
   edit (the dataclass) but every consumer's destructuring code must
   be updated by hand. Mitigation: dataclass field access is
   typecheckable; pyright/mypy will catch the rename mismatches at
   the Stage boundary.
4. **The chunked backward is awkward.** The current code does
   `backward_loss.backward(retain_graph=not is_last_chunk)` inside a
   loop; the typed-dataflow model wants stages to produce a single
   bundle out, not N. Resolution: the chunked path becomes a nested
   `ChunkedRecoBackwardPipeline` driven by the outer `BackwardStage`.
   This adds one additional pipeline-driver class but localizes the
   chunking concern. Alternative: declare the framewise/microbatch
   strategies dead and force `batched` everywhere. The session note
   on the Apr 29 multicam fix already noted the multicam trainer
   uses `batched` exclusively — this may be the right answer.
5. **Bundle-typed pipeline is harder to gradually adopt.** Unlike
   strategy-pattern (Proposer 2) or pure-helper (Proposer 1), the
   typed-dataflow pipeline only really works once enough stages
   exist to cover one full pipeline. There is no "extract one
   helper" win. Mitigation: Step 2 of the migration plan (single-cam
   F=3 baseline) is the smallest viable pipeline; once that runs,
   incremental adds are mechanical.
6. **PipelineContext is a god object in disguise.** It carries
   model, optimizer, RNG, config, multicam_bundle, sequences,
   eval_sequences, feature_cache, camera_rig. The justification:
   these are the exact things that cannot live in bundles because
   they have lifetime exceeding one step. Mitigation: keep the
   context minimal; explicitly forbid stages from adding fields to
   it; document its membership in the codebase guide.
7. **Risk of over-engineering.** Are 10 bundle types pulling weight,
   or are they bureaucratic? My answer: each bundle named in this
   proposal has a real consumer with a real consumed field. Drop a
   bundle and a consumer must reach across multiple stages — exactly
   the implicit dataflow the proposal is trying to remove. But this
   is a real risk; the implementation pass should be willing to
   collapse a bundle if it has no daylight from its predecessor.

## Tradeoffs vs the other two proposals

vs Proposer 1 (pure-function helpers):

- I give up small-grained reuse. A helper like `compose_rendered_rgb`
  could be lifted out tomorrow with zero refactoring of trainers; my
  proposal forces every caller into the pipeline framework.
- I gain enforced dataflow. Proposer 1's helpers can still be called
  in the wrong order, with the wrong arguments, by the wrong trainer.
  My pipeline is checked at construction: if a stage list is
  inconsistent, the bundle types do not unify.
- Proposer 1's smaller blast radius makes it more pragmatic for a
  team that is mid-research and not ready for a structural rewrite.

vs Proposer 2 (strategy / Protocol composition):

- Proposer 2 keeps the trainer class shape but factors out
  composition/render/loss as plug-in protocols. Lower migration cost
  than mine; preserves the "trainer is the agent" mental model.
- I argue that the bug class the Apr 29 session uncovered (multicam
  composition path silently bypassed) is the kind of bug that
  protocol composition does not prevent: the multicam trainer has a
  protocol slot for "compose", that slot was filled with the wrong
  implementation, and nothing in the type system caught it. My
  pipeline catches it because there is only one ComposeStage and one
  way to wire it.
- The cost: my proposal forces the bundle abstraction onto every
  contributor. Proposer 2 keeps the contributor interface "implement
  a Trainer subclass," which is what every contributor already knows.

vs both: my proposal is the only one that makes the dataflow
typecheckable end to end. The others are honest about the data shape;
mine encodes it.

## What this proposal commits to

- Bundles are immutable. No `bundle.foo = bar` anywhere in stage
  bodies. All stage outputs are constructed with `BundleType(...)`.
- The pipeline is data, not code. Adding a new "trainer" is a registry
  entry, not a new class.
- Every stage interface is one bundle in, one bundle out. No multi-arg
  call signatures. No optional callbacks.
- The Apr 29 bug class (multicam silently bypassing alpha-aware
  composition) becomes structurally impossible: there is one
  `ComposeStage`, the multicam pipeline uses it, and there is no
  multicam-only compose path to forget to update.

End of proposal.
