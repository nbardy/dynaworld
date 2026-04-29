# Proposal 02 — Composition Over Inheritance

> Wave 2 / Proposer 2 of three. Sibling proposals: 01 (pure-function helpers) and 03 (typed dataflow pipeline DSL). This proposal targets the same problem space but from a different angle.

## TL;DR

- Replace the entire trainer subclass tree with **one concrete `Trainer` class** that takes a tuple of pluggable strategy objects. No inheritance. No subclasses.
- The bug we just fixed (multicam `step` overriding the parent and silently bypassing alpha-aware composition) becomes structurally impossible: a multicam config swaps the `RenderHarness` strategy, not the trainer class. Strategies cannot "skip" compositional steps because each strategy owns exactly one decision.
- Six strategy `Protocol`s carve the trainer at the right joints: `ClipSampler`, `RenderHarness`, `LossFn`, `Validator`, `VideoLogger`, `ModelBuilder`. The `Trainer` class becomes a ~200-line shell that wires them together.
- All seven trainer files (`train_video_token_implicit_dynamic.py` (`Trainer`+`KnownCameraTrainer`), `train_precomputed_feature_implicit_dynamic.py`, `train_multicam_precomputed_feature_implicit_dynamic.py`, `train_ltx_feature_implicit_dynamic.py`, `train_camera_implicit_dynamic.py`, `dynamicTokenGS.py`, `tokenGS.py`) collapse into one trainer + a strategy registry. Trainer "kinds" become *configurations* of strategy combos.
- `key_learnings.md:18` warns "A single shared `BaseTrainer` would hide real differences..." — this proposal honours that note. `Protocol` is **structural**, not nominal: there is no shared base class to leak invariants. Each strategy is an independent unit; the contract is the protocol, not the lineage.

## Module layout

```
src/train/
├── trainer.py                       # the one Trainer class (~200 lines)
├── pipeline_strategies/
│   ├── __init__.py                  # re-exports + Protocol typing
│   ├── types.py                     # dataclasses flowing between strategies
│   ├── sampler.py                   # ClipSampler protocol + 3 impls
│   ├── render_harness.py            # RenderHarness protocol + 3 impls
│   ├── loss_fn.py                   # LossFn protocol + 2 impls
│   ├── validator.py                 # Validator protocol + 3 impls
│   ├── video_logger.py              # VideoLogger protocol + 2 impls
│   └── model_builder.py             # ModelBuilder protocol + 2 impls
├── pipeline_registry.py             # build_strategies_from_config + name table
├── pipeline_strategies_legacy/      # 1-call-site adapters for the 3 legacy paths
│   ├── prebaked_camera.py           # absorbs dynamicTokenGS.py
│   ├── image_encoder_baseline.py    # absorbs train_camera_implicit_dynamic.py
│   └── single_image_overfit.py      # absorbs tokenGS.py
├── train_runtime.py                 # pick_device / fast_attn_context / select_window_indices (lifted out of dynamicTokenGS.py so legacy file can die)
├── runtime_types.py                 # GaussianSequence (unchanged)
├── colorize.py                      # FeatureToColor (unchanged module; ownership moves to RenderHarness)
├── losses.py                        # reconstruction_loss_per_image (unchanged kernel)
└── ...
```

What's gone:

- `train_video_token_implicit_dynamic.py` (the 2 072-line monolith): renamed to `trainer.py`, body collapses to ~200 lines after extraction; module-level helpers (`render_clip_sequence`, `viewport_cameras`, `colorize_view_dirs_for_features`) move into the relevant strategy modules.
- `train_precomputed_feature_implicit_dynamic.py`, `train_multicam_precomputed_feature_implicit_dynamic.py`, `train_ltx_feature_implicit_dynamic.py`, `train_camera_implicit_dynamic.py`, `dynamicTokenGS.py`, `tokenGS.py`: all become strategy combos. The files themselves are deleted; their distinctive logic survives in named strategies under `pipeline_strategies/`.
- `KnownCameraTrainer`: deleted. Its only structural difference (cameras come from `sequence_data` not `decoded`) becomes a `KnownCameraSampler` returning a `ClipBatch` with `provided_cameras` populated, plus the model-builder selecting `DynamicVideoTokenGSKnownCamera`.
- `train_image_encoder_implicit_camera_baseline.py`, `train_camera_implict_dynamic.py` (typo), `dynamicTokenGS_shared.py`, `dynamicTokenGS_tiled.py`, `tokenGS_shared.py`, `tokenGS_tiled.py`: deleted unconditionally.

## Wire-format dataclasses

These are the typed bundles flowing between strategies. Every field has a type and a documented invariant. The Trainer never inspects fields it cannot pass straight through — the invariants live in the constructor of each dataclass.

### `ClipBatch` — `ClipSampler.next_clip()` output

```python
@dataclass(frozen=True)
class ClipBatch:
    """One training step's worth of clip data + bookkeeping.

    Output of ClipSampler.next_clip(); input of RenderHarness.render() and
    LossFn.compute(). The Trainer never constructs this directly.
    """

    sequence_data: SequenceData
    """The full sequence the clip was sampled from. Carries cameras, GT
    frames, fps. RenderHarness uses sequence_data.cameras as the default
    camera source unless provided_cameras is set."""

    clip_indices: torch.Tensor
    """Long tensor [T] of frame indices into sequence_data. Used for
    indexing precomputed feature caches and GT frames."""

    clip_frames: torch.Tensor
    """[T, 3, H, W] GT pixel frames. Already at model-input resolution
    (model_cfg.size). RenderHarness re-resizes to render_size internally."""

    clip_times: torch.Tensor
    """[T] (or [T, F] for multicam) normalized time in [0, 1]. Fed to the
    model as decode_times."""

    views: tuple[int, ...] | None
    """Multicam-only. Indices into sequence_data.train_views. None for
    single-cam configs."""

    provided_cameras: tuple[CameraSpec, ...] | None
    """Optional ground-truth cameras (known-camera training). When set,
    RenderHarness uses these instead of decoded.cameras. None for
    implicit-camera training."""

    model_input: torch.Tensor | dict[str, torch.Tensor]
    """Whatever the ModelBuilder declared the model wants. Live encoder:
    clip_frames. Precomputed: cached features keyed by clip_indices.
    Multicam: per-view cached features. Opaque to the Trainer."""

    extras: Mapping[str, Any] = field(default_factory=dict)
    """Strategy-specific bookkeeping (cache hit counts, sampling flags,
    multicam_bundle handle, etc.). Passed to the VideoLogger so it can
    surface diagnostic counters."""
```

Invariants (enforced in `__post_init__`):

- `clip_indices.dim() == 1` and `clip_indices.dtype in {int32, int64}`.
- `clip_frames.shape[0] == clip_indices.shape[0]`.
- `(views is None) == (sequence_data.is_multicam is False)` — a multicam sampler must populate `views`; a single-cam one must not.
- `(provided_cameras is None) or (len(provided_cameras) == clip_indices.shape[0])`.

### `DecodedSplats`

```python
@dataclass(frozen=True)
class DecodedSplats:
    """Output of model.forward(...). Reuses the existing GaussianSequence
    type as the underlying carrier; this dataclass adds nothing — the
    name exists to make pipeline call sites read better.
    """

    sequence: GaussianSequence
    """Includes xyz, scales, quats, opacities, rgbs (= F-channel features
    when feature_dim != 3; the field name is grandfathered), cameras,
    camera_state, auxiliary."""
```

We do not introduce a new bag — `GaussianSequence` already carries everything. The wrapper exists so `RenderHarness.render(decoded: DecodedSplats, ...)` reads as a typed seam rather than `render(decoded: Any, ...)`.

### `RenderedClipBundle` — `RenderHarness.render()` output

```python
@dataclass(frozen=True)
class RenderedClipBundle:
    """Composited per-clip render. RenderHarness owns alpha composition,
    colorize MLP, and per-step random background. By the time this leaves
    the strategy, the rendered tensor is in the colorimetric space the
    LossFn expects (RGB-3 for standard recon loss, F-channel pass-through
    for any future direct-feature loss)."""

    rgb: torch.Tensor
    """[T, 3, H, W] (single-cam) or [V, T, 3, H, W] (multicam). Already
    composited against the chosen background. This is what the LossFn
    consumes."""

    alpha: torch.Tensor | None
    """[T, H, W] or [V, T, H, W]. None when the renderer cannot expose
    alpha (dense / tiled / taichi modes; F=3 v5 path). Used by the
    VideoLogger for the alpha-mask panel; LossFn never reads this."""

    cameras: tuple[CameraSpec, ...] | tuple[tuple[CameraSpec, ...], ...]
    """Cameras used for the render, in the same order as the rgb tensor's
    leading dim(s). Multicam: tuple-of-tuples per view. Single-cam:
    flat tuple."""

    pre_composite_features: torch.Tensor | None
    """The raw F-channel rasterizer output before colorize, kept ONLY
    when VideoLogger is the FeaturePcaVideoLogger. Otherwise None to free
    memory."""

    aux: Mapping[str, Any] = field(default_factory=dict)
    """Diagnostic side-channel. Bank-rate auxiliary, dense-renderer
    return_aux dict, debug overlays. Trainer threads it into the
    LossFn and VideoLogger."""
```

Invariants:

- `rgb.shape[-3] == 3` always. The colorize step is mandatory inside `RenderHarness`; no caller ever sees raw F-channel features unless they explicitly opt in via `pre_composite_features`.
- `(alpha is None) or (alpha.shape[:-2] == rgb.shape[:-3])` — leading dims must match.

### `LossOutput`

```python
@dataclass(frozen=True)
class LossOutput:
    """Output of LossFn.compute(). Trainer calls .backward() on .total
    and posts .breakdown to the VideoLogger / W&B scalar payload."""

    total: torch.Tensor
    """Scalar tensor. Has grad. This is what gets backpropagated."""

    breakdown: Mapping[str, torch.Tensor]
    """Per-term scalar tensors for logging. May be detached or live;
    VideoLogger calls .detach() before aggregating. Required keys:
    'recon_loss'. Conventional keys: 'camera_loss', 'bank_rate_loss',
    'rig_loss'. New strategies may add their own keys; logger is
    permissive."""

    backward_strategy: BackwardStrategy
    """Tells the Trainer how to backprop:
    - SingleShot: one .backward() on .total, then optimizer.step().
    - Chunked(chunks): N partial backwards with retain_graph=True except
      on last; Trainer obeys the chunks list. This is the only place
      the temporal_microbatch_size knob lives.

    The Trainer does not pick the strategy; the LossFn does, because
    chunking is a property of how the loss was assembled (per-frame vs
    full-clip)."""
```

`BackwardStrategy` is a small sum:

```python
@dataclass(frozen=True)
class SingleShot:
    pass

@dataclass(frozen=True)
class Chunked:
    """Sequence of pre-computed scalar losses, in order. Trainer calls
    .backward(retain_graph=True) on every entry except the last."""
    chunk_losses: list[torch.Tensor]
    last_step_extras: torch.Tensor | None  # camera/bank reg added to last chunk

BackwardStrategy = SingleShot | Chunked
```

This is the part the bug we fixed lived in. The single-cam path used per-chunk backward inside `recon_backward`; the multicam path used `loss.backward()` once at the bottom. In this proposal, that decision is *the LossFn's*, not the trainer's. The Trainer just receives a `LossOutput` and follows the recipe.

### `StepResult`

`StepResult` already exists in `runtime_types.py`. It survives unchanged except: it now contains the `LossOutput.breakdown` directly (instead of named scalar fields like `camera_motion_loss`, `bank_rate_loss`). Logging stays uniform via the breakdown dict.

```python
@dataclass(frozen=True)
class StepResult:
    step: int
    decoded: DecodedSplats
    rendered: RenderedClipBundle
    loss: LossOutput
    elapsed_ms: float
```

### `ValidationPayload` — `Validator.compute()` output

```python
@dataclass(frozen=True)
class ValidationPayload:
    """Output of Validator.compute_metrics(). VideoLogger consumes this
    plus the freshly-rendered clip bundle to build the W&B post."""

    scalar_metrics: Mapping[str, float]
    """Eval/L1, Eval/PSNR, Eval/DSSIM, etc. Keys are W&B-style namespaced.
    Multicam validators add 'TrainView{i}/Eval/PSNR' and
    'Heldout{i}_{name}/Eval/PSNR' as needed."""

    rendered_clips: Mapping[str, RenderedClipBundle]
    """Keyed by panel name (e.g. 'train_view_0', 'heldout_canon_l').
    VideoLogger renders each into a panel."""

    gt_clips: Mapping[str, torch.Tensor]
    """Same keys as rendered_clips, value is [T, 3, H, W] GT."""

    sequence_fps: float

    extras: Mapping[str, Any] = field(default_factory=dict)
    """Anything else the VideoLogger might want (PCA basis, alpha
    masks, decoded temporal payload). Strategy-specific."""
```

## The six Protocols — full specs

`Protocol` types (`typing.Protocol`, `runtime_checkable=False` — purely structural). Implementations do not inherit from these; they just match the shape.

### `ClipSampler`

```python
class ClipSampler(Protocol):
    """Produces training clips. The Trainer calls next_clip() once per
    step. Owns: train manifest, eval manifest, sequence-data lifecycle,
    feature cache (when relevant), view sampling for multicam.

    Lifecycle:
        sampler = ClipSampler(...)
        sampler.load(device)               # called once at trainer __init__
        for step in range(steps):
            batch = sampler.next_clip()    # called per step
            ...
        sampler.eval_sequences()           # called for validation
    """

    def load(self, device: torch.device) -> None:
        """Materialize sequences onto device. Builds the feature cache if
        the sampler is a precomputed-feature variant. Called once."""
        ...

    def next_clip(self) -> ClipBatch:
        """Returns a typed bundle of clip frames + cameras + bookkeeping.
        May sample a different sequence each call (manifest training); may
        sample views (multicam). Trainer treats this opaquely."""
        ...

    def sequence_count(self) -> int:
        """For progress reporting. Number of distinct sequences this
        sampler will visit across an epoch."""
        ...

    def eval_sequences(self) -> Sequence[SequenceData]:
        """Sequences to run validation on. Single-cam returns the eval
        manifest entries; multicam returns the single multicam bundle."""
        ...

    def feature_cache_handle(self) -> FeatureCacheHandle | None:
        """For VideoLogger diagnostic counters (cache hit rate). None if
        no cache is in play."""
        ...
```

### `RenderHarness`

```python
class RenderHarness(Protocol):
    """Owns everything between decoded splats and a 3-channel image.
    Specifically: viewport scaling, the rasterizer call, the colorize
    MLP, the alpha-aware composition, AND the per-step random
    background sampling.

    The composition + random-bg + colorize triplet is THE LOAD-BEARING
    DUPLICATION the multicam trainer was missing. Putting it inside the
    RenderHarness means a multicam config gets it the moment it picks
    the multicam-flavored harness, and CANNOT bypass it.

    Multiple harness implementations differ ONLY in:
    - how cameras are picked per render call (single-cam: decoded.cameras;
      multicam: rig.cameras_for_view; known-camera: clip_batch.provided_cameras)
    - how many renders happen per call (1 vs V)
    - whether the loop is wrapped in a chunked-backward generator
    """

    def __init__(
        self,
        *,
        renderer_cfg: RenderConfig,
        colorize: FeatureToColor | None,
        train_chunk_size: int | Literal["framewise", "batched"],
    ) -> None:
        ...

    def render(
        self,
        decoded: DecodedSplats,
        clip_batch: ClipBatch,
        *,
        training: bool,
    ) -> RenderedClipBundle:
        """Renders the splats. The `training` flag controls whether to
        apply random per-step background (training=True; freshly sampled
        torch.rand(3) broadcast across all chunks) or fixed white
        (training=False).

        Multicam harnesses iterate over clip_batch.views and stack into
        the leading dim of the output tensors.

        On training=True with a chunked-backward strategy, this method
        returns one RenderedClipBundle whose rgb tensor still requires
        grad and is THE concatenation of chunk-renders; the chunked
        backward is then driven by the LossFn. (Alternative: this method
        returns a generator. Going with the eager concatenation for
        simplicity; see Risk Analysis.)
        """
        ...

    def render_full_sequence(
        self,
        decoded: DecodedSplats,
        sequence_data: SequenceData,
    ) -> RenderedClipBundle:
        """Eval-only path. Renders all frames of a full sequence (not
        clip-bounded) for video logging. Always uses fixed white bg
        (training=False). Multicam version renders all train + heldout
        views."""
        ...
```

### `LossFn`

```python
class LossFn(Protocol):
    """Computes the training loss given GT and rendered. Owns: the
    per-image recon kernel choice, camera regularizers, bank-rate
    regularizers, rig regularizers, AND the backward strategy.

    The backward-strategy decision lives here (not on the Trainer)
    because chunking is a property of how the loss was assembled.
    Single-cam standard loss returns Chunked when temporal_microbatch
    is set; multicam loss returns SingleShot.
    """

    def compute(
        self,
        rendered: RenderedClipBundle,
        clip_batch: ClipBatch,
        decoded: DecodedSplats,
    ) -> LossOutput:
        """Builds and returns the LossOutput. This method does NOT
        call .backward(); the Trainer does, following LossOutput's
        backward_strategy. The chunked variant pre-computes the per-chunk
        scalar losses inside compute() and packages them into Chunked()."""
        ...
```

### `Validator`

```python
class Validator(Protocol):
    """Owns the held-out-camera novel-view metric (multicam) or
    source-view eval (single-cam). Runs at validation cadence (every
    image_log_every / video_log_every).

    Multicam validators return per-train-view AND per-heldout-view
    metrics + clips. Single-cam validators return one set of metrics +
    clips per eval sequence."""

    def compute(
        self,
        model: nn.Module,
        sampler: ClipSampler,
        render_harness: RenderHarness,
    ) -> ValidationPayload:
        """Materializes a ValidationPayload. The Trainer wires this to
        VideoLogger.payload(). Calls into the same RenderHarness as
        training so the alpha/composition path is bit-identical between
        train and eval (only the bg differs: random vs white)."""
        ...
```

### `VideoLogger`

```python
class VideoLogger(Protocol):
    """Owns W&B media payload assembly. Single-cam logger builds:
    Render_GT_Video, optional Alpha_Mask_Video, optional Feature_PCA_Video,
    optional Render_Composite_Video. Multicam logger builds per-view
    rendered + GT panels.

    The logger is the ONLY component that knows W&B. Trainer.validate()
    calls logger.payload(...) and then calls a thin wandb.log(payload)
    at the boundary."""

    def scalar_payload(
        self,
        step: int,
        result: StepResult,
        sampler: ClipSampler,
    ) -> Mapping[str, float]:
        """Per-step scalars. Includes loss breakdown + cache stats +
        rig regularization weight + anything the strategy tracks. Called
        every log_every steps."""
        ...

    def image_payload(
        self,
        step: int,
        result: StepResult,
        sampler: ClipSampler,
    ) -> Mapping[str, Any]:
        """Periodic preview images. Single-cam: one preview from the
        most recent step. Multicam: per-view previews."""
        ...

    def video_payload(
        self,
        step: int,
        validation_payload: ValidationPayload,
    ) -> Mapping[str, Any]:
        """Periodic full-sequence videos. Builds wandb.Video objects.
        This is the panel the multicam trainer was missing. The single-cam
        logger composes Alpha_Mask_Video / Feature_PCA_Video /
        Render_Composite_Video per the existing rules; the multicam
        logger composes per-view + per-heldout-view panels."""
        ...
```

### `ModelBuilder`

```python
class ModelBuilder(Protocol):
    """Builds the model from config. Single method; lifecycle is
    one-shot at trainer __init__.

    The builder is also the answer to 'where does the FeatureToColor
    MLP live?'. The builder constructs both the model AND the colorize
    MLP, returning them as a pair, so a model variant that needs no
    colorize (F=3 RGB direct) returns (model, None) and the
    RenderHarness handles the None case explicitly."""

    def build(
        self,
        cfg: Mapping[str, Any],
        device: torch.device,
    ) -> tuple[nn.Module, FeatureToColor | None]:
        """Returns (model, optional FeatureToColor). The trainer registers
        both with its optimizer."""
        ...
```

### Helper protocol: `FeatureCacheHandle`

```python
class FeatureCacheHandle(Protocol):
    """Diagnostic surface for precomputed-feature samplers. The logger
    reads cache hit / size / version off this for telemetry."""

    def hit_count(self) -> int: ...
    def total_count(self) -> int: ...
    def cache_version(self) -> str: ...
```

## The Trainer class — full body

```python
class Trainer:
    """The one trainer. Composes six strategies; runs the training loop.

    No subclasses. No overrideable hooks. If you want different
    behavior, you swap a strategy. If a strategy combo doesn't exist,
    you write a new one (likely 30-100 lines) and register it.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = resolved_config(config)
        self.device = pick_device(self.config["train"].get("device"))
        self.train_cfg = self.config["train"]
        self.logging_cfg = self.config["logging"]

        # Strategy wiring. The registry decides; the Trainer doesn't
        # know what's inside the tuple.
        strategies = build_strategies_from_config(self.config, self.device)
        self.sampler: ClipSampler = strategies.sampler
        self.render_harness: RenderHarness = strategies.render_harness
        self.loss_fn: LossFn = strategies.loss_fn
        self.validator: Validator = strategies.validator
        self.video_logger: VideoLogger = strategies.video_logger
        self.model_builder: ModelBuilder = strategies.model_builder

        # Build the model + colorize. The builder picks the variant.
        self.model, self.colorize = self.model_builder.build(
            self.config, self.device
        )
        # The render harness needs the colorize handle; we resolve the
        # circular dep by injecting after build.
        self.render_harness.bind_colorize(self.colorize)

        # Optimizer. Includes colorize params if present. Multicam-rig
        # param group is added by the multicam ClipSampler via a
        # post-load hook (the sampler exposes .extra_param_groups()).
        params: list[Any] = list(self.model.parameters())
        if self.colorize is not None:
            params += list(self.colorize.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=self.train_cfg["lr"], fused=_fused_supported(self.device)
        )
        for group in self.sampler.extra_param_groups():
            self.optimizer.add_param_group(group)

        # Sampler load happens last so feature-cache prebake can see the
        # built model if it needs to.
        self.sampler.load(self.device)

        self.step_idx = 0
        self.steps = self.train_cfg["steps"]

    def step(self) -> StepResult:
        """One training iteration. ZERO branching on trainer kind."""
        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        clip_batch = self.sampler.next_clip()
        with fast_attn_context(self.device), self._autocast_context():
            decoded_seq = self.model(
                clip_batch.model_input,
                decode_times=clip_batch.clip_times,
                cameras=clip_batch.provided_cameras,  # None unless known-camera
            )
        decoded = DecodedSplats(sequence=decoded_seq)

        rendered = self.render_harness.render(
            decoded, clip_batch, training=True,
        )

        loss_out = self.loss_fn.compute(rendered, clip_batch, decoded)

        # Backward strategy is a property of the loss, not the trainer.
        match loss_out.backward_strategy:
            case SingleShot():
                loss_out.total.backward()
            case Chunked(chunk_losses=chunks, last_step_extras=extras):
                last = len(chunks) - 1
                for i, chunk_loss in enumerate(chunks):
                    loss_chunk = chunk_loss
                    if i == last and extras is not None:
                        loss_chunk = loss_chunk + extras
                    loss_chunk.backward(retain_graph=(i != last))

        self.optimizer.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.step_idx += 1
        return StepResult(
            step=self.step_idx,
            decoded=decoded,
            rendered=rendered,
            loss=loss_out,
            elapsed_ms=elapsed_ms,
        )

    def validate(self) -> None:
        """Run validation, post W&B logs. Called at video_log_every cadence."""
        with torch.no_grad():
            payload = self.validator.compute(
                self.model, self.sampler, self.render_harness,
            )
        media = self.video_logger.video_payload(self.step_idx, payload)
        scalars = {f"Eval/{k}": v for k, v in payload.scalar_metrics.items()}
        wandb.log({**scalars, **media}, step=self.step_idx)

    def run(self) -> None:
        """Train loop. ~25 lines."""
        # Initial step result, scalar dump, and validation at step 0
        self._log_initial()

        while self.step_idx < self.steps:
            result = self.step()

            if self._should_log_scalars():
                wandb.log(
                    self.video_logger.scalar_payload(
                        self.step_idx, result, self.sampler,
                    ),
                    step=self.step_idx,
                )

            if self._should_log_images():
                wandb.log(
                    self.video_logger.image_payload(
                        self.step_idx, result, self.sampler,
                    ),
                    step=self.step_idx,
                )

            if self._should_log_videos():
                self.validate()

            self._print_progress(result)

        self._maybe_export_browser_bundle()

    # ---- small private helpers (cadence, autocast, progress) ----

    def _autocast_context(self):
        if self.train_cfg["amp"] and self.device.type == "cuda":
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    def _should_log_scalars(self) -> bool:
        every = max(1, self.logging_cfg["log_every"])
        return self.step_idx % every == 0 or self._is_last_step()

    def _should_log_images(self) -> bool:
        every = max(1, self.logging_cfg["image_log_every"])
        return self.step_idx % every == 0 or self._is_last_step()

    def _should_log_videos(self) -> bool:
        every = max(1, self.logging_cfg["video_log_every"])
        return self.step_idx % every == 0 or self._is_last_step()

    def _is_last_step(self) -> bool:
        return (
            self.logging_cfg.get("always_log_last_step", True)
            and self.step_idx == self.steps
        )

    def _log_initial(self) -> None:
        """Step-0 sanity check + initial validation pass."""
        with torch.no_grad():
            clip_batch = self.sampler.next_clip()
            decoded_seq = self.model(
                clip_batch.model_input,
                decode_times=clip_batch.clip_times,
                cameras=clip_batch.provided_cameras,
            )
            decoded = DecodedSplats(sequence=decoded_seq)
            rendered = self.render_harness.render(
                decoded, clip_batch, training=False,
            )
            initial_result = StepResult(
                step=0, decoded=decoded, rendered=rendered,
                loss=self.loss_fn.compute(rendered, clip_batch, decoded),
                elapsed_ms=0.0,
            )
        wandb.log(
            self.video_logger.scalar_payload(0, initial_result, self.sampler),
            step=0,
        )
        self.validate()

    def _print_progress(self, result: StepResult) -> None:
        if self.step_idx % max(1, self.logging_cfg["log_every"]) == 0:
            recon = result.loss.breakdown.get("recon_loss", 0.0)
            print(
                f"step {self.step_idx}/{self.steps} "
                f"loss={float(result.loss.total):.4f} "
                f"recon={float(recon):.4f} "
                f"elapsed={result.elapsed_ms:.1f}ms"
            )

    def _maybe_export_browser_bundle(self) -> None:
        export_cfg = self.config.get("export", {})
        if export_cfg.get("enabled"):
            from export_dynaworld_browser_bundle import export
            export(self.config, self.model, self.sampler.eval_sequences())
```

That's the whole trainer. ~180 lines including the helpers. Compare to the 2 072-line current file.

## Concrete strategy combos (today's trainers, replaced)

Each row maps a current trainer to its strategy tuple. **Bold** entries indicate a strategy that is unique to this combo; everything else is shared.

| Today's trainer (file) | Sampler | RenderHarness | LossFn | Validator | VideoLogger | ModelBuilder |
|---|---|---|---|---|---|---|
| `Trainer` (single-cam, `train_video_token_implicit_dynamic.py`) | `SingleClipSampler` | `SingleViewRenderHarness` | `StandardLoss` | `SourceViewValidator` | `StandardVideoLogger` | `FromVariantModelBuilder` |
| `KnownCameraTrainer` | **`KnownCameraClipSampler`** | `SingleViewRenderHarness` | `StandardLoss` (camera weights forced 0) | `SourceViewValidator` | `StandardVideoLogger` | `FromVariantModelBuilder` (picks `DynamicVideoTokenGSKnownCamera`) |
| `PrecomputedFeatureImplicitTrainer` | **`PrecomputedFeatureSampler`** (extends single + cache prebake) | `SingleViewRenderHarness` | `StandardLoss` | `SourceViewValidator` | `StandardVideoLogger` | `FromVariantModelBuilder` |
| `MulticamPrecomputedFeatureImplicitTrainer` | **`MulticamPrecomputedSampler`** (multicam + cache prebake + rig) | **`MultiViewRenderHarness`** | **`MulticamLoss`** (per-view sum, includes rig regularization, returns SingleShot) | **`HeldoutCameraValidator`** | **`MulticamVideoLogger`** | `FromVariantModelBuilder` |
| `LTXFeatureImplicitTrainer` | `PrecomputedFeatureSampler` | `SingleViewRenderHarness` | `StandardLoss` | `SourceViewValidator` | `StandardVideoLogger` | `FromVariantModelBuilder` |
| `dynamicTokenGS.py` (legacy prebaked-camera) | **`PrebakedCameraSampler`** | `SingleViewRenderHarness` | **`PrebakedCameraLoss`** (no camera reg, has clip_grad / lr_schedule) | `SourceViewValidator` | `StandardVideoLogger` | **`PrebakedModelBuilder`** (builds `DynamicTokenGS`) |
| `train_camera_implicit_dynamic.py` (image-encoder baseline) | `SingleClipSampler` (no cache) | **`PerFrameRenderHarness`** | `StandardLoss` | `SourceViewValidator` | `StandardVideoLogger` | **`ImageEncoderModelBuilder`** (builds `DynamicTokenGSImplicitCamera` / `...Separated`) |
| `tokenGS.py` (single-image) | **`SingleImageSampler`** | `SingleViewRenderHarness` | `StandardLoss` | **`NullValidator`** (no eval) | `StandardVideoLogger` | **`SingleImageModelBuilder`** (builds `TokenGS`) |

Eight current trainers → eight strategy combos. All share the **same one** `Trainer` class. The combos differ in 1-3 strategies each.

## How alpha-aware composition + random bg lives

It lives **inside `RenderHarness`**, not as its own strategy. Reasons:

1. **The bug we just fixed proves it.** The multicam trainer's `step` method ran `loss.backward()` on a tensor that came from `render_view_clip()` which forwarded `render_clip_sequence()` which already returned `(features, alpha)` — and the multicam author bypassed the alpha branch entirely. If alpha composition is a separate `Compositor` strategy, the multicam author has to *remember* to wire it. If it's inside `RenderHarness`, swapping `SingleViewRenderHarness → MultiViewRenderHarness` automatically gets it; the multicam harness uses the same composition method internally.

2. **Composition is not orthogonal to rendering.** It needs the same per-step `random_bg`, the same `cameras` for view-conditioning, the same `colorize` MLP, and the same chunk loop. Splitting these across two strategies forces them to share too much state through a third typed bag.

3. **The view-conditioning ray dirs need camera scaling that already lives in the renderer.** `colorize_view_dirs_for_features` calls `viewport_cameras` and `camera_center_ray_dirs` — both are renderer-internal.

The `RenderHarness` Protocol method shown above takes `training: bool`, which controls whether to sample `random_bg` (training=True) or use fixed white (training=False). Internally, every harness implementation calls one shared protected helper:

```python
class _RenderHarnessBase:  # NOT exposed in pipeline_strategies/__init__.py
    """Internal mixin shared by harness implementations. NOT inheritance —
    just a shared file. Each harness composes-by-call into _composite()."""

    @staticmethod
    def _composite(
        features: torch.Tensor,           # [..., F, H, W]
        alpha: torch.Tensor | None,       # [..., H, W] or None
        *,
        colorize: FeatureToColor | None,
        cameras: tuple[CameraSpec, ...],
        background: torch.Tensor | float,
        view_condition: str,
        detach_view_condition: bool,
        input_size: int,
        render_size: int,
    ) -> torch.Tensor:
        """The one composition kernel. All four current duplicates
        (recon_backward L1357, initial_step_result L1412, render_full_sequence
        L1604, KnownCameraTrainer.render_full_sequence L1974) collapse here."""
        ...
```

Each `RenderHarness` implementation then calls `self._composite(...)` from its `render()` body. Single-view calls it once per chunk; multi-view calls it once per `(view, chunk)` pair. The shared kernel is reachable by call from every harness without inheritance.

## Configuration schema

A single top-level `pipeline:` field names the combo, plus an optional `strategies:` block that overrides individual strategies for ablation:

```jsonc
{
  // The named combo. Default and ergonomic for the 90% case.
  "pipeline": "multicam_vjepa_alpha",

  // Optional override (ablation only). If both pipeline and strategies
  // are present, strategies wins on the keys it sets.
  "strategies": {
    "render_harness": "multi_view_alpha_aware",
    "loss_fn": "l1_dssim_multicam"
  },

  "data": { ... },
  "model": { ... },
  "render": { ... },
  "train": { ... },
  "losses": { ... },
  "logging": { ... },
  "colorize": { ... },
  "features": { ... }   // optional, only when sampler == "precomputed_*"
}
```

The pipeline registry (`pipeline_registry.py`) holds the name → combo table:

```python
@dataclass(frozen=True)
class StrategyTuple:
    sampler: ClipSampler
    render_harness: RenderHarness
    loss_fn: LossFn
    validator: Validator
    video_logger: VideoLogger
    model_builder: ModelBuilder

# Each entry maps a pipeline name to a function(cfg, device) -> StrategyTuple.
PIPELINE_REGISTRY: dict[str, Callable[[Mapping, torch.device], StrategyTuple]] = {
    "single_cam_implicit_camera": _build_single_cam_implicit,
    "single_cam_known_camera": _build_single_cam_known,
    "single_cam_precomputed": _build_single_cam_precomputed,
    "multicam_vjepa_alpha": _build_multicam_alpha,
    "image_encoder_baseline": _build_image_encoder_baseline,
    "prebaked_camera_legacy": _build_prebaked_legacy,
    "single_image_overfit": _build_single_image,
    "ltx_feature_precomputed": _build_ltx_precomputed,  # alias of single_cam_precomputed
}

# Per-strategy registries for fine-grained override.
SAMPLER_REGISTRY: dict[str, Callable[[Mapping, torch.device], ClipSampler]] = {
    "single_clip": _make_single_clip_sampler,
    "known_camera": _make_known_camera_sampler,
    "precomputed_feature": _make_precomputed_feature_sampler,
    "multicam_precomputed": _make_multicam_precomputed_sampler,
    "single_image": _make_single_image_sampler,
    "prebaked_camera": _make_prebaked_camera_sampler,
}
# ... and the same for RENDER_HARNESS_REGISTRY, LOSS_FN_REGISTRY, etc.

def build_strategies_from_config(
    cfg: Mapping[str, Any],
    device: torch.device,
) -> StrategyTuple:
    """Resolve config -> StrategyTuple. Preference order:
    1. cfg['strategies'] if present, individual strategy names.
    2. cfg['pipeline'] for the combo.
    3. Fail loud on mismatch (e.g. precomputed_feature sampler with
       'features' missing from config).
    """
    pipeline_name = cfg.get("pipeline")
    overrides = cfg.get("strategies", {})

    if pipeline_name:
        base = PIPELINE_REGISTRY[pipeline_name](cfg, device)
    else:
        # All six strategies must be in overrides if no pipeline name.
        required = {"sampler", "render_harness", "loss_fn",
                    "validator", "video_logger", "model_builder"}
        missing = required - overrides.keys()
        if missing:
            raise KeyError(f"No 'pipeline' key and missing strategies: {missing}")
        base = StrategyTuple(
            sampler=SAMPLER_REGISTRY[overrides["sampler"]](cfg, device),
            render_harness=RENDER_HARNESS_REGISTRY[overrides["render_harness"]](cfg, device),
            ...
        )

    # Apply individual overrides on top of the named combo.
    for key, name in overrides.items():
        registry = _registry_for(key)
        base = replace(base, **{key: registry[name](cfg, device)})

    return base
```

The launcher script becomes a single command: `python -m train.run_trainer config.jsonc` — no more "which `.sh` do I run" lookup. The `arch` field in current configs is deleted (it was already dead per investigator 04).

## Migration plan

The 2 072-line monolith can't be flipped in one PR. The order below extracts strategies leaf-first so each PR is small and the trainer keeps running throughout.

**PR 1: Lift `pick_device`, `fast_attn_context`, `configure_fast_attn`, `select_window_indices` out of `dynamicTokenGS.py` into a new `train_runtime.py`.**

- Why first: investigator 05 calls these out as the only reason `dynamicTokenGS.py` cannot be deleted.
- Changes: new file, update 4 imports.
- Smoke: existing trainers continue to launch.
- Safe in isolation: pure refactor, no behavior change.

**PR 2: Extract `VideoLogger` from `Trainer.scalar_payload / render_preview_image / validation_video_payload`.**

- Why early: leaf of the call graph, no model / sampler / render coupling.
- Changes: new `pipeline_strategies/video_logger.py` with `StandardVideoLogger`. `Trainer` calls `video_logger.scalar_payload(...)` etc. Multicam trainer's `validation_video_payload` becomes `MulticamVideoLogger`.
- Smoke: run `local_mac_overfit_video_token_smoke.jsonc` and confirm W&B panels are bit-identical.
- Safe in isolation: trainer subclasses still exist; we just pass them a logger.

**PR 3: Extract `LossFn` from `recon_backward + compute_camera_losses + build_bank_rate_loss`.**

- Why next: the load-bearing duplication. Pulling this out forces `Chunked` vs `SingleShot` to be a typed value, which exposes the multicam trainer's gap immediately.
- Changes: `pipeline_strategies/loss_fn.py` with `StandardLoss` (Chunked when temporal_microbatch is set; SingleShot otherwise) and `MulticamLoss` (SingleShot, includes rig). `Trainer.recon_backward` becomes a private dispatcher driven by `LossOutput.backward_strategy`.
- Smoke: 30-step overfit on `local_mac_overfit_video_token_smoke.jsonc` (single-cam) and `local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_..._16f_8192splats.jsonc` (multicam). Same loss curves.
- Safe in isolation: backward strategy is now data, but trainer still inherits.

**PR 4: Extract `RenderHarness` (and fold composition + random bg into it).**

- Why now: with `LossFn` extracted, the render -> compose -> loss pipeline has a clean seam. The four duplicate composition blocks collapse into `_RenderHarnessBase._composite`.
- Changes: `pipeline_strategies/render_harness.py` with `SingleViewRenderHarness`, `MultiViewRenderHarness`, `PerFrameRenderHarness`. The bug in `KnownCameraTrainer.initial_step_result` (line 1897) and the three multicam tuple-vs-tensor bugs are fixed *automatically* because every harness goes through the shared `_composite` kernel.
- Smoke: same as PR 3, plus `local_mac_compare_local_video_encoder_16f_known_camera_..._fast_mac_8192splats.jsonc` — the known-camera config that previously had the latent tuple bug.
- Safe in isolation: the four composition sites already have unit-test fodder (see Test Surface).

**PR 5: Extract `Validator` from `Trainer.render_full_sequence + initial_step_result + the multicam render_full_external_views`.**

- Why now: with `LossFn` and `RenderHarness` extracted, `Validator.compute()` is "run the harness on eval clips, compute Eval/* metrics."
- Changes: `pipeline_strategies/validator.py` with `SourceViewValidator`, `HeldoutCameraValidator`, `NullValidator`.
- Smoke: same configs as PR 4. Same Eval/PSNR / Eval/DSSIM values.
- Safe in isolation: validation is invoked at fixed cadence; bisect-friendly.

**PR 6: Extract `ClipSampler` from `Trainer.sample_clip + sample_sequence + load_*_sequences + on_sequences_loaded`.**

- Why now: this is where the precomputed-feature lifecycle lives, and it's the heaviest extraction. Doing it after `LossFn` and `RenderHarness` means the sampler doesn't need to know about composition or backward strategy.
- Changes: `pipeline_strategies/sampler.py` with five sampler implementations. Multicam sampler exposes `extra_param_groups()` for the rig.
- Smoke: full matrix.
- Safe in isolation: the sampler interface is one method (`next_clip()`); easy to verify by side-by-side run.

**PR 7: Extract `ModelBuilder` from `build_model_from_config`.**

- Why now: model building is the simplest extraction, but it's also the one that depends on every other strategy being in place to know its caller's expectations (e.g. F-channel awareness).
- Changes: `pipeline_strategies/model_builder.py` with `FromVariantModelBuilder` (current 10-variant dispatch), `PrebakedModelBuilder`, `ImageEncoderModelBuilder`, `SingleImageModelBuilder`. Also fixes the `**_unused`/`feature_dim` silent-drop bug in `FreeGaussianBankImplicitCamera` because the builder passes `feature_dim` explicitly.
- Smoke: build every model variant once at step 0, verify shapes.

**PR 8: Collapse the trainer hierarchy into one `Trainer` class.**

- Why last: with all six strategies extracted, the subclass overrides have nothing left to override. They become empty bodies that just call `super()`.
- Changes: delete `KnownCameraTrainer`, `PrecomputedFeatureImplicitTrainer`, `MulticamPrecomputedFeatureImplicitTrainer`, `LTXFeatureImplicitTrainer`. Move the remaining body of `Trainer` into `trainer.py`. Delete the parent file's imports and 1k+ lines of helpers (now lifted into strategy modules). The `arch` and `model.variant` dispatch in `build_strategies_from_config` replaces `trainer_class_for_config`.
- Smoke: every config in `src/train_configs/` runs at least one step.
- Safe in isolation: nothing — this PR is the cutover. But after PRs 2-7 it is mostly file deletion and config rename.

**PR 9: Absorb the legacy file trainers (`dynamicTokenGS.py`, `train_camera_implicit_dynamic.py`, `tokenGS.py`).**

- Why last: these have the most distinctive sampler / model-builder / loss combos and the smallest config footprints.
- Changes: write `PrebakedCameraSampler`, `PrebakedCameraLoss`, `PrebakedModelBuilder`, `ImageEncoderModelBuilder`, `SingleImageSampler`, `SingleImageModelBuilder`. Add 3 entries to `PIPELINE_REGISTRY`. Delete the three legacy trainer files and the six shim files.
- Smoke: each legacy config (`local_mac_overfit_prebaked_camera_*.jsonc`, `local_mac_overfit_image_implicit_camera*.jsonc`, `local_mac_overfit_single_image*.jsonc`) runs end-to-end on the new path.

Each PR is independently revertable. The bug we fixed gets a permanent structural fix at PR 4; the rest is mostly mechanical cleanup.

## What gets deleted

After PR 9 lands:

| File | Status |
|---|---|
| `src/train/train_video_token_implicit_dynamic.py` | renamed to `trainer.py`, body shrinks ~10x |
| `src/train/train_precomputed_feature_implicit_dynamic.py` | deleted (logic in `pipeline_strategies/sampler.py::PrecomputedFeatureSampler`) |
| `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` | deleted (logic in `MulticamPrecomputedSampler`, `MultiViewRenderHarness`, `MulticamLoss`, `MulticamVideoLogger`, `HeldoutCameraValidator`) |
| `src/train/train_ltx_feature_implicit_dynamic.py` | deleted (was empty body anyway) |
| `src/train/train_camera_implicit_dynamic.py` | deleted (logic in `pipeline_strategies_legacy/image_encoder_baseline.py`) |
| `src/train/train_image_encoder_implicit_camera_baseline.py` | deleted (was a 4-line shim) |
| `src/train/train_camera_implict_dynamic.py` | deleted (typo file, never used) |
| `src/train/dynamicTokenGS.py` | deleted (utilities lifted to `train_runtime.py`; trainer logic in `pipeline_strategies_legacy/prebaked_camera.py`) |
| `src/train/dynamicTokenGS_shared.py` | deleted (dead shim) |
| `src/train/dynamicTokenGS_tiled.py` | deleted (dead shim) |
| `src/train/tokenGS.py` | deleted (logic in `pipeline_strategies_legacy/single_image_overfit.py`) |
| `src/train/tokenGS_shared.py` | deleted (dead shim) |
| `src/train/tokenGS_tiled.py` | deleted (dead shim) |

12 trainer files collapse into 1 `trainer.py` + 6 strategy modules + 3 legacy strategy modules + 1 registry + 1 runtime utility module = 11 files, but each is 50-300 lines and has one reason to change.

The `arch` field in every `src/train_configs/*.jsonc` is deleted in the same PR; configs gain a `pipeline:` field. (The 96 configs are mechanical edits — likely a small Python script that maps the launcher script + `model.variant` to the right `pipeline:` value.)

## Test surface

Strategy-based design unlocks unit tests at the strategy level. Following the AGENTS.md rule "tests should catch a bug a future engineer would otherwise introduce," not "tests that mirror the implementation":

- **`tests/test_render_harness_composition.py`** — feed a controlled `(features, alpha)` tuple into `_composite` with `background=0.0` and again with `background=1.0`; assert the formula `α·rgb + (1-α)·bg` to 1e-6. This is a falsifiable math test that catches any future drift in the composition kernel.
- **`tests/test_render_harness_known_camera_alpha.py`** — regression guard for the bug in this session. Build a known-camera config, run `Trainer.step()` for 1 step, assert that the rendered RGB has been composited (i.e. `RenderedClipBundle.alpha is not None` for `fast_mac` F!=3) and that `LossOutput.total` is finite. Without this test, the next refactor could silently re-break the bug.
- **`tests/test_loss_fn_chunked_vs_singleshot.py`** — feed the same synthetic batch through `StandardLoss` with `temporal_microbatch_size=4` (Chunked) and `=full` (SingleShot); assert the gradients are identical to 1e-5 after `optimizer.step()`. This guards against any future change to `BackwardStrategy` accidentally giving different math.
- **`tests/test_multicam_loss_no_silent_skip.py`** — architectural invariant. `mock.patch('pipeline_strategies.render_harness._composite', side_effect=AssertionError)` and run `MulticamLoss.compute(...)`; assert it raises. This guarantees that the multicam path *must* go through the shared composition kernel and cannot regress to the bypass-bug shape.
- **`tests/test_pipeline_registry_resolution.py`** — feed every entry in `PIPELINE_REGISTRY` a minimum-viable config; assert it resolves to a `StrategyTuple` with all six fields non-None. Catches missing combo entries.
- **`tests/test_clip_sampler_invariants.py`** — feed every sampler a minimum-viable config; for each `next_clip()` output, assert the `ClipBatch` invariants from `__post_init__` (which already raise; this just exercises every variant).
- **`tests/test_feature_dim_thread.py`** — for every model variant in `MODEL_VARIANT_REGISTRY`, build with `feature_dim=32`; call forward; assert `decoded.sequence.rgbs.shape[-1] == 32`. Catches the `FreeGaussianBankImplicitCamera` `**_unused` silent-drop bug from investigator 05.

What we do NOT write:

- `test_trainer_init_succeeds.py` — type system catches this.
- `test_clip_batch_has_field_X.py` — pure schema mirror.
- `test_video_logger_returns_dict_with_key_Y.py` — the protocol is the spec.

## Risk analysis

Honest tradeoffs:

**1. This is a bigger one-shot refactor than Proposer 1's (pure-function helpers).** Proposer 1 can land the composition-kernel extraction in one PR. This proposal needs 9 PRs to land cleanly. The migration plan above breaks the work down so each PR is reviewable in isolation, but the total volume is larger. Mitigation: PRs 1-3 alone fix the bug we just shipped, and the rest can be paused at any time without leaving the codebase in a worse state.

**2. Strategy combinations could explode (NxM problem).** Six strategies with 3-5 implementations each gives ~3⁶ = 729 possible combos. The vast majority are nonsensical (e.g. `MulticamLoss` + `SingleViewRenderHarness`). Mitigation: `PIPELINE_REGISTRY` is the canonical surface; ad-hoc combos are gated by an explicit `strategies:` block in config that's reviewed in code. Plus a `pipeline_registry.py::validate(tuple)` guard at trainer init that asserts compatible pairs (e.g. multicam loss requires multicam harness; precomputed sampler requires `features:` config section). Failures are loud at init, not at step 1.

**3. Type checking via `Protocol` is weaker than abstract base classes.** `Protocol` is structural — a class only needs to *look right* to satisfy. There's no nominal type check at construction. Mitigation: `runtime_checkable=True` on every Protocol so we get an `isinstance(strategy, ClipSampler)` smoke at registry-build time. mypy / pyright catch protocol mismatches at static-check time. The looseness is a feature for adapter classes that wrap legacy code (e.g. wrapping `dynamicTokenGS.run_training` in a `PrebakedCameraSampler` adapter).

**4. The user's `key_learnings.md:18` warns against `BaseTrainer`-style sharing.** Direct quote: *"A single shared `BaseTrainer` would hide real differences between known-camera, image-implicit, and video-token implicit training. Shared payload contracts are cleaner than shared trainer inheritance."* This proposal honours that note, NOT violates it. The `Trainer` class has no inheritance and no overrideable hooks. The differences between trainer kinds live in the strategies, where they are explicit and named. The shared payload contracts (`ClipBatch`, `RenderedClipBundle`, `LossOutput`, `ValidationPayload`) are exactly what the note approves. `Protocol` is not inheritance — it's a structural contract, equivalent to "this object has these methods." No protocol class supplies any behavior; implementations stand alone.

**5. The chunked-backward strategy now lives inside `LossFn`, not `Trainer`.** This is the most controversial design choice. The current code has `recon_backward` calling `.backward()` per chunk inside `Trainer`; this proposal makes it a `Chunked` payload that the Trainer drives. The Trainer's `step()` body has a `match` statement on `BackwardStrategy`, which is a small structural branch. We accept it because: (a) chunking is a property of how the loss was assembled (per-chunk reconstruction), not a property of the trainer; (b) the `match` is exhaustive and shallow; (c) it makes the multicam case (single-shot) and single-cam case (chunked) explicit.

**6. The eager-concatenation render contract.** `RenderHarness.render()` returns a single `RenderedClipBundle` rather than a generator of per-chunk bundles. This means peak memory for the rendered tensor is `T * 3 * H * W` even when chunked-backward is in effect. The current code holds chunk renders only one at a time. Mitigation: the chunked path stores per-chunk losses (small scalars) inside the `LossFn`, not per-chunk renders inside the `RenderHarness`; the renders are released after `LossFn.compute()` returns. The total peak is bounded by the largest chunk size, not the full clip — same as today. (See PR 4 smoke test.)

**7. `render_harness.bind_colorize(self.colorize)` is a circular-dependency hack.** The Trainer builds the model + colorize via `ModelBuilder`, then injects colorize into the `RenderHarness`. An alternative is for `RenderHarness` to receive a `ModelBuilder` and call `.build()` itself, but that pushes model lifecycle into a strategy. Mitigation: keep the `bind_colorize` hook but assert it's called exactly once before any `render()` call.

## Tradeoffs vs the other two proposals

**vs Proposer 1 (pure-function helpers):**

- Pros: this proposal makes "what each trainer does" *visible* — a multicam config has a named `MultiViewRenderHarness`, not "the multicam trainer's `step` method." Bugs of the kind we just fixed cannot recur because there's no Method Resolution Order to bypass. Also yields stronger isolation for testing (one strategy, one test file).
- Cons: bigger refactor. Proposer 1 lands the composition fix in one PR; this lands it in PR 4. If the team wants to ship the bug fix and stop, Proposer 1 wins.

**vs Proposer 3 (typed dataflow pipeline DSL):**

- Pros: simpler operationally. `Trainer.step()` is straight Python with explicit method calls, not a graph executor or DAG runtime. Easier to debug with pdb. No new framework concepts to learn beyond `Protocol` + dataclass.
- Cons: less expressive for novel pipeline shapes. Proposer 3's DSL would let you compose "render -> compose -> downsample -> loss" by stitching nodes; this proposal requires modifying the strategy interface (e.g. `RenderHarness.render()` always returns rgb-3 already composed). Proposer 3 wins if we expect to add many new pipeline shapes; this proposal wins if the current 8 trainer kinds + ~5 future ones are the steady state.

**The bug we fixed:**

- Proposer 1: fixed by extracting `compose_rendered_rgb(...)` and calling it from both single-cam and multicam paths. Method dispatch still possible to bypass.
- Proposer 2 (this): fixed structurally — the multicam path can only invoke `MultiViewRenderHarness.render()`, which internally calls `_composite()`; there's no alternate path through inheritance.
- Proposer 3: fixed by making the composition node a required edge in the dataflow graph; bypass requires explicitly removing the node, which is visible in the DSL config.

All three fix the bug. This proposal makes the fix structural rather than disciplinary.

---

End of proposal.
