# Proposal A: Shared Render Objective Pipeline

Date: 2026-04-30
Writer: Proposal Writer A
Scope: render, colorize, alpha composition, reconstruction loss, and validation artifacts

## Thesis

`render_view_batch` / `RenderObjective` must become the only boundary where
rasterized `(features, alpha)` becomes final RGB.

The recent F=32 feature-splatting fix proved the core issue: training behavior
depends on a subtle sequence:

1. rasterize Gaussian splat features
2. colorize F-channel features into RGB
3. alpha-compose the colorized RGB against the configured background
4. compute reconstruction loss against target RGB
5. preserve feature/alpha buffers for diagnostics

That sequence currently lives in one mostly-correct method
(`Trainer.recon_backward`) and then gets copied, partially copied, or bypassed in
validation, known-camera, multicam, and legacy helper paths. The next cleanup
should not be "fix every override again." The cleanup should make it impossible
for a trainer override to compare raw F-channel features to RGB, forget alpha,
forget random train background, or log held-out views without the alpha/PCA
diagnostics.

The proposed design is a small functional objective layer with typed data
carriers. Trainers decode `GaussianSequence`; samplers emit target views; the
objective renders/losses/logs those views. The objective owns all RGB formation.

## Current Evidence

Observed facts from the live code:

- `render_clip_sequence(...)` now returns `tuple[torch.Tensor, torch.Tensor | None]`
  at `src/train/train_video_token_implicit_dynamic.py:556`. The first tensor is
  `[T, F, H, W]`; the second is `[T, H, W]` only for fast-mac F>3.
- `render_gaussian_frames_alpha_aware(...)` exposes alpha only through the
  fast-mac F!=3 path at `src/train/rendering.py:363`.
- `render_fast_mac_3dgs(...)` dispatches by `rgbs.shape[-1]`: F=3 uses v5 and
  returns `(image_rgb, None)`, while F!=3 uses `v5_features` and returns
  `(image_features, alpha_hw)` at `src/train/renderers/fast_mac.py:286`.
- `FeatureToColor.forward(...)` maps `[N, F, H, W]` or `[B, T, F, H, W]` to RGB
  at `src/train/colorize.py:190`.
- `reconstruction_loss_per_image(...)` expects prediction and target tensors
  with the same RGB shape at `src/train/losses.py:58`.
- `Trainer.recon_backward(...)` now has the desired training logic: rasterize,
  colorize, compose `alpha * splat_rgb + (1 - alpha) * random_bg`, and then
  compute the loss at `src/train/train_video_token_implicit_dynamic.py:1309`.
- `Trainer.initial_step_result(...)` and `Trainer.render_full_sequence(...)`
  repeat similar composition logic with a fixed white eval background at
  `src/train/train_video_token_implicit_dynamic.py:1388` and `:1550`.
- `KnownCameraTrainer.initial_step_result(...)` is stale: it assigns the entire
  `(features, alpha)` tuple to `rendered_features` and then passes it through
  colorization/loss as if it were a tensor at
  `src/train/train_video_token_implicit_dynamic.py:1880`.
- `MulticamPrecomputedFeatureImplicitTrainer.render_view_clip(...)` returns the
  tuple from `render_clip_sequence(...)` while annotated as `torch.Tensor`, and
  `multicam_recon_loss(...)` passes that tuple into
  `reconstruction_loss_per_image(...)` at
  `src/train/train_multicam_precomputed_feature_implicit_dynamic.py:177`.
- Multicam validation also calls `.detach().cpu()` on tuple-returning render
  calls at `src/train/train_multicam_precomputed_feature_implicit_dynamic.py:288`.
- Single-cam validation logs `Alpha_Mask_Video`, `Feature_PCA_Video`, and
  `Render_Composite_Video` at `src/train/train_video_token_implicit_dynamic.py:1649`.
  Multicam validation does not have equivalent train-view or held-out-view
  diagnostics.

Current belief:
    The bug class is not an isolated tuple arity bug. It is a boundary bug:
    "render" means different things in different trainers. Sometimes it means
    rasterized features, sometimes colorized RGB, sometimes final composited RGB,
    and sometimes a tuple. The same word is used for incompatible tensor
    contracts.

Confidence:
    High. The F=32 feature path already broke multicam and known-camera preview
    through signature drift. Future loss/background changes will repeat this
    unless the boundary is made explicit.

## Why Inheritance Failed Here

The current trainer hierarchy looks convenient but encodes the wrong ownership.

```text
Trainer
  owns single-cam sampling, decoding, render/loss, validation videos

  KnownCameraTrainer(Trainer)
    overrides sampling, decode call, step, initial result, render_full_sequence

  PrecomputedFeatureImplicitTrainer(Trainer)
    overrides feature cache input, inherits base render/loss

    MulticamPrecomputedFeatureImplicitTrainer(PrecomputedFeatureImplicitTrainer)
      overrides sampling, decode wrapper, render, loss, validation videos
```

The inheritance problem is not that subclassing is always wrong. It is that the
overrides are slicing through a behavior that must remain atomic:

```text
rasterize -> colorize -> compose -> loss -> diagnostics
```

The base class changed `render_clip_sequence` from "returns tensor" to "returns
tuple". Subclasses had no compiler-visible contract forcing them to unpack it,
colorize it, and compose it. The multicam subclass overrides exactly the two
methods that matter (`render_view_clip` and `multicam_recon_loss`), so it bypasses
the good path. Known-camera partially inherits the good path for `step`, but not
for `initial_step_result`, so its preview/eval path drifts.

Specific inheritance failure modes:

1. Return type drift is hidden behind `torch.Tensor` annotations.
   `render_view_clip(...) -> torch.Tensor` can return a tuple and no local type
   checker catches it because the runtime function is imported from the base
   module.

2. Render semantics are overloaded.
   `rendered`, `rendered_features`, and `rendered_clip` are used for raw features,
   splat RGB, and final RGB in different scopes.

3. Background policy is not a config-owned object.
   Training random background currently exists as a hardcoded tensor allocation
   inside `Trainer.recon_backward`. W&B/config provenance cannot prove whether a
   run used random background, and eval paths duplicate fixed-white behavior.

4. Diagnostics are downstream of a specific trainer.
   Single-cam eval has alpha/PCA/composite videos. Multicam held-out eval, which
   is the main pressure test, does not.

5. Model/data differences are allowed to fork objective logic.
   Single-cam, precomputed, known-camera, and multicam differ in how they build
   conditioning input and cameras. They should not differ in how a decoded
   `GaussianSequence` is turned into RGB/loss.

Conclusion:
    Inheritance should be reduced to compatibility shims and legacy entrypoints.
    The render/loss behavior should be a composable object with plain data inputs.

## Non-Negotiable Behavioral Contracts

These contracts are the reason this proposal exists.

### C1: Final RGB Is The Only Loss Input

`reconstruction_loss_per_image(prediction, target, loss_cfg)` must receive:

```text
prediction: [T, 3, H, W], float
target:     [T, 3, H, W], float
```

No trainer should call reconstruction loss with:

```text
[T, F, H, W] where F != 3
tuple[features, alpha]
uncomposited splat_rgb when alpha is available and composition is configured
```

### C2: F=3 Legacy Path Remains Boring

When `GaussianSequence.rgbs.shape[-1] == 3`:

```text
rasterized.features: [T, 3, H, W]
rasterized.alpha:    None for current fast-mac v5 path
rendered.rgb:        rasterized.features
rendered.splat_rgb:  None unless colorize is explicitly forced for diagnostics
```

The objective should not add a second composition step for F=3 unless a future
rasterizer exposes alpha for the F=3 path and the config explicitly opts into
post-raster background composition. The immediate migration must preserve legacy
RGB behavior.

### C3: F>3 Feature Splatting Requires Colorization

When `GaussianSequence.rgbs.shape[-1] != 3`:

```text
rasterized.features: [T, F, H, W]
colorizer:           required
splat_rgb:           [T, 3, H, W]
```

If colorizer is missing, fail during objective construction or at the first
render call with a clear error:

```text
feature_dim=32 requires FeatureToColor; got colorizer=None
```

### C4: Alpha-Aware Composition Is Centralized

When F>3 and `rasterized.alpha is not None`:

```text
alpha_expanded = rasterized.alpha.unsqueeze(1)  # [T, 1, H, W]
rgb = alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * background_rgb
```

No trainer, validation method, or probe script should open-code that equation.

### C5: Random Training Background Is A Policy, Not A Local Variable

Training can use a random RGB background sampled once per objective call:

```text
background_rgb: [1, 3, 1, 1]
```

It should broadcast across:

```text
views
chunks
frames
pixels
```

within the same optimization step, unless the config explicitly asks for per-view
or per-frame randomness. The default for the current F=32 anti-cheating fix
should be "one RGB per train step".

Eval should use a stable background:

```text
white: [1, 3, 1, 1] filled with 1.0
fixed: [1, 3, 1, 1] from config
black: [1, 3, 1, 1] filled with 0.0
```

### C6: Diagnostics Travel With The Rendered View

Every rendered view should carry:

```text
rgb:       [T, 3, H, W]
features:  [T, F, H, W]
alpha:     [T, H, W] | None
splat_rgb: [T, 3, H, W] | None
background_rgb: [1, 3, 1, 1] | [T, 3, H, W] | None
```

Validation logging should not have to re-render just to get PCA or alpha videos.

### C7: Held-Out Multicam Is A First-Class Target Role

The objective does not care whether a target view is:

```text
single_train
known_camera_train
multicam_train_view_0
multicam_heldout_camera_0040
```

It receives a `TargetView` with frames and cameras. Role affects metrics prefix
and whether gradients are enabled, not RGB composition.

## Proposed Module Boundary

The implementation should be split into small modules. Names are proposals, not
requirements, but the type boundary is the important part.

```text
src/train/objective_types.py
    Pure dataclasses, Protocols, and Literal aliases.

src/train/render_objective.py
    RenderObjective, background policy, composition, loss aggregation.

src/train/validation_artifacts.py
    Feature PCA, alpha grayscale, side-by-side composites, W&B payload helpers.

src/train/train.py
    Future unified runner/router. Not required for Proposal A phase 1, but the
    objective should be designed so it can be called from that runner.
```

The immediate step can keep old trainer files. They should delegate to
`RenderObjective` rather than own render/loss semantics.

## Proposed Types And Interfaces

All dataclasses should be `frozen=True` unless they intentionally carry mutable
state. Tensors remain mutable by PyTorch semantics; frozen dataclasses prevent
accidental field replacement.

### Shape Aliases

These are documentation aliases, not runtime classes.

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

Tensor = torch.Tensor

# Common shape comments:
# FrameRGB:        [T, 3, H, W]
# FeatureImage:    [T, F, H, W]
# AlphaImage:      [T, H, W]
# BackgroundRGB:   [1, 3, 1, 1] or [T, 3, H, W]
# FrameIndices:    [T]
# FrameTimes:      [T, 1]
# Gaussian attrs:  [T, G, C]
```

### Phase And Roles

```python
ObjectivePhase = Literal["train", "eval", "preview"]
TargetRole = Literal["train", "heldout", "source", "debug"]
FeatureMode = Literal["legacy_rgb", "feature_splat"]
BackgroundMode = Literal["white", "black", "fixed_rgb", "random_rgb", "none"]
BackgroundSampleScope = Literal["step", "view", "frame"]
```

Semantics:

- `train`: gradients may flow; random train background is allowed.
- `eval`: no gradients; background must be deterministic unless explicitly
  configured otherwise.
- `preview`: no gradients; same composition as eval, but may render only a
  subset and may keep fewer artifacts.

### RenderSpec

`RenderSpec` is the normalized render section. It replaces repeated unrolling of
`renderer_mode`, `render_cfg`, `input_size`, `render_size`, and `dense_grid`.

```python
@dataclass(frozen=True)
class RenderSpec:
    mode: str
    input_size: int
    render_size: int
    tile_size: int
    bound_scale: float
    alpha_threshold: float
    near_plane: float
    fast_mac: Mapping[str, Any]
    camera_projection: str | None
    dense_grid: Tensor | None = None
```

Shape rules:

- `dense_grid` is renderer-specific and may be `None`.
- `render_size` controls both output height and width for current square
  training configs. If rectangular output becomes needed, split this into
  `height` and `width` in the normalized spec.

### BackgroundSpec

This makes random background visible in config and W&B.

```python
@dataclass(frozen=True)
class BackgroundSpec:
    train_mode: BackgroundMode = "white"
    eval_mode: BackgroundMode = "white"
    fixed_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_scope: BackgroundSampleScope = "step"
    apply_when_alpha_missing: bool = False
```

Recommended current F=32 config:

```jsonc
"losses": {
  "background": {
    "train_mode": "random_rgb",
    "eval_mode": "white",
    "sample_scope": "step"
  }
}
```

Notes:

- `apply_when_alpha_missing=False` preserves current behavior for non-fast-mac
  feature paths: if no alpha exists, `splat_rgb` is used directly.
- F=3 legacy ignores this policy for now because alpha is absent and the
  renderer owns its legacy background semantics.

### TargetView

`TargetView` is the common target object for single-cam, known-camera, multicam
train, and multicam held-out validation.

```python
@dataclass(frozen=True)
class TargetView:
    view_id: str
    role: TargetRole
    frames: Tensor                         # [T, 3, H_src, W_src], float 0..1
    cameras: tuple["CameraSpec", ...]      # length T, render-space camera source
    frame_indices: Tensor                  # [T], source frame ids
    frame_times: Tensor                    # [T, 1], normalized/source times
    video_fps: float
    loss_weight: float = 1.0
    metrics_prefix: str | None = None
    source_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Invariants:

```text
frames.shape       == [T, 3, H_src, W_src]
len(cameras)       == T
frame_indices.shape == [T]
frame_times.shape   == [T, 1]
loss_weight >= 0
```

Mapping from current paths:

- Single-cam implicit train: one `TargetView(role="train", view_id="source")`
  using `decoded.cameras`.
- Known-camera train: one `TargetView(role="train", view_id="known_camera")`
  using `SequenceData.cameras[clip_indices]`.
- Multicam train: N `TargetView(role="train", view_id=f"train_view_{i}")`
  using `camera_rig.cameras_for_view(i, clip_indices)`.
- Multicam held-out eval: M `TargetView(role="heldout", view_id=camera_name)`
  using `camera_rig.heldout_cameras_for(i, frame_indices)`.

### RasterizedView

This is the direct output of the rasterizer. It is not RGB unless
`feature_dim == 3`.

```python
@dataclass(frozen=True)
class RasterizedView:
    view_id: str
    role: TargetRole
    features: Tensor                       # [T, F, H, W]
    alpha: Tensor | None                   # [T, H, W] for fast-mac F>3, else None
    cameras: tuple["CameraSpec", ...]      # length T, viewport-adjusted or source cameras
    frame_indices: Tensor                  # [T]
    feature_dim: int
    render_size: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Invariants:

```text
features.ndim == 4
features.shape == [T, F, H, W]
feature_dim == F
alpha is None or alpha.shape == [T, H, W]
len(cameras) == T
```

### ColorizedView

This is the output of `FeatureToColor` when it is used. It is optional for F=3.

```python
@dataclass(frozen=True)
class ColorizedView:
    view_id: str
    splat_rgb: Tensor                      # [T, 3, H, W]
    logits: Tensor | None = None           # [T, 3, H, W], optional diagnostics
    view_dirs: Tensor | None = None        # [T, 3, H, W], when view-conditioned
```

### BackgroundSample

Background is sampled once by policy, then passed into every compose call that
belongs to the same objective step.

```python
@dataclass(frozen=True)
class BackgroundSample:
    rgb: Tensor | None                     # None means no post-raster composition
    mode: BackgroundMode
    phase: ObjectivePhase
    scope: BackgroundSampleScope
```

Shape rules:

```text
scope == "step":  rgb.shape == [1, 3, 1, 1]
scope == "view":  rgb.shape == [V, 3, 1, 1] before selecting per view
scope == "frame": rgb.shape == [T, 3, 1, 1] or [T, 3, H, W]
```

Initial implementation should support `"step"` only. The enum exists so future
behavior does not need another signature break.

### RenderedView

`RenderedView.rgb` is the only tensor that can enter RGB reconstruction loss.

```python
@dataclass(frozen=True)
class RenderedView:
    view_id: str
    role: TargetRole
    rgb: Tensor                            # [T, 3, H, W], final composited RGB
    target_rgb: Tensor | None              # [T, 3, H, W], resized target if supplied
    rasterized: RasterizedView
    colorized: ColorizedView | None
    background: BackgroundSample
    phase: ObjectivePhase
    metrics_prefix: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Invariants:

```text
rgb.shape == [T, 3, H, W]
target_rgb is None or target_rgb.shape == rgb.shape
rasterized.features.shape[0] == rgb.shape[0]
```

### ViewLoss

```python
@dataclass(frozen=True)
class ViewLoss:
    view_id: str
    role: TargetRole
    total: Tensor                          # scalar
    per_image: Tensor                      # [T]
    weight: float
    metrics: Mapping[str, float] = field(default_factory=dict)
```

### ObjectiveLoss

This is the return object for train/eval loss aggregation.

```python
@dataclass(frozen=True)
class ObjectiveLoss:
    total: Tensor                          # scalar, weighted mean over views
    reconstruction: Tensor                 # scalar, before external regularizers
    view_losses: tuple[ViewLoss, ...]
    rendered_views: tuple[RenderedView, ...]
```

### ValidationArtifactBundle

This keeps videos out of trainer subclasses.

```python
@dataclass(frozen=True)
class ValidationArtifactBundle:
    videos: Mapping[str, Any]              # W&B video/image values or plain tensors
    metrics: Mapping[str, float]
    rendered_views: tuple[RenderedView, ...]
```

### RasterizerProtocol

```python
class RasterizerProtocol(Protocol):
    def rasterize(
        self,
        decoded: "GaussianSequence",
        cameras: tuple["CameraSpec", ...],
        *,
        render_spec: RenderSpec,
    ) -> tuple[Tensor, Tensor | None]:
        """Return (features, alpha).

        features:
            [T, F, H, W]
        alpha:
            [T, H, W] for alpha-aware feature rasterizers, else None
        """
```

The first adapter can wrap existing `render_clip_sequence(...)`.

### ColorizerProtocol

```python
class ColorizerProtocol(Protocol):
    feature_dim: int

    def forward(self, features: Tensor, view_dirs: Tensor | None = None) -> Tensor:
        """Map [T, F, H, W] -> [T, 3, H, W]."""

    def forward_with_logits(
        self,
        features: Tensor,
        view_dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return (rgb, logits), both [T, 3, H, W]."""
```

`FeatureToColor` already satisfies the important part of this protocol for
`forward`.

### BackgroundPolicyProtocol

```python
class BackgroundPolicyProtocol(Protocol):
    def sample(
        self,
        *,
        phase: ObjectivePhase,
        like: Tensor,
        view_count: int,
        frame_count: int,
        generator: torch.Generator | None = None,
    ) -> BackgroundSample:
        """Return a background sample on like.device/like.dtype."""
```

Initial concrete class:

```python
@dataclass(frozen=True)
class BackgroundPolicy:
    spec: BackgroundSpec

    def sample(
        self,
        *,
        phase: ObjectivePhase,
        like: Tensor,
        view_count: int,
        frame_count: int,
        generator: torch.Generator | None = None,
    ) -> BackgroundSample: ...
```

### RenderObjective

```python
@dataclass
class RenderObjective:
    render_spec: RenderSpec
    loss_cfg: Mapping[str, Any]
    background_policy: BackgroundPolicyProtocol
    colorizer: ColorizerProtocol | None = None
    rasterizer: RasterizerProtocol | None = None
    feature_pca_enabled: bool = False
    alpha_video_enabled: bool = True
    composite_video_enabled: bool = True

    def rasterize_view(
        self,
        decoded: "GaussianSequence",
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
        phase: ObjectivePhase,
    ) -> RenderedView: ...

    def render_view(
        self,
        decoded: "GaussianSequence",
        target: TargetView,
        *,
        phase: ObjectivePhase,
        background: BackgroundSample | None = None,
    ) -> RenderedView: ...

    def render_view_batch(
        self,
        decoded: "GaussianSequence",
        targets: Sequence[TargetView],
        *,
        phase: ObjectivePhase,
        background: BackgroundSample | None = None,
    ) -> tuple[RenderedView, ...]: ...

    def loss_for_view(
        self,
        rendered: RenderedView,
        *,
        weight: float = 1.0,
    ) -> ViewLoss: ...

    def loss_for_batch(
        self,
        decoded: "GaussianSequence",
        targets: Sequence[TargetView],
        *,
        phase: ObjectivePhase,
        background: BackgroundSample | None = None,
    ) -> ObjectiveLoss: ...

    def validation_payload(
        self,
        rendered_views: Sequence[RenderedView],
        *,
        video_fps: float,
        include_gt: bool,
    ) -> ValidationArtifactBundle: ...
```

`RenderObjective` is allowed to hold the colorizer module because it is part of
the trainable objective. It should not own model decode, optimizer, camera
regularization, bank-rate loss, W&B run state, or sampler state.

### Functional Convenience API

For trainers that should not instantiate a class directly:

```python
def render_view_batch(
    decoded: "GaussianSequence",
    targets: Sequence[TargetView],
    *,
    objective: RenderObjective,
    phase: ObjectivePhase,
    background: BackgroundSample | None = None,
) -> tuple[RenderedView, ...]:
    return objective.render_view_batch(
        decoded,
        targets,
        phase=phase,
        background=background,
    )
```

### Factory Boundary

`**kwargs` is acceptable here, and only here, after validation.

```python
def build_render_objective(
    *,
    render_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    loss_cfg: Mapping[str, Any],
    colorize_cfg: Mapping[str, Any] | None,
    dense_grid: Tensor | None,
    device: torch.device,
    **validated_factory_kwargs: Any,
) -> RenderObjective: ...
```

Rules:

- Unknown config keys should fail in the factory, not be ignored in warm paths.
- The factory should normalize the background config once.
- The factory should enforce `feature_dim != 3 -> colorizer required`.
- The factory should record a small summary suitable for W&B config/logging:

```python
def objective_summary(objective: RenderObjective) -> dict[str, Any]: ...
```

Example summary:

```python
{
    "objective/feature_mode": "feature_splat",
    "objective/feature_dim": 32,
    "objective/colorizer": "FeatureToColor(hidden_dim=None, pre_norm=True, ...)",
    "objective/background/train_mode": "random_rgb",
    "objective/background/eval_mode": "white",
    "objective/background/sample_scope": "step",
    "objective/alpha_required_for_composition": True,
}
```

## Core Algorithms

### `rasterize_view`

```python
def rasterize_view(
    self,
    decoded: GaussianSequence,
    target: TargetView,
) -> RasterizedView:
    features, alpha = self.rasterizer.rasterize(
        decoded,
        target.cameras,
        render_spec=self.render_spec,
    )
    assert_render_features(features, alpha, target)
    return RasterizedView(...)
```

Validation:

```text
features.shape[0] == len(target.cameras)
features.shape[-2:] == (render_size, render_size)
alpha is None or alpha.shape == (features.shape[0], render_size, render_size)
```

Important:

- This function should not resize targets.
- This function should not colorize.
- This function should not compose background.

### `colorize_view`

```python
def colorize_view(
    self,
    rasterized: RasterizedView,
) -> ColorizedView | None:
    if rasterized.feature_dim == 3 and self.colorizer is None:
        return None
    if self.colorizer is None:
        raise ValueError(f"feature_dim={rasterized.feature_dim} requires a colorizer")
    splat_rgb = self.colorizer.forward(rasterized.features)
    return ColorizedView(view_id=rasterized.view_id, splat_rgb=splat_rgb)
```

Validation:

```text
splat_rgb.shape == [T, 3, H, W]
splat_rgb.device == rasterized.features.device
splat_rgb.dtype == rasterized.features.dtype
```

Future view-conditioned colorization:

```python
view_dirs = build_view_condition_dirs(
    cameras=rasterized.cameras,
    height=rasterized.features.shape[-2],
    width=rasterized.features.shape[-1],
    mode=colorize_cfg["view_condition"],
)
splat_rgb = self.colorizer.forward(rasterized.features, view_dirs=view_dirs)
```

The colorizer should receive all view-conditioning inputs from the objective,
not from arbitrary trainer code.

### `compose_view`

```python
def compose_view(
    self,
    rasterized: RasterizedView,
    colorized: ColorizedView | None,
    *,
    target_rgb: Tensor | None,
    background: BackgroundSample,
    phase: ObjectivePhase,
) -> RenderedView:
    if rasterized.feature_dim == 3 and colorized is None:
        rgb = rasterized.features
    else:
        if colorized is None:
            raise ValueError("feature splatting requires colorized RGB")
        splat_rgb = colorized.splat_rgb
        if rasterized.alpha is not None:
            if background.rgb is None:
                raise ValueError("alpha-aware composition requires non-None background")
            alpha = rasterized.alpha.unsqueeze(1)
            rgb = alpha * splat_rgb + (1.0 - alpha) * background.rgb
        else:
            if background.rgb is not None and self.background_policy.spec.apply_when_alpha_missing:
                raise ValueError("cannot apply RGB background without alpha")
            rgb = splat_rgb
    return RenderedView(...)
```

This single function replaces all open-coded variants of:

```python
alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * 1.0
alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * random_bg
rendered_clip = splat_rgb
rendered_clip = rendered_features
```

### `render_view_batch`

```python
def render_view_batch(
    self,
    decoded: GaussianSequence,
    targets: Sequence[TargetView],
    *,
    phase: ObjectivePhase,
    background: BackgroundSample | None = None,
) -> tuple[RenderedView, ...]:
    if not targets:
        return ()
    if background is None:
        background = self.background_policy.sample(
            phase=phase,
            like=decoded.rgbs,
            view_count=len(targets),
            frame_count=targets[0].frames.shape[0],
        )
    rendered = []
    for target in targets:
        rasterized = self.rasterize_view(decoded, target)
        target_rgb = resize_target_rgb(target.frames, self.render_spec.render_size)
        colorized = self.colorize_view(rasterized)
        rendered.append(
            self.compose_view(
                rasterized,
                colorized,
                target_rgb=target_rgb,
                background=background,
                phase=phase,
            )
        )
    return tuple(rendered)
```

Important detail:

- `background` can be passed in by a chunked training loop so every chunk in a
  step sees the same random background.
- The first implementation can call this for one view at a time. The contract
  still makes multi-view aggregation explicit.

### `loss_for_batch`

```python
def loss_for_batch(
    self,
    decoded: GaussianSequence,
    targets: Sequence[TargetView],
    *,
    phase: ObjectivePhase,
    background: BackgroundSample | None = None,
) -> ObjectiveLoss:
    rendered_views = self.render_view_batch(
        decoded,
        targets,
        phase=phase,
        background=background,
    )
    view_losses = tuple(
        self.loss_for_view(rendered, weight=target.loss_weight)
        for rendered, target in zip(rendered_views, targets, strict=True)
    )
    weight_sum = sum(loss.weight for loss in view_losses)
    if weight_sum <= 0:
        raise ValueError("at least one target view must have positive loss_weight")
    reconstruction = sum(loss.total * loss.weight for loss in view_losses) / weight_sum
    return ObjectiveLoss(
        total=reconstruction,
        reconstruction=reconstruction,
        view_losses=view_losses,
        rendered_views=rendered_views,
    )
```

`regularizer_loss`, camera loss, rig regularization, and bank-rate loss stay
outside this objective and are added by the training loop:

```python
objective_loss = objective.loss_for_batch(decoded, targets, phase="train", background=step_bg)
loss = objective_loss.total + camera_loss + bank_rate_loss + rig_loss
```

This keeps `RenderObjective` focused on RGB reconstruction, not all model
regularization.

## How Current Trainers Would Use It

### Single-Cam Implicit Trainer

Current warm path:

```text
sample_clip -> model_input_for_clip -> forward_clip -> recon_backward
```

Proposed warm path:

```python
clip_target = TargetView(
    view_id="source",
    role="train",
    frames=clip_frames[0],                 # [T, 3, H, W]
    cameras=decoded.cameras,
    frame_indices=clip_indices,
    frame_times=clip_times.squeeze(0) if clip_times.ndim == 3 else clip_times,
    video_fps=sequence_data.video_fps,
)

step_bg = self.objective.background_policy.sample(
    phase="train",
    like=decoded.rgbs,
    view_count=1,
    frame_count=clip_target.frames.shape[0],
)

objective_loss = self.objective.loss_for_batch(
    decoded,
    [clip_target],
    phase="train",
    background=step_bg,
)
```

Chunked backward still works:

```python
step_bg = objective.background_policy.sample(...)
for chunk_start, chunk_end in chunks:
    chunk_decoded = gaussian_sequence_slice(decoded, chunk_start, chunk_end)
    chunk_target = target_view_slice(clip_target, chunk_start, chunk_end)
    chunk_loss = objective.loss_for_batch(
        chunk_decoded,
        [chunk_target],
        phase="train",
        background=step_bg,
    )
    (chunk_loss.total / frame_count_scale + regularizers_if_last).backward(...)
```

The objective owns RGB formation. The trainer owns chunk scheduling and backward.

### Precomputed Feature Trainer

`PrecomputedFeatureImplicitTrainer` only changes conditioning:

```text
model_input_for_clip(sequence_data, clip_frames, clip_times)
```

It should not override render/loss. It can inherit the same single-cam path once
the base path delegates to `RenderObjective`.

### Known-Camera Trainer

Known-camera differs in decode input and target cameras:

```python
clip_cameras = tuple(sequence_data.cameras[i] for i in clip_indices.tolist())
decoded = self.forward_known_clip(clip_frames, clip_times, clip_cameras)

target = TargetView(
    view_id="known_camera",
    role="train",
    frames=clip_frames[0],
    cameras=clip_cameras,
    frame_indices=clip_indices,
    frame_times=clip_times.squeeze(0) if clip_times.ndim == 3 else clip_times,
    video_fps=sequence_data.video_fps,
)

objective_loss = self.objective.loss_for_batch(decoded, [target], phase="train")
```

This fixes the stale `initial_step_result` by deleting its local render/colorize
logic. The same objective path is used for preview and validation.

### Multicam Precomputed Feature Trainer

Multicam differs in target view count and camera source:

```python
targets = []
for view in selected_train_views:
    targets.append(
        TargetView(
            view_id=f"train_view_{view}",
            role="train",
            frames=self.multicam_bundle.train_frames[view, clip_indices],
            cameras=self.camera_rig.cameras_for_view(view, clip_indices),
            frame_indices=clip_indices,
            frame_times=clip_times.squeeze(0) if clip_times.ndim == 3 else clip_times,
            video_fps=self.sequence_data.video_fps,
            metrics_prefix=f"TrainView{view}",
        )
    )

objective_loss = self.objective.loss_for_batch(decoded, targets, phase="train")
recon_loss = objective_loss.reconstruction
```

Held-out validation:

```python
heldout_targets = []
for view in range(self.multicam_bundle.heldout_view_count):
    camera_name = self.multicam_bundle.heldout_camera_names[view]
    heldout_targets.append(
        TargetView(
            view_id=camera_name,
            role="heldout",
            frames=self.multicam_bundle.heldout_frames[view],
            cameras=self.camera_rig.heldout_cameras_for(view, frame_indices),
            frame_indices=frame_indices,
            frame_times=clip_times.squeeze(0) if clip_times.ndim == 3 else clip_times,
            video_fps=self.sequence_data.video_fps,
            metrics_prefix=f"Heldout{view}_{camera_name}",
        )
    )

rendered_heldout = self.objective.render_view_batch(
    decoded,
    heldout_targets,
    phase="eval",
)
payload = self.objective.validation_payload(
    rendered_heldout,
    video_fps=self.sequence_data.video_fps,
    include_gt=not self.gt_video_logged,
)
```

This is the main payoff: held-out camera rendering gets exactly the same F=32
feature colorize, alpha composition, fixed eval background, alpha mask video, PCA
video, and composite video as source-view validation.

## Validation Artifact API

The artifact layer should operate on `RenderedView`, not on trainer-specific
tuples.

```python
def alpha_to_rgb(alpha: Tensor) -> Tensor:
    """[T, H, W] -> [T, 3, H, W] grayscale."""

def feature_pca_video(features: Tensor) -> Tensor:
    """[T, F, H, W] -> [T, 3, H, W]."""

def render_composite_columns(rendered: RenderedView) -> tuple[tuple[str, Tensor], ...]:
    """Return named video columns for GT, Pred, Alpha, FeaturePCA, optional SplatRGB."""

def build_rendered_view_video_payload(
    rendered: RenderedView,
    *,
    video_fps: float,
    include_gt: bool,
    prefix: str,
    include_alpha: bool = True,
    include_feature_pca: bool = True,
    include_composite: bool = True,
) -> dict[str, Any]: ...

def build_validation_video_payload_for_views(
    rendered_views: Sequence[RenderedView],
    *,
    video_fps: float,
    include_gt: bool,
) -> dict[str, Any]: ...
```

Naming convention:

Single-cam first sequence can keep existing names for continuity:

```text
GT_Video
Render_Video
Render_GT_Video
Alpha_Mask_Video
Feature_PCA_Video
Render_Composite_Video
```

Multicam should prefix every view:

```text
TrainView0_GT_Video
TrainView0_Rendered_Video
TrainView0_Alpha_Mask_Video
TrainView0_Feature_PCA_Video
TrainView0_Render_Composite_Video

Heldout0_camera_0040_GT_Video
Heldout0_camera_0040_Rendered_Video
Heldout0_camera_0040_Alpha_Mask_Video
Heldout0_camera_0040_Feature_PCA_Video
Heldout0_camera_0040_Render_Composite_Video
```

Composite column order:

```text
GT | Pred | Alpha | FeaturePCA
```

Optional future columns:

```text
SplatRGB          # colorized splat before alpha/background composition
Background        # broadcast background expanded to video shape
AbsError          # abs(Pred - GT), useful for held-out debugging
```

## Loss And Metric API

The loss code should remain small, but shape checks must move before the loss.

```python
def resize_target_rgb(frames: Tensor, render_size: int) -> Tensor:
    """Resize [T, 3, H, W] -> [T, 3, render_size, render_size]."""

def assert_rgb_loss_shapes(prediction: Tensor, target: Tensor, *, context: str) -> None:
    """Fail before broadcasting or tuple errors hide the real issue."""

def reconstruction_view_loss(
    rendered: RenderedView,
    loss_cfg: Mapping[str, Any],
    *,
    weight: float = 1.0,
) -> ViewLoss:
    if rendered.target_rgb is None:
        raise ValueError("RenderedView.target_rgb is required for reconstruction loss")
    assert_rgb_loss_shapes(rendered.rgb, rendered.target_rgb, context=rendered.view_id)
    per_image = reconstruction_loss_per_image(rendered.rgb, rendered.target_rgb, dict(loss_cfg))
    return ViewLoss(
        view_id=rendered.view_id,
        role=rendered.role,
        total=per_image.mean(),
        per_image=per_image,
        weight=weight,
    )
```

Multicam aggregation should be explicit:

```python
def aggregate_view_losses(view_losses: Sequence[ViewLoss]) -> Tensor:
    total_weight = sum(loss.weight for loss in view_losses)
    if total_weight <= 0:
        raise ValueError("No positive target view weights")
    return sum(loss.total * loss.weight for loss in view_losses) / total_weight
```

Do not bake "divide by selected view count" into multicam trainer code.

## Config Contract

Minimum new normalized schema:

```jsonc
{
  "render": {
    "renderer": "fast_mac",
    "render_size": 256,
    "tile_size": 16,
    "bound_scale": 3.0,
    "alpha_threshold": 0.0039215686,
    "near_plane": 0.0001,
    "camera_projection": "auto",
    "fast_mac": {}
  },
  "model": {
    "feature_dim": 32
  },
  "colorize": {
    "hidden_dim": null,
    "activation": "sigmoid",
    "pre_norm": true,
    "weight_init": "kaiming",
    "weight_init_gain": 4.0,
    "view_condition": "none"
  },
  "losses": {
    "type": "l1",
    "background": {
      "train_mode": "random_rgb",
      "eval_mode": "white",
      "sample_scope": "step"
    }
  },
  "logging": {
    "feature_pca_video": true,
    "alpha_mask_video": true,
    "render_composite_video": true
  }
}
```

Backward compatibility:

- If `losses.background` is missing, normalize to:

```python
BackgroundSpec(train_mode="white", eval_mode="white")
```

- For explicitly migrated F=32 configs, set train random background in the
  checked-in config, not as hidden trainer code.
- The factory should write normalized background fields into W&B config or scalar
  payload.

## Migration Plan

### Phase 0: Preserve Current Working Runs

No behavior change. Add only docs and then code in a separate implementation
change.

Before implementation, record the two current reference runs in the migration
issue/notes:

```text
F=32 alpha-aware white/fixed path: run 3reqcya9, final loss 0.0653
F=32 random-bg path: run 9gr2dm3v, final loss 0.0665 / recon 0.0660
```

These numbers are not acceptance thresholds for refactor parity because random
background is stochastic. They are sanity anchors.

### Phase 1: Add Types And Objective Adapter

Add:

```text
src/train/objective_types.py
src/train/render_objective.py
src/train/validation_artifacts.py
```

Keep old trainer code callable. First adapter implementation can wrap existing
functions:

```python
class ExistingRasterizer:
    def rasterize(self, decoded, cameras, *, render_spec):
        return render_clip_sequence(
            decoded,
            cameras,
            renderer_mode=render_spec.mode,
            render_cfg=render_spec_to_legacy_dict(render_spec),
            input_size=render_spec.input_size,
            render_size=render_spec.render_size,
            dense_grid=render_spec.dense_grid,
        )
```

Add unit-level shape tests for:

```text
F=3 rasterized -> rendered.rgb == features
F=32 alpha present -> alpha composition shape and values
F=32 alpha missing -> splat_rgb direct path
loss rejects [T,F,H,W] when F != 3
```

These are behavior tests, not implementation-mechanic tests.

### Phase 2: Switch Base Trainer Warm Path

Change `Trainer.recon_backward`, `Trainer.initial_step_result`, and
`Trainer.render_full_sequence` to call `RenderObjective`.

Delete local open-coded composition from those methods. Keep chunked backward in
the trainer, but pass a shared `BackgroundSample` into each chunk.

Acceptance:

```text
F=3 single-cam 1-step smoke passes.
F=32 single-cam 1-step smoke passes.
F=32 single-cam validation logs Alpha_Mask_Video, Feature_PCA_Video,
Render_Composite_Video.
```

### Phase 3: Fix Known-Camera By Delegation

Change `KnownCameraTrainer.initial_step_result` and `render_full_sequence` to
construct `TargetView` and call the same objective methods.

Acceptance:

```text
known-camera 1-step smoke exercises initial result and final validation.
No tuple is passed to FeatureToColor.
No raw F-channel tensor reaches reconstruction_loss_per_image.
```

### Phase 4: Fix Multicam By Deleting Local Objective Logic

Replace:

```text
render_view_clip
multicam_recon_loss
render_full_external_views
validation_video_payload
```

with target construction plus objective calls.

Important: do not merely unpack the tuple in `render_view_clip`. That leaves the
same design smell and will miss the next objective change. The multicam trainer
should produce `TargetView` objects and ask `RenderObjective` for loss/rendered
views.

Acceptance:

```text
multicam F32 1-step smoke passes.
held-out rendered video exists.
held-out alpha mask video exists.
held-out feature PCA video exists.
held-out composite video exists.
metrics are prefixed per held-out camera.
```

### Phase 5: Normalize Background Config

Move hardcoded random background out of `Trainer.recon_backward` and into
`losses.background`.

Migration rule:

- Existing configs missing `losses.background` get white train/eval background.
- Current F=32 alpha configs get explicit:

```jsonc
"background": {
  "train_mode": "random_rgb",
  "eval_mode": "white",
  "sample_scope": "step"
}
```

W&B should show the normalized background policy.

### Phase 6: Retire Footguns

After smokes are green:

- Deprecate module-level `render_full_sequence` in
  `train_video_token_implicit_dynamic.py` or make it delegate to the objective.
- Remove any local alpha composition in trainer subclasses.
- Make `render_clip_sequence` usage private to the rasterizer adapter.
- Add `rg` guard in review checklist:

```bash
rg "render_clip_sequence\\(" src/train
rg "reconstruction_loss_per_image\\(" src/train
rg "alpha.*splat_rgb|splat_rgb.*alpha" src/train
```

Expected post-cleanup:

```text
render_clip_sequence         -> only rasterizer adapter and maybe probes
reconstruction_loss_per_image -> only objective/loss helper
alpha * splat_rgb             -> only compose_view
```

## Smoke Matrix

These smokes should run after the implementation, not after this doc-only
proposal.

### S1: F=3 Single-Cam Legacy RGB

Goal:
    Prove legacy path did not regress.

Command shape:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py /tmp/smoke_f3.jsonc
```

Expected:

```text
train.steps = 1
rendered.rgb shape [T, 3, H, W]
rasterized.alpha is None
no FeatureToColor required
val_log(0) and val_log(1) complete
```

### S2: F=32 Single-Cam Feature Splat Random Background

Goal:
    Prove the current fixed bug still works through the new objective boundary.

Command shape:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py /tmp/smoke_f32_random_bg.jsonc
```

Expected:

```text
feature_dim == 32
colorizer is constructed
background.train_mode == random_rgb
rasterized.alpha shape [T, H, W]
rendered.rgb shape [T, 3, H, W]
Alpha_Mask_Video logged
Feature_PCA_Video logged
Render_Composite_Video logged
```

### S3: Precomputed V-JEPA Single-Cam

Goal:
    Prove changing the objective did not break the feature-cache input adapter.

Command shape:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_precomputed_feature_implicit_dynamic.py /tmp/smoke_precomputed.jsonc
```

Expected:

```text
feature cache loads or bakes
model_input_for_clip returns precomputed features
objective path handles rendered RGB/loss
```

### S4: Known-Camera Preview And Validation

Goal:
    Catch the current stale tuple path in `KnownCameraTrainer.initial_step_result`.

Expected:

```text
initial_step_result uses RenderObjective
render_full_sequence uses RenderObjective
no tuple reaches FeatureToColor
no tuple reaches reconstruction_loss_per_image
```

### S5: Multicam F32 Ultimate Smoke

Goal:
    Make the desired multicam V-JEPA + F32 + random-bg + held-out-view path real.

Command shape:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  /tmp/smoke_multicam_f32_ultimate.jsonc
```

Expected:

```text
train step samples train views
each train view becomes TargetView
loss averages view losses
held-out camera renders in eval
held-out Alpha_Mask_Video exists
held-out Feature_PCA_Video exists
held-out Render_Composite_Video exists
```

### S6: Shape-Failure Synthetic Test

Goal:
    Prove the objective fails loudly before PyTorch broadcasts or tuple errors.

Cases:

```text
F=32, colorizer=None -> ValueError at objective build/render
features [T,32,H,W], target [T,3,H,W] -> never passed to loss
alpha [T,H,W], splat_rgb [T,3,H,W], bg [1,3,1,1] -> ok
alpha [T,H,W], bg None -> ValueError
```

## Risks And Counterarguments

### Counterargument: Just Fix Multicam Tuple Unpacking

This is the smallest patch, but it leaves the failure class intact. The next
change to background, alpha regularization, colorizer view-conditioning, or
diagnostic videos will require touching the same trainer forks again.

Cheap test:
    Add a new objective behavior, such as `splat_rgb` video logging. If it must
    be implemented separately in single-cam and multicam, the patch failed the
    cleanup goal.

### Counterargument: Keep Inheritance, Add A Base Helper

A helper is better than copy/paste, but only if it owns the full atomic boundary.
A helper named `compose_features` is too small; a trainer can still bypass loss
or diagnostics. The boundary should return `RenderedView` and carry target,
alpha, feature, background, and metrics context together.

Acceptable compromise:
    `RenderObjective` can be a small helper object rather than a large framework.
    The important thing is the data contract, not the class size.

### Counterargument: F=3 And F>3 Are Too Different For One Objective

They differ in colorization and alpha availability, but they share the contract:
loss sees final RGB and diagnostics should know what intermediate buffers exist.
The objective can branch on `feature_dim` internally. Letting every trainer
branch independently is the bug.

### Risk: The Objective Becomes A God Object

Prevent this by keeping ownership narrow:

`RenderObjective` owns:

```text
rasterize
colorize
background sample
alpha compose
RGB reconstruction loss
render diagnostics
```

It does not own:

```text
model construction
feature cache baking
camera rig optimization
optimizer stepping
bank-rate loss
camera regularization
W&B run lifecycle
export browser bundle
sampler policy
```

### Risk: Random Background Changes Eval Videos

The config separates train and eval background. Eval defaults to white/fixed for
stable videos. Train random background should not leak into validation unless
explicitly configured.

### Risk: Multicam Decode Once, Render Many Views

The design supports this. `decoded` is view-independent; each `TargetView` has
its own cameras. The objective loops over target views and reuses the same
decoded splats.

### Risk: Chunked Backward Needs Special Scaling

The trainer can still own chunking. The objective returns per-view/per-frame
losses, and the trainer can scale by total frame count for backward parity:

```python
chunk_losses = objective.loss_for_batch(...)
chunk_recon_loss = chunk_losses.reconstruction * (chunk_frame_count / total_frame_count)
```

The key is that chunking must pass the same `BackgroundSample` for the whole
step when sample scope is `"step"`.

## Implementation Checklist For Future Agent

Do this in order:

1. Add `objective_types.py` with the dataclasses above.
2. Add `render_objective.py` with `ExistingRasterizer`, `BackgroundPolicy`,
   `RenderObjective`, and strict shape checks.
3. Add `validation_artifacts.py` operating only on `RenderedView`.
4. Add `build_render_objective(...)` to the current trainer construction path.
5. Convert base `Trainer.recon_backward` to call the objective.
6. Convert base `Trainer.initial_step_result` and `render_full_sequence`.
7. Convert `KnownCameraTrainer.initial_step_result` and `render_full_sequence`.
8. Convert `MulticamPrecomputedFeatureImplicitTrainer` train and validation
   render/loss to build `TargetView` and call the objective.
9. Move random background into normalized config and W&B summary.
10. Run the smoke matrix.
11. Only after smokes pass, remove local duplicate composition helpers.

## Review Checklist

Before merging the implementation:

```bash
rg "render_clip_sequence\\(" src/train
rg "reconstruction_loss_per_image\\(" src/train
rg "alpha_expanded|splat_rgb \\+|alpha.*splat" src/train
rg "Feature_PCA_Video|Alpha_Mask_Video|Render_Composite_Video" src/train
```

Expected:

- `render_clip_sequence` is called by the rasterizer adapter, not directly by
  trainers.
- `reconstruction_loss_per_image` is called by objective/loss helpers, not by
  multicam trainer code.
- Alpha composition appears only in `compose_view`.
- Validation video names are produced by artifact helpers for both single-cam and
  multicam/held-out views.

## Definition Of Done

This proposal is implemented when:

1. F=3 legacy single-cam smoke is green.
2. F=32 single-cam random-bg smoke is green and logs alpha/PCA/composite videos.
3. Precomputed V-JEPA single-cam smoke is green.
4. Known-camera initial preview and validation no longer have tuple drift.
5. Multicam F32 smoke is green.
6. Multicam held-out validation logs rendered, GT, alpha, feature PCA, and
   composite videos per held-out camera.
7. `losses.background` is explicit in migrated F=32 configs and visible in W&B.
8. No trainer method manually converts `(features, alpha)` into final RGB.

The core acceptance criterion is not "fewer files." It is that the next feature
change to colorization, alpha composition, or background policy can be made in
one objective module and automatically apply to single-cam, precomputed,
known-camera, multicam train, and multicam held-out validation.
