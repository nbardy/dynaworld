# Proposal 01 — Minimum-disruption functional helpers

> Author: Proposer 1.
> Philosophy: extract pure functions and dataclasses; keep every existing
> trainer class. No new base class, no DI, no DSL.
> Scope: redesign of trainer/render/loss seam, driven by the alpha-aware
> composition bug that didn't propagate through the multicam trainer.

## TL;DR

- Create one new package `src/train/training_common/` with **6 modules of
  pure helpers + dataclasses** (no classes that own state, no inheritance).
- Introduce two frozen dataclasses: `RenderedClipBundle` (the alpha+features
  tuple as a typed value) and `BackgroundPolicy` (named bg shape: random,
  white, scalar, custom). No `LossInputs` bundle; loss helpers take plain
  tensors.
- The four existing trainer classes (`Trainer`, `KnownCameraTrainer`,
  `PrecomputedFeatureImplicitTrainer`, `MulticamPrecomputedFeatureImplicitTrainer`)
  stay. Each gets slimmed: `recon_backward` collapses from ~63 lines to ~25;
  `multicam_recon_loss` collapses from ~17 lines to ~10 *and gets the
  alpha-aware composition for free*. `validation_video_payload` keeps its
  per-trainer skeleton but the column-assembly logic becomes a helper.
- Delete: `train_camera_implict_dynamic.py` (typo file), the four trampoline
  shims (`*_shared.py`, `*_tiled.py` that no script invokes),
  `train_ltx_feature_implicit_dynamic.py` (empty subclass alias), and the
  module-level `render_full_sequence` at `train_video_token_implicit_dynamic.py:743`
  if `train_camera_implicit_dynamic.py` is also retired (out of scope here).
- Migrate in **8 numbered steps**, each one runnable end-to-end. The
  "alpha-aware multicam" fix lands in step 5; everything before that is
  prep that doesn't change behaviour.

## Module layout proposal

```text
src/train/training_common/
├── __init__.py               # re-exports the public surface
├── render_bundle.py          # RenderedClipBundle dataclass + BackgroundPolicy
├── compose.py                # compose_rendered_rgb, sample_random_background
├── colorize_factory.py       # build_colorize_module_from_config (extracted from Trainer.__init__)
├── recon_loss.py             # compute_recon_loss + chunked variant (no .backward in helper)
├── video_logging.py          # build_validation_video_payload_columns (consolidated)
└── render_clip.py            # render_clip_with_alpha (typed wrapper around render_clip_sequence)
```

Files **edited** by this proposal:

- `src/train/train_video_token_implicit_dynamic.py` — `Trainer.recon_backward`,
  `Trainer.initial_step_result`, `Trainer.render_full_sequence`,
  `Trainer.validation_video_payload`, `Trainer.__init__` (colorize block),
  `KnownCameraTrainer.initial_step_result` (bug fix),
  `KnownCameraTrainer.render_full_sequence`. The two trainer classes still
  live in this file; they just call helpers.
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` —
  `render_view_clip` (return type fix), `multicam_recon_loss` (alpha-aware,
  uses helper), `initial_step_result`, `render_full_external_views`,
  `validation_video_payload` (column assembly via helper).
- `src/train/rendering.py` — no edits in this proposal. The bundle wraps
  `render_gaussian_frames_alpha_aware` at the trainer-side wrapper layer,
  not inside `rendering.py`. (This keeps the v5/v5_features tuple convention
  alive but contained; renderer-side unification is a separate proposal.)

Files **deleted** by this proposal (zero callers each, all confirmed by
investigators 01 and 05):

- `src/train/train_camera_implict_dynamic.py` (typo of `implicit`)
- `src/train/dynamicTokenGS_shared.py`, `src/train/dynamicTokenGS_tiled.py`
- `src/train/tokenGS_shared.py`, `src/train/tokenGS_tiled.py`
- `src/train/train_ltx_feature_implicit_dynamic.py` (empty body, the only
  effect is a print prefix in `run`; dispatching directly to
  `PrecomputedFeatureImplicitTrainer` works)

Out of scope for this proposal (left for follow-up): the `dynamicTokenGS.py`
legacy trainer, the `train_camera_implicit_dynamic.py` legacy trainer, the
gauge-fields trainer family, and the renderer-side v5/v5_features unification.
Investigator 03 calls those out as bigger questions.

## Every new dataclass — full spec

### `RenderedClipBundle`

```python
# src/train/training_common/render_bundle.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch

@dataclass(frozen=True)
class RenderedClipBundle:
    """The raw output of a single rasterizer pass over a clip.

    Replaces the ad-hoc `tuple[Tensor, Tensor | None]` that the v5 (RGB-only)
    and v5_features (F-channel + alpha) bridges return through
    `render_clip_sequence`. A frozen dataclass is the cheapest way to give
    the tuple a type — every caller now binds named fields and the
    'is this a tensor or a tuple?' question is answered at construction.

    Invariants (enforced in __post_init__):
      - features.dim() == 4                # [T, F, H, W]
      - alpha is None  OR  alpha.shape == features.shape[0:1] + features.shape[2:]
      - features.device == alpha.device   when alpha is not None
      - len(cameras) == features.shape[0]
    """
    features: torch.Tensor                # [T, F, H, W]; F==3 for legacy RGB, F!=3 for feature splatting
    alpha: torch.Tensor | None            # [T, H, W] when v5_features path returned alpha, else None
    cameras: tuple[Any, ...]              # the cameras used for this render (post-viewport scaling)

    def __post_init__(self) -> None:
        if self.features.dim() != 4:
            raise ValueError(f"RenderedClipBundle.features must be [T, F, H, W]; got {tuple(self.features.shape)}")
        if self.alpha is not None:
            expected = (self.features.shape[0], self.features.shape[2], self.features.shape[3])
            if tuple(self.alpha.shape) != expected:
                raise ValueError(
                    f"RenderedClipBundle.alpha shape mismatch: features={tuple(self.features.shape)}, "
                    f"alpha={tuple(self.alpha.shape)}, expected {expected}"
                )
            if self.alpha.device != self.features.device:
                raise ValueError("RenderedClipBundle.alpha and .features must be on the same device.")
        if len(self.cameras) != self.features.shape[0]:
            raise ValueError(
                f"RenderedClipBundle: {len(self.cameras)} cameras but {self.features.shape[0]} feature frames."
            )

    @property
    def frame_count(self) -> int:
        return self.features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]

    @property
    def has_alpha(self) -> bool:
        return self.alpha is not None
```

Notes:
- Frozen dataclass for cheap hashing in tests; `torch.Tensor` fields are
  identity-hashed by Python — fine for our use, no hashing needed in prod.
- `cameras` is intentionally `tuple[Any, ...]` not `tuple[CameraSpec, ...]`
  — the codebase uses `Any` for cameras everywhere because of the legacy
  `CameraSpec` import path. Tightening the type is out of scope.
- No `feature_background` or `aux` fields. The investigator 03 audit shows
  `feature_background` is a renderer-config concern that gets overwritten
  downstream by alpha composition; it doesn't need to ride in the bundle.

### `BackgroundPolicy`

```python
# src/train/training_common/render_bundle.py (same module)
from typing import Literal, Union
import torch

@dataclass(frozen=True)
class _RandomRGBBackground:
    """Per-step random RGB sampled at compose time. Standard 3DGS trick:
    drives `alpha = 1 + splat_rgb = GT` as the only solution that survives
    different bg every step. Used at training time only."""
    generator: torch.Generator | None = None  # optional reproducibility hook

@dataclass(frozen=True)
class _ScalarBackground:
    """Fixed scalar broadcast across all RGB channels. value=1.0 = white,
    0.0 = black. Used at eval time. Cheap, deterministic, matches the
    published metric convention (white-bg)."""
    value: float

@dataclass(frozen=True)
class _TensorBackground:
    """Caller-provided pre-built bg tensor, broadcast-compatible with
    [T, 3, H, W]. Escape hatch for things like a custom env map; nothing
    in the codebase uses this today but it costs nothing to allow."""
    tensor: torch.Tensor

BackgroundPolicy = Union[_RandomRGBBackground, _ScalarBackground, _TensorBackground]

# Shorthand constructors for ergonomic call sites.
def random_bg(*, generator: torch.Generator | None = None) -> BackgroundPolicy:
    return _RandomRGBBackground(generator=generator)

def white_bg() -> BackgroundPolicy:
    return _ScalarBackground(value=1.0)

def scalar_bg(value: float) -> BackgroundPolicy:
    return _ScalarBackground(value=float(value))

def custom_bg(tensor: torch.Tensor) -> BackgroundPolicy:
    return _TensorBackground(tensor=tensor)
```

Why a typed sum and not a `bool training`? The investigator 02 report
flags asymmetry: training uses random per-step bg, eval uses white. A
boolean conflates *who decides* with *what gets sampled*. Naming each
case as a dataclass means the trainer says `white_bg()` at eval-time
explicitly, the helper picks the path, and a future "black-bg eval"
ablation is a one-line change at the call site, not a config knob.

This is **not** dependency injection: the trainer constructs the policy
inline at the call site, doesn't pass it as a constructor argument.

## Every new pure function — full signature

### `compose_rendered_rgb`

```python
# src/train/training_common/compose.py
from __future__ import annotations
import torch
from .render_bundle import RenderedClipBundle, BackgroundPolicy, _RandomRGBBackground, _ScalarBackground, _TensorBackground

def compose_rendered_rgb(
    bundle: RenderedClipBundle,
    *,
    colorize: torch.nn.Module | None,        # FeatureToColor, but typed as Module to avoid import cycle
    view_dirs: torch.Tensor | None,          # already-built [T, V, H, W] view conditioning, or None
    background: BackgroundPolicy,
) -> torch.Tensor:
    """Single source of truth for: features -> colorize -> alpha-composite.

    Replaces the 4 copies of the same 6-line block in
    train_video_token_implicit_dynamic.py at lines 1346-1361, 1408-1416,
    1599-1609, 1969-1979, AND fixes the missing-composition bug at
    KnownCameraTrainer.initial_step_result:1899-1902.

    Returns: [T, 3, H, W] RGB ready for reconstruction loss / video logging.

    Behaviour by case:
      colorize=None, F=3, alpha=None      -> bundle.features                       (legacy RGB pass-through)
      colorize=None, F!=3, alpha=anything -> ValueError                            (caller bug; mirror the trainer's fail-loud)
      colorize=mod,  F=any, alpha=None    -> colorize(features, view_dirs)         (no compositing; rasterizer composited internally)
      colorize=mod,  F=any, alpha!=None   -> alpha * colorize(...) + (1-alpha) * bg  (the canonical case)
    """
    if colorize is None:
        if bundle.feature_dim != 3:
            raise ValueError(
                f"compose_rendered_rgb: feature_dim={bundle.feature_dim} requires a colorize module. "
                "Pass FeatureToColor or set model.feature_dim=3."
            )
        return bundle.features

    splat_rgb = colorize(bundle.features, view_dirs=view_dirs)
    if bundle.alpha is None:
        return splat_rgb

    alpha_expanded = bundle.alpha.unsqueeze(1)               # [T, 1, H, W]
    bg = _materialize_background(background, alpha_expanded, splat_rgb)
    return alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * bg


def _materialize_background(
    policy: BackgroundPolicy,
    alpha_expanded: torch.Tensor,                            # [T, 1, H, W]; reference shape
    splat_rgb: torch.Tensor,                                 # [T, 3, H, W]; reference dtype/device
) -> torch.Tensor:
    """Internal. Materialize the BackgroundPolicy into a tensor that broadcasts
    against [T, 3, H, W]. Sampling for random bg happens HERE, once per call.
    Caller is responsible for calling once per step (training) or once per
    render (eval) — the helper is stateless."""
    device, dtype = splat_rgb.device, splat_rgb.dtype
    if isinstance(policy, _RandomRGBBackground):
        return torch.rand(3, device=device, dtype=dtype, generator=policy.generator).view(1, 3, 1, 1)
    if isinstance(policy, _ScalarBackground):
        return splat_rgb.new_tensor(policy.value)
    if isinstance(policy, _TensorBackground):
        return policy.tensor.to(device=device, dtype=dtype)
    raise TypeError(f"compose_rendered_rgb: unknown BackgroundPolicy {type(policy).__name__}")
```

Caller snippet (replaces the 4 duplicates):

```python
# inside Trainer.recon_backward, per chunk
random_bg_policy = random_bg()        # sampled once per step, outside the chunk loop
# ... build view_dirs once per chunk via colorize_view_dirs_for_features ...
chunk_renders = compose_rendered_rgb(bundle, colorize=self.colorize, view_dirs=view_dirs, background=random_bg_policy)
```

Note: `random_bg_policy` is sampled per-CALL, not per-policy-instance. The
"once per step" semantics are enforced by where the trainer hoists the
`compose_rendered_rgb` call relative to the chunk loop — not by the policy
type. We preserve the existing behaviour. (Investigator 02 raises this as
an open question; this proposal punts on per-frame vs per-chunk diversity.)

### `sample_random_background`

```python
# src/train/training_common/compose.py
def sample_random_background(
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one [1, 3, 1, 1] random RGB background. Exposed for callers
    that want to pre-sample once per step and reuse across multiple
    compose_rendered_rgb calls (e.g. multicam, where every view in a step
    must share the same bg or the gradient signal cancels itself out).

    Returns: [1, 3, 1, 1] tensor, broadcast-compatible with [T, 3, H, W].

    NOTE on torch.Generator: per investigator 02, the existing single-cam
    path uses default global RNG, so seed reproducibility silently breaks.
    This helper takes generator=None for parity, but every new call site
    can opt into a per-trainer Generator for reproducibility.
    """
    return torch.rand(3, device=device, dtype=dtype, generator=generator).view(1, 3, 1, 1)
```

Caller snippet (multicam case):

```python
# inside MulticamPrecomputedFeatureImplicitTrainer.multicam_recon_loss
shared_bg_tensor = sample_random_background(device=self.device, dtype=clip_frames.dtype)
shared_bg = custom_bg(shared_bg_tensor)
for view in views:
    bundle = render_clip_with_alpha(decoded, cameras=..., ...)
    rendered_rgb = compose_rendered_rgb(bundle, colorize=self.colorize, view_dirs=..., background=shared_bg)
    recon_loss = recon_loss + reconstruction_loss_per_image(rendered_rgb, target, self.loss_cfg).mean()
```

The "share one bg across views per step" is the same pattern as "share
one bg across chunks per step" in the existing single-cam recon_backward.
Multicam needs to do the same thing or the per-view gradients fight.

### `render_clip_with_alpha`

```python
# src/train/training_common/render_clip.py
from __future__ import annotations
from typing import Any
import torch
from .render_bundle import RenderedClipBundle

def render_clip_with_alpha(
    sequence,                               # GaussianSequence; not type-hinted to avoid import cycle
    cameras: tuple[Any, ...],
    *,
    renderer_mode: str,
    render_cfg: dict[str, Any],
    input_size: int,
    render_size: int,
    dense_grid: torch.Tensor,
) -> RenderedClipBundle:
    """Typed wrapper around `render_clip_sequence`. Same arguments, same
    backend; only difference is the return type is a RenderedClipBundle
    instead of a `tuple[Tensor, Tensor | None]`.

    The 6 trainer-side wrappers (`render_clip_sequence`, `Trainer.render_decoded_clip`,
    `MulticamTrainer.render_view_clip`, etc.) all collapse to one call to
    this helper.

    Importantly, the bundle includes the post-viewport-scaled cameras; this
    is what the colorize view-dirs helper needs anyway, so we hand it back
    rather than making each caller re-compute viewport_cameras.
    """
    from train_video_token_implicit_dynamic import render_clip_sequence, viewport_cameras  # local import: existing module
    features, alpha = render_clip_sequence(
        sequence,
        cameras,
        renderer_mode=renderer_mode,
        render_cfg=render_cfg,
        input_size=input_size,
        render_size=render_size,
        dense_grid=dense_grid,
    )
    render_cameras = viewport_cameras(cameras, input_size=input_size, render_size=render_size)
    return RenderedClipBundle(features=features, alpha=alpha, cameras=tuple(render_cameras))
```

The local import inside the function avoids a circular import at module
load. This is OK because `render_clip_with_alpha` is called inside
trainer methods, which run after both modules are loaded. The cleaner
fix (move `render_clip_sequence` and `viewport_cameras` into a shared
module) is deferred — it's a chunk of work and not load-bearing for the
alpha-aware fix.

### `apply_colorize`

```python
# src/train/training_common/compose.py
def apply_colorize(
    features: torch.Tensor,                  # [T, F, H, W]
    colorize: torch.nn.Module | None,        # FeatureToColor or None
    *,
    view_dirs: torch.Tensor | None,
) -> torch.Tensor:
    """Thin wrapper around the colorize MLP forward, matching the existing
    `Trainer.colorize_features` semantics (None means pass-through).

    Exposed mainly for tests and for the (rare) caller that wants to
    colorize without compositing — e.g. preview rendering when alpha is
    None and the caller already knows it doesn't need a bg."""
    if colorize is None:
        return features
    return colorize(features, view_dirs=view_dirs)
```

This is `Trainer.colorize_features` minus the view-dirs construction. The
view-dirs construction lives in `colorize_view_dirs_for_features` (which
is already a free function in `train_video_token_implicit_dynamic.py:501`)
and stays where it is — we don't move it. The trainers will call:

```python
view_dirs = colorize_view_dirs_for_features(bundle.features, bundle.cameras, ...)
rgb = compose_rendered_rgb(bundle, colorize=self.colorize, view_dirs=view_dirs, background=...)
```

### `compute_recon_loss`

```python
# src/train/training_common/recon_loss.py
from __future__ import annotations
from typing import Any
import torch
from losses import reconstruction_loss_per_image  # existing pure helper

def compute_recon_loss(
    rendered_rgb: torch.Tensor,              # [T, 3, H, W]
    gt_rgb: torch.Tensor,                    # [T, 3, H, W]; resize_images already applied
    *,
    loss_cfg: dict[str, Any],
    frame_count_for_normalization: int | None = None,
) -> torch.Tensor:
    """Single recon-loss helper. Wraps existing reconstruction_loss_per_image
    with the per-frame-normalization that the chunked backward needs.

    When frame_count_for_normalization is set, divides by that count so a
    summed-over-chunks total matches a mean-over-frames total. This is the
    semantic the existing Trainer.recon_backward relies on; without this
    parameter the helper is just a `.mean()` over per-image losses.

    Critically: this helper does NOT call .backward(). Backward stays in
    the trainer (where retain_graph and chunked-backward semantics live).
    """
    per_image = reconstruction_loss_per_image(rendered_rgb, gt_rgb, loss_cfg)
    if frame_count_for_normalization is None:
        return per_image.mean()
    return per_image.sum() / float(frame_count_for_normalization)
```

Caller (Trainer.recon_backward, simplified):

```python
chunk_loss = compute_recon_loss(chunk_renders, target_chunk, loss_cfg=self.loss_cfg, frame_count_for_normalization=frame_count)
backward_loss = chunk_loss + (regularizer_loss if is_last_chunk else 0.0)
backward_loss.backward(retain_graph=not is_last_chunk)   # backward stays in trainer
```

Caller (multicam):

```python
# multicam wants the simple .mean() shape — no chunk normalization
view_loss = compute_recon_loss(rendered_rgb, target_view, loss_cfg=self.loss_cfg)
recon_loss = recon_loss + view_loss
```

The investigator 02 report is explicit that the chunked-backward strategy
is single-cam-only; multicam treats each view as already a "chunk" and
does one final backward. This proposal does NOT push multicam onto the
chunked-backward strategy — it's a separate question. The helper
parametrizes over the normalization choice; trainers stay in charge of
backward.

### `build_validation_video_payload`

```python
# src/train/training_common/video_logging.py
from __future__ import annotations
from typing import Any
import torch
import wandb
from train_logging import build_validation_video_payload as _legacy_two_panel, make_wandb_video

def build_validation_video_payload(
    *,
    gt: torch.Tensor,                        # [T, 3, H, W] CPU tensor
    rendered: torch.Tensor,                  # [T, 3, H, W] CPU tensor
    alpha: torch.Tensor | None,              # [T, H, W] CPU tensor or None
    features_pca: torch.Tensor | None,       # [T, 3, H, W] CPU tensor (post-PCA-to-RGB) or None
    fps: float,
    log_gt: bool,                            # True only for the very first time we log GT for this run
) -> dict[str, Any]:
    """Single video-logging assembly. Takes whatever we have (some of these
    are None for some configs) and builds the dict of wandb.Video panels
    that the trainer will splat into its scalar payload.

    Returns keys:
      Render_Video, Render_GT_Video                         (always)
      GT_Video                                              (only if log_gt=True)
      Alpha_Mask_Video                                      (only if alpha is not None)
      Feature_PCA_Video                                     (only if features_pca is not None)
      Render_Composite_Video                                (only if any of alpha/PCA were present)

    Replaces the inline column-building logic at
    train_video_token_implicit_dynamic.py:1696-1731 and the simpler
    inline logic in MulticamPrecomputedFeatureImplicitTrainer.validation_video_payload.
    """
    payload: dict[str, Any] = dict(_legacy_two_panel(rendered, gt, fps))
    if log_gt:
        payload["GT_Video"] = make_wandb_video(gt, fps)
    composite_columns: list[torch.Tensor] = [gt, rendered]
    if alpha is not None:
        alpha_grayscale = alpha.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        payload["Alpha_Mask_Video"] = make_wandb_video(alpha_grayscale, fps)
        composite_columns.append(alpha_grayscale)
    if features_pca is not None:
        payload["Feature_PCA_Video"] = make_wandb_video(features_pca, fps)
        composite_columns.append(features_pca)
    if len(composite_columns) > 2:
        composite = torch.cat(composite_columns, dim=-1)
        payload["Render_Composite_Video"] = make_wandb_video(composite, fps)
    return payload
```

Caller (Trainer.validation_video_payload, sequence_index==0 branch):

```python
panels = build_validation_video_payload(
    gt=gt_sequence,
    rendered=rendered_sequence,
    alpha=alpha_sequence,
    features_pca=feature_pca_to_rgb(feature_sequence) if feature_sequence is not None else None,
    fps=sequence_data.video_fps,
    log_gt=not self.gt_video_logged,
)
payload.update(panels)
self.gt_video_logged = True
```

Multicam will use the same helper per-view, with `alpha=None,
features_pca=None` for now (multicam doesn't yet collect alpha into its
eval — that's a follow-up).

### `build_colorize_module_from_config`

```python
# src/train/training_common/colorize_factory.py
from __future__ import annotations
from typing import Any
import torch
from colorize import FeatureToColor, normalize_view_condition

def build_colorize_module_from_config(
    cfg: dict[str, Any],
    *,
    feature_dim: int,
    device: torch.device,
) -> tuple[FeatureToColor | None, str, bool]:
    """Extract the colorize-construction block from Trainer.__init__ into a
    pure factory.

    Returns: (module_or_None, view_condition_str, detach_view_condition_bool).

    The trainer still owns the `self.colorize`, `self.colorize_view_condition`,
    `self.colorize_detach_view_condition` attributes — this just moves the
    construction into one tested place. Per investigator 04 §5: the inline
    `cfg.get('colorize').get(key, default)` chain at trainer line 1014-1023
    is one of the project's flagged smells. Factor it out without changing
    semantics.
    """
    colorize_cfg = cfg.get("colorize")
    if colorize_cfg is None:
        if feature_dim != 3:
            raise ValueError(
                f"model.feature_dim={feature_dim} requires a 'colorize' config section. "
                "Add `\"colorize\": {\"hidden_dim\": null, \"activation\": \"sigmoid\"}` to the train config, "
                "or set model.feature_dim=3 for the legacy RGB path."
            )
        return None, "none", True

    view_condition = normalize_view_condition(colorize_cfg.get("view_condition", "none"))
    detach = bool(colorize_cfg.get("detach_view_condition", True))
    module = FeatureToColor(
        feature_dim=feature_dim,
        hidden_dim=colorize_cfg.get("hidden_dim"),
        activation=colorize_cfg.get("activation", "sigmoid"),
        pre_norm=bool(colorize_cfg.get("pre_norm", False)),
        weight_init=str(colorize_cfg.get("weight_init", "kaiming")).lower(),
        weight_init_gain=float(colorize_cfg.get("weight_init_gain", 1.0)),
        view_condition=view_condition,
    ).to(device)
    return module, view_condition, detach
```

This is the only helper that **takes raw config**. Every other helper
takes already-validated tensors and modules. The factory is a one-shot
boundary; once it returns, the trainer doesn't ask config questions about
colorize again.

## Every existing function that gets changed

### `Trainer.recon_backward` (was: 63 lines)

After the refactor (~28 lines, no behaviour change for single-cam):

```python
def recon_backward(
    self,
    clip_frames: torch.Tensor,
    decoded: GaussianSequence,
    regularizer_loss: torch.Tensor,
    keep_preview: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if decoded.cameras is None:
        raise ValueError("Implicit-camera video decode must include cameras.")
    frame_count = len(decoded.cameras)
    chunk_size = self.temporal_recon_chunk_size(frame_count)
    target_frames = resize_images(clip_frames[0], self.render_size)
    bg_policy = random_bg()
    recon_loss = clip_frames.new_tensor(0.0)
    preview_render = None
    preview_features = None

    for chunk_start in range(0, frame_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, frame_count)
        chunk_seq = gaussian_sequence_slice(decoded, chunk_start, chunk_end)
        bundle = render_clip_with_alpha(
            chunk_seq, tuple(decoded.cameras[chunk_start:chunk_end]),
            renderer_mode=self.renderer_mode, render_cfg=self.render_cfg,
            input_size=self.model_cfg["size"], render_size=self.render_size, dense_grid=self.dense_grid,
        )
        view_dirs = colorize_view_dirs_for_features(
            bundle.features, bundle.cameras,
            view_condition=self.colorize_view_condition,
            input_size=self.model_cfg["size"], render_size=self.render_size,
            detach=self.colorize_detach_view_condition,
        )
        chunk_renders = compose_rendered_rgb(
            bundle, colorize=self.colorize, view_dirs=view_dirs, background=bg_policy,
        )
        if keep_preview and self.feature_pca_log and preview_features is None:
            preview_features = bundle.features[0].detach()
        if keep_preview and preview_render is None:
            preview_render = chunk_renders[0].detach()
        chunk_loss = compute_recon_loss(
            chunk_renders, target_frames[chunk_start:chunk_end],
            loss_cfg=self.loss_cfg, frame_count_for_normalization=frame_count,
        )
        recon_loss = recon_loss + chunk_loss.detach()
        is_last_chunk = chunk_end == frame_count
        backward_loss = chunk_loss + (regularizer_loss if is_last_chunk else 0.0)
        backward_loss.backward(retain_graph=not is_last_chunk)

    return recon_loss, preview_render, preview_features
```

Same control flow, same `.backward(retain_graph=...)` semantics, same
chunk loop. The four duplicated lines collapse into one
`compose_rendered_rgb` call. The bg sampling moved out of the loop and
into `random_bg()` (sampled once when the policy is materialized inside
`compose_rendered_rgb` per chunk — which IS once per step in the existing
code because the random tensor is broadcast across chunks; this proposal
slightly weakens that to per-chunk sampling. **Caveat: this is a small
semantic change.** See "Risk analysis" below.)

### `MulticamPrecomputedFeatureImplicitTrainer.multicam_recon_loss` (was: 17 lines, broken)

After the refactor (~14 lines, alpha-aware, no longer broken):

```python
def multicam_recon_loss(
    self,
    decoded,
    *,
    clip_indices: torch.Tensor,
    views: list[int],
    keep_preview: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    recon_loss = self.multicam_bundle.train_frames.new_zeros(())
    preview_render = None
    # one bg shared across all views per step (matches single-cam "one bg per step" semantic)
    shared_bg_tensor = sample_random_background(device=self.device, dtype=self.multicam_bundle.train_frames.dtype)
    bg_policy = custom_bg(shared_bg_tensor)
    for view in views:
        bundle = self.render_view_clip_bundle(decoded, view=int(view), clip_indices=clip_indices)
        view_dirs = colorize_view_dirs_for_features(
            bundle.features, bundle.cameras,
            view_condition=self.colorize_view_condition,
            input_size=self.model_cfg["size"], render_size=self.render_size,
            detach=self.colorize_detach_view_condition,
        )
        rendered_rgb = compose_rendered_rgb(bundle, colorize=self.colorize, view_dirs=view_dirs, background=bg_policy)
        target = resize_images(self.multicam_bundle.train_frames[int(view), clip_indices], self.render_size)
        recon_loss = recon_loss + compute_recon_loss(rendered_rgb, target, loss_cfg=self.loss_cfg)
        if keep_preview and preview_render is None:
            preview_render = rendered_rgb[0].detach()
    return recon_loss / float(max(len(views), 1)), preview_render
```

Note `render_view_clip_bundle`: a thin rename of the existing
`render_view_clip` that returns a `RenderedClipBundle` instead of the raw
tuple. Its body is one line: `return render_clip_with_alpha(decoded, ...,
cameras=self.camera_rig.cameras_for_view(view, clip_indices), ...)`. The
old method name is removed.

### `Trainer.validation_video_payload` (was: 89 lines)

After (~35 lines; payload assembly is the helper):

```python
def validation_video_payload(self) -> dict[str, Any]:
    sequences = self.eval_sequences or [self.sequence_data]
    metric_payloads = []
    payload: dict[str, Any] = {"Eval/SequenceCount": len(sequences)}
    for sequence_index, sequence_data in enumerate(sequences):
        rendered_sequence, eval_camera_state, decoded_metrics, feature_sequence, alpha_sequence = (
            self.render_full_sequence(sequence_data)
        )
        gt_sequence = resize_images(sequence_data.frames, self.render_size).detach().cpu()
        metrics = {
            **eval_metric_payload(rendered_sequence, gt_sequence, self.loss_cfg),
            **temporal_similarity_payload(rendered_sequence, gt_sequence, self.loss_cfg),
            **decoded_metrics,
        }
        if eval_camera_state is not None:
            metrics.update(
                {
                    "Camera/EvalFOVDegrees": eval_camera_state.fov_degrees.item(),
                    "Camera/EvalRadius": eval_camera_state.radius.item(),
                    "Camera/EvalRotationDeltaMeanDegrees":
                        torch.rad2deg(torch.linalg.norm(eval_camera_state.rotation_delta, dim=-1)).mean().item(),
                    "Camera/EvalTranslationDeltaMean":
                        torch.linalg.norm(eval_camera_state.translation_delta, dim=-1).mean().item(),
                }
            )
            metrics.update(camera_temporal_payload(eval_camera_state))
        metric_payloads.append(metrics)
        if sequence_index == 0:
            features_pca = feature_pca_to_rgb(feature_sequence) if feature_sequence is not None else None
            payload.update(build_validation_video_payload(
                gt=gt_sequence, rendered=rendered_sequence, alpha=alpha_sequence,
                features_pca=features_pca, fps=sequence_data.video_fps,
                log_gt=not self.gt_video_logged,
            ))
            self.gt_video_logged = True

    metric_keys = sorted({key for item in metric_payloads for key in item})
    for key in metric_keys:
        values = [item[key] for item in metric_payloads if key in item]
        payload[key] = sum(values) / len(values)
    return payload
```

`render_full_sequence` is unchanged (still returns the 5-tuple) — that's
intentional, it's the only place that does the multi-clip
camera-state-merge work and rewriting it adds risk for no payoff.

### `Trainer.initial_step_result` (was: 50 lines)

The colorize+composite block at lines 1408-1416 collapses to:

```python
bundle = render_clip_with_alpha(decoded, decoded.cameras, ...)  # same args as before
view_dirs = colorize_view_dirs_for_features(bundle.features, bundle.cameras, ...)
rendered_clip = compose_rendered_rgb(bundle, colorize=self.colorize, view_dirs=view_dirs, background=white_bg())
preview_features = bundle.features[0].detach() if self.feature_pca_log else None
target_frames = resize_images(clip_frames[0], self.render_size)
recon_loss = compute_recon_loss(rendered_clip, target_frames, loss_cfg=self.loss_cfg)
```

Six lines, no branching, no tuple-unpack. Background is `white_bg()`
explicitly because this is eval-time. The `1.0` literal vanishes from
the source.

### `KnownCameraTrainer.initial_step_result` (was: buggy)

Same shape as `Trainer.initial_step_result`. The latent tuple-as-tensor
bug at line 1897 is fixed because `render_decoded_clip`'s replacement
returns a `RenderedClipBundle` and the bundle's `.features` is always
a tensor. The missing alpha composition is fixed because
`compose_rendered_rgb(... background=white_bg())` is called
unconditionally — same code path as the implicit-camera trainer.

The investigator 02 audit said two things were wrong here at once
(tuple-unpack + missing composition); both are fixed by routing through
the same helper.

## Every file that gets deleted

Investigator-confirmed dead files (no callers, no scripts):

| File | Investigator | Prerequisite |
|---|---|---|
| `src/train/train_camera_implict_dynamic.py` (typo) | 01 | none — unconditional delete |
| `src/train/dynamicTokenGS_shared.py` | 01 §"Status / deletion candidates" | `git grep "from dynamicTokenGS_shared"` returns 0 hits |
| `src/train/dynamicTokenGS_tiled.py` | 01 | `git grep "from dynamicTokenGS_tiled"` returns 0 hits |
| `src/train/tokenGS_shared.py` | 05 | `git grep "from tokenGS_shared"` returns 0 hits |
| `src/train/tokenGS_tiled.py` | 05 | `git grep "from tokenGS_tiled"` returns 0 hits |
| `src/train/train_ltx_feature_implicit_dynamic.py` | 01 | the `arch=ltx_feature_implicit_camera` config (`local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`) must dispatch to `train_precomputed_feature_implicit_dynamic.py` instead. Investigator 01 notes the launcher script already does this; only the empty subclass is dead. |

The `train_image_encoder_implicit_camera_baseline.py` 4-line shim is left
alone — it's harmless and keeping it costs nothing.

## Migration plan, step by step

Each step is independently runnable. After each step, the smoke test
named below MUST pass before moving on. The user explicitly called out
"py_compile-clean-but-broken" mid-cascade states; each step ends with a
real `python <trainer>.py <config>` smoke that exercises the alpha-aware
path with both F=3 and F!=3 to catch tuple-unpack bugs at the call graph.

**Step 1 — Build the package skeleton.**
- New: `src/train/training_common/__init__.py`,
  `src/train/training_common/render_bundle.py` (with `RenderedClipBundle`
  + `BackgroundPolicy`).
- Edited: zero existing files.
- Smoke: `uv run pytest tests/test_render_bundle.py` (new file, see "Test
  surface").
- Risk: zero — nothing is imported by trainers yet.

**Step 2 — Add `compose.py`, `recon_loss.py`, `render_clip.py`.**
- New: 3 modules with the helpers above. Each one's body is small enough
  to write in a single sitting. None of them are imported yet.
- Smoke: `uv run pytest tests/test_compose.py` and the existing trainer
  smoke tests still pass (the trainers haven't been touched).
- Risk: zero — still pure addition.

**Step 3 — Re-point `Trainer.recon_backward` to the new helpers.**
- Edit: `src/train/train_video_token_implicit_dynamic.py` —
  `Trainer.recon_backward`. Body shrinks per the snippet above. The
  trainer's `self.colorize`, `self.colorize_view_condition`, etc. stay
  exactly where they are (the colorize factory comes in step 6).
- Smoke 1 (F=3): `uv run python src/train/train_video_token_implicit_dynamic.py
  src/train_configs/local_mac_overfit_video_token_smoke.jsonc` for 1 step.
- Smoke 2 (F=32 alpha): `uv run python src/train/train_precomputed_feature_implicit_dynamic.py
  src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc`
  with `train.steps=1` overridden.
- Both must produce a finite scalar loss and a non-NaN preview.
- Risk: medium. This is the biggest single edit. Mitigation:
  side-by-side diff against the original to confirm the only change is
  "duplicated 6-line block becomes one helper call." Step is reversible
  by `git checkout`.

**Step 4 — Re-point `Trainer.initial_step_result` and
`Trainer.render_full_sequence` (the trainer-method one at line 1550).**
- Edit: same file. Both methods now construct `RenderedClipBundle` via
  `render_clip_with_alpha` and call `compose_rendered_rgb(... white_bg())`
  for compositing.
- Fix: the latent
  `KnownCameraTrainer.initial_step_result` tuple-unpack bug is now
  unreachable because `render_decoded_clip` returns a bundle.
- Smoke 1: same as step 3 smokes.
- Smoke 2 (known camera): `uv run python src/train/train_video_token_implicit_dynamic.py
  src/train_configs/local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc`
  for 1 step. **This was previously broken** for any F!=3 use of
  `KnownCameraTrainer.initial_step_result`; the F=3 config tests the
  non-bug path. (Adding a known-camera F!=3 config is out of scope.)
- Risk: low — the changes are local and the smokes catch tuple-unpack
  errors.

**Step 5 — Fix the multicam trainer.** (THE alpha-aware fix.)
- Edit: `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
  — `render_view_clip` becomes `render_view_clip_bundle` returning a
  `RenderedClipBundle`. `multicam_recon_loss` body is replaced per the
  snippet above. `initial_step_result` calls the same. `render_full_external_views`
  unpacks the bundle. `validation_video_payload` keeps its existing shape
  (per-view, per-heldout) but builds video panels via
  `build_validation_video_payload` (step 7).
- The multicam trainer NOW gets alpha-aware composition for free. F!=3
  multicam configs that previously crashed (per investigator 02) now
  work.
- Smoke 1 (F=3 multicam):
  `uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py
  src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc`
  for 2 steps. Must produce finite loss and preview.
- Smoke 2 (F=32 alpha multicam — the previously-broken case):
  `uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py
  src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc`
  with `train.steps=2`. Must produce finite loss; previously this would
  have crashed at `reconstruction_loss_per_image(rendered, ...)` because
  `rendered` was a tuple.
- Risk: medium. This is the change we wrote the proposal for; step 5
  validates the whole investment. Mitigation: smokes catch every branch
  of the bug.

**Step 6 — Move colorize construction into the factory.**
- Edit: `src/train/train_video_token_implicit_dynamic.py:1007-1036`
  becomes a single call to `build_colorize_module_from_config`. The
  `self.feature_pca_log` validation stays inline (it's a different
  invariant).
- New file used: `src/train/training_common/colorize_factory.py`.
- Smoke: every previous trainer smoke still passes.
- Risk: very low — this is just a refactor of the constructor; behaviour
  is verbatim.

**Step 7 — Move video payload assembly into the helper.**
- Edit: both `Trainer.validation_video_payload` and the multicam
  validation method now call `build_validation_video_payload` from
  `training_common.video_logging`. The two replicated column-builds
  collapse into one helper.
- Smoke (single-cam): one full validation cycle on the F=32 alpha config
  — confirm `Render_Composite_Video`, `Alpha_Mask_Video`,
  `Feature_PCA_Video` all appear in the payload.
- Smoke (multicam): one full validation cycle on the F=3 multicam smoke;
  confirm `TrainView0_Rendered_Video` and `TrainView0_GT_Video` appear,
  no composite (no alpha collected yet on multicam).
- Risk: low — the helper accepts None for any column; multicam side
  doesn't need to start collecting alpha to pass.

**Step 8 — Delete dead files.**
- Delete the 6 files in the "deleted" list.
- Edit:
  `src/train_configs/local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`
  is unchanged; the launcher already targets the precomputed trainer
  directly per investigator 01.
- Smoke: a global `git grep` for each deleted module name shows zero
  hits. `uv run python -c "import src.train.tokenGS"` confirms no
  trainer config imports the deleted shims.
- Risk: very low — these files are dead per two investigators.

After step 8, the user can decide whether to extend this proposal to the
gauge-fields / legacy trainer territory, or stop here and call the
multicam-fix-plus-cleanup good.

## Test surface

Three new pure-helper test files. Each is small (each test under 30
lines) and targets a specific invariant.

### `tests/test_render_bundle.py`

```python
def test_bundle_rejects_wrong_alpha_shape():
    """RenderedClipBundle.__post_init__ catches a [T, H, W] alpha against
    [T, F, H', W'] features. Regression guard for the multicam
    tuple-as-tensor bug — if someone constructs a bundle from a stale
    tuple again, the dataclass complains at construction, not at the
    next .detach() call deep in the trainer."""
    with pytest.raises(ValueError, match="alpha shape mismatch"):
        RenderedClipBundle(
            features=torch.zeros(2, 3, 8, 8),
            alpha=torch.zeros(2, 16, 16),    # wrong H,W
            cameras=(_dummy_camera(), _dummy_camera()),
        )

def test_bundle_is_frozen():
    """Catches a future refactor that adds bundle.alpha = ... mutation
    inside a trainer method (the temptation will be real once the
    bundle is the central type)."""
    bundle = _make_valid_bundle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.alpha = None
```

Two tests, no implementation mirrors. The first catches the literal bug
the multicam trainer had today; the second catches a future
foot-gun.

### `tests/test_compose.py`

```python
def test_compose_alpha_blend_formula():
    """Numerical: alpha=1 returns splat exactly; alpha=0 returns bg
    exactly; alpha=0.5 returns the midpoint to within 1e-6. This is the
    actual math we care about, falsifiable, and a future refactor of
    compose_rendered_rgb that breaks the formula will be caught here."""
    bundle = _bundle(features=torch.full((1, 3, 4, 4), 0.5),
                     alpha=torch.tensor([[[0.0, 0.5, 1.0, ...]]]))
    rgb = compose_rendered_rgb(bundle, colorize=_identity_colorize(), view_dirs=None,
                               background=scalar_bg(0.0))
    # alpha=1 -> 0.5 (splat), alpha=0 -> 0.0 (bg), alpha=0.5 -> 0.25
    expected = torch.tensor([0.0, 0.25, 0.5])
    assert torch.allclose(rgb[0, 0, 0, :3], expected, atol=1e-6)

def test_compose_no_colorize_F3_passthrough():
    """F=3 + colorize=None preserves the legacy RGB path bit-for-bit."""
    bundle = _bundle(features=torch.rand(2, 3, 8, 8), alpha=None)
    out = compose_rendered_rgb(bundle, colorize=None, view_dirs=None, background=white_bg())
    assert torch.equal(out, bundle.features)

def test_compose_no_colorize_F32_raises():
    """F!=3 + colorize=None is a caller bug (the trainer would have
    raised in __init__, but the helper validates anyway). This is the
    invariant Trainer.__init__:1025-1030 already enforces; we mirror it
    so a future caller that bypasses the trainer constructor still gets
    the message."""
    with pytest.raises(ValueError, match="requires a colorize module"):
        compose_rendered_rgb(_bundle(F=32), colorize=None, view_dirs=None, background=white_bg())
```

### `tests/test_sample_random_background.py`

```python
def test_random_bg_shape_and_range():
    """Output is [1, 3, 1, 1] in [0, 1)."""
    bg = sample_random_background(device=torch.device("cpu"), dtype=torch.float32)
    assert bg.shape == (1, 3, 1, 1)
    assert bg.min() >= 0.0 and bg.max() < 1.0

def test_random_bg_deterministic_with_generator():
    """Two calls with the same seed produce the same output. This is the
    reproducibility hook investigator 02 flagged as missing today."""
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    bg1 = sample_random_background(device=torch.device("cpu"), dtype=torch.float32, generator=g1)
    bg2 = sample_random_background(device=torch.device("cpu"), dtype=torch.float32, generator=g2)
    assert torch.equal(bg1, bg2)
```

Six tests total. Every one of them catches a specific bug; none of them
is a "the field exists" mirror. Total test budget: <10 minutes to write
each, <1 second each at runtime.

## Risk analysis

### Risk 1: zombie code (helpers added, duplicated logic not removed)

The biggest failure mode for an additive refactor is "we added the
helper but the trainers never started calling it." Two compounding
defenses:

1. The migration plan has each trainer-edit step (3, 4, 5) **gated by a
   real smoke test**. We don't move on until the trainer is calling the
   helper and producing the same loss it produced before.
2. After step 5, a single `git grep "alpha_expanded \\* "` search in
   `src/train/` should return ZERO hits. The compositing math is no
   longer inline anywhere; it lives once in `compose.py`. A reviewer can
   run that grep as the sanity check.

Mitigation: include the grep in step 8's checklist.

### Risk 2: bg sampling moved from "per-step" to "per-chunk"

The rewritten `Trainer.recon_backward` materializes the random bg inside
`compose_rendered_rgb`, which is called once per chunk. The original
code samples ONE bg per step and broadcasts across chunks. For the
default `recon_backward_strategy=batched` (single chunk), these are
identical. For `microbatch` and `framewise`, the new behaviour samples
a fresh bg per chunk.

Investigator 02 §"The random per-step background" calls this out as an
open question. Two paths:

- **Path A (preserve verbatim):** sample once outside the loop, pass a
  `custom_bg(tensor)` to `compose_rendered_rgb`. Same semantics as
  today. ~2 lines extra in `recon_backward`.
- **Path B (new behaviour):** `random_bg()` policy, fresh sample per
  chunk. Slightly more bg diversity per step.

This proposal recommends **Path A** for the migration to keep
behaviour-identical, then a separate one-line change later if Path B is
preferred. Mitigation: spell this out in the step-3 smoke (compare
loss curves to a recent W&B baseline; should be identical to within
RNG noise on the F=3 path).

### Risk 3: helper signatures churn after callers are pointed at them

If the `compose_rendered_rgb` signature changes after the trainers have
adopted it, every trainer needs a coordinated edit. Mitigation: the
proposal pins the signatures *before* step 3, every helper has a unit
test that locks the contract, and the trainer code only knows the
helper through these signatures. Adding a new feature later
(e.g. depth output) is an additive new helper, not a signature change.

### Risk 4: eval-time fixed-bg vs train-time random-bg toggle gets confused

Today, the trainer has an implicit "if I'm in `recon_backward` it's
training, otherwise it's eval" rule. After the refactor, the trainer
explicitly says `random_bg()` in `recon_backward` and `white_bg()`
everywhere else. The `BackgroundPolicy` sum type makes the asymmetry
*visible*. Future ablations (e.g. "what if eval also uses random bg?")
become a one-line config change at the call site, not a ten-line search
through the trainer.

Mitigation: the policy type is its own argument (not derived from a
`training: bool` flag), so a reader can grep `compose_rendered_rgb(`
and see what bg every site uses.

### Risk 5: `RenderedClipBundle.cameras` shape drift

The bundle stores `cameras` (post-viewport-scaling), and downstream
helpers like `colorize_view_dirs_for_features` expect post-scaling
cameras. The investigator 02 audit shows a few ad-hoc sites that pass
`decoded.cameras` (pre-scaling) into colorize — these go away with the
refactor because the bundle hands back the right cameras. But there's a
scenario where someone wants to colorize against the original cameras
for some debugging reason; the bundle's `cameras` is the wrong choice
in that scenario.

Mitigation: the dataclass docstring says "post-viewport scaling" and
the test `test_bundle_cameras_match_features` (added if needed) locks
it. Any new caller that wants pre-scaling cameras has access to the
sequence's own `decoded.cameras` field — they don't have to dig the
pre-scaling cameras out of the bundle.

### Risk 6: deleting `train_ltx_feature_implicit_dynamic.py` changes a print prefix

Investigator 01 notes the empty subclass body was kept "for the
backward-compat alias." The only behavioural difference is that the
subclass's `run()` prints `"LTX feature implicit camera trainer ..."`
before calling `super().run()`. Mitigation: if the prefix is
load-bearing for someone's terminal grep, restore it as a one-line
print at the top of `PrecomputedFeatureImplicitTrainer.run` gated on
`config.get("arch") == "ltx_feature_implicit_camera"`. Cheap.

## Tradeoffs vs the other two proposals

What this proposal **gets right**:

- **Smallest blast radius.** Every step is a few hundred lines of diff.
  The class hierarchy is unchanged; the four trainers stay; the people
  who already know their way around `Trainer` and
  `MulticamPrecomputedFeatureImplicitTrainer` don't have to relearn
  anything.
- **The bug we wrote the proposal for is fixed by step 5.** Nothing
  else has to land for the multicam alpha-aware fix to ship.
- **No new abstraction layer.** Helpers are plain functions; dataclasses
  are plain frozen dataclasses; there's no registry, no plugin, no
  pipeline graph. A new contributor reads `compose_rendered_rgb` and
  understands it in 30 seconds.
- **Honors the project's `key_learnings.md:18`:** "A single shared
  `BaseTrainer` would hide real differences ... shared payload
  contracts are cleaner than shared trainer inheritance." This proposal
  does exactly that — `RenderedClipBundle` is a shared payload, not a
  shared trainer.

What this proposal **gives up**:

- **Doesn't unify the legacy trainers.** `dynamicTokenGS.py`,
  `train_camera_implicit_dynamic.py`, `tokenGS.py` keep their own
  parallel paths. The proposer-2 (DI / strategy) approach could pull
  those into a shared trainer; this proposal explicitly does not. If
  the user wants one trainer to rule them all, this isn't it.
- **Doesn't fix the renderer-side dispatch.** Investigator 03 §
  "feature_dim dispatch in fast_mac.py" notes that v5 vs v5_features
  routing is silent and structural. This proposal lives one layer
  above; the trainer-side bundle hides the dispatch from trainers but
  doesn't reshape it. A proposer-3-style typed dataflow could push
  through to the renderer.
- **Doesn't solve the config-schema mess.** Investigator 04 documents
  30+ required-but-not-defaulted keys, the `arch` field that's read by
  no one, and the per-trainer DEFAULTS dicts. This proposal adds
  exactly one factory (`build_colorize_module_from_config`) and otherwise
  leaves config alone. A bigger config-driven approach is a different
  proposal.
- **Doesn't solve the `feature_dim`-silently-dropped bug in
  `FreeGaussianBankImplicitCamera`.** Investigator 05 §"feature_dim
  thread audit" flags two model classes with broken `**_unused`. This
  proposal touches no model code. A proposer that takes a model-side
  view of typing could catch it.
- **Per-chunk bg sampling drift (Risk 2).** Resolvable but the user
  must pick a path before the merge.
- **The two `_camera_scalar_vector` copies remain duplicated.**
  Investigator 03 notes them; this proposal doesn't extend to renderer
  helpers.

In short: this proposal solves the load-bearing bug with the lowest
possible cost and leaves every other smell for a follow-up. If the user
needs a one-shot consolidation, proposers 2 and 3 are the right
trade-off; if the user needs to ship the multicam fix this week, this
is.

agent_notes/apr_30th_clean/proposal_01_functional_helpers.md
