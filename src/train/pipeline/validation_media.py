"""Validation-time W&B media-logging helpers.

Extracted from `train_video_token_implicit_dynamic.py` (single-cam) and
`train_multicam_precomputed_feature_implicit_dynamic.py` (multicam) so both
trainers go through one set of free functions for:

  - alpha-mask-to-grayscale-video conversion
  - the GT|Pred|Alpha|FeaturePCA side-by-side grid
  - scalar W&B payloads (per training step)
  - validation video W&B payloads (per validation gate)

Wave 1 — additive only. The trainer methods are not yet rewired to call into
this module; that wiring lands in wave 2 alongside smoke tests. None of these
helpers depend on `self`; everything is passed in explicitly so the same
function can serve single-cam and multicam paths.

Style notes (keep these in sync with `objective/objective.py`):
  - free functions, no classes
  - no `Optional[T]` for "didn't decide" — `T | None` only when None has
    semantic meaning (e.g. "no precomputed PCA available")
  - no silent fallbacks: if the caller asks for a column that requires data
    we don't have, raise loud
  - explicit deps as keyword-only args; no global state
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import wandb

from train_logging import (
    build_validation_video_payload,
    make_preview_image,
    make_wandb_video,
)

# `RenderedClip` is the typed bundle being introduced in `pipeline/render.py`
# by a parallel agent (see agent_notes/apr_30th_clean/FINAL_DESIGN.md). It is
# not yet on disk at the time this file is added; the import is a TODO so the
# wave-2 wiring agent can resolve it. Until then the public functions in this
# module take the raw tensors (rendered RGB + alpha + features) directly so
# they can be exercised without `RenderedClip`.
# TODO(wave-2): from pipeline.render import RenderedClip


__all__ = [
    "alpha_to_grayscale_video",
    "compose_gt_pred_alpha_pca_grid",
    "scalar_payload",
    "render_preview_image",
    "render_diagnostics_payload",
    "single_cam_validation_video_payload",
    "multicam_validation_video_payload",
    "make_wandb_video",
]


# --------------------------------------------------------------------------- #
# Tensor helpers
# --------------------------------------------------------------------------- #


def alpha_to_grayscale_video(alpha: torch.Tensor) -> torch.Tensor:
    """Convert an alpha sequence [T, H, W] into a 3-channel grayscale clip
    [T, 3, H, W], suitable for direct W&B video logging or appending to a
    composite grid via :func:`compose_gt_pred_alpha_pca_grid`.

    Both single-cam and multicam paths used to do this inline; this is the
    one canonical implementation.
    """
    if alpha.dim() != 3:
        raise ValueError(
            "alpha_to_grayscale_video expects [T, H, W]; "
            f"got shape {tuple(alpha.shape)}."
        )
    return alpha.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


def compose_gt_pred_alpha_pca_grid(
    *,
    gt: torch.Tensor,
    pred: torch.Tensor,
    alpha_video: torch.Tensor | None = None,
    pca_video: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build a width-concatenated [T, 3, H, K*W] composite of the validation
    media columns. Returns ``None`` when only GT and Pred are available, since
    that two-column case is already covered by ``Render_GT_Video``.

    All inputs must already be 4-D ``[T, 3, H, W]`` and aligned in T/H/W.
    Pass ``alpha_video`` already-grayscaled (run :func:`alpha_to_grayscale_video`
    first) and ``pca_video`` already-projected (use
    ``feature_pca_viz.feature_pca_to_rgb``).
    """
    columns: list[torch.Tensor] = [gt, pred]
    if alpha_video is not None:
        columns.append(alpha_video)
    if pca_video is not None:
        columns.append(pca_video)
    if len(columns) <= 2:
        return None
    return torch.cat(columns, dim=-1)


# --------------------------------------------------------------------------- #
# Per-step scalar payload
# --------------------------------------------------------------------------- #


def _camera_state_metrics(camera_state) -> dict[str, float]:
    """Mirrors `Trainer.camera_metrics` from the single-cam monolith."""
    return {
        "fov_degrees": camera_state.fov_degrees.item(),
        "radius": camera_state.radius.item(),
        "rotation_delta_mean_degrees": (
            torch.rad2deg(torch.linalg.norm(camera_state.rotation_delta, dim=-1)).mean().item()
        ),
        "translation_delta_mean": torch.linalg.norm(camera_state.translation_delta, dim=-1).mean().item(),
    }


def scalar_payload(
    result,
    *,
    train_frame_count: int,
    train_sequence_count: int,
    eval_sequence_count: int,
    input_size: int,
    render_size: int,
) -> dict[str, Any]:
    """Build the per-step W&B scalar payload from a `StepResult`.

    Extracted verbatim from `Trainer.scalar_payload` in
    `train_video_token_implicit_dynamic.py` (lines 1380-1407). The original
    method pulled `train_frame_count`, sequence counts, `input_size`, and
    `render_size` off `self.model_cfg` / `self.train_sequences` etc.; this
    free-function variant takes them as keyword-only deps so it can serve
    both single-cam and multicam without `self`.

    NOTE for wiring agent: the original prompt suggested signature
    ``scalar_payload(result, *, model_input, opacity_stats)``. Neither
    `model_input` nor `opacity_stats` is referenced by the existing
    `Trainer.scalar_payload` body. I followed the actual deps; if the wiring
    agent wants those wired in for a richer scalar dump later, add them as
    additional keyword args and append to the returned dict.
    """
    payload: dict[str, Any] = {
        "Loss": result.loss.item(),
        "Loss/Reconstruction": result.recon_loss.item(),
        "Loss/CameraMotion": result.camera_motion_loss.item(),
        "Loss/CameraTemporal": result.camera_temporal_loss.item(),
        "Loss/CameraGlobal": result.camera_global_loss.item(),
        "Loss/BankRate": result.bank_rate_loss.item(),
        "TrainFrameCount": int(train_frame_count),
        "SequenceFrames": result.sequence_frame_count,
        "TrainSequenceCount": int(train_sequence_count),
        "EvalSequenceCount": int(eval_sequence_count),
        "InputSize": int(input_size),
        "RenderSize": int(render_size),
    }
    if result.camera_state is not None:
        metrics = _camera_state_metrics(result.camera_state)
        payload.update(
            {
                "Camera/FOVDegrees": metrics["fov_degrees"],
                "Camera/Radius": metrics["radius"],
                "Camera/RotationDeltaMeanDegrees": metrics["rotation_delta_mean_degrees"],
                "Camera/TranslationDeltaMean": metrics["translation_delta_mean"],
            }
        )
    for key, value in result.bank_rate_terms.items():
        payload[f"BankRate/{key}"] = value.item()
    return payload


# --------------------------------------------------------------------------- #
# Per-step preview image
# --------------------------------------------------------------------------- #


def render_preview_image(result, *, render_size: int, step: int) -> wandb.Image:
    """Build the GT-vs-Pred preview image for the per-step image-log gate.

    Extracted from `Trainer.render_preview_image`
    (`train_video_token_implicit_dynamic.py` lines 1409-1413). The original
    pulled `self.render_size` directly; this version takes it as a kwarg.

    `resize_images` is imported lazily so this module can be imported in a
    smoke that doesn't have the renderer dependency stack on PYTHONPATH.
    """
    if result.preview_render is None:
        raise ValueError(
            "Preview render was requested for logging but was not retained "
            "during the training step."
        )
    from rendering import resize_images

    target = resize_images(result.clip_frames[0, 0], render_size)
    return make_preview_image(target, result.preview_render, caption=f"Step {step}")


# --------------------------------------------------------------------------- #
# Render-diagnostics block (Alpha mask + Feature PCA + composite grid)
# --------------------------------------------------------------------------- #


def render_diagnostics_payload(
    *,
    prefix: str,
    target: torch.Tensor,
    pred: torch.Tensor,
    alpha: torch.Tensor | None,
    features: torch.Tensor | None,
    feature_pca_log: bool,
    fps: float,
) -> dict[str, Any]:
    """Build the per-view diagnostic W&B payload (alpha mask video, feature
    PCA video, composite grid) and return it as a dict keyed by the given
    ``prefix``.

    This is the canonical extraction of the multicam
    `add_render_diagnostics` method (lines 406-427 of
    `train_multicam_precomputed_feature_implicit_dynamic.py`); the single-cam
    `validation_video_payload` had the same logic inlined. Both go through
    this helper now.

    Args:
        prefix: Key prefix for the W&B dict (e.g. ``"TrainView0"`` or
            ``""`` for single-cam where there's no view prefix).
        target: GT clip ``[T, 3, H, W]`` already at render resolution.
        pred: Predicted RGB clip ``[T, 3, H, W]``.
        alpha: Optional alpha mask ``[T, H, W]`` from the alpha-aware
            renderer; when ``None``, no alpha video / no alpha column.
        features: Optional ``[T, F, H, W]`` feature buffer. Only consulted
            when ``feature_pca_log`` is True.
        feature_pca_log: Whether to compute & log the F-channel PCA video.
        fps: Output video FPS (typically ``sequence_data.video_fps``).
    """
    payload: dict[str, Any] = {}
    composite_columns: list[torch.Tensor] = [target, pred]

    def _key(suffix: str) -> str:
        return f"{prefix}_{suffix}" if prefix else suffix

    if alpha is not None:
        alpha_video = alpha_to_grayscale_video(alpha)
        payload[_key("Alpha_Mask_Video")] = make_wandb_video(alpha_video, fps)
        composite_columns.append(alpha_video)

    if feature_pca_log:
        if features is None:
            raise ValueError(
                "feature_pca_log=True requires F-channel features; got None. "
                f"prefix={prefix!r}"
            )
        # Lazy import: feature_pca_viz pulls in torch.linalg.svd which can
        # be slow to import on test machines that don't need PCA.
        from feature_pca_viz import feature_pca_to_rgb

        pca = feature_pca_to_rgb(features)
        payload[_key("Feature_PCA_Video")] = make_wandb_video(pca, fps)
        composite_columns.append(pca)

    if len(composite_columns) > 2:
        composite = torch.cat(composite_columns, dim=-1)
        payload[_key("Render_Composite_Video")] = make_wandb_video(composite, fps)

    return payload


# --------------------------------------------------------------------------- #
# Single-cam validation payload
# --------------------------------------------------------------------------- #


def single_cam_validation_video_payload(
    *,
    sequence_index: int,
    rendered_sequence: torch.Tensor,
    gt_sequence: torch.Tensor,
    feature_sequence: torch.Tensor | None,
    alpha_sequence: torch.Tensor | None,
    eval_payload: Mapping[str, float],
    feature_pca_log: bool,
    gt_video_logged: bool,
    fps: float,
) -> tuple[dict[str, Any], bool]:
    """Build the per-sequence W&B media payload for a single-cam validator.

    Extracted from `Trainer.validation_video_payload` lines 1507-1595 of
    `train_video_token_implicit_dynamic.py`. The original method ran the
    full eval-render loop, computed metrics, AND built the W&B payload in
    one body; this free function only handles the W&B media + side-by-side
    composition step. The caller is responsible for:

      - rendering the full sequence (giving us ``rendered_sequence``,
        ``alpha_sequence``, ``feature_sequence``)
      - resizing the GT (giving us ``gt_sequence``)
      - computing eval metrics (``eval_payload``)

    Only the first sequence (``sequence_index == 0``) gets media; the
    remaining sequences only contribute their metrics. This matches the
    existing trainer behaviour exactly.

    The ``gt_video_logged`` flag is passed in and a new value returned so
    the caller can keep the one-shot semantics without us touching ``self``.

    Returns:
        ``(payload, gt_video_logged_after)`` — payload includes the
        eval-metric scalars merged in; the second value is the updated
        flag (True iff GT was logged this call or previously).
    """
    payload: dict[str, Any] = dict(eval_payload)

    if sequence_index != 0:
        return payload, gt_video_logged

    payload.update(
        build_validation_video_payload(rendered_sequence, gt_sequence, fps)
    )
    if not gt_video_logged:
        payload["GT_Video"] = make_wandb_video(gt_sequence, fps)
        gt_video_logged_after = True
    else:
        gt_video_logged_after = gt_video_logged

    diagnostics = render_diagnostics_payload(
        prefix="",
        target=gt_sequence,
        pred=rendered_sequence,
        alpha=alpha_sequence,
        features=feature_sequence,
        feature_pca_log=feature_pca_log,
        fps=fps,
    )
    payload.update(diagnostics)
    return payload, gt_video_logged_after


# --------------------------------------------------------------------------- #
# Multicam validation payload
# --------------------------------------------------------------------------- #


def multicam_validation_video_payload(
    *,
    train_rendered: list,           # list[RenderedView] — one per training view
    heldout_rendered: list,         # list[RenderedView] — one per held-out view (may be empty)
    train_targets: list[torch.Tensor],          # [V_train] of [T, 3, H, W]
    heldout_targets: list[torch.Tensor],        # [V_heldout] of [T, 3, H, W]
    train_camera_names: list[str] | tuple[str, ...],
    heldout_camera_names: list[str] | tuple[str, ...],
    decoded_metrics: Mapping[str, float],
    eval_metric_fn,                 # callable(pred, target, loss_cfg) -> dict
    temporal_metric_fn,              # callable(pred, target, loss_cfg) -> dict
    prefix_eval_metrics_fn,         # callable(prefix, metrics) -> dict
    loss_cfg: Mapping[str, Any],
    camera_rig_metrics: Mapping[str, float],
    feature_pca_log: bool,
    gt_video_logged: bool,
    fps: float,
) -> tuple[dict[str, Any], bool]:
    """Build the multicam validation W&B payload.

    Extracted from
    `MulticamPrecomputedFeatureImplicitTrainer.validation_video_payload`
    (lines 429-484 of
    `train_multicam_precomputed_feature_implicit_dynamic.py`). The original
    method ran the full multi-view render then formatted the payload in one
    body; this free function only does the formatting. The caller renders
    and resizes ahead of time.

    Both metric-helpers (`eval_metric_fn`, `temporal_metric_fn`,
    `prefix_eval_metrics_fn`) are taken as callables instead of imported
    here so this module stays decoupled from the single-cam monolith. The
    wiring agent should pass `eval_metric_payload`, `temporal_similarity_payload`,
    and the local `_prefix_eval_metrics` from
    `train_multicam_precomputed_feature_implicit_dynamic.py:62`.

    Returns:
        ``(payload, gt_video_logged_after)`` — see
        :func:`single_cam_validation_video_payload`.
    """
    payload: dict[str, Any] = {
        "Eval/SequenceCount": 1,
        **camera_rig_metrics,
    }

    # Train views
    for view, rendered in enumerate(train_rendered):
        target = train_targets[view]
        metrics = {
            **eval_metric_fn(rendered.rgb, target, loss_cfg),
            **temporal_metric_fn(rendered.rgb, target, loss_cfg),
        }
        if view == 0:
            metrics.update(decoded_metrics)
        payload.update(prefix_eval_metrics_fn(f"TrainView{view}", metrics))
        if view == 0:
            payload["TrainView0_Rendered_Video"] = make_wandb_video(rendered.rgb, fps)
            payload.update(
                render_diagnostics_payload(
                    prefix="TrainView0",
                    target=target,
                    pred=rendered.rgb,
                    alpha=rendered.alpha,
                    features=rendered.features if feature_pca_log else None,
                    feature_pca_log=feature_pca_log,
                    fps=fps,
                )
            )
            if not gt_video_logged:
                payload["TrainView0_GT_Video"] = make_wandb_video(target, fps)

    # Held-out views (novel-view metrics)
    if heldout_rendered:
        for view, rendered in enumerate(heldout_rendered):
            heldout_target = heldout_targets[view]
            camera_name = (
                heldout_camera_names[view]
                if view < len(heldout_camera_names)
                else f"view{view}"
            )
            prefix = f"Heldout{view}_{camera_name}"
            heldout_metrics = {
                **eval_metric_fn(rendered.rgb, heldout_target, loss_cfg),
                **temporal_metric_fn(rendered.rgb, heldout_target, loss_cfg),
            }
            payload.update(prefix_eval_metrics_fn(prefix, heldout_metrics))
            payload[f"{prefix}_Rendered_Video"] = make_wandb_video(rendered.rgb, fps)
            payload.update(
                render_diagnostics_payload(
                    prefix=prefix,
                    target=heldout_target,
                    pred=rendered.rgb,
                    alpha=rendered.alpha,
                    features=rendered.features if feature_pca_log else None,
                    feature_pca_log=feature_pca_log,
                    fps=fps,
                )
            )
            if not gt_video_logged:
                payload[f"{prefix}_GT_Video"] = make_wandb_video(heldout_target, fps)

    gt_video_logged_after = True

    summary = {
        key: round(float(value), 4)
        for key, value in payload.items()
        if key.endswith("/Eval/PSNR") or key.endswith("/Eval/SSIM")
    }
    if summary:
        print({"multicam_eval": summary})
    return payload, gt_video_logged_after
