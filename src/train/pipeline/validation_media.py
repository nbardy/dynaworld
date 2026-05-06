"""Validation-time W&B scalar and media payload helpers.

The single-cam and multicam trainers call this module for preview images,
validation videos, alpha-mask videos, feature-PCA videos, and composite
GT|Pred|Alpha|FeaturePCA grids. The helpers are pure payload builders: callers
pass the rendered tensors, targets, metrics, and logging config explicitly, and
missing required diagnostic inputs raise instead of silently dropping columns.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import wandb

from pipeline.diagnostics import eval_metric_payload, temporal_similarity_payload
from train_logging import (
    build_validation_video_payload,
    make_preview_image,
    make_wandb_video,
)

__all__ = [
    "alpha_to_grayscale_video",
    "compose_gt_pred_alpha_pca_grid",
    "compose_multicam_feature_gt_render_grid",
    "compose_multicam_diagnostic_grid",
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


def compose_multicam_diagnostic_grid(rows: list[torch.Tensor]) -> torch.Tensor | None:
    """Stack per-view GT|Pred|Alpha|Feature rows into one multicam video.

    Each row is a [T, 3, H, W_total] tensor, normally produced by
    :func:`compose_gt_pred_alpha_pca_grid`. The output stacks views vertically:
    [T, 3, rows*H, W_total].
    """
    if not rows:
        return None
    reference_shape = rows[0].shape
    for index, row in enumerate(rows):
        if row.shape != reference_shape:
            raise ValueError(
                "Multicam diagnostic rows must have identical shape; "
                f"row0={tuple(reference_shape)}, row{index}={tuple(row.shape)}."
            )
    return torch.cat(rows, dim=-2)


def compose_multicam_feature_gt_render_grid(
    *,
    feature_videos: list[torch.Tensor],
    gt_videos: list[torch.Tensor],
    render_videos: list[torch.Tensor],
) -> torch.Tensor | None:
    """Build a camera-column grid with rows FeaturePCA / GT / Render.

    Each input clip must be [T, 3, H, W]. The output is
    [T, 3, 3*H, camera_count*W], with cameras concatenated left-to-right and
    media types stacked top-to-bottom. Returns None when feature videos are not
    available, since the GT/Render two-row case is already covered by the
    existing per-view and GT|Pred media.
    """
    if not feature_videos:
        return None
    camera_count = len(gt_videos)
    if camera_count == 0:
        return None
    if len(feature_videos) != camera_count or len(render_videos) != camera_count:
        raise ValueError(
            "Multicam Feature/GT/Render grid requires matching camera counts: "
            f"features={len(feature_videos)}, gt={len(gt_videos)}, render={len(render_videos)}."
        )
    reference_shape = feature_videos[0].shape
    for label, videos in (("feature", feature_videos), ("gt", gt_videos), ("render", render_videos)):
        for index, video in enumerate(videos):
            if video.shape != reference_shape:
                raise ValueError(
                    "Multicam Feature/GT/Render grid clips must have identical shape; "
                    f"feature0={tuple(reference_shape)}, {label}{index}={tuple(video.shape)}."
                )
    return torch.cat(
        [
            torch.cat(feature_videos, dim=-1),
            torch.cat(gt_videos, dim=-1),
            torch.cat(render_videos, dim=-1),
        ],
        dim=-2,
    )


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
    cfg: Mapping[str, Any],
    result,
    *,
    train_sequence_count: int,
    eval_sequence_count: int,
) -> dict[str, Any]:
    """Build the per-step W&B scalar payload from a `StepResult`."""
    payload: dict[str, Any] = {
        "Loss": result.loss.item(),
        "Loss/Reconstruction": result.recon_loss.item(),
        "Loss/CameraMotion": result.camera_motion_loss.item(),
        "Loss/CameraTemporal": result.camera_temporal_loss.item(),
        "Loss/CameraGlobal": result.camera_global_loss.item(),
        "Loss/BankRate": result.bank_rate_loss.item(),
        "TrainFrameCount": int(cfg["model"]["train_frame_count"]),
        "SequenceFrames": result.sequence_frame_count,
        "TrainSequenceCount": int(train_sequence_count),
        "EvalSequenceCount": int(eval_sequence_count),
        "InputSize": int(cfg["model"]["size"]),
        "RenderSize": int(cfg["render"]["render_size"]),
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


def render_preview_image(cfg: Mapping[str, Any], result, step: int) -> wandb.Image:
    """Build the GT-vs-Pred preview image for the per-step image-log gate.

    `resize_images` is imported lazily so this module can be imported in a
    smoke that doesn't have the renderer dependency stack on PYTHONPATH.
    """
    if result.preview_render is None:
        raise ValueError(
            "Preview render was requested for logging but was not retained "
            "during the training step."
        )
    from rendering import resize_images

    target = resize_images(result.clip_frames[0, 0], int(cfg["render"]["render_size"]))
    return make_preview_image(target, result.preview_render, caption=f"Step {step}")


# --------------------------------------------------------------------------- #
# Render-diagnostics block (Alpha mask + Feature PCA + composite grid)
# --------------------------------------------------------------------------- #


def _diagnostic_media(
    cfg: Mapping[str, Any],
    *,
    target: torch.Tensor,
    pred: torch.Tensor,
    alpha: torch.Tensor | None,
    features: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    feature_pca_log = bool(cfg["logging"]["feature_pca_log"])
    alpha_video = alpha_to_grayscale_video(alpha) if alpha is not None else None
    pca_video = None
    if feature_pca_log:
        if features is None:
            raise ValueError("feature_pca_log=True requires F-channel features.")
        # Lazy import: feature_pca_viz pulls in torch.linalg.svd which can
        # be slow to import on test machines that don't need PCA.
        from feature_pca_viz import feature_pca_to_rgb

        pca_video = feature_pca_to_rgb(features)
    composite = compose_gt_pred_alpha_pca_grid(
        gt=target,
        pred=pred,
        alpha_video=alpha_video,
        pca_video=pca_video,
    )
    return alpha_video, pca_video, composite


def _diagnostics_payload_from_media(
    *,
    prefix: str,
    alpha_video: torch.Tensor | None,
    pca_video: torch.Tensor | None,
    composite: torch.Tensor | None,
    fps: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    def _key(suffix: str) -> str:
        return f"{prefix}_{suffix}" if prefix else suffix

    if alpha_video is not None:
        payload[_key("Alpha_Mask_Video")] = make_wandb_video(alpha_video, fps)
    if pca_video is not None:
        payload[_key("Feature_PCA_Video")] = make_wandb_video(pca_video, fps)
    if composite is not None:
        payload[_key("Render_Composite_Video")] = make_wandb_video(composite, fps)
    return payload


def render_diagnostics_payload(
    cfg: Mapping[str, Any],
    *,
    prefix: str,
    target: torch.Tensor,
    pred: torch.Tensor,
    alpha: torch.Tensor | None,
    features: torch.Tensor | None,
    fps: float,
) -> dict[str, Any]:
    """Per-view diagnostic W&B payload.

    Emits alpha-mask, feature-PCA, and composite videos when their inputs are
    available. PCA-video gating reads ``cfg["logging"]["feature_pca_log"]``.
    """
    alpha_video, pca_video, composite = _diagnostic_media(
        cfg,
        target=target,
        pred=pred,
        alpha=alpha,
        features=features,
    )
    return _diagnostics_payload_from_media(
        prefix=prefix,
        alpha_video=alpha_video,
        pca_video=pca_video,
        composite=composite,
        fps=fps,
    )


# --------------------------------------------------------------------------- #
# Single-cam validation payload
# --------------------------------------------------------------------------- #


def single_cam_validation_video_payload(
    cfg: Mapping[str, Any],
    *,
    sequence_index: int,
    rendered_sequence: torch.Tensor,
    gt_sequence: torch.Tensor,
    feature_sequence: torch.Tensor | None,
    alpha_sequence: torch.Tensor | None,
    eval_payload: Mapping[str, float],
    gt_video_logged: bool,
    fps: float,
) -> tuple[dict[str, Any], bool]:
    """Per-sequence W&B media payload for a single-cam validator.

    Only the first sequence (``sequence_index == 0``) gets media; later ones
    only contribute eval metrics. ``gt_video_logged`` is threaded through so
    the caller owns one-shot GT-video state.
    """
    payload: dict[str, Any] = dict(eval_payload)

    if sequence_index != 0:
        return payload, gt_video_logged

    payload.update(build_validation_video_payload(rendered_sequence, gt_sequence, fps))
    if not gt_video_logged:
        payload["GT_Video"] = make_wandb_video(gt_sequence, fps)
        gt_video_logged = True

    payload.update(
        render_diagnostics_payload(
            cfg,
            prefix="",
            target=gt_sequence,
            pred=rendered_sequence,
            alpha=alpha_sequence,
            features=feature_sequence,
            fps=fps,
        )
    )
    return payload, gt_video_logged


# --------------------------------------------------------------------------- #
# Multicam validation payload
# --------------------------------------------------------------------------- #


def _prefixed(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def multicam_validation_video_payload(
    cfg: Mapping[str, Any],
    *,
    train_rendered: list,
    heldout_rendered: list,
    train_targets: list[torch.Tensor],
    heldout_targets: list[torch.Tensor],
    heldout_camera_names: list[str] | tuple[str, ...],
    decoded_metrics: Mapping[str, float],
    camera_rig_metrics: Mapping[str, float],
    gt_video_logged: bool,
    fps: float,
) -> tuple[dict[str, Any], bool]:
    """Multicam validation W&B payload — train views + held-out views with
    PSNR/SSIM, alpha mask, feature PCA, composite grids, and GT one-shots.
    """
    feature_pca_log = bool(cfg["logging"]["feature_pca_log"])
    payload: dict[str, Any] = {"Eval/SequenceCount": 1, **camera_rig_metrics}
    comparison_rows: list[torch.Tensor] = []
    camera_grid_targets: list[torch.Tensor] = []
    camera_grid_renders: list[torch.Tensor] = []
    camera_grid_feature_sources: list[torch.Tensor] = []

    for view, rendered in enumerate(train_rendered):
        target = train_targets[view]
        prefix = f"TrainView{view}"
        metrics = {
            **eval_metric_payload(rendered.rgb, target, cfg["losses"]),
            **temporal_similarity_payload(rendered.rgb, target, cfg["losses"]),
        }
        if view == 0:
            metrics.update(decoded_metrics)
        payload.update(_prefixed(prefix, metrics))
        payload[f"{prefix}_Rendered_Video"] = make_wandb_video(rendered.rgb, fps)
        alpha_video, pca_video, composite = _diagnostic_media(
            cfg,
            target=target,
            pred=rendered.rgb,
            alpha=rendered.alpha,
            features=rendered.features if feature_pca_log else None,
        )
        payload.update(
            _diagnostics_payload_from_media(
                prefix=prefix,
                alpha_video=alpha_video,
                pca_video=pca_video,
                composite=composite,
                fps=fps,
            )
        )
        if composite is not None:
            comparison_rows.append(composite)
        if pca_video is not None:
            camera_grid_feature_sources.append(rendered.features)
            camera_grid_targets.append(target)
            camera_grid_renders.append(rendered.rgb)
        if not gt_video_logged:
            payload[f"{prefix}_GT_Video"] = make_wandb_video(target, fps)

    for view, rendered in enumerate(heldout_rendered):
        heldout_target = heldout_targets[view]
        camera_name = heldout_camera_names[view] if view < len(heldout_camera_names) else f"view{view}"
        prefix = f"Heldout{view}_{camera_name}"
        payload.update(_prefixed(prefix, {
            **eval_metric_payload(rendered.rgb, heldout_target, cfg["losses"]),
            **temporal_similarity_payload(rendered.rgb, heldout_target, cfg["losses"]),
        }))
        payload[f"{prefix}_Rendered_Video"] = make_wandb_video(rendered.rgb, fps)
        alpha_video, pca_video, composite = _diagnostic_media(
            cfg,
            target=heldout_target,
            pred=rendered.rgb,
            alpha=rendered.alpha,
            features=rendered.features if feature_pca_log else None,
        )
        payload.update(
            _diagnostics_payload_from_media(
                prefix=prefix,
                alpha_video=alpha_video,
                pca_video=pca_video,
                composite=composite,
                fps=fps,
            )
        )
        if composite is not None:
            comparison_rows.append(composite)
        if pca_video is not None:
            camera_grid_feature_sources.append(rendered.features)
            camera_grid_targets.append(heldout_target)
            camera_grid_renders.append(rendered.rgb)
        if not gt_video_logged:
            payload[f"{prefix}_GT_Video"] = make_wandb_video(heldout_target, fps)

    comparison_grid = compose_multicam_diagnostic_grid(comparison_rows)
    if comparison_grid is not None:
        payload["Multicam_GT_Splat_Alpha_Feature_Grid_Video"] = make_wandb_video(comparison_grid, fps)
    camera_grid_features: list[torch.Tensor] = []
    if camera_grid_feature_sources:
        from feature_pca_viz import feature_pca_to_rgb

        pca_batch = feature_pca_to_rgb(torch.stack(camera_grid_feature_sources, dim=0))
        camera_grid_features = [pca_batch[index] for index in range(pca_batch.shape[0])]
    feature_gt_render_grid = compose_multicam_feature_gt_render_grid(
        feature_videos=camera_grid_features,
        gt_videos=camera_grid_targets,
        render_videos=camera_grid_renders,
    )
    if feature_gt_render_grid is not None:
        payload["Multicam_Feature_GT_Render_ByCamera_Grid_Video"] = make_wandb_video(
            feature_gt_render_grid,
            fps,
        )

    summary = {
        key: round(float(value), 4)
        for key, value in payload.items()
        if key.endswith("/Eval/PSNR") or key.endswith("/Eval/SSIM")
    }
    if summary:
        print({"multicam_eval": summary})
    return payload, True
