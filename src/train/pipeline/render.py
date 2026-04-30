"""Shared render orchestration for trainer validation paths.

The trainers own model state, optimizer state, and objective policy. This
module owns clip slicing, rasterization dispatch, colorizer view-conditioning
directions, and full-sequence render stitching. Trainer-specific decode/render
policy enters through small typed callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from camera import build_camera_rays_batch
from objective.types import RenderedView
from pipeline.diagnostics import (
    decoded_temporal_payload,
    fill_decoded_frame_buffers,
    init_decoded_frame_buffers,
)
from rendering import camera_for_viewport, render_gaussian_frames_alpha_aware
from runtime_types import CameraState, GaussianSequence, SequenceData


def prepare_clip(sequence_data: SequenceData, clip_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clip_frames = sequence_data.frames[clip_indices]
    time_denominator = max(sequence_data.frame_count - 1, 1)
    clip_times = (clip_indices.to(dtype=torch.float32) / float(time_denominator)).reshape(1, -1)
    return clip_frames.unsqueeze(0), clip_times


def stack_complete_frame_list(frames: list[torch.Tensor | None], *, name: str) -> torch.Tensor:
    missing = [i for i, f in enumerate(frames) if f is None]
    if missing:
        preview = ", ".join(str(i) for i in missing[:8]) + (", ..." if len(missing) > 8 else "")
        raise RuntimeError(f"{name} did not cover all frames; missing indices: {preview}")
    complete = [f for f in frames if f is not None]
    if not complete:
        raise RuntimeError(f"{name} did not produce any frames.")
    return torch.stack(complete, dim=0)


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedClip:
    """The 5-tuple that `render_full_sequence` used to return, but as a
    frozen typed bundle so future signature changes don't reintroduce
    tuple-arity bugs.

    NOTE: This is a clip-level result (full sequence rendered chunk-by-chunk
    and stitched). It is intentionally distinct from
    `objective.types.RenderedView`, which is a per-clip per-target view
    result with its own (TargetView, RasterizedView, ColorizedView) chain.
    The two coexist; do not confuse them.

    Fields:
        rgb_sequence: [F, 3, H, W] CPU tensor — the full clip rendered to RGB.
        camera_state: aggregate `CameraState` for implicit-camera trainers,
            or `None` for known-camera trainers (whose camera state is
            empty by construction).
        temporal_metrics: scalar payload from `decoded_temporal_payload`.
        feature_sequence: [F, F_dim, H, W] CPU tensor when `feature_pca_log`
            is on, else `None`.
        alpha_sequence: [F, H, W] CPU tensor when the renderer produced
            alpha (F!=3 fast_mac path), else `None`.
    """

    rgb_sequence: torch.Tensor
    camera_state: CameraState | None
    temporal_metrics: dict[str, float]
    feature_sequence: torch.Tensor | None
    alpha_sequence: torch.Tensor | None


DecodeClipFn = Callable[[SequenceData, torch.Tensor, torch.Tensor, torch.Tensor], GaussianSequence]
RenderClipFn = Callable[..., RenderedView]


# ---------------------------------------------------------------------------
# Helper: viewport_cameras lives in the monolith; we accept it as a
# callable to avoid pulling it (out of scope for wave 1).
# ---------------------------------------------------------------------------


def _camera_center_ray_dirs(
    cameras: tuple[Any, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dirs = torch.stack(
        [camera.camera_to_world.to(device=device, dtype=dtype)[:3, 2] for camera in cameras],
        dim=0,
    )
    return dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _viewport_cameras(
    cameras: tuple[Any, ...],
    *,
    input_size: int,
    render_size: int,
) -> tuple[Any, ...]:
    """Map source-resolution cameras onto the configured render viewport."""
    return tuple(
        camera_for_viewport(
            camera,
            source_height=input_size,
            source_width=input_size,
            target_height=render_size,
            target_width=render_size,
        )
        for camera in cameras
    )


# ---------------------------------------------------------------------------
# Public render helpers
# ---------------------------------------------------------------------------


def colorize_view_dirs_for_features(
    cfg: Mapping[str, Any],
    colorize: Any,
    features: torch.Tensor,
    cameras: tuple[Any, ...],
) -> torch.Tensor | None:
    """View-conditioning vector field for the colorize MLP.

    Returns `None` when the colorize module's view_condition is "none".
    Returns `[T, 3, H, W]` directions for `camera_center_ray`/`pixel_ray`.
    """
    view_condition = colorize.view_condition
    if view_condition == "none":
        return None
    if features.dim() != 4:
        raise ValueError(
            f"View-conditioned colorize expects [T, F, H, W] features, got {tuple(features.shape)}."
        )
    if len(cameras) != features.shape[0]:
        raise ValueError(
            f"Expected {features.shape[0]} cameras for colorize view conditioning, got {len(cameras)}."
        )

    render_cameras = _viewport_cameras(
        cameras, input_size=cfg["model"]["size"], render_size=cfg["render"]["render_size"]
    )
    frame_count, _, height, width = features.shape
    if view_condition == "camera_center_ray":
        dirs = _camera_center_ray_dirs(render_cameras, device=features.device, dtype=features.dtype)
        view_dirs = dirs.reshape(frame_count, 3, 1, 1).expand(-1, -1, height, width)
    elif view_condition == "pixel_ray":
        _, dirs_hw3 = build_camera_rays_batch(
            list(render_cameras),
            height=height,
            width=width,
            device=features.device,
            dtype=features.dtype,
            pixel_center=0.5,
        )
        view_dirs = dirs_hw3.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(
            "colorize.view_condition must be one of none, camera_center_ray, pixel_ray; "
            f"got {view_condition!r}."
        )
    return view_dirs.detach() if colorize.detach_view_condition else view_dirs


def gaussian_sequence_slice(sequence: GaussianSequence, start: int, end: int) -> GaussianSequence:
    """Return a clip-window slice [start:end] of a GaussianSequence,
    propagating cameras / camera_state / auxiliary verbatim.
    """
    cameras = None
    if sequence.cameras is not None:
        cameras = tuple(sequence.cameras[start:end])
    return GaussianSequence(
        xyz=sequence.xyz[start:end],
        scales=sequence.scales[start:end],
        quats=sequence.quats[start:end],
        opacities=sequence.opacities[start:end],
        rgbs=sequence.rgbs[start:end],
        cameras=cameras,
        camera_state=sequence.camera_state,
        auxiliary=sequence.auxiliary,
    )


def render_clip_sequence(
    cfg: Mapping[str, Any],
    sequence: GaussianSequence,
    cameras: tuple[Any, ...],
    *,
    renderer_mode: str,
    dense_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns `(rendered_features, alpha_mask)`. Alpha is `None` for
    non-fast_mac modes and for the F=3 v5 legacy path; a `[T, H, W]`
    tensor for F!=3 fast_mac.
    """
    return render_gaussian_frames_alpha_aware(
        cfg,
        sequence,
        _viewport_cameras(
            cameras,
            input_size=int(cfg["model"]["size"]),
            render_size=int(cfg["render"]["render_size"]),
        ),
        mode=renderer_mode,
        dense_grid=dense_grid,
    )


# ---------------------------------------------------------------------------
# Full-sequence rendering (was Trainer.render_full_sequence /
# KnownCameraTrainer.render_full_sequence)
# ---------------------------------------------------------------------------


def _new_frame_buffer(num_frames: int) -> list[torch.Tensor | None]:
    return [None] * num_frames


def _render_full_sequence_chunked(
    cfg: Mapping[str, Any],
    sequence_data: SequenceData,
    decode_clip: DecodeClipFn,
    render_clip: RenderClipFn,
    *,
    device: torch.device,
    view_id_prefix: str,
    require_camera_state: bool,
) -> tuple[
    list[torch.Tensor | None],
    list[torch.Tensor | None] | None,
    list[torch.Tensor | None] | None,
    dict[str, list[torch.Tensor | None]],
    list[CameraState],
]:
    """Iterate the sequence in clips, decode + render each, accumulate buffers."""
    train_frame_count = int(cfg["model"]["train_frame_count"])
    feature_pca_log = bool(cfg["logging"]["feature_pca_log"])
    num_frames = sequence_data.frame_count
    rendered_frames: list[torch.Tensor | None] = _new_frame_buffer(num_frames)
    feature_frames: list[torch.Tensor | None] | None = (
        _new_frame_buffer(num_frames) if feature_pca_log else None
    )
    alpha_frames: list[torch.Tensor | None] | None = None
    decoded_buffers = init_decoded_frame_buffers(num_frames)
    camera_states: list[CameraState] = []

    for end in range(train_frame_count, num_frames + train_frame_count, train_frame_count):
        clip_end = min(end, num_frames)
        clip_start = max(0, clip_end - train_frame_count)
        clip_indices = torch.arange(clip_start, clip_end, device=device)
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        decoded = decode_clip(sequence_data, clip_indices, clip_frames, clip_times)
        if decoded.cameras is None:
            raise ValueError("Video decode must include cameras for full-sequence render.")
        if require_camera_state:
            if decoded.camera_state is None:
                raise ValueError("Implicit-camera video decode must include camera_state.")
            camera_states.append(decoded.camera_state)
        fill_decoded_frame_buffers(decoded_buffers, decoded, clip_indices)
        rendered = render_clip(
            decoded,
            clip_frames=clip_frames[0],
            clip_indices=clip_indices,
            clip_times=clip_times,
            view_id=f"{view_id_prefix}_{clip_start}_{clip_end}",
        )
        features_cpu = rendered.features.detach().cpu() if feature_frames is not None else None
        alpha_cpu = rendered.alpha.detach().cpu() if rendered.alpha is not None else None
        if alpha_cpu is not None and alpha_frames is None:
            alpha_frames = _new_frame_buffer(num_frames)
        rendered_clip = rendered.rgb.cpu()

        for local_index, frame_index in enumerate(clip_indices.tolist()):
            if rendered_frames[frame_index] is not None:
                continue
            rendered_frames[frame_index] = rendered_clip[local_index]
            if feature_frames is not None and features_cpu is not None:
                feature_frames[frame_index] = features_cpu[local_index]
            if alpha_frames is not None and alpha_cpu is not None:
                alpha_frames[frame_index] = alpha_cpu[local_index]

    return rendered_frames, feature_frames, alpha_frames, decoded_buffers, camera_states


def _aggregate_camera_state(camera_states: list[CameraState]) -> CameraState:
    """Merge per-clip CameraStates into one whole-sequence CameraState.
    Mirrors the implicit-camera Trainer's existing aggregation.
    """
    if not camera_states:
        raise ValueError("Cannot aggregate an empty list of CameraStates.")
    path_residuals: list[torch.Tensor] = []
    for state in camera_states:
        if state.path_residuals is None:
            raise ValueError("path_residuals required for aggregation; got None.")
        path_residuals.append(state.path_residuals)
    return CameraState(
        fov_degrees=torch.stack([state.fov_degrees for state in camera_states]).mean(),
        radius=torch.stack([state.radius for state in camera_states]).mean(),
        global_residuals=torch.stack(
            [state.global_residuals for state in camera_states]
        ).mean(dim=0),
        rotation_delta=torch.cat([state.rotation_delta for state in camera_states], dim=0),
        translation_delta=torch.cat(
            [state.translation_delta for state in camera_states], dim=0
        ),
        path_residuals=torch.cat(path_residuals, dim=0),
    )


@torch.no_grad()
def render_full_sequence(
    cfg: Mapping[str, Any],
    model: torch.nn.Module,
    sequence_data: SequenceData,
    decode_clip: DecodeClipFn,
    render_clip: RenderClipFn,
) -> RenderedClip:
    """Render a sequence chunk-by-chunk and stitch.

    Mode is auto-selected from ``sequence_data.cameras``:
      - ``None``  -> implicit-camera; aggregates per-clip ``CameraState``
      - populated -> known-camera; returns ``camera_state=None``
    """
    is_implicit = sequence_data.cameras is None
    was_training = model.training
    model.eval()
    try:
        rendered_frames, feature_frames, alpha_frames, decoded_buffers, camera_states = (
            _render_full_sequence_chunked(
                cfg,
                sequence_data,
                decode_clip,
                render_clip,
                device=next(model.parameters()).device,
                view_id_prefix="eval_sequence" if is_implicit else "known_eval",
                require_camera_state=is_implicit,
            )
        )
    finally:
        if was_training:
            model.train()

    return RenderedClip(
        rgb_sequence=stack_complete_frame_list(rendered_frames, name="validation render"),
        camera_state=_aggregate_camera_state(camera_states) if is_implicit else None,
        temporal_metrics=decoded_temporal_payload(decoded_buffers),
        feature_sequence=(
            stack_complete_frame_list(feature_frames, name="validation feature PCA")
            if feature_frames is not None else None
        ),
        alpha_sequence=(
            stack_complete_frame_list(alpha_frames, name="validation alpha mask")
            if alpha_frames is not None else None
        ),
    )


__all__ = [
    "RenderedClip",
    "DecodeClipFn",
    "RenderClipFn",
    "prepare_clip",
    "stack_complete_frame_list",
    "colorize_view_dirs_for_features",
    "gaussian_sequence_slice",
    "render_clip_sequence",
    "render_full_sequence",
]
