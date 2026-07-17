from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from objective.types import BackgroundSample
from pipeline.render import _viewport_cameras, gaussian_sequence_slice
from renderers.fast_mac import (
    FastMacRendererConfig,
    _rasterize_features_projected,
    _rasterize_rgb_projected,
    project_for_fast_mac_batch,
)
from rendering import _camera_scalar_vector, _resolve_camera_projection_mode
from runtime_types import GaussianSequence
from train_devices import sync_torch_device


class PhaseTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.elapsed_ms: dict[str, float] = defaultdict(float)

    @contextmanager
    def measure(self, phase: str):
        sync_torch_device(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            sync_torch_device(self.device)
            self.elapsed_ms[phase] += (time.perf_counter() - start) * 1000.0


@dataclass(frozen=True)
class RasterGraph:
    features: torch.Tensor
    alpha: torch.Tensor | None
    projected: tuple[torch.Tensor, ...]
    projection_inputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class FixedRenderChunk:
    sequence: GaussianSequence
    target: Any
    background: Any | None = None


@dataclass(frozen=True)
class FixedRenderCase:
    chunks: tuple[FixedRenderChunk, ...]
    background: BackgroundSample | None
    total_frames: int
    setup_phases_ms: dict[str, float]
    temporal_chunk_size: int | None = None


def fast_mac_project_and_rasterize(
    trainer,
    sequence: GaussianSequence,
    cameras: tuple[Any, ...],
    timer: PhaseTimer,
) -> RasterGraph:
    if trainer.renderer_mode != "fast_mac":
        raise ValueError(
            f"fixed-render graph split currently supports renderer='fast_mac' only; "
            f"resolved renderer_mode={trainer.renderer_mode!r}."
        )
    if sequence.frame_count != len(cameras):
        raise ValueError(f"Expected {sequence.frame_count} cameras, got {len(cameras)}.")

    cfg = trainer.cfg
    height = int(cfg["render"]["render_size"])
    width = int(cfg["render"]["render_size"])
    render_cameras = _viewport_cameras(cameras, input_size=int(cfg["model"]["size"]), render_size=height)
    device = sequence.xyz.device
    projection_mode = _resolve_camera_projection_mode(render_cameras, cfg["render"]["camera_projection"])
    fx = _camera_scalar_vector(render_cameras, "fx", device)
    fy = _camera_scalar_vector(render_cameras, "fy", device)
    cx = _camera_scalar_vector(render_cameras, "cx", device)
    cy = _camera_scalar_vector(render_cameras, "cy", device)
    camera_to_world = torch.stack(
        [camera.camera_to_world.to(device=device, dtype=torch.float32) for camera in render_cameras],
        dim=0,
    )

    xyz = sequence.xyz.float()
    scales = sequence.scales.float()
    quats = sequence.quats.float()
    opacities = sequence.opacities.float()
    colors_in = sequence.rgbs.float()
    with timer.measure("project"):
        means2d, conics, colors, projected_opacities, depths = project_for_fast_mac_batch(
            xyz,
            scales,
            quats,
            opacities,
            colors_in,
            fx,
            fy,
            cx,
            cy,
            cameras=render_cameras,
            projection_mode=projection_mode,
            camera_to_world=camera_to_world,
            near_plane=cfg["render"]["near_plane"],
        )

    fast_mac_config = FastMacRendererConfig.from_mapping(
        cfg["render"]["fast_mac"],
        fallback_tile_size=cfg["render"]["tile_size"],
        fallback_alpha_threshold=cfg["render"]["alpha_threshold"],
    )
    feature_dim = int(colors.shape[-1])
    with timer.measure("raster_forward"):
        if feature_dim == 3:
            image_bhwc = _rasterize_rgb_projected(
                means2d,
                conics,
                colors,
                projected_opacities,
                depths,
                fast_mac_config,
                height,
                width,
            )
            features = image_bhwc.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()
            alpha = None
        else:
            rasterize_out = _rasterize_features_projected(
                means2d,
                conics,
                colors,
                projected_opacities,
                depths,
                fast_mac_config,
                height,
                width,
                feature_dim,
            )
            image_bhwf, alpha = rasterize_out
            features = image_bhwf.permute(0, 3, 1, 2).contiguous()
    return RasterGraph(
        features=features,
        alpha=alpha,
        projected=(means2d, conics, colors, projected_opacities, depths),
        projection_inputs=(xyz, scales, quats, opacities, colors_in, fx, fy, cx, cy, camera_to_world),
    )


def singlecam_sample_and_encode(trainer, timer: PhaseTimer):
    with timer.measure("sample"):
        sampled = trainer.sample_clip()
    if len(sampled) == 4:
        sequence_data, clip_frames, clip_times, clip_cameras = sampled
        with timer.measure("encode"):
            decoded = trainer.forward_known_clip(clip_frames, clip_times, clip_cameras)
        frame_count = int(clip_frames.shape[1])
        targets = [
            trainer.make_target_view(
                view_id="benchmark_train_clip",
                frames=clip_frames[0],
                frame_indices=torch.arange(frame_count, device=trainer.device),
                frame_times=clip_times.reshape(-1),
                cameras=clip_cameras,
                role="train",
                camera_owner="external_rig",
            )
        ]
        return sequence_data, clip_frames, clip_times, decoded, targets

    sequence_data, clip_frames, clip_times = sampled
    with timer.measure("encode"):
        model_input = trainer.model_input_for_clip(sequence_data, clip_frames, clip_times)
        decoded = trainer.forward_clip(model_input, clip_times)
    if decoded.cameras is None:
        raise ValueError("Implicit-camera decode did not produce cameras.")
    frame_count = int(clip_frames.shape[1])
    targets = [
        trainer.make_target_view(
            view_id="benchmark_train_clip",
            frames=clip_frames[0],
            frame_indices=torch.arange(frame_count, device=trainer.device),
            frame_times=clip_times.reshape(-1),
            cameras=tuple(decoded.cameras),
            role="train",
            camera_owner="model",
        )
    ]
    return sequence_data, clip_frames, clip_times, decoded, targets


def multicam_sample_and_encode(trainer, timer: PhaseTimer):
    with timer.measure("sample"):
        sequence_data, clip_indices, clip_frames, clip_times, views = trainer.sample_multicam_clip()
    with timer.measure("encode"):
        decoded = trainer._decode_clip(sequence_data, clip_frames, clip_times)
    targets = []
    for view in views:
        view_i = int(view)
        targets.append(
            trainer.make_target_view(
                view_id=f"benchmark_train_view_{view_i}",
                frames=trainer.multicam_bundle.train_frames[view_i, clip_indices],
                frame_indices=clip_indices,
                frame_times=trainer.frame_times_for_indices(clip_indices),
                cameras=trainer.camera_rig.cameras_for_view(view_i, clip_indices),
                role="train",
                camera_owner="external_rig",
                camera_name=trainer.multicam_bundle.train_camera_names[view_i],
            )
        )
    return sequence_data, clip_frames, clip_times, decoded, targets


def iter_target_chunks(
    trainer,
    decoded: GaussianSequence,
    target,
    *,
    use_microbatch: bool,
    chunk_size_override: int | None = None,
):
    frame_count = target.frame_count
    if chunk_size_override is not None:
        chunk_size = min(max(1, int(chunk_size_override)), frame_count)
    else:
        chunk_size = trainer.temporal_recon_chunk_size(frame_count) if use_microbatch else frame_count
    for chunk_start in range(0, frame_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, frame_count)
        chunk_sequence = gaussian_sequence_slice(decoded, chunk_start, chunk_end)
        chunk_target = trainer.make_target_view(
            view_id=target.view_id,
            frames=target.frames[chunk_start:chunk_end],
            frame_indices=target.frame_indices[chunk_start:chunk_end],
            frame_times=target.frame_times.reshape(-1)[chunk_start:chunk_end],
            cameras=tuple(target.cameras[chunk_start:chunk_end]),
            role=target.role,
            camera_owner=target.camera_owner,
            camera_name=target.camera_name,
            metrics_prefix=target.metrics_prefix,
        )
        yield chunk_start, chunk_end, chunk_sequence, chunk_target


def detach_sequence_for_fixed_render(sequence: GaussianSequence) -> GaussianSequence:
    return GaussianSequence(
        xyz=sequence.xyz.detach(),
        scales=sequence.scales.detach(),
        quats=sequence.quats.detach(),
        opacities=sequence.opacities.detach(),
        rgbs=sequence.rgbs.detach(),
        cameras=sequence.cameras,
        camera_state=None,
        auxiliary=sequence.auxiliary,
    )


def clone_sequence_for_fixed_render(sequence: GaussianSequence, *, freeze_colors: bool) -> GaussianSequence:
    def leaf(tensor: torch.Tensor, *, requires_grad: bool = True) -> torch.Tensor:
        out = tensor.detach().clone()
        return out.requires_grad_(requires_grad)

    return GaussianSequence(
        xyz=leaf(sequence.xyz),
        scales=leaf(sequence.scales),
        quats=leaf(sequence.quats),
        opacities=leaf(sequence.opacities),
        rgbs=leaf(sequence.rgbs, requires_grad=not freeze_colors),
        cameras=sequence.cameras,
        camera_state=None,
        auxiliary=sequence.auxiliary,
    )


def _slice_background_tensor(
    value: torch.Tensor | None,
    *,
    chunk_start: int,
    chunk_end: int,
    label: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    chunk_len = int(chunk_end - chunk_start)
    if int(value.shape[0]) not in {1, chunk_len}:
        if int(value.shape[0]) < chunk_end:
            raise ValueError(
                f"Cannot slice {label} background with shape {tuple(value.shape)} "
                f"for chunk [{chunk_start}, {chunk_end})."
            )
        return value[chunk_start:chunk_end]
    return value


def background_for_chunk(background: BackgroundSample, *, chunk_start: int, chunk_end: int) -> BackgroundSample:
    return BackgroundSample(
        rgb=_slice_background_tensor(
            background.rgb,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            label="RGB",
        ),
        mode=background.mode,
        phase=background.phase,
        feature=_slice_background_tensor(
            background.feature,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            label="feature",
        ),
        step=background.step,
        feature_mode=background.feature_mode,
    )


def prepare_fixed_render_case(
    trainer,
    *,
    multicam: bool,
    temporal_chunk_size: int | None = None,
) -> FixedRenderCase:
    setup_timer = PhaseTimer(trainer.device)
    if multicam:
        _sequence_data, _clip_frames, _clip_times, decoded, targets = multicam_sample_and_encode(trainer, setup_timer)
    else:
        _sequence_data, _clip_frames, _clip_times, decoded, targets = singlecam_sample_and_encode(trainer, setup_timer)

    use_microbatch = not multicam
    total_frames = sum(target.frame_count for target in targets)
    feature_background_active = trainer.rgb_objective.background_policy.spec.feature_train_mode != "none"
    background = (
        None
        if feature_background_active
        else trainer.rgb_objective.sample_background(
            phase="train",
            like=targets[0].frames,
            frame_count=targets[0].frame_count,
        )
    )
    chunks = []
    for target in targets:
        for chunk_start, chunk_end, chunk_sequence, chunk_target in iter_target_chunks(
            trainer,
            decoded,
            target,
            use_microbatch=use_microbatch,
            chunk_size_override=temporal_chunk_size,
        ):
            chunk_background = (
                None
                if background is None
                else background_for_chunk(background, chunk_start=chunk_start, chunk_end=chunk_end)
            )
            chunks.append(
                FixedRenderChunk(
                    sequence=detach_sequence_for_fixed_render(chunk_sequence),
                    target=chunk_target,
                    background=chunk_background,
                )
            )
    return FixedRenderCase(
        chunks=tuple(chunks),
        background=background.detach() if torch.is_tensor(background) else background,
        total_frames=total_frames,
        setup_phases_ms={phase: float(setup_timer.elapsed_ms.get(phase, 0.0)) for phase in ("sample", "encode")},
        temporal_chunk_size=temporal_chunk_size,
    )
