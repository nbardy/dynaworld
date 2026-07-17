from __future__ import annotations

import torch

from colorize import FeatureToColor
from star_uvt_common import target_grid_slice_for_render_chunk
from star_uvt_feature_targets import FEATURE_TARGET_GRID_ADAPTERS
from star_uvt_sparse_grid import _sparse_target_grid_pixel_ids
from star_uvt_sparse_visual_losses import (
    _compose_sparse_visual_rgb,
    _gather_sparse_visual_rgb_values,
    _sparse_visual_rgb_loss_and_grads,
)
from star_uvt_sparse_visual_sampling import _sparse_visual_pixel_ids_for_chunk


RENDERED_FEATURE_PROBE_PIXEL_SOURCES = {"target_grid", "stratified_grid"}
RENDERED_FEATURE_PROBE_GRID_ADAPTERS = FEATURE_TARGET_GRID_ADAPTERS


def _target_grid_pixel_ids_for_chunk(
    *,
    chunk_frames: int,
    feature_dim: int,
    height: int,
    width: int,
    render_frames: int,
    frame_start: int,
    sample_grid_shape: tuple[int, int, int],
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    target_start, target_frames = target_grid_slice_for_render_chunk(
        target_frames=int(sample_grid_shape[0]),
        render_frames=render_frames,
        frame_start=frame_start,
        chunk_frames=chunk_frames,
    )
    del target_start
    return _sparse_target_grid_pixel_ids(
        input_shape=(chunk_frames, feature_dim, height, width),
        target_shape=(target_frames, feature_dim, int(sample_grid_shape[1]), int(sample_grid_shape[2])),
        mode=mode,
        device=device,
    )


def _stratified_grid_pixel_ids_for_chunk(
    *,
    chunk_frames: int,
    height: int,
    width: int,
    render_frames: int,
    frame_start: int,
    sample_grid_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    return _sparse_visual_pixel_ids_for_chunk(
        pixel_source="stratified_grid",
        chunk_frames=chunk_frames,
        height=height,
        width=width,
        render_frames=render_frames,
        frame_start=frame_start,
        sample_grid_shape=sample_grid_shape,
        device=device,
    )


def _pixel_ids_for_chunk(
    *,
    pixel_source: str,
    chunk_frames: int,
    feature_dim: int,
    height: int,
    width: int,
    render_frames: int,
    frame_start: int,
    sample_grid_shape: tuple[int, int, int],
    sample_grid_adapter: str,
    device: torch.device,
) -> torch.Tensor:
    if pixel_source == "target_grid":
        return _target_grid_pixel_ids_for_chunk(
            chunk_frames=chunk_frames,
            feature_dim=feature_dim,
            height=height,
            width=width,
            render_frames=render_frames,
            frame_start=frame_start,
            sample_grid_shape=sample_grid_shape,
            mode=sample_grid_adapter,
            device=device,
        )
    if pixel_source == "stratified_grid":
        return _stratified_grid_pixel_ids_for_chunk(
            chunk_frames=chunk_frames,
            height=height,
            width=width,
            render_frames=render_frames,
            frame_start=frame_start,
            sample_grid_shape=sample_grid_shape,
            device=device,
        )
    expected = ", ".join(sorted(RENDERED_FEATURE_PROBE_PIXEL_SOURCES))
    raise ValueError(f"pixel_source must be one of: {expected}")


def gather_target_rgb_values(target_rgb_chunk: torch.Tensor, pixel_ids: torch.Tensor) -> torch.Tensor:
    return _gather_sparse_visual_rgb_values(target_rgb_chunk, pixel_ids)


def compose_sparse_rgb(
    feature_values: torch.Tensor,
    alpha_values: torch.Tensor,
    colorizer: FeatureToColor,
) -> torch.Tensor:
    return _compose_sparse_visual_rgb(feature_values, alpha_values, colorizer, composition="black")


def sparse_rgb_loss_and_grads(
    feature_values: torch.Tensor,
    alpha_values: torch.Tensor,
    target_values: torch.Tensor,
    colorizer: FeatureToColor,
    *,
    total_loss_elems: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        target_values,
        colorizer,
        total_loss_elems=total_loss_elems,
        loss_weight=1.0,
        loss_basis="pixel",
        vjp_mode="autograd",
        composition="black",
    )


__all__ = [
    "RENDERED_FEATURE_PROBE_GRID_ADAPTERS",
    "RENDERED_FEATURE_PROBE_PIXEL_SOURCES",
    "_pixel_ids_for_chunk",
    "_stratified_grid_pixel_ids_for_chunk",
    "_target_grid_pixel_ids_for_chunk",
    "compose_sparse_rgb",
    "gather_target_rgb_values",
    "sparse_rgb_loss_and_grads",
]
