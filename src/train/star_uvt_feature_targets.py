from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from star_uvt_common import target_grid_slice_for_render_chunk


FEATURE_TARGET_GRID_ADAPTERS = {"nearest", "trilinear"}


@dataclass(frozen=True)
class FeatureTargetTensor:
    materialization: str
    frames: int
    height: int
    width: int
    feature_dim: int
    grid_mode: str
    normalization: str
    dense: torch.Tensor | None
    source: torch.Tensor | None
    chunks: tuple[torch.Tensor, ...] | None
    chunk_size: int | None
    mean: torch.Tensor | None
    std: torch.Tensor | None
    meta: dict[str, Any]

    @property
    def numel(self) -> int:
        if self.materialization == "target_grid":
            if self.source is None:
                raise RuntimeError("target-grid feature target is missing its source tensor")
            return int(self.source.numel())
        return int(self.frames * self.feature_dim * self.height * self.width)

    def chunk(self, frame_start: int, chunk_frames: int) -> torch.Tensor:
        if self.materialization == "target_grid":
            if self.source is None:
                raise RuntimeError("target-grid feature target is missing its source tensor")
            target_start, target_frames = target_grid_slice_for_render_chunk(
                target_frames=int(self.source.shape[0]),
                render_frames=self.frames,
                frame_start=frame_start,
                chunk_frames=chunk_frames,
            )
            return self.source[target_start : target_start + target_frames].contiguous()
        if self.dense is not None:
            return self.dense[frame_start : frame_start + chunk_frames]
        if self.chunks is not None:
            if self.chunk_size is None:
                raise RuntimeError("cached feature target chunks are missing chunk_size")
            if frame_start % self.chunk_size != 0:
                raise ValueError(
                    f"cached feature target chunk request frame_start={frame_start} is not aligned "
                    f"to chunk_size={self.chunk_size}"
                )
            index = frame_start // self.chunk_size
            if index < 0 or index >= len(self.chunks):
                raise IndexError(f"cached feature target chunk index out of range: {index}")
            cached = self.chunks[index]
            if int(cached.shape[0]) != int(chunk_frames):
                raise ValueError(
                    f"cached feature target chunk has {int(cached.shape[0])} frames, "
                    f"requested {int(chunk_frames)}"
                )
            return cached
        if self.source is None:
            raise RuntimeError("chunked feature target is missing its source tensor")
        target = _adapt_feature_target_grid_chunk(
            self.source,
            frames=self.frames,
            height=self.height,
            width=self.width,
            frame_start=frame_start,
            chunk_frames=chunk_frames,
            mode=self.grid_mode,
        )
        return _normalize_feature_target_with_stats(
            target,
            mode=self.normalization,
            mean=self.mean,
            std=self.std,
        ).detach()


def _adapt_rgb_to_grid(
    frames_rgb: torch.Tensor,
    *,
    target_shape: tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    if frames_rgb.dim() != 4 or int(frames_rgb.shape[1]) != 3:
        raise ValueError(f"frames_rgb must have shape [T,3,H,W], got {tuple(frames_rgb.shape)}")
    if mode == "nearest":
        return F.interpolate(
            frames_rgb.permute(1, 0, 2, 3).unsqueeze(0),
            size=target_shape,
            mode="nearest",
        )[0].permute(1, 0, 2, 3).contiguous()
    if mode == "trilinear":
        return F.interpolate(
            frames_rgb.permute(1, 0, 2, 3).unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )[0].permute(1, 0, 2, 3).contiguous()
    raise ValueError(f"Unsupported RGB probe adapter={mode!r}.")


def adapt_rgb_to_grid(
    frames_rgb: torch.Tensor,
    *,
    target_shape: tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    return _adapt_rgb_to_grid(frames_rgb, target_shape=target_shape, mode=mode)


def _upsample_grid_rgb(
    grid_rgb: torch.Tensor,
    *,
    target_shape: tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    if grid_rgb.dim() != 4 or int(grid_rgb.shape[1]) != 3:
        raise ValueError(f"grid_rgb must have shape [T,3,H,W], got {tuple(grid_rgb.shape)}")
    if mode == "nearest":
        return F.interpolate(
            grid_rgb.permute(1, 0, 2, 3).unsqueeze(0),
            size=target_shape,
            mode="nearest",
        )[0].permute(1, 0, 2, 3).contiguous()
    if mode == "trilinear":
        return F.interpolate(
            grid_rgb.permute(1, 0, 2, 3).unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )[0].permute(1, 0, 2, 3).contiguous()
    raise ValueError(f"Unsupported RGB probe adapter={mode!r}.")


def upsample_grid_rgb(
    grid_rgb: torch.Tensor,
    *,
    target_shape: tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    return _upsample_grid_rgb(grid_rgb, target_shape=target_shape, mode=mode)


def mean_rgb_grid_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().mean()


def _feature_tensor_to_tchw(
    value: torch.Tensor,
    *,
    layer: str,
    token_grid_shape: list[int] | tuple[int, int, int] | None,
) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"Feature target layer {layer!r} must have batch size 1, got {tuple(value.shape)}.")
        if token_grid_shape is None:
            raise ValueError(
                f"Feature target layer {layer!r} is token-shaped {tuple(value.shape)}; "
                "feature_target.token_grid_shape=[T,H,W] is required."
            )
        grid_t, grid_h, grid_w = [int(item) for item in token_grid_shape]
        expected_tokens = grid_t * grid_h * grid_w
        if int(value.shape[1]) != expected_tokens:
            raise ValueError(
                f"feature_target.token_grid_shape={list(token_grid_shape)} expects {expected_tokens} tokens, "
                f"but layer {layer!r} has {int(value.shape[1])}."
            )
        return value[0].reshape(grid_t, grid_h, grid_w, int(value.shape[2])).permute(0, 3, 1, 2).contiguous()
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError(f"Feature target layer {layer!r} must have batch size 1, got {tuple(value.shape)}.")
        return value[0].permute(1, 0, 2, 3).contiguous()
    if value.ndim == 4:
        return value.contiguous()
    raise ValueError(f"Feature target layer {layer!r} must be [1,C,T,H,W] or [T,C,H,W], got {tuple(value.shape)}.")


def _adapt_feature_target_grid(
    target: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    if (int(target.shape[0]), int(target.shape[2]), int(target.shape[3])) == (frames, height, width):
        return target
    if mode == "nearest":
        return F.interpolate(
            target.permute(1, 0, 2, 3).unsqueeze(0),
            size=(frames, height, width),
            mode="nearest",
        )[0].permute(1, 0, 2, 3).contiguous()
    if mode == "trilinear":
        return F.interpolate(
            target.permute(1, 0, 2, 3).unsqueeze(0),
            size=(frames, height, width),
            mode="trilinear",
            align_corners=False,
        )[0].permute(1, 0, 2, 3).contiguous()
    raise ValueError(f"Unsupported feature target temporal_spatial_adapter={mode!r}.")


def _interpolate_axis_grid(
    *,
    source_size: int,
    target_size: int,
    start: int,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    index = torch.arange(start, start + length, device=device, dtype=torch.float32)
    coord = (index + 0.5) * (float(source_size) / float(target_size)) - 0.5
    return (2.0 * (coord + 0.5) / float(source_size)) - 1.0


def _nearest_axis_index(
    *,
    source_size: int,
    target_size: int,
    start: int,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    index = torch.arange(start, start + length, device=device, dtype=torch.float32)
    return torch.floor(index * (float(source_size) / float(target_size))).to(torch.long).clamp(0, source_size - 1)


def _adapt_feature_target_grid_chunk(
    target: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    frame_start: int,
    chunk_frames: int,
    mode: str,
) -> torch.Tensor:
    if (int(target.shape[0]), int(target.shape[2]), int(target.shape[3])) == (frames, height, width):
        return target[frame_start : frame_start + chunk_frames].contiguous()
    if mode not in {"nearest", "trilinear"}:
        raise ValueError(f"Unsupported feature target temporal_spatial_adapter={mode!r}.")

    src_frames, _, src_h, src_w = [int(item) for item in target.shape]
    if mode == "nearest":
        z_index = _nearest_axis_index(
            source_size=src_frames,
            target_size=frames,
            start=frame_start,
            length=chunk_frames,
            device=target.device,
        )
        y_index = _nearest_axis_index(
            source_size=src_h,
            target_size=height,
            start=0,
            length=height,
            device=target.device,
        )
        x_index = _nearest_axis_index(
            source_size=src_w,
            target_size=width,
            start=0,
            length=width,
            device=target.device,
        )
        return target.index_select(0, z_index).index_select(2, y_index).index_select(3, x_index).contiguous()

    z = _interpolate_axis_grid(
        source_size=src_frames,
        target_size=frames,
        start=frame_start,
        length=chunk_frames,
        device=target.device,
    )
    y = _interpolate_axis_grid(
        source_size=src_h,
        target_size=height,
        start=0,
        length=height,
        device=target.device,
    )
    x = _interpolate_axis_grid(
        source_size=src_w,
        target_size=width,
        start=0,
        length=width,
        device=target.device,
    )
    grid = torch.stack(
        torch.meshgrid(z, y, x, indexing="ij"),
        dim=-1,
    )[..., [2, 1, 0]].unsqueeze(0)
    sampled = F.grid_sample(
        target.permute(1, 0, 2, 3).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0].permute(1, 0, 2, 3).contiguous()


def _adapt_render_to_feature_target(
    rendered: torch.Tensor,
    *,
    target_shape: tuple[int, int, int, int],
    mode: str,
) -> torch.Tensor:
    if tuple(rendered.shape) == target_shape:
        return rendered
    if mode == "nearest":
        return F.interpolate(
            rendered.permute(1, 0, 2, 3).unsqueeze(0),
            size=(target_shape[0], target_shape[2], target_shape[3]),
            mode="nearest",
        )[0].permute(1, 0, 2, 3).contiguous()
    if mode == "trilinear":
        return F.interpolate(
            rendered.permute(1, 0, 2, 3).unsqueeze(0),
            size=(target_shape[0], target_shape[2], target_shape[3]),
            mode="trilinear",
            align_corners=False,
        )[0].permute(1, 0, 2, 3).contiguous()
    raise ValueError(f"Unsupported feature target temporal_spatial_adapter={mode!r}.")


def _adapt_feature_target_channels(target: torch.Tensor, *, feature_dim: int, mode: str) -> torch.Tensor:
    channels = int(target.shape[1])
    if channels == feature_dim:
        return target
    if mode == "truncate_or_pad":
        if channels > feature_dim:
            return target[:, :feature_dim].contiguous()
        pad = target.new_zeros((target.shape[0], feature_dim - channels, target.shape[2], target.shape[3]))
        return torch.cat([target, pad], dim=1).contiguous()
    if mode == "repeat_truncate":
        repeats = (feature_dim + channels - 1) // channels
        return target.repeat(1, repeats, 1, 1)[:, :feature_dim].contiguous()
    raise ValueError(
        f"Feature target channel count {channels} does not match feature_dim {feature_dim}; "
        f"set channel_adapter to truncate_or_pad or repeat_truncate."
    )


def _normalize_feature_target(target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return target
    if mode == "channel_standardize":
        mean = target.mean(dim=(0, 2, 3), keepdim=True)
        std = target.std(dim=(0, 2, 3), keepdim=True).clamp_min(1.0e-6)
        return (target - mean) / std
    raise ValueError(f"Unsupported feature target normalization={mode!r}.")


def _normalize_feature_target_with_stats(
    target: torch.Tensor,
    *,
    mode: str,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mode == "none":
        return target
    if mode == "channel_standardize":
        if mean is None or std is None:
            raise RuntimeError("channel_standardize target chunk is missing mean/std")
        return (target - mean) / std
    raise ValueError(f"Unsupported feature target normalization={mode!r}.")


def _feature_target_channel_stats(
    target: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    grid_mode: str,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    channel_count = int(target.shape[1])
    sums = target.new_zeros((channel_count,))
    sum_squares = target.new_zeros((channel_count,))
    count = int(frames * height * width)
    for frame_start in range(0, frames, chunk_size):
        chunk_frames = min(chunk_size, frames - frame_start)
        chunk = _adapt_feature_target_grid_chunk(
            target,
            frames=frames,
            height=height,
            width=width,
            frame_start=frame_start,
            chunk_frames=chunk_frames,
            mode=grid_mode,
        )
        sums = sums + chunk.sum(dim=(0, 2, 3))
        sum_squares = sum_squares + chunk.square().sum(dim=(0, 2, 3))
    mean = sums / float(count)
    if count > 1:
        variance = (sum_squares - sums.square() / float(count)) / float(count - 1)
    else:
        variance = sum_squares.new_zeros(sum_squares.shape)
    std = variance.clamp_min(0.0).sqrt().clamp_min(1.0e-6)
    return mean.view(1, channel_count, 1, 1), std.view(1, channel_count, 1, 1)


def _load_cached_feature_target(
    *,
    cfg: dict[str, Any],
    sequence_data: Any,
    device: torch.device,
    frames: int,
    height: int,
    width: int,
    feature_dim: int,
) -> FeatureTargetTensor:
    from video_feature_cache import VideoFeatureCache

    target_cfg = cfg["feature_target"]
    cache = VideoFeatureCache(cfg["features"], device)
    features = cache.load_or_bake(sequence_data)
    layer = str(target_cfg["layer"])
    materialization = str(target_cfg.get("materialization", "dense"))
    if layer not in features:
        available = ", ".join(sorted(features))
        raise KeyError(f"Feature target layer {layer!r} missing from cache. Available layers: {available}")
    raw = features[layer].detach().to(device=device, dtype=torch.float32)
    raw_source_shape = list(raw.shape)
    target = _feature_tensor_to_tchw(raw, layer=layer, token_grid_shape=target_cfg.get("token_grid_shape"))
    target = _adapt_feature_target_channels(
        target,
        feature_dim=feature_dim,
        mode=str(target_cfg["channel_adapter"]),
    )
    grid_mode = str(target_cfg["temporal_spatial_adapter"])
    normalization = str(target_cfg["normalization"])
    meta = {
        "layer": layer,
        "source_shape": raw_source_shape,
        "channel_adapted_source_shape": list(target.shape),
        "adapted_shape": [frames, feature_dim, height, width],
        "cache_key": cache.cache_key(sequence_data),
        "cache_path": str(cache.cache_path(sequence_data)),
        "extractor": str(cfg["features"].get("extractor")),
        "channel_adapter": str(target_cfg["channel_adapter"]),
        "channel_adapter_applied_before_grid": True,
        "temporal_spatial_adapter": grid_mode,
        "normalization": normalization,
        "token_grid_shape": target_cfg.get("token_grid_shape"),
        "materialization": materialization,
    }
    if materialization == "dense":
        dense = _adapt_feature_target_grid(
            target,
            frames=frames,
            height=height,
            width=width,
            mode=grid_mode,
        )
        dense = _normalize_feature_target(dense, normalization).detach()
        meta["dense_tensor_shape"] = list(dense.shape)
        return FeatureTargetTensor(
            materialization=materialization,
            frames=frames,
            height=height,
            width=width,
            feature_dim=feature_dim,
            grid_mode=grid_mode,
            normalization=normalization,
            dense=dense,
            source=None,
            chunks=None,
            chunk_size=None,
            mean=None,
            std=None,
            meta=meta,
        )

    if materialization == "target_grid":
        target_grid = _normalize_feature_target(target, normalization).detach().contiguous()
        meta["target_grid_shape"] = list(target_grid.shape)
        meta["target_grid_bytes"] = int(target_grid.numel() * target_grid.element_size())
        meta["target_grid_mib"] = meta["target_grid_bytes"] / float(1024 * 1024)
        return FeatureTargetTensor(
            materialization=materialization,
            frames=frames,
            height=height,
            width=width,
            feature_dim=feature_dim,
            grid_mode=grid_mode,
            normalization=normalization,
            dense=None,
            source=target_grid,
            chunks=None,
            chunk_size=None,
            mean=None,
            std=None,
            meta=meta,
        )

    chunk_size = cfg["train"]["frame_chunk_size"]
    stats_chunk_size = int(target_cfg.get("materialization_chunk_size") or chunk_size or frames)
    if stats_chunk_size <= 0:
        raise ValueError("feature_target.materialization_chunk_size must be positive")
    stats_chunk_size = min(stats_chunk_size, frames)
    train_chunk_size = frames if chunk_size is None else min(int(chunk_size), frames)
    if materialization == "cached_chunks" and stats_chunk_size != train_chunk_size:
        raise ValueError(
            "feature_target.materialization=cached_chunks requires materialization_chunk_size "
            "to match train.frame_chunk_size"
        )
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    if normalization == "channel_standardize":
        mean, std = _feature_target_channel_stats(
            target,
            frames=frames,
            height=height,
            width=width,
            grid_mode=grid_mode,
            chunk_size=stats_chunk_size,
        )
    if materialization == "cached_chunks":
        chunks: list[torch.Tensor] = []
        cached_bytes = 0
        for frame_start in range(0, frames, stats_chunk_size):
            chunk_frames = min(stats_chunk_size, frames - frame_start)
            chunk = _adapt_feature_target_grid_chunk(
                target,
                frames=frames,
                height=height,
                width=width,
                frame_start=frame_start,
                chunk_frames=chunk_frames,
                mode=grid_mode,
            )
            chunk = _normalize_feature_target_with_stats(
                chunk,
                mode=normalization,
                mean=mean,
                std=std,
            ).detach().contiguous()
            cached_bytes += int(chunk.numel() * chunk.element_size())
            chunks.append(chunk)
        meta["materialization_chunk_size"] = stats_chunk_size
        meta["normalization_streaming_stats"] = bool(normalization == "channel_standardize")
        meta["cached_chunk_count"] = len(chunks)
        meta["cached_target_bytes"] = cached_bytes
        meta["cached_target_mib"] = cached_bytes / float(1024 * 1024)
        return FeatureTargetTensor(
            materialization=materialization,
            frames=frames,
            height=height,
            width=width,
            feature_dim=feature_dim,
            grid_mode=grid_mode,
            normalization=normalization,
            dense=None,
            source=None,
            chunks=tuple(chunks),
            chunk_size=stats_chunk_size,
            mean=None if mean is None else mean.detach(),
            std=None if std is None else std.detach(),
            meta=meta,
        )
    meta["materialization_chunk_size"] = stats_chunk_size
    meta["normalization_streaming_stats"] = bool(normalization == "channel_standardize")
    return FeatureTargetTensor(
        materialization=materialization,
        frames=frames,
        height=height,
        width=width,
        feature_dim=feature_dim,
        grid_mode=grid_mode,
        normalization=normalization,
        dense=None,
        source=target.detach(),
        chunks=None,
        chunk_size=None,
        mean=None if mean is None else mean.detach(),
        std=None if std is None else std.detach(),
        meta=meta,
    )


__all__ = [
    "FEATURE_TARGET_GRID_ADAPTERS",
    "FeatureTargetTensor",
    "adapt_rgb_to_grid",
    "mean_rgb_grid_loss",
    "upsample_grid_rgb",
    "_adapt_feature_target_grid",
    "_adapt_feature_target_grid_chunk",
    "_adapt_render_to_feature_target",
    "_adapt_rgb_to_grid",
    "_feature_target_channel_stats",
    "_feature_tensor_to_tchw",
    "_load_cached_feature_target",
    "_normalize_feature_target",
    "_normalize_feature_target_with_stats",
    "_upsample_grid_rgb",
]
