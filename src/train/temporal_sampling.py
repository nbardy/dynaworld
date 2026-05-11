from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


TEMPORAL_DILATION_OFFSETS = (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
FRAME_SAMPLING_MODE_ALIASES = {
    "random": "random",
    "contiguous": "contiguous",
    "contigous": "contiguous",
    "temporal-dilation": "temporal-dilation",
    "temporal_dilation": "temporal-dilation",
}


def normalize_frame_sampling_config(config: Any) -> dict[str, Any]:
    if config is None:
        config = {"mode": "contiguous"}
    elif isinstance(config, str):
        config = {"mode": config}
    elif isinstance(config, Mapping):
        config = dict(config)
    else:
        raise TypeError("train.frame_sampling must be a string, object, or null.")

    mode = str(config.get("mode", "contiguous")).lower()
    if mode not in FRAME_SAMPLING_MODE_ALIASES:
        known = ", ".join(sorted(FRAME_SAMPLING_MODE_ALIASES))
        raise ValueError(f"Unknown train.frame_sampling.mode={mode!r}. Expected one of: {known}.")
    normalized = {"mode": FRAME_SAMPLING_MODE_ALIASES[mode]}
    offsets = config.get("offsets", TEMPORAL_DILATION_OFFSETS)
    normalized["offsets"] = _normalize_offsets(offsets)
    return normalized


def validate_frame_sampling_config(config: Mapping[str, Any], sample_count: int) -> None:
    sample_count = int(sample_count)
    if sample_count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count}.")
    if config["mode"] == "temporal-dilation" and len(config["offsets"]) != sample_count:
        raise ValueError(
            "train.frame_sampling.mode='temporal-dilation' returns one frame per offset; "
            f"got {len(config['offsets'])} offsets but model.train_frame_count={sample_count}. "
            "Set model.train_frame_count to the offsets length or provide train.frame_sampling.offsets."
        )


def select_frame_indices(
    num_frames: int,
    sample_count: int,
    frame_sampling: Mapping[str, Any] | str | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    config = normalize_frame_sampling_config(frame_sampling)
    validate_frame_sampling_config(config, int(sample_count))
    mode = config["mode"]
    if mode == "random":
        return select_random_indices(num_frames, sample_count, device=device)
    if mode == "contiguous":
        return select_contiguous_indices(num_frames, sample_count, device=device)
    if mode == "temporal-dilation":
        return select_temporal_dilation_indices(num_frames, config["offsets"], device=device)
    raise AssertionError(f"Unhandled frame sampling mode {mode!r}.")


def select_contiguous_indices(
    num_frames: int,
    sample_count: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_num_frames(num_frames)
    sample_count = _validate_sample_count(sample_count)
    window = min(sample_count, int(num_frames))
    if window >= int(num_frames):
        return torch.arange(int(num_frames), device=device)
    start = torch.randint(0, int(num_frames) - window + 1, (1,), device=device).item()
    return torch.arange(start, start + window, device=device)


def select_random_indices(
    num_frames: int,
    sample_count: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_num_frames(num_frames)
    sample_count = _validate_sample_count(sample_count)
    if sample_count >= int(num_frames):
        return torch.arange(int(num_frames), device=device)
    return torch.sort(torch.randperm(int(num_frames), device=device)[:sample_count]).values


def select_temporal_dilation_indices(
    num_frames: int,
    offsets: Sequence[int] = TEMPORAL_DILATION_OFFSETS,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_num_frames(num_frames)
    offsets = _normalize_offsets(offsets)
    center = int(torch.randint(int(num_frames), (1,), device=device).item())
    return temporal_dilation_indices_for_center(int(num_frames), center, offsets, device=device)


def temporal_dilation_indices_for_center(
    num_frames: int,
    center: int,
    offsets: Sequence[int] = TEMPORAL_DILATION_OFFSETS,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_num_frames(num_frames)
    offsets = _normalize_offsets(offsets)
    offset_tensor = torch.tensor(offsets, dtype=torch.long, device=device)
    return (int(center) + offset_tensor).remainder(int(num_frames))


def _normalize_offsets(offsets: Any) -> tuple[int, ...]:
    if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
        raise TypeError("train.frame_sampling.offsets must be a sequence of integers.")
    normalized = tuple(int(offset) for offset in offsets)
    if not normalized:
        raise ValueError("train.frame_sampling.offsets must contain at least one offset.")
    return normalized


def _validate_num_frames(num_frames: int) -> None:
    if int(num_frames) < 1:
        raise ValueError("Need at least one frame to sample frame indices.")


def _validate_sample_count(sample_count: int) -> int:
    sample_count = int(sample_count)
    if sample_count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count}.")
    return sample_count
