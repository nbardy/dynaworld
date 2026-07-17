from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from checkpoint_utils import load_checkpoint_mapping


def load_training_sequence(cfg: dict[str, Any], device: torch.device) -> Any:
    from sequence_data import load_video_sequence, load_video_window_sequence

    data = cfg["data"]
    video_path = Path(data["video_path"])
    if data["start_seconds"] is not None or data["fps"] is not None or data["duration_seconds"] is not None:
        sequence = load_video_window_sequence(
            video_path,
            target_size=int(data["target_size"]),
            start_seconds=0.0 if data["start_seconds"] is None else float(data["start_seconds"]),
            duration_seconds=data["duration_seconds"],
            fps=0.0 if data["fps"] is None else float(data["fps"]),
            frame_count=int(data["max_frames"]),
            image_crop_mode=str(data["image_crop_mode"]),
        )
    else:
        sequence = load_video_sequence(
            video_path,
            target_size=int(data["target_size"]),
            max_frames=int(data["max_frames"]),
            image_crop_mode=str(data["image_crop_mode"]),
        )
    return sequence.to(device)


def load_colorizer_init_checkpoint(
    path: Path,
    *,
    colorizer: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    payload = load_checkpoint_mapping(path, map_location=device, label="Colorizer init checkpoint")
    if "colorizer" not in payload:
        raise ValueError(f"Colorizer init checkpoint {path} must contain a colorizer state dict")
    colorizer_state = payload["colorizer"]
    if not isinstance(colorizer_state, Mapping):
        raise ValueError(f"Colorizer init checkpoint {path} has invalid colorizer state")
    colorizer.load_state_dict(colorizer_state)
    return {"path": str(path), "loaded": True}


def grad_norms(model: nn.Module, colorizer: nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix, module in (("model", model), ("colorizer", colorizer)):
        for name, param in module.named_parameters():
            if param.grad is not None:
                out[f"{prefix}.{name}"] = float(param.grad.detach().norm().cpu().item())
    return out


def target_grid_slice_for_render_chunk(
    *,
    target_frames: int,
    render_frames: int,
    frame_start: int,
    chunk_frames: int,
) -> tuple[int, int]:
    start = float(frame_start) * float(target_frames) / float(render_frames)
    end = float(frame_start + chunk_frames) * float(target_frames) / float(render_frames)
    start_i = int(round(start))
    end_i = int(round(end))
    if abs(start - float(start_i)) > 1.0e-6 or abs(end - float(end_i)) > 1.0e-6 or end_i <= start_i:
        raise ValueError(
            "feature_target.materialization=target_grid requires train.frame_chunk_size to align "
            "with feature_target.token_grid_shape[0]"
        )
    return start_i, end_i - start_i


__all__ = [
    "grad_norms",
    "load_colorizer_init_checkpoint",
    "load_training_sequence",
    "target_grid_slice_for_render_chunk",
]
