from __future__ import annotations

from typing import Any

import torch


def flatten_rgb_pixels(images: torch.Tensor) -> torch.Tensor:
    return images.permute(0, 2, 3, 1).reshape(-1, images.shape[1]).to(dtype=torch.float32)


def add_bias_column(values: torch.Tensor) -> torch.Tensor:
    return torch.cat([values, torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device)], dim=1)


def fit_channel_affine(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rows = []
    for channel in range(rendered.shape[1]):
        x = flatten_rgb_pixels(rendered[:, channel : channel + 1])
        y = flatten_rgb_pixels(target[:, channel : channel + 1])
        rows.append(torch.linalg.lstsq(add_bias_column(x), y).solution[:, 0])
    return torch.stack(rows, dim=0)


def apply_channel_affine(rendered: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    corrected = rendered.clone()
    for channel, row in enumerate(transform):
        corrected[:, channel] = rendered[:, channel] * row[0] + row[1]
    return corrected.clamp(0.0, 1.0)


def fit_rgb_matrix_affine(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(add_bias_column(flatten_rgb_pixels(rendered)), flatten_rgb_pixels(target)).solution


def apply_rgb_matrix_affine(rendered: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    shape = rendered.shape
    corrected = add_bias_column(flatten_rgb_pixels(rendered)) @ transform
    return corrected.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2).clamp(0.0, 1.0)


def fit_eval_color_calibration(
    render_cfg: dict[str, Any],
    train_renders: torch.Tensor,
    train_targets: torch.Tensor,
) -> dict[str, Any] | None:
    mode = str(render_cfg["eval_color_calibration"])
    if mode == "none":
        return None
    if mode == "train_fit_channel_affine":
        transform = fit_channel_affine(train_renders, train_targets)
    elif mode == "train_fit_rgb_matrix_affine":
        transform = fit_rgb_matrix_affine(train_renders, train_targets)
    else:
        raise ValueError(f"Unknown eval color calibration mode {mode!r}")
    return {"mode": mode, "transform": transform}


def apply_eval_color_calibration(rendered: torch.Tensor, calibration: dict[str, Any] | None) -> torch.Tensor:
    if calibration is None:
        return rendered
    mode = str(calibration["mode"])
    transform = calibration["transform"]
    if mode == "train_fit_channel_affine":
        return apply_channel_affine(rendered, transform)
    if mode == "train_fit_rgb_matrix_affine":
        return apply_rgb_matrix_affine(rendered, transform)
    raise ValueError(f"Unknown eval color calibration mode {mode!r}")


def frame_index_summary(frame_indices: torch.Tensor | None) -> dict[str, Any] | None:
    if frame_indices is None:
        return None
    values = [int(value) for value in frame_indices.detach().cpu().reshape(-1).tolist()]
    return {
        "count": len(values),
        "unique": sorted(set(values)),
    }


def serialize_eval_color_calibration(
    calibration: dict[str, Any] | None,
    *,
    step: int | None = None,
    train_frame_indices: torch.Tensor | None = None,
    heldout_frame_indices: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    if calibration is None:
        return None
    transform = calibration["transform"]
    payload: dict[str, Any] = {
        "mode": str(calibration["mode"]),
        "transform": transform.detach().cpu().tolist(),
        "fit_scope": "train_render_to_train_target",
        "heldout_blind": True,
    }
    if step is not None:
        payload["step"] = int(step)
    if train_frame_indices is not None:
        payload["train_frame_indices"] = frame_index_summary(train_frame_indices)
    if heldout_frame_indices is not None:
        payload["heldout_frame_indices"] = frame_index_summary(heldout_frame_indices)
    return payload


__all__ = [
    "add_bias_column",
    "apply_channel_affine",
    "apply_eval_color_calibration",
    "apply_rgb_matrix_affine",
    "fit_channel_affine",
    "fit_eval_color_calibration",
    "fit_rgb_matrix_affine",
    "flatten_rgb_pixels",
    "frame_index_summary",
    "serialize_eval_color_calibration",
]
