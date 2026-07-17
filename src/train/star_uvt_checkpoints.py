from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from checkpoint_utils import atomic_torch_save, load_checkpoint_mapping
from colorize import FeatureToColor
from config_utils import path_or_none, serialize_config_value
from star_uvt_colorizers import build_feature_colorizer, set_module_trainable


def optimizer_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _required_state_dict(payload: Mapping[str, Any], key: str, *, path: Path, label: str) -> Mapping[str, Any]:
    state = payload.get(key)
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} {path} is missing {key} state")
    return state


def load_star_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    colorizer: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    resume_optimizer: bool,
    resume_colorizer: bool = True,
) -> dict[str, Any]:
    payload = load_checkpoint_mapping(path, map_location=device, label="Training checkpoint")
    model.load_state_dict(_required_state_dict(payload, "model", path=path, label="Training checkpoint"))
    colorizer_loaded = False
    if resume_colorizer:
        colorizer.load_state_dict(_required_state_dict(payload, "colorizer", path=path, label="Training checkpoint"))
        colorizer_loaded = True
    optimizer_loaded = False
    optimizer_lrs_loaded: list[float] = []
    if resume_optimizer:
        optimizer.load_state_dict(_required_state_dict(payload, "optimizer", path=path, label="Training checkpoint"))
        optimizer_loaded = True
        optimizer_lrs_loaded = optimizer_lrs(optimizer)
    return {
        "path": str(path),
        "loaded": True,
        "colorizer_loaded": colorizer_loaded,
        "optimizer_loaded": optimizer_loaded,
        "optimizer_lrs_loaded": optimizer_lrs_loaded,
        "steps": payload.get("steps"),
    }


def load_star_model_from_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    device: torch.device,
    freeze_model: bool = True,
) -> dict[str, Any]:
    payload = load_checkpoint_mapping(path, map_location=device, label="STAR checkpoint")
    model.load_state_dict(_required_state_dict(payload, "model", path=path, label="STAR checkpoint"))
    set_module_trainable(model, not freeze_model)
    row = payload.get("row")
    row_payload = row if isinstance(row, Mapping) else {}
    return {
        "path": str(path),
        "steps": payload.get("steps"),
        "row_pass": row_payload.get("pass"),
        "row_end_feature_target_loss": row_payload.get("end_feature_target_loss"),
        "row_end_rgb_probe_psnr": row_payload.get("end_rgb_probe_psnr"),
    }


def load_feature_rgb_probe_checkpoint(
    path: Path,
    *,
    device: torch.device,
    feature_dim: int,
) -> tuple[FeatureToColor, dict[str, Any]]:
    payload = load_checkpoint_mapping(path, map_location=device, label="RGB probe checkpoint")
    colorizer_state = payload.get("colorizer")
    if not isinstance(colorizer_state, Mapping):
        raise ValueError(f"RGB probe checkpoint {path} must contain a colorizer state dict")
    probe_cfg = payload.get("config")
    if not isinstance(probe_cfg, Mapping):
        raise ValueError(f"RGB probe checkpoint {path} is missing serialized config")
    feature_uvt_cfg = probe_cfg.get("feature_uvt", {})
    if not isinstance(feature_uvt_cfg, Mapping):
        raise ValueError(f"RGB probe checkpoint {path} has invalid feature_uvt config")
    probe_feature_dim = int(feature_uvt_cfg.get("feature_dim", feature_dim))
    if probe_feature_dim != int(feature_dim):
        raise ValueError(
            f"RGB probe feature_dim={probe_feature_dim} does not match STAR feature_dim={int(feature_dim)}"
        )
    colorize_cfg = probe_cfg.get("colorize")
    if not isinstance(colorize_cfg, Mapping):
        raise ValueError(f"RGB probe checkpoint {path} is missing colorize config")
    probe = build_feature_colorizer(colorize_cfg, feature_dim=int(feature_dim), device=device)
    probe.load_state_dict(colorizer_state)
    set_module_trainable(probe, False)
    meta = {
        "checkpoint": str(path),
        "feature_dim": int(feature_dim),
        "hidden_dim": colorize_cfg["hidden_dim"],
        "activation": str(colorize_cfg["activation"]),
        "pre_norm": bool(colorize_cfg["pre_norm"]),
        "weight_init": str(colorize_cfg["weight_init"]),
        "weight_init_gain": float(colorize_cfg["weight_init_gain"]),
        "grid_loss": payload.get("grid_loss"),
        "full_loss": payload.get("full_loss"),
        "target_grid_shape": payload.get("target_grid_shape"),
        "target_rgb_shape": payload.get("target_rgb_shape"),
    }
    return probe, meta


def load_feature_to_rgb_probe(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    feature_dim: int,
) -> tuple[FeatureToColor | None, dict[str, Any] | None]:
    path = path_or_none(cfg["feature_target"].get("rgb_probe_checkpoint"))
    if path is None:
        return None, None
    return load_feature_rgb_probe_checkpoint(path, device=device, feature_dim=feature_dim)


def save_feature_rgb_probe_checkpoint(
    path: Path,
    *,
    colorizer: nn.Module,
    cfg: dict[str, Any],
    feature_target_meta: Mapping[str, Any],
    target_grid_shape: tuple[int, ...] | list[int] | torch.Size,
    target_rgb_shape: tuple[int, ...] | list[int] | torch.Size,
    grid_loss: float,
    full_loss: float,
) -> None:
    atomic_torch_save(
        {
            "colorizer": colorizer.state_dict(),
            "config": serialize_config_value(cfg),
            "feature_target": serialize_config_value(dict(feature_target_meta)),
            "target_grid_shape": list(target_grid_shape),
            "target_rgb_shape": list(target_rgb_shape),
            "grid_loss": float(grid_loss),
            "full_loss": float(full_loss),
        },
        path,
    )


def save_rendered_feature_rgb_probe_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    colorizer: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    resume_state: Mapping[str, Any],
    colorizer_init_state: Mapping[str, Any],
    train_star_model: bool,
    sparse_sample_loss: float | None,
    full_loss: float,
) -> None:
    atomic_torch_save(
        {
            "colorizer": colorizer.state_dict(),
            "model": model.state_dict() if bool(train_star_model) else None,
            "optimizer": optimizer.state_dict(),
            "config": serialize_config_value(cfg),
            "resume_state": serialize_config_value(dict(resume_state)),
            "colorizer_init_state": serialize_config_value(dict(colorizer_init_state)),
            "sparse_sample_loss": None if sparse_sample_loss is None else float(sparse_sample_loss),
            "full_loss": float(full_loss),
        },
        path,
    )


def save_star_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    colorizer: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    row: dict[str, Any],
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "colorizer": colorizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": serialize_config_value(cfg),
            "row": serialize_config_value(row),
            "steps": int(row["steps"]),
            "losses": list(row["losses"]),
            "rgb_losses": list(row["rgb_losses"]),
            "feature_target_losses": list(row["feature_target_losses"]),
            "rgb_grid_losses": list(row.get("rgb_grid_losses", [])),
            "rgb_probe_losses": list(row["rgb_probe_losses"]),
            "dense_alpha_losses": list(row.get("dense_alpha_losses", [])),
            "sparse_visual_losses": list(row.get("sparse_visual_losses", [])),
            "sparse_visual_alpha_losses": list(row.get("sparse_visual_alpha_losses", [])),
            "sparse_visual_black_hole_losses": list(row.get("sparse_visual_black_hole_losses", [])),
            "visibility_proxy_losses": list(row.get("visibility_proxy_losses", [])),
        },
        path,
    )


__all__ = [
    "load_feature_rgb_probe_checkpoint",
    "load_feature_to_rgb_probe",
    "save_feature_rgb_probe_checkpoint",
    "save_rendered_feature_rgb_probe_checkpoint",
    "load_star_model_from_training_checkpoint",
    "load_star_training_checkpoint",
    "optimizer_lrs",
    "save_star_training_checkpoint",
    "set_optimizer_lr",
]
