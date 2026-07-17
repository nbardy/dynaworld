from __future__ import annotations

import math
from typing import Any

import torch

from powerfoam_metal_config import LR_GROUP_SPECS


def cosine_scheduled_lr(
    initial: float,
    final: float,
    step: int,
    total_steps: int,
    *,
    warmup_steps: int = 0,
) -> float:
    warmup = max(int(warmup_steps), 0)
    step_f = float(step)
    max_steps = max(int(total_steps), 1)
    if warmup and step < warmup:
        return float(initial) * step_f / float(warmup)
    if step > max_steps:
        return float(final)
    denom = max(float(max_steps - warmup), 1.0)
    progress = (step_f - float(warmup)) / denom
    lr_cos = float(final) + 0.5 * (float(initial) - float(final)) * (1.0 + math.cos(math.pi * progress))
    return float(lr_cos)


def powerfoam_group_initial_lr(train_cfg: dict[str, Any], group_name: str) -> float:
    multiplier_key, official_key, _warmup = LR_GROUP_SPECS[group_name]
    if official_key is not None:
        explicit = train_cfg[f"{official_key}_lr_init"]
        if explicit is not None:
            return float(explicit)
    return float(train_cfg["lr"]) * float(train_cfg[multiplier_key])


def powerfoam_group_final_lr(train_cfg: dict[str, Any], group_name: str, initial_lr: float) -> float:
    _multiplier_key, official_key, _warmup = LR_GROUP_SPECS[group_name]
    if official_key is None:
        return float(initial_lr)
    explicit = train_cfg[f"{official_key}_lr_final"]
    if explicit is None:
        return float(initial_lr)
    return float(explicit)


def powerfoam_group_warmup_steps(train_cfg: dict[str, Any], group_name: str) -> int:
    _multiplier_key, _official_key, default_warmup_steps = LR_GROUP_SPECS[group_name]
    overrides = train_cfg["lr_warmup_steps"]
    if group_name in overrides:
        return int(overrides[group_name])
    return int(default_warmup_steps)


def powerfoam_group_lr_metadata(train_cfg: dict[str, Any], group_name: str) -> dict[str, float | int]:
    initial_lr = powerfoam_group_initial_lr(train_cfg, group_name)
    return {
        "lr": initial_lr,
        "initial_lr": initial_lr,
        "final_lr": powerfoam_group_final_lr(train_cfg, group_name, initial_lr),
        "warmup_steps": powerfoam_group_warmup_steps(train_cfg, group_name),
    }


def update_powerfoam_learning_rates(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    *,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    if str(train_cfg["lr_schedule"]) == "cosine":
        for group in optimizer.param_groups:
            if "initial_lr" not in group or "final_lr" not in group:
                continue
            group["lr"] = cosine_scheduled_lr(
                float(group["initial_lr"]),
                float(group["final_lr"]),
                int(step),
                int(total_steps),
                warmup_steps=int(group.get("warmup_steps", 0)),
            )
    return {str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}
