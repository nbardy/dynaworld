from __future__ import annotations

import random
from typing import Any

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def tensor_diff_stats(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {
            "both_none": a is None and b is None,
            "max_abs": None,
            "mean_abs": None,
            "shape": None,
        }
    if tuple(a.shape) != tuple(b.shape):
        return {
            "both_none": False,
            "shape_mismatch": [list(a.shape), list(b.shape)],
            "max_abs": None,
            "mean_abs": None,
            "shape": None,
        }
    delta = (a.detach() - b.detach()).abs()
    return {
        "both_none": False,
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "shape": list(a.shape),
    }


def max_tensor_diff(diff_by_key: dict[str, dict[str, Any]]) -> float:
    return max((float(value.get("max_abs") or 0.0) for value in diff_by_key.values()), default=0.0)


def grad_diff_stats(
    base: dict[str, torch.Tensor | None],
    candidate: dict[str, torch.Tensor | None],
) -> dict[str, dict[str, Any]]:
    keys = sorted(set(base) | set(candidate))
    return {key: tensor_diff_stats(base.get(key), candidate.get(key)) for key in keys}
