from __future__ import annotations

from typing import Literal

import torch

from runtime_types import GaussianSequence


SEQUENCE_GRAD_FIELDS = ("xyz", "scales", "quats", "opacities", "rgbs")
MissingGradPolicy = Literal["none", "zero"]


def sequence_leaf_grads(
    sequence: GaussianSequence,
    *,
    missing: MissingGradPolicy = "none",
    clone: bool = False,
) -> dict[str, torch.Tensor | None]:
    grads: dict[str, torch.Tensor | None] = {}
    for name in SEQUENCE_GRAD_FIELDS:
        tensor = getattr(sequence, name)
        grad = tensor.grad
        if grad is None:
            if missing == "none":
                grads[name] = None
            elif missing == "zero":
                grads[name] = torch.zeros_like(tensor)
            else:
                raise ValueError(f"Unsupported missing gradient policy {missing!r}.")
        else:
            grads[name] = grad.detach().clone() if clone else grad
    return grads


def module_parameter_grads(module: torch.nn.Module | None, *, clone: bool = True) -> dict[str, torch.Tensor | None]:
    if module is None:
        return {}
    return {
        name: None if param.grad is None else (param.grad.detach().clone() if clone else param.grad)
        for name, param in module.named_parameters()
    }


def named_module_parameter_grads(
    modules: dict[str, torch.nn.Module | None],
    *,
    clone: bool = True,
) -> dict[str, torch.Tensor | None]:
    grads: dict[str, torch.Tensor | None] = {}
    for prefix, module in modules.items():
        for name, grad in module_parameter_grads(module, clone=clone).items():
            grads[f"{prefix}.{name}"] = grad
    return grads
