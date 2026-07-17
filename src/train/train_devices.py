from __future__ import annotations

import gc

import torch


def resolve_torch_device(
    value: str,
    *,
    auto_cuda: bool = False,
    auto_prefer_cuda: bool = False,
    validate_requested: bool = False,
) -> torch.device:
    if value == "auto":
        if auto_cuda and auto_prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if auto_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(value)
    if validate_requested and device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if validate_requested and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def sync_torch_device(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def clear_torch_device_cache(device: torch.device | str, *, sync: bool = False) -> None:
    gc.collect()
    resolved = torch.device(device)
    if resolved.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if sync:
        sync_torch_device(resolved)


__all__ = ["clear_torch_device_cache", "resolve_torch_device", "sync_torch_device"]
