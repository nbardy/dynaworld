from __future__ import annotations

from typing import Callable

import torch

from device_memory import DeviceMemorySampler, device_memory_stats
from train_devices import clear_torch_device_cache


def clear_device_cache(device: torch.device) -> None:
    clear_torch_device_cache(device, sync=True)


def run_with_memory_sampling(
    device: torch.device,
    *,
    interval_ms: float,
    clear_cache: bool,
    fn: Callable[[], dict[str, float | int]],
) -> dict[str, float | int]:
    if clear_cache:
        clear_device_cache(device)
    start_stats = device_memory_stats(device)
    start_stats = {f"start_{key}": value for key, value in start_stats.items()}
    with DeviceMemorySampler(device, interval_ms=interval_ms) as sampler:
        sample = fn()
    end_stats = device_memory_stats(device)
    end_stats = {f"end_{key}": value for key, value in end_stats.items()}
    return {**sample, **start_stats, **end_stats, **sampler.stats()}
