from __future__ import annotations

import threading
from typing import Callable

import torch

from train_devices import clear_torch_device_cache


def device_memory_stats(device: torch.device) -> dict[str, int]:
    if device.type == "mps" and hasattr(torch, "mps"):
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    if device.type == "cuda":
        return {
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "driver_allocated_bytes": int(torch.cuda.memory_reserved(device)),
        }
    return {}


def clear_device_cache(device: torch.device) -> None:
    clear_torch_device_cache(device, sync=True)


class DeviceMemorySampler:
    def __init__(self, device: torch.device, *, interval_ms: float) -> None:
        self.device = device
        self.interval_s = float(interval_ms) / 1000.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.max_current_allocated_bytes = 0
        self.max_driver_allocated_bytes = 0
        self.samples = 0
        self.enabled = self.interval_s > 0.0 and bool(device_memory_stats(device))

    def _sample_once(self) -> None:
        stats = device_memory_stats(self.device)
        if not stats:
            return
        self.max_current_allocated_bytes = max(
            self.max_current_allocated_bytes,
            int(stats["current_allocated_bytes"]),
        )
        self.max_driver_allocated_bytes = max(
            self.max_driver_allocated_bytes,
            int(stats["driver_allocated_bytes"]),
        )
        self.samples += 1

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample_once()
            self.stop_event.wait(self.interval_s)
        self._sample_once()

    def __enter__(self) -> "DeviceMemorySampler":
        if self.enabled:
            self._sample_once()
            self.thread = threading.Thread(target=self._run, name="device-memory-sampler", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if not self.enabled:
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample_once()

    def stats(self) -> dict[str, int]:
        if not self.enabled:
            return {}
        return {
            "sampled_peak_current_allocated_bytes": int(self.max_current_allocated_bytes),
            "sampled_peak_driver_allocated_bytes": int(self.max_driver_allocated_bytes),
            "memory_sample_count": int(self.samples),
        }


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
