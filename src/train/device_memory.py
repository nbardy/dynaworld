from __future__ import annotations

import threading

import torch


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


class DeviceMemorySampler:
    """Sample allocator state in the background so short-lived peaks are retained."""

    def __init__(self, device: torch.device, *, interval_ms: float = 5.0) -> None:
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
        self.start()
        return self

    def start(self) -> None:
        if self.enabled:
            self._sample_once()
            self.thread = threading.Thread(target=self._run, name="device-memory-sampler", daemon=True)
            self.thread.start()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.stop()

    def stop(self) -> None:
        if not self.enabled:
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample_once()

    def stats(self) -> dict[str, int]:
        if not self.enabled:
            return {
                "sampled_peak_current_allocated_bytes": 0,
                "sampled_peak_driver_allocated_bytes": 0,
                "memory_sample_count": 0,
            }
        return {
            "sampled_peak_current_allocated_bytes": int(self.max_current_allocated_bytes),
            "sampled_peak_driver_allocated_bytes": int(self.max_driver_allocated_bytes),
            "memory_sample_count": int(self.samples),
        }


__all__ = ["DeviceMemorySampler", "device_memory_stats"]
