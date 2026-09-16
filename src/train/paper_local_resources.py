"""Bounded local-playground resources; no Torch import in launcher checks.

The publication profile keeps its existing launch rules. The opt-in playground
charges a CPU process tree, a capped MPS allocator, and OS headroom against an
8-GiB machine. Existing swapped-out desktop pages are recorded; new swap growth
during the job is a stop condition, rather than requiring a reboot first.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

GIB = 1024**3
RESOURCE_KEYS = (
    "process_tree_rss_limit_bytes", "mps_allocator_limit_bytes",
    "host_memory_reserve_bytes", "min_available_memory_bytes",
    "max_swap_growth_bytes", "min_disk_free_bytes", "output_limit_bytes",
    "disk_budget_bytes", "max_load_per_cpu",
)


def normalize_local_resources(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != set(RESOURCE_KEYS):
        raise ValueError("local_resources must specify the complete playground resource contract")
    result = dict(raw)
    for key in RESOURCE_KEYS:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"local_resources.{key} must be positive and finite")
        if key != "max_load_per_cpu" and not isinstance(value, int):
            raise ValueError(f"local_resources.{key} must be integer bytes")
    total = sum(result[k] for k in (
        "process_tree_rss_limit_bytes", "mps_allocator_limit_bytes", "host_memory_reserve_bytes",
    ))
    if total > 8 * GIB or result["disk_budget_bytes"] > 16 * GIB:
        raise ValueError("playground budgets must fit 8 GiB RAM and 16 GiB disk")
    if result["output_limit_bytes"] >= result["disk_budget_bytes"]:
        raise ValueError("playground disk budget must leave room for data and dependencies")
    return result


def require_local_host(snapshot: Mapping[str, Any], limits: Mapping[str, Any]) -> None:
    failures = []
    if snapshot.get("platform") != "darwin":
        failures.append("macOS required")
    if snapshot.get("available_memory_bytes", 0) < limits["min_available_memory_bytes"]:
        failures.append("available memory")
    if snapshot.get("disk_free_bytes", 0) < limits["min_disk_free_bytes"]:
        failures.append("free disk")
    if snapshot.get("load_1m_per_logical_cpu", math.inf) > limits["max_load_per_cpu"]:
        failures.append("CPU load")
    if "swap_used_bytes" not in snapshot or "swap_probe_error" in snapshot:
        failures.append("swap probe unavailable")
    if failures:
        raise RuntimeError("local playground host gate: " + ", ".join(failures))


def configure_local_mps(raw_protocol: Mapping[str, Any], device: str) -> None:
    limits = normalize_local_resources(raw_protocol.get("local_resources"))
    if limits is None or str(device) != "mps":
        return
    import torch

    fraction = limits["mps_allocator_limit_bytes"] / torch.mps.recommended_max_memory()
    torch.mps.set_per_process_memory_fraction(min(1.0, fraction))


def directory_bytes(path: Path) -> int:
    """Logical file bytes, including hardlinks, without following directory links."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for folder, _, files in os.walk(path):
        for name in files:
            file = Path(folder) / name
            try:
                total += file.stat().st_size
            except FileNotFoundError:
                pass  # A temporary file may disappear between list and stat.
    return total


def process_tree_rss(pid: int) -> int:
    """RSS of the lane, descendants, and the small observing launcher."""
    output = subprocess.check_output(("ps", "-axo", "pid=,ppid=,rss="), text=True)
    rows = [tuple(map(int, line.split())) for line in output.splitlines() if line.strip()]
    children = {pid, os.getpid()}
    while True:
        enlarged = children | {p for p, parent, _ in rows if parent in children}
        if enlarged == children:
            break
        children = enlarged
    return sum(rss * 1024 for p, _, rss in rows if p in children)


def check_running_limits(
    limits: Mapping[str, Any], *, tree_rss: int, output_bytes: int,
    initial_swap: int, snapshot: Mapping[str, Any],
) -> None:
    if tree_rss > limits["process_tree_rss_limit_bytes"]:
        raise RuntimeError("local playground exceeded process-tree RSS limit")
    if output_bytes > limits["output_limit_bytes"]:
        raise RuntimeError("local playground exceeded output disk limit")
    if "swap_probe_error" in snapshot or "swap_used_bytes" not in snapshot:
        raise RuntimeError("local playground lost the swap resource probe")
    if snapshot["swap_used_bytes"] - initial_swap > limits["max_swap_growth_bytes"]:
        raise RuntimeError("local playground caused excessive host swap growth")
    if snapshot.get("available_memory_bytes", 0) < limits["host_memory_reserve_bytes"]:
        raise RuntimeError("local playground exhausted host memory headroom")
    if snapshot.get("disk_free_bytes", 0) < limits["min_disk_free_bytes"]:
        raise RuntimeError("local playground exhausted free disk headroom")


class LocalResourceMonitor:
    """One-second host/disk checks plus a process-tree sample at every RSS poll."""

    def __init__(self, limits, output_root, snapshot_fn):
        if output_root is None:
            raise ValueError("playground execution requires an output root for its disk guard")
        self.limits = limits
        self.output_root = output_root
        self.snapshot_fn = snapshot_fn
        self.initial = self.snapshot_fn()
        require_local_host(self.initial, limits)
        self.snapshot = self.initial
        self.next_host_sample = 0.0
        self.peak_tree_rss = 0
        self.peak_output_bytes = 0
        self.peak_swap_growth = 0

    def sample(self, pid):
        self.peak_tree_rss = max(self.peak_tree_rss, process_tree_rss(pid))
        if time.monotonic() >= self.next_host_sample:
            self.snapshot = self.snapshot_fn()
            self.peak_output_bytes = max(self.peak_output_bytes, directory_bytes(self.output_root))
            self.peak_swap_growth = max(
                self.peak_swap_growth,
                self.snapshot["swap_used_bytes"] - self.initial["swap_used_bytes"],
            )
            self.next_host_sample = time.monotonic() + 1.0
        check_running_limits(
            self.limits, tree_rss=self.peak_tree_rss, output_bytes=self.peak_output_bytes,
            initial_swap=self.initial["swap_used_bytes"], snapshot=self.snapshot,
        )

    def receipt(self):
        # Include files flushed at shutdown, after the last running sample.
        self.peak_output_bytes = max(self.peak_output_bytes, directory_bytes(self.output_root))
        check_running_limits(
            self.limits, tree_rss=self.peak_tree_rss, output_bytes=self.peak_output_bytes,
            initial_swap=self.initial["swap_used_bytes"], snapshot=self.snapshot_fn(),
        )
        return {
            "measurement": "process_tree_and_launcher_rss_plus_host_and_output_polling",
            "peak_process_tree_and_launcher_rss_bytes": self.peak_tree_rss,
            "peak_output_bytes": self.peak_output_bytes,
            "peak_host_swap_growth_bytes": self.peak_swap_growth,
            "limits": self.limits,
        }
