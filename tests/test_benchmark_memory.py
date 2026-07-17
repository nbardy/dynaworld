from __future__ import annotations

import torch

from benchmark_memory import device_memory_stats, run_with_memory_sampling


def test_cpu_memory_sampling_is_a_noop_but_preserves_sample_payload() -> None:
    payload = run_with_memory_sampling(
        torch.device("cpu"),
        interval_ms=1.0,
        clear_cache=False,
        fn=lambda: {"elapsed_ms": 1.5, "count": 2},
    )
    assert payload == {"elapsed_ms": 1.5, "count": 2}


def test_device_memory_stats_returns_empty_for_cpu() -> None:
    assert device_memory_stats(torch.device("cpu")) == {}
