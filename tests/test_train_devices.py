from __future__ import annotations

import torch
import pytest

from train_devices import clear_torch_device_cache, resolve_torch_device, sync_torch_device
from fast_attn import pick_device


def test_resolve_torch_device_auto_prefers_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_torch_device("auto", auto_cuda=True) == torch.device("mps")


def test_resolve_torch_device_preserves_cpu_fallback_when_cuda_auto_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_torch_device("auto", auto_cuda=False) == torch.device("cpu")


def test_resolve_torch_device_can_use_cuda_auto_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_torch_device("auto", auto_cuda=True) == torch.device("cuda")


def test_resolve_torch_device_can_prefer_cuda_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_torch_device("auto", auto_cuda=True, auto_prefer_cuda=True) == torch.device("cuda")


def test_fast_attn_pick_device_uses_shared_cuda_first_auto_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert pick_device() == resolve_torch_device("auto", auto_cuda=True, auto_prefer_cuda=True)


def test_resolve_torch_device_validates_requested_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_torch_device("cuda", validate_requested=True)


def test_sync_torch_device_uses_cuda_synchronize(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[torch.device] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))

    sync_torch_device(torch.device("cuda:0"))

    assert calls == [torch.device("cuda:0")]


def test_sync_torch_device_skips_cpu() -> None:
    sync_torch_device(torch.device("cpu"))


def test_sync_torch_device_skips_unavailable_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append(None))

    sync_torch_device(torch.device("mps"))

    assert calls == []


def test_sync_torch_device_skips_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[torch.device] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))

    sync_torch_device(torch.device("cuda:0"))

    assert calls == []


def test_clear_torch_device_cache_clears_and_syncs_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str | torch.device] = []

    monkeypatch.setattr("train_devices.gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))

    clear_torch_device_cache(torch.device("cuda:0"), sync=True)

    assert calls == ["gc", "empty_cache", torch.device("cuda:0")]


def test_clear_torch_device_cache_skips_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("train_devices.gc.collect", lambda: calls.append("gc"))

    clear_torch_device_cache(torch.device("cpu"), sync=True)

    assert calls == ["gc"]
