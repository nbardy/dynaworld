from __future__ import annotations

import pytest
import torch

import train_optim
from train_optim import adam_with_device_fused, optimizer_backward_step


def test_adam_with_device_fused_uses_plain_adam_on_cpu() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    optimizer = adam_with_device_fused([parameter], lr=0.01, device=torch.device("cpu"))

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 0.01
    assert optimizer.defaults["fused"] is False


def test_adam_with_device_fused_enables_fused_for_gpu_like_devices(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeAdam:
        def __init__(self, params, **kwargs):
            calls.append({"params": list(params), **kwargs})

    monkeypatch.setattr(train_optim.torch.optim, "Adam", FakeAdam)
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    optimizer = adam_with_device_fused([parameter], lr=0.02, device="mps")

    assert isinstance(optimizer, FakeAdam)
    assert calls == [{"params": [parameter], "lr": 0.02, "fused": True}]


def test_optimizer_backward_step_runs_zero_grad_backward_clip_and_step(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    calls: list[tuple[str, float]] = []

    def fake_clip(params, max_norm: float):
        calls.append(("clip", float(max_norm)))
        return torch.tensor(0.0)

    monkeypatch.setattr(train_optim.torch.nn.utils, "clip_grad_norm_", fake_clip)
    loss = parameter.square()

    optimizer_backward_step(optimizer, loss, clip_grad_params=[parameter], max_grad_norm=1.0)

    assert calls == [("clip", 1.0)]
    assert parameter.grad is not None
    assert float(parameter.detach()) == pytest.approx(1.6)


def test_optimizer_backward_step_requires_clip_params_when_clipping() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    try:
        optimizer_backward_step(optimizer, parameter.square(), max_grad_norm=1.0)
    except ValueError as exc:
        assert "clip_grad_params" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert parameter.grad is None
    assert float(parameter.detach()) == 2.0
