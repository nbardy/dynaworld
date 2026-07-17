from __future__ import annotations

import torch

from vjepa_feature_loss import TorchHubVJEPAFeatureLoss


class _FakeVJEPAEncoder(torch.nn.Module):
    embed_dim = 3

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # Input arrives as [B, C, T, H, W]. Return [B, T, C] tokens.
        return video.mean(dim=(-2, -1)).permute(0, 2, 1)


def test_vjepa_feature_loss_keeps_prediction_gradient(monkeypatch) -> None:
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _FakeVJEPAEncoder())
    loss_fn = TorchHubVJEPAFeatureLoss(
        model_id="fake_vjepa",
        crop_size=4,
        dtype="float32",
        temporal_stride=2,
        normalize_features=False,
        loss_type="mse",
    )
    prediction = torch.full((4, 3, 4, 4), 0.25, dtype=torch.float32, requires_grad=True)
    target = torch.full((4, 3, 4, 4), 0.75, dtype=torch.float32)

    loss = loss_fn(prediction, target)
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0
