from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.postprocess_dof import depth_aware_defocus_blur


def test_depth_aware_defocus_blur_shape_finiteness_and_gradients() -> None:
    torch.manual_seed(7)
    rgb = torch.rand(2, 3, 7, 8, requires_grad=True)
    depth = (torch.rand(2, 1, 7, 8) + 0.25).requires_grad_()
    alpha = torch.rand(2, 1, 7, 8)
    inv_focus_depth = torch.tensor(1.1, requires_grad=True)
    log_q = torch.tensor(0.2, requires_grad=True)

    out = depth_aware_defocus_blur(
        rgb,
        depth,
        alpha=alpha,
        inv_focus_depth=inv_focus_depth,
        log_q=log_q,
        max_radius=2,
        depth_edge_sigma=0.75,
        detach_depth=True,
    )
    loss = out.square().mean()
    loss.backward()

    assert out.shape == rgb.shape
    assert torch.isfinite(out).all()
    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()
    assert inv_focus_depth.grad is not None
    assert torch.isfinite(inv_focus_depth.grad)
    assert log_q.grad is not None
    assert torch.isfinite(log_q.grad)
    assert depth.grad is None


def test_depth_aware_defocus_blur_can_backprop_to_depth_when_enabled() -> None:
    torch.manual_seed(11)
    rgb = torch.rand(1, 3, 6, 6, requires_grad=True)
    depth = (torch.rand(1, 1, 6, 6) + 0.5).requires_grad_()
    blur_strength = torch.tensor(1.5, requires_grad=True)

    out = depth_aware_defocus_blur(
        rgb,
        depth,
        inv_focus_depth=1.0,
        blur_strength=blur_strength,
        max_radius=1,
        detach_depth=False,
    )
    out.mean().backward()

    assert out.shape == rgb.shape
    assert torch.isfinite(out).all()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()
    assert blur_strength.grad is not None
    assert torch.isfinite(blur_strength.grad)
