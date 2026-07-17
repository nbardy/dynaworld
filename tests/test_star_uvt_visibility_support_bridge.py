from __future__ import annotations

import torch

from research_experiments.star_uvt_feature_tubes.visibility_support_bridge_prototype import (
    BridgeConfig,
    _make_miss_model,
    _support_proxy_loss,
    _target_mask,
    _target_points,
)
from research_experiments.star_uvt_feature_tubes.visibility_support_birth_split_prototype import (
    BirthSplitConfig,
    apply_support_birth_split,
)
from star_uvt_feature_tube_model import (
    FeatureTubeRenderConfig,
    render_model_features,
)


def test_support_proxy_sends_geometry_gradients_from_zero_hit_target() -> None:
    bridge = BridgeConfig(frames=4, height=20, width=20, tubes=4, steps=2, seed=3)
    render_cfg = FeatureTubeRenderConfig(
        frames=bridge.frames,
        height=bridge.height,
        width=bridge.width,
        feature_dim=bridge.feature_dim,
        alpha_threshold=1.0 / 255.0,
        max_alpha=0.99,
    )
    device = torch.device("cpu")
    mask = _target_mask(render_cfg, radius=bridge.target_radius, device=device)
    points = _target_points(mask, frames=bridge.frames)
    model = _make_miss_model(render_cfg, bridge, device=device)

    loss = _support_proxy_loss(
        model,
        points,
        scale_px=bridge.proxy_scale_px,
        temperature=bridge.proxy_temperature,
    )
    loss.backward()

    assert model.center_uv.grad is not None
    assert model.velocity_uv.grad is not None
    assert bool((model.center_uv.grad.abs() > 0).any())
    assert bool((model.velocity_uv.grad.abs() > 0).any())


def test_birth_split_reuses_fixed_budget_and_increases_target_support() -> None:
    config = BirthSplitConfig(frames=4, height=20, width=20, tubes=8, birth_tubes=4, seed=5)
    bridge = BridgeConfig(
        frames=config.frames,
        height=config.height,
        width=config.width,
        tubes=config.tubes,
        feature_dim=config.feature_dim,
        seed=config.seed,
        target_radius=config.target_radius,
    )
    render_cfg = FeatureTubeRenderConfig(
        frames=config.frames,
        height=config.height,
        width=config.width,
        feature_dim=config.feature_dim,
        alpha_threshold=1.0 / 255.0,
        max_alpha=0.99,
    )
    device = torch.device("cpu")
    mask = _target_mask(render_cfg, radius=config.target_radius, device=device)
    model = _make_miss_model(render_cfg, bridge, device=device)

    initial = model.tube_count
    apply_support_birth_split(model, mask, config)
    _feature_image, alpha = render_model_features(model)

    assert model.tube_count == initial
    target_alpha = alpha[mask].detach()
    assert float(target_alpha.mean()) > 0.50
    assert float((target_alpha > 0.10).to(torch.float32).mean()) > 0.80
