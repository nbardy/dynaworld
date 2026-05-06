from __future__ import annotations

import torch

from dynamic_gauge_foam import DynamicGaugeFoamVideo


def test_dynamic_gauge_foam_renderer_backward_smoke() -> None:
    frames = torch.zeros(2, 3, 8, 8)
    frames[:, 0] = 0.2
    frames[:, 1] = 0.5
    frames[:, 2] = 0.8
    model = DynamicGaugeFoamVideo(
        frame_times=torch.tensor([0.0, 1.0]),
        init_frames=frames,
        primitive_count=16,
        feature_dim=4,
        atlas_res=2,
        num_time_ctrl=3,
        render_size=8,
        fov_degrees=55.0,
        init_depth=2.0,
        radius_scale=1.8,
        opacity_init=0.9,
        feature_noise=0.01,
        color_hidden_dim=16,
        rgb_skip=True,
        seed=3,
    )
    out = model(
        torch.tensor([0, 1]),
        chunk_pixels=16,
        max_hits=4,
        near=0.05,
        far=10.0,
        falloff=2.5,
        min_alpha=1.0e-4,
    )
    assert out.rgb.shape == (2, 8, 8, 3)
    assert out.alpha.shape == (2, 8, 8, 1)
    assert float(out.alpha.detach().max()) > 0.0
    loss = out.rgb.mean() + out.alpha.mean()
    loss.backward()
    assert model.p0.grad is not None
    assert torch.isfinite(model.p0.grad).all()
    assert model.atlas.grad is not None
    assert torch.isfinite(model.atlas.grad).all()
    assert model.twist_ctrl.grad is not None
    assert torch.isfinite(model.twist_ctrl.grad).all()
