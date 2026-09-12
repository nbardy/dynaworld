"""Training must not receive an image with silently dropped primitives."""
import pytest
import torch

from star_uvt_runtime import ensure_star_uvt_on_path

ensure_star_uvt_on_path(include_dynaworld_root=False)
from torch_gsplat_bridge_star_uvt import UVTRenderConfig, render_uvt_tubes, render_uvt_tubes_gated


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("gated", [False, True])
def test_overfull_tile_is_rejected_before_training_receives_rgb(monkeypatch, gated):
    monkeypatch.setenv("STAR_UVT_TILE_CAPACITY", "128")
    cfg = UVTRenderConfig(height=1, width=1, frames=1, tile_capacity=128)
    count = cfg.tile_capacity + 1
    inputs = (
        torch.tensor([[0.5, 0.5, 0.0]], device="mps").repeat(count, 1),
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device="mps").repeat(count, 1),
        torch.arange(1, count + 1, dtype=torch.float32, device="mps"),
        torch.zeros((count, 3), device="mps"),
        torch.full((count,), 0.01, device="mps"),
        torch.tensor([[1.0, 0.0, 0.0]], device="mps").repeat(count, 1),
    )
    # Explicit diagnostic rendering still retains the negative result and counts.
    diagnostic = render_uvt_tubes(*inputs, cfg, return_aux=True)
    assert int(diagnostic.tile_counts.max().cpu()) == count
    assert bool(diagnostic.tile_overflow.any().cpu())
    assert abs(float(diagnostic.image[0, 0, 0, 0].cpu()) - (1.0 - 0.99**count)) > 0.001
    with pytest.raises(RuntimeError, match="tile capacity.*overflow"):
        if gated:
            render_uvt_tubes_gated(
                *inputs, torch.zeros(count, dtype=torch.int32, device="mps"),
                torch.ones(count, dtype=torch.int32, device="mps"), cfg,
            )
        else:
            render_uvt_tubes(*inputs, cfg)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("alpha_mode", ["peak_splat", "beer_lambert"])
def test_moving_tubes_keep_complete_rgb_and_gradients_without_lifetime_box_overflow(monkeypatch, alpha_mode):
    """Disjoint time slices must not exhaust a tile with lifetime-only candidates."""
    from torch_gsplat_bridge_star_uvt import brute_force_render_uvt_tubes
    from research_project.trainer_harness.tile_metal_autograd import render_uvt_tubes_metal_tile_backward

    monkeypatch.setenv("STAR_UVT_TILE_CAPACITY", "128")
    cfg = UVTRenderConfig(height=2, width=3, frames=3, background=(0.1, 0.2, 0.3), alpha_mode=alpha_mode)
    count = 129
    inputs = (
        torch.stack((torch.linspace(-100, 100, count), torch.full((count,), 0.8), torch.zeros(count)), dim=1),
        torch.tensor([[2.0, 0.0, -80.0, 2.0, 0.0, 3201.0]]).repeat(count, 1),
        torch.linspace(1, 3, count),
        torch.tensor([[0.01, -0.02, 0.03]]).repeat(count, 1),
        torch.full((count,), 0.18),
        torch.stack((torch.linspace(0.1, 0.9, count), torch.linspace(0.8, 0.2, count), torch.full((count,), 0.4)), dim=1),
    )
    reference_inputs = tuple(t.clone().requires_grad_() for t in inputs)
    actual_inputs = tuple(t.to("mps").requires_grad_() for t in inputs)
    expected = brute_force_render_uvt_tubes(*reference_inputs, cfg)
    actual = render_uvt_tubes_metal_tile_backward(
        *actual_inputs, cfg, reduction_mode="index_add", sample_emission_mode="direct_atomic",
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
    weights = torch.linspace(-0.7, 1.1, expected.numel()).reshape(expected.shape)
    (expected * weights).sum().backward()
    (actual * weights.to("mps")).sum().backward()
    for index in (0, 1, 4, 5):
        torch.testing.assert_close(actual_inputs[index].grad.cpu(), reference_inputs[index].grad, atol=3e-5, rtol=5e-4)

    starts = torch.arange(count, dtype=torch.int32) % 2
    stops = torch.full((count,), cfg.frames, dtype=torch.int32)
    gated = render_uvt_tubes_gated(*actual_inputs, starts.to("mps"), stops.to("mps"), cfg)
    gated_expected = brute_force_render_uvt_tubes(*inputs, cfg, active_start=starts, active_stop=stops)
    torch.testing.assert_close(gated.cpu(), gated_expected, atol=2e-5, rtol=2e-4)
