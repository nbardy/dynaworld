from __future__ import annotations

import pytest
import torch

from research_experiments.softmax_gs.reference import softmax_gs_bounded_contribution_tape
from renderers.fast_mac import FastMacRendererConfig, _ensure_fast_mac_v5_softmax_gs_on_path, render_fast_mac_3dgs


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Softmax-GS Metal forward regression requires MPS",
)


def _config(*, enabled: bool) -> FastMacRendererConfig:
    return FastMacRendererConfig.from_mapping(
        {
            "rgb_variant": "v5_softmax_gs",
            "depth_mode": "center_camera_z",
            "softmax_gs_enabled": enabled,
            "softmax_gs_beta": 12.0,
            "softmax_gs_gamma": 0.0,
        },
        fallback_tile_size=16,
        fallback_alpha_threshold=1.0 / 255.0,
    )


def _render_pair(colors: torch.Tensor, config: FastMacRendererConfig) -> torch.Tensor:
    device = colors.device
    means = torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 1.5]], device=device)
    scales = torch.full((2, 3), 0.16, device=device)
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        device=device,
    )
    opacities = torch.full((2, 1), 0.7, device=device)
    image, _alpha = render_fast_mac_3dgs(
        means,
        scales,
        quats,
        opacities,
        colors,
        height=32,
        width=32,
        fx=35.0,
        fy=35.0,
        cx=16.0,
        cy=16.0,
        projection_mode="legacy_pinhole",
        config=config,
    )
    return image


def test_softmax_gs_same_depth_two_splats_are_order_invariant_on_metal() -> None:
    colors_ab = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device="mps",
    )
    colors_ba = colors_ab.flip(0).contiguous()

    with torch.no_grad():
        vanilla_ab = _render_pair(colors_ab, _config(enabled=False))
        vanilla_ba = _render_pair(colors_ba, _config(enabled=False))
        softmax_ab = _render_pair(colors_ab, _config(enabled=True))
        softmax_ba = _render_pair(colors_ba, _config(enabled=True))

    assert (vanilla_ab - vanilla_ba).abs().max().item() > 0.1
    torch.testing.assert_close(softmax_ab, softmax_ba, atol=1e-5, rtol=1e-5)


def test_softmax_gs_training_path_matches_metal_forward_and_backprops() -> None:
    device = torch.device("mps")
    torch.manual_seed(7)
    gaussian_count = 6
    means = torch.randn(gaussian_count, 3, device=device) * 0.08
    means[:, 2] = torch.linspace(1.0, 1.8, gaussian_count, device=device)
    scales = torch.full((gaussian_count, 3), 0.12, device=device)
    quats = torch.zeros(gaussian_count, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((gaussian_count, 1), 0.6, device=device)
    colors = torch.rand(gaussian_count, 3, device=device)
    config = FastMacRendererConfig.from_mapping(
        {
            "rgb_variant": "v5_softmax_gs",
            "depth_mode": "center_camera_z",
            "softmax_gs_enabled": True,
            "softmax_gs_beta": 8.0,
            "softmax_gs_gamma": 0.0,
        },
        fallback_tile_size=16,
        fallback_alpha_threshold=1.0 / 255.0,
    )
    kwargs = dict(
        height=32,
        width=32,
        fx=35.0,
        fy=35.0,
        cx=16.0,
        cy=16.0,
        projection_mode="legacy_pinhole",
        config=config,
    )

    with torch.no_grad():
        metal_image, _alpha = render_fast_mac_3dgs(means, scales, quats, opacities, colors, **kwargs)

    train_means = means.detach().clone().requires_grad_(True)
    train_scales = scales.detach().clone().requires_grad_(True)
    train_quats = quats.detach().clone().requires_grad_(True)
    train_opacities = opacities.detach().clone().requires_grad_(True)
    train_colors = colors.detach().clone().requires_grad_(True)
    train_image, _alpha = render_fast_mac_3dgs(
        train_means,
        train_scales,
        train_quats,
        train_opacities,
        train_colors,
        **kwargs,
    )
    torch.testing.assert_close(train_image, metal_image, atol=1e-5, rtol=1e-5)

    train_image.square().mean().backward()
    for tensor in (train_means, train_scales, train_quats, train_opacities, train_colors):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_softmax_gs_native_backward_matches_torch_recompute_reference() -> None:
    _ensure_fast_mac_v5_softmax_gs_on_path()
    from torch_gsplat_bridge_v5_softmax_gs import RasterConfig, rasterize_projected_gaussians
    from torch_gsplat_bridge_v5_softmax_gs.rasterize import _normalize_inputs, _rasterize_softmax_gs_torch_train

    device = torch.device("mps")
    torch.manual_seed(3)
    gaussian_count = 5
    means = torch.tensor(
        [[10.0, 10.0], [12.0, 10.5], [15.0, 15.0], [18.0, 12.0], [20.0, 20.0]],
        device=device,
        requires_grad=True,
    )
    conics = torch.tensor(
        [[0.08, 0.0, 0.08], [0.07, 0.01, 0.06], [0.08, 0.0, 0.07], [0.06, 0.0, 0.08], [0.07, -0.01, 0.07]],
        device=device,
        requires_grad=True,
    )
    colors = torch.rand(gaussian_count, 3, device=device, requires_grad=True)
    opacities = torch.full((gaussian_count,), 0.65, device=device, requires_grad=True)
    depths = torch.linspace(1.0, 1.4, gaussian_count, device=device, requires_grad=True)
    config = RasterConfig(
        height=32,
        width=32,
        tile_size=16,
        max_fast_pairs=128,
        alpha_threshold=1.0 / 255.0,
        transmittance_threshold=1.0e-4,
        background=(0.1, 0.2, 0.3),
        enable_overflow_fallback=False,
        inputs_sorted_by_depth=True,
        softmax_gs_enabled=True,
        softmax_gs_beta=4.0,
        softmax_gs_gamma=3.0,
    )

    native = rasterize_projected_gaussians(means, conics, colors, opacities, depths, config)
    grad = torch.randn_like(native)
    native_grads = torch.autograd.grad(native, (means, conics, colors, opacities, depths), grad)

    ref_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (means, conics, colors, opacities, depths))
    means_b, conics_b, colors_b, opacities_b, depths_b, _was_batched = _normalize_inputs(*ref_inputs)
    reference = _rasterize_softmax_gs_torch_train(means_b, conics_b, colors_b, opacities_b, depths_b, config)[0]
    reference_grads_b = torch.autograd.grad(reference, (means_b, conics_b, colors_b, opacities_b, depths_b), grad)
    reference_grads = (
        reference_grads_b[0][0],
        reference_grads_b[1][0],
        reference_grads_b[2][0],
        reference_grads_b[3][0],
        reference_grads_b[4][0],
    )

    torch.testing.assert_close(native, reference, atol=1.0e-5, rtol=1.0e-5)
    for actual, expected in zip(native_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-4)


def _bounded_tape_fixture(max_fast_pairs: int):
    _ensure_fast_mac_v5_softmax_gs_on_path()
    from torch_gsplat_bridge_v5_softmax_gs import RasterConfig, rasterize_softmax_gs_bounded_tape

    device = torch.device("mps")
    means = torch.tensor(
        [[8.5, 8.5], [8.9, 8.4], [7.8, 8.7], [9.2, 8.9], [8.2, 7.9]],
        device=device,
    )
    conics = torch.tensor(
        [[0.08, 0.0, 0.07], [0.06, 0.01, 0.08], [0.07, -0.01, 0.06], [0.09, 0.0, 0.05], [0.05, 0.01, 0.09]],
        device=device,
    )
    colors = torch.rand(5, 3, device=device)
    opacities = torch.tensor([0.35, 0.72, 0.28, 0.63, 0.42], device=device)
    depths = torch.tensor([1.0, 1.02, 1.04, 1.08, 1.25], device=device)
    config = RasterConfig(
        height=16,
        width=16,
        tile_size=16,
        max_fast_pairs=max_fast_pairs,
        alpha_threshold=1.0 / 255.0,
        transmittance_threshold=1.0e-4,
        background=(0.0, 0.0, 0.0),
        enable_overflow_fallback=True,
        inputs_sorted_by_depth=True,
        softmax_gs_enabled=True,
        softmax_gs_beta=5.0,
        softmax_gs_gamma=3.0,
    )
    tape = rasterize_softmax_gs_bounded_tape(means, conics, colors, opacities, depths, config, k_limit=3)
    pixel = torch.tensor([8.5, 8.5], device=device)
    delta = pixel.view(1, 2) - means
    power = -0.5 * (
        conics[:, 0] * delta[:, 0].square()
        + 2.0 * conics[:, 1] * delta[:, 0] * delta[:, 1]
        + conics[:, 2] * delta[:, 1].square()
    )
    absorbance = torch.minimum(opacities * torch.exp(power), torch.full_like(opacities, 0.99))
    reference = softmax_gs_bounded_contribution_tape(
        absorbance.cpu().to(torch.float64),
        power.cpu().to(torch.float64),
        depths.cpu().to(torch.float64),
        beta=5.0,
        gamma=3.0,
        k_limit=3,
    )
    return tape, reference


def test_softmax_gs_fast_bounded_tape_matches_reference() -> None:
    (selected_ids, selected_weights, residual_weight, final_alpha), reference = _bounded_tape_fixture(max_fast_pairs=128)
    y = x = 8

    torch.testing.assert_close(selected_ids[y, x].cpu().to(torch.long), reference.selected_indices)
    torch.testing.assert_close(selected_weights[y, x].cpu().to(torch.float64), reference.selected_weights, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(residual_weight[y, x].cpu().to(torch.float64), reference.residual_weight, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(final_alpha[y, x].cpu().to(torch.float64), reference.final_alpha, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.parametrize("max_fast_pairs", [128, 1])
def test_softmax_gs_full_tape_backward_matches_reference(max_fast_pairs: int) -> None:
    _ensure_fast_mac_v5_softmax_gs_on_path()
    from torch_gsplat_bridge_v5_softmax_gs import RasterConfig, rasterize_projected_gaussians
    from torch_gsplat_bridge_v5_softmax_gs.rasterize import _normalize_inputs, _rasterize_softmax_gs_torch_train

    device = torch.device("mps")
    torch.manual_seed(23)
    gaussian_count = 5
    means = torch.tensor(
        [[10.0, 10.0], [12.0, 10.5], [15.0, 15.0], [18.0, 12.0], [20.0, 20.0]],
        device=device,
        requires_grad=True,
    )
    conics = torch.tensor(
        [[0.08, 0.0, 0.08], [0.07, 0.01, 0.06], [0.08, 0.0, 0.07], [0.06, 0.0, 0.08], [0.07, -0.01, 0.07]],
        device=device,
        requires_grad=True,
    )
    colors = torch.rand(gaussian_count, 3, device=device, requires_grad=True)
    opacities = torch.full((gaussian_count,), 0.65, device=device, requires_grad=True)
    depths = torch.linspace(1.0, 1.4, gaussian_count, device=device, requires_grad=True)
    config = RasterConfig(
        height=32,
        width=32,
        tile_size=16,
        max_fast_pairs=max_fast_pairs,
        alpha_threshold=1.0 / 255.0,
        transmittance_threshold=1.0e-4,
        background=(0.1, 0.2, 0.3),
        enable_overflow_fallback=True,
        inputs_sorted_by_depth=True,
        softmax_gs_enabled=True,
        softmax_gs_beta=4.0,
        softmax_gs_gamma=3.0,
        softmax_gs_tape_k=gaussian_count,
    )

    native = rasterize_projected_gaussians(means, conics, colors, opacities, depths, config)
    grad = torch.randn_like(native)
    native_grads = torch.autograd.grad(native, (means, conics, colors, opacities, depths), grad)

    ref_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (means, conics, colors, opacities, depths))
    means_b, conics_b, colors_b, opacities_b, depths_b, _was_batched = _normalize_inputs(*ref_inputs)
    reference = _rasterize_softmax_gs_torch_train(means_b, conics_b, colors_b, opacities_b, depths_b, config)[0]
    reference_grads_b = torch.autograd.grad(reference, (means_b, conics_b, colors_b, opacities_b, depths_b), grad)
    reference_grads = (
        reference_grads_b[0][0],
        reference_grads_b[1][0],
        reference_grads_b[2][0],
        reference_grads_b[3][0],
        reference_grads_b[4][0],
    )

    torch.testing.assert_close(native, reference, atol=1.0e-5, rtol=1.0e-5)
    for actual, expected in zip(native_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=3.0e-5, rtol=3.0e-4)


def test_softmax_gs_overflow_bounded_tape_matches_reference() -> None:
    (selected_ids, selected_weights, residual_weight, final_alpha), reference = _bounded_tape_fixture(max_fast_pairs=1)
    y = x = 8

    torch.testing.assert_close(selected_ids[y, x].cpu().to(torch.long), reference.selected_indices)
    torch.testing.assert_close(selected_weights[y, x].cpu().to(torch.float64), reference.selected_weights, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(residual_weight[y, x].cpu().to(torch.float64), reference.residual_weight, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(final_alpha[y, x].cpu().to(torch.float64), reference.final_alpha, atol=2.0e-5, rtol=2.0e-5)


def test_softmax_gs_overflow_backward_matches_torch_recompute_reference() -> None:
    _ensure_fast_mac_v5_softmax_gs_on_path()
    from torch_gsplat_bridge_v5_softmax_gs import RasterConfig, rasterize_projected_gaussians
    from torch_gsplat_bridge_v5_softmax_gs.rasterize import _normalize_inputs, _rasterize_softmax_gs_torch_train

    device = torch.device("mps")
    torch.manual_seed(13)
    gaussian_count = 5
    means = torch.tensor(
        [[10.0, 10.0], [11.0, 10.0], [10.5, 11.0], [11.5, 11.5], [12.0, 10.5]],
        device=device,
        requires_grad=True,
    )
    conics = torch.tensor(
        [[0.05, 0.0, 0.05], [0.06, 0.01, 0.05], [0.05, -0.01, 0.06], [0.05, 0.0, 0.05], [0.06, 0.0, 0.06]],
        device=device,
        requires_grad=True,
    )
    colors = torch.rand(gaussian_count, 3, device=device, requires_grad=True)
    opacities = torch.full((gaussian_count,), 0.7, device=device, requires_grad=True)
    depths = torch.linspace(1.0, 1.3, gaussian_count, device=device, requires_grad=True)
    config = RasterConfig(
        height=16,
        width=16,
        tile_size=16,
        max_fast_pairs=1,
        alpha_threshold=1.0 / 255.0,
        transmittance_threshold=1.0e-4,
        background=(0.04, 0.05, 0.06),
        enable_overflow_fallback=True,
        inputs_sorted_by_depth=True,
        softmax_gs_enabled=True,
        softmax_gs_beta=5.0,
        softmax_gs_gamma=2.0,
    )

    native = rasterize_projected_gaussians(means, conics, colors, opacities, depths, config)
    grad = torch.randn_like(native)
    native_grads = torch.autograd.grad(native, (means, conics, colors, opacities, depths), grad)

    ref_inputs = tuple(t.detach().clone().requires_grad_(True) for t in (means, conics, colors, opacities, depths))
    means_b, conics_b, colors_b, opacities_b, depths_b, _was_batched = _normalize_inputs(*ref_inputs)
    reference = _rasterize_softmax_gs_torch_train(means_b, conics_b, colors_b, opacities_b, depths_b, config)[0]
    reference_grads_b = torch.autograd.grad(reference, (means_b, conics_b, colors_b, opacities_b, depths_b), grad)
    reference_grads = (
        reference_grads_b[0][0],
        reference_grads_b[1][0],
        reference_grads_b[2][0],
        reference_grads_b[3][0],
        reference_grads_b[4][0],
    )

    torch.testing.assert_close(native, reference, atol=1.0e-5, rtol=1.0e-5)
    for actual, expected in zip(native_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=3.0e-5, rtol=3.0e-4)
