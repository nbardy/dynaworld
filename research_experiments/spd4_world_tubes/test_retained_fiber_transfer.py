from __future__ import annotations

import math

import pytest
import torch

from research_experiments.spd4_world_tubes.hybrid_transfer import (
    render_variance_certified_hybrid_metal,
)
from research_experiments.spd4_world_tubes.retained_fiber_metal import (
    RetainedFiberMetal,
    render_retained_fiber_metal,
)
from research_experiments.spd4_world_tubes.retained_fiber_transfer import (
    render_retained_fiber_reference,
)


def _scene(dtype: torch.dtype = torch.float64):
    ma = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=dtype)
    q_uvt = torch.tensor(
        [
            [0.8, 0.0, 0.0, 0.8, 0.0, 1.0],
            [0.8, 0.0, 0.0, 0.8, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    depth0 = torch.tensor([-0.18, 0.16], dtype=dtype)
    depth_beta = torch.tensor(
        [[0.03, -0.02, 0.04], [-0.01, 0.025, -0.03]], dtype=dtype
    )
    depth_variance = torch.tensor([0.20, 0.16], dtype=dtype)
    optical = torch.tensor([0.72, 0.61], dtype=dtype)
    color = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.25, 0.95]], dtype=dtype)
    times = torch.tensor([-0.25, 0.35], dtype=dtype)
    return ma, q_uvt, depth0, depth_beta, depth_variance, optical, color, times


def test_single_atom_matches_integrated_beer_lambert_transfer() -> None:
    values = list(_scene())
    values = [value[:1] if index < 7 else value[:1] for index, value in enumerate(values)]
    ma, q_uvt, depth0, depth_beta, variance, optical, color, times = values
    image = render_retained_fiber_reference(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        variance,
        optical,
        color,
        times,
        height=1,
        width=1,
        depth_samples=64,
    )
    coordinate = torch.tensor([0.5, 0.5, float(times[0])], dtype=ma.dtype)
    delta = coordinate - ma[0]
    q = torch.tensor(
        [
            [q_uvt[0, 0], q_uvt[0, 1], q_uvt[0, 2]],
            [q_uvt[0, 1], q_uvt[0, 3], q_uvt[0, 4]],
            [q_uvt[0, 2], q_uvt[0, 4], q_uvt[0, 5]],
        ],
        dtype=ma.dtype,
    )
    tau = optical[0] * torch.exp(-0.5 * (delta @ q @ delta))
    expected = (1.0 - torch.exp(-tau)) * color[0]
    assert torch.allclose(image[0, 0, 0], expected, rtol=0.0, atol=2.0e-7)


def test_overlap_converges_and_is_not_mean_depth_splat_sort() -> None:
    scene = _scene()
    coarse = render_retained_fiber_reference(
        *scene,
        height=1,
        width=1,
        depth_samples=64,
    )
    dense = render_retained_fiber_reference(
        *scene,
        height=1,
        width=1,
        depth_samples=1024,
    )
    assert torch.allclose(coarse, dense, rtol=0.0, atol=7.0e-5)

    optical = scene[5]
    colors = scene[6]
    ordered = (1.0 - torch.exp(-optical[0])) * colors[0]
    ordered = ordered + torch.exp(-optical[0]) * (1.0 - torch.exp(-optical[1])) * colors[1]
    assert float((dense[0, 0, 0] - ordered).abs().max()) > 1.0e-3


def test_reference_gradients_are_finite_and_nonzero() -> None:
    scene = [value.clone().requires_grad_(index < 7) for index, value in enumerate(_scene())]
    image = render_retained_fiber_reference(
        *scene,
        height=1,
        width=1,
        depth_samples=48,
    )
    weights = torch.tensor([0.3, -0.2, 0.5], dtype=image.dtype)
    loss = torch.sum(image * weights)
    gradients = torch.autograd.grad(loss, scene[:7])
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    assert all(float(value.abs().sum()) > 0.0 for value in gradients)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple Metal/MPS",
)
def test_tiny_metal_forward_and_vjp_match_reference() -> None:
    cpu_scene = [value.float() for value in _scene(torch.float32)]
    cpu_scene[7] = cpu_scene[7][:1]
    differentiable = [
        value.clone().requires_grad_(index < 7)
        for index, value in enumerate(cpu_scene)
    ]
    reference = render_retained_fiber_reference(
        *differentiable,
        height=2,
        width=2,
        depth_samples=32,
    )
    grad_output = torch.tensor(
        [
            [
                [[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]],
                [[-0.3, 0.2, 0.1], [0.25, -0.15, 0.35]],
            ]
        ],
        dtype=torch.float32,
    )
    reference_gradients = torch.autograd.grad(
        torch.sum(reference * grad_output),
        differentiable[:7],
    )

    mps_scene = [value.detach().to("mps").contiguous() for value in cpu_scene]
    metal = RetainedFiberMetal()
    actual = metal.forward(
        *mps_scene,
        height=2,
        width=2,
        depth_samples=32,
    )
    actual_gradients = metal.vjp(
        grad_output.to("mps").contiguous(),
        *mps_scene,
        height=2,
        width=2,
        depth_samples=32,
    )
    torch.mps.synchronize()
    assert torch.allclose(actual.cpu(), reference.detach(), rtol=0.0, atol=3.0e-5)
    for name, expected in zip(
        (
            "ma",
            "q_uvt",
            "depth0",
            "depth_beta",
            "depth_variance",
            "optical_thickness",
            "color",
        ),
        reference_gradients,
        strict=True,
    ):
        got = actual_gradients[name].cpu()
        denominator = torch.maximum(torch.ones_like(expected), expected.abs())
        normalized = ((got - expected).abs() / denominator).max()
        assert float(normalized) <= 2.0e-4, (name, float(normalized))
    torch.mps.empty_cache()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple Metal/MPS",
)
def test_differentiable_metal_boundary_uses_native_vjp() -> None:
    cpu_scene = [value.float() for value in _scene(torch.float32)]
    cpu_scene[7] = cpu_scene[7][:1]
    reference_inputs = [
        value.clone().requires_grad_(index < 7)
        for index, value in enumerate(cpu_scene)
    ]
    reference = render_retained_fiber_reference(
        *reference_inputs,
        height=1,
        width=1,
        depth_samples=24,
    )
    weights = torch.tensor([[[[0.31, -0.17, 0.43]]]], dtype=torch.float32)
    expected_gradients = torch.autograd.grad(
        torch.sum(reference * weights),
        reference_inputs[:7],
    )

    mps_inputs = [
        value.detach().to("mps").contiguous().requires_grad_(index < 7)
        for index, value in enumerate(cpu_scene)
    ]
    actual = render_retained_fiber_metal(
        *mps_inputs,
        height=1,
        width=1,
        depth_samples=24,
    )
    actual_gradients = torch.autograd.grad(
        torch.sum(actual * weights.to("mps")),
        mps_inputs[:7],
    )
    torch.mps.synchronize()

    assert torch.allclose(actual.cpu(), reference.detach(), rtol=0.0, atol=3.0e-5)
    for got, expected in zip(actual_gradients, expected_gradients, strict=True):
        denominator = torch.maximum(torch.ones_like(expected), expected.abs())
        normalized = ((got.cpu() - expected).abs() / denominator).max()
        assert float(normalized) <= 2.0e-4
    assert all(float(gradient.abs().sum().cpu()) > 0.0 for gradient in actual_gradients)
    torch.mps.empty_cache()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple Metal/MPS",
)
def test_metal_tile_certificate_routes_only_ambiguous_depth_bands() -> None:
    scene = [value.float().to("mps").contiguous() for value in _scene(torch.float32)]
    ma, q_uvt, _depth0, depth_beta, _variance, optical, _color, _times = scene
    depth_fit_error = torch.zeros((2,), dtype=torch.float32, device="mps")
    metal = RetainedFiberMetal()

    separated = metal.certify_tiles(
        ma,
        q_uvt,
        torch.tensor([-2.0, 2.0], dtype=torch.float32, device="mps"),
        depth_beta * 0.0,
        torch.full((2,), 0.01, dtype=torch.float32, device="mps"),
        optical,
        frames=1,
        height=1,
        width=1,
        tile_x=1,
        tile_y=1,
        tile_t=1,
        alpha_threshold=1.0 / 255.0,
        sigma_multiplier=3.0,
        depth_fit_error=depth_fit_error,
    )
    ambiguous = metal.certify_tiles(
        ma,
        q_uvt,
        torch.tensor([-0.1, 0.1], dtype=torch.float32, device="mps"),
        depth_beta * 0.0,
        torch.full((2,), 0.25, dtype=torch.float32, device="mps"),
        optical,
        frames=1,
        height=1,
        width=1,
        tile_x=1,
        tile_y=1,
        tile_t=1,
        alpha_threshold=1.0 / 255.0,
        sigma_multiplier=3.0,
        depth_fit_error=depth_fit_error,
    )
    torch.mps.synchronize()

    assert int(separated.fallback_tiles.cpu().item()) == 0
    assert int(separated.reason_bits.cpu().item()) == 0
    assert int(separated.active_counts.cpu().item()) == 2
    assert float(separated.minimum_pair_separation.cpu().item()) > 0.0
    assert int(ambiguous.fallback_tiles.cpu().item()) == 1
    assert int(ambiguous.reason_bits.cpu().item()) & 4
    assert int(ambiguous.active_counts.cpu().item()) == 2
    assert float(ambiguous.minimum_pair_separation.cpu().item()) < 0.0
    torch.mps.empty_cache()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple Metal/MPS",
)
def test_masked_retained_fiber_skips_certified_pixels_and_gradients() -> None:
    cpu_scene = [value.float() for value in _scene(torch.float32)]
    cpu_scene[7] = cpu_scene[7][:1]
    mps_inputs = [
        value.to("mps").contiguous().requires_grad_(index < 7)
        for index, value in enumerate(cpu_scene)
    ]
    mask = torch.zeros((1, 1, 1), dtype=torch.int32, device="mps")
    image = render_retained_fiber_metal(
        *mps_inputs,
        height=1,
        width=1,
        depth_samples=16,
        fallback_mask=mask,
        alpha_threshold=1.0 / 255.0,
    )
    gradients = torch.autograd.grad(image.sum(), mps_inputs[:7])
    torch.mps.synchronize()

    assert torch.count_nonzero(image).item() == 0
    assert all(torch.count_nonzero(gradient).item() == 0 for gradient in gradients)
    torch.mps.empty_cache()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple Metal/MPS",
)
def test_hybrid_routes_certified_and_ambiguous_tiles_to_different_vjps() -> None:
    cpu_scene = [value.float() for value in _scene(torch.float32)]
    cpu_scene[7] = cpu_scene[7][:1]
    mps_scene = [
        value.to("mps").contiguous().requires_grad_(index < 7)
        for index, value in enumerate(cpu_scene)
    ]
    ma, q_uvt, _depth0, depth_beta, _variance, optical, color, times = mps_scene

    separated_depth = torch.tensor(
        [-4.0, 4.0],
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )
    separated_variance = torch.full(
        (2,),
        0.01,
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )
    fast_leaf = torch.tensor(
        [[[[0.12, 0.34, 0.56]]]],
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )
    certified = render_variance_certified_hybrid_metal(
        fast_rgb=fast_leaf,
        ma=ma,
        q_uvt=q_uvt,
        depth0=separated_depth,
        depth_beta=depth_beta * 0.0,
        depth_variance=separated_variance,
        optical_thickness=optical,
        color=color,
        times=times,
        height=1,
        width=1,
        tile_x=1,
        tile_y=1,
        tile_t=1,
        alpha_threshold=1.0 / 255.0,
        depth_samples=16,
        sigma_extent=3.0,
        certificate_sigma=3.0,
    )
    grad_fast = torch.autograd.grad(certified.rgb.sum(), fast_leaf)[0]
    assert int(certified.fallback_mask.cpu().item()) == 0
    torch.testing.assert_close(certified.rgb, fast_leaf)
    assert torch.equal(grad_fast, torch.ones_like(grad_fast))

    ambiguous_depth = torch.tensor(
        [-0.1, 0.1],
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )
    ambiguous_variance = torch.full(
        (2,),
        0.25,
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )
    rejected_fast = torch.full(
        (1, 1, 1, 3),
        0.95,
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )
    fallback = render_variance_certified_hybrid_metal(
        fast_rgb=rejected_fast,
        ma=ma,
        q_uvt=q_uvt,
        depth0=ambiguous_depth,
        depth_beta=depth_beta * 0.0,
        depth_variance=ambiguous_variance,
        optical_thickness=optical,
        color=color,
        times=times,
        height=1,
        width=1,
        tile_x=1,
        tile_y=1,
        tile_t=1,
        alpha_threshold=1.0 / 255.0,
        depth_samples=24,
        sigma_extent=3.0,
        certificate_sigma=3.0,
    )
    grad_rejected, grad_depth, grad_variance = torch.autograd.grad(
        fallback.rgb.sum(),
        (rejected_fast, ambiguous_depth, ambiguous_variance),
    )
    torch.mps.synchronize()

    assert int(fallback.fallback_mask.cpu().item()) == 1
    assert torch.count_nonzero(grad_rejected).item() == 0
    assert float(grad_depth.abs().sum().cpu()) > 0.0
    assert float(grad_variance.abs().sum().cpu()) > 0.0
    assert not torch.allclose(fallback.rgb, rejected_fast)
    torch.mps.empty_cache()
