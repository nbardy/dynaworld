from __future__ import annotations

import itertools
from pathlib import Path
import sys

import torch

STAR_UVT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "star_uvt_v0"
)
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    uvt_tubes_to_projective_trace_cell_atlas,
)

from research_experiments.spd4_world_tubes import (
    AffineRayGauge,
    WorldAtomBatch,
    analytic_fiber_optical_depth,
    block_cholesky_from_covariance,
    certify_confidence_band_order,
    covariance_from_block_cholesky,
    dense_retained_fiber_render,
    pushforward_world_atoms,
    unpack_symmetric_3x3,
)


DTYPE = torch.float64


def _identity_gauge() -> AffineRayGauge:
    return AffineRayGauge(
        gauge_from_world=torch.eye(4, dtype=DTYPE),
        gauge_offset=torch.zeros(4, dtype=DTYPE),
        fiber_measure_scale=torch.ones((), dtype=DTYPE),
    )


def _joint_covariance_from_depth_condition(
    marginal_uvt: torch.Tensor,
    depth_beta: torch.Tensor,
    depth_variance: torch.Tensor,
) -> torch.Tensor:
    """Build covariance in ordered gauge coordinates (u,v,depth,t)."""

    covariance_a_depth = marginal_uvt @ depth_beta
    depth_joint_variance = (
        depth_variance + depth_beta @ marginal_uvt @ depth_beta
    )
    rows = (
        torch.stack(
            (
                marginal_uvt[0, 0],
                marginal_uvt[0, 1],
                covariance_a_depth[0],
                marginal_uvt[0, 2],
            )
        ),
        torch.stack(
            (
                marginal_uvt[1, 0],
                marginal_uvt[1, 1],
                covariance_a_depth[1],
                marginal_uvt[1, 2],
            )
        ),
        torch.stack(
            (
                covariance_a_depth[0],
                covariance_a_depth[1],
                depth_joint_variance,
                covariance_a_depth[2],
            )
        ),
        torch.stack(
            (
                marginal_uvt[2, 0],
                marginal_uvt[2, 1],
                covariance_a_depth[2],
                marginal_uvt[2, 2],
            )
        ),
    )
    return torch.stack(rows)


def test_block_cholesky_chart_is_spd_and_lossless() -> None:
    spatial_cholesky = torch.tensor(
        [
            [[0.8, 0.0, 0.0], [0.2, 1.1, 0.0], [-0.1, 0.3, 0.6]],
            [[1.3, 0.0, 0.0], [-0.4, 0.7, 0.0], [0.2, -0.1, 0.9]],
        ],
        dtype=DTYPE,
    )
    tilt = torch.tensor([[0.7, -0.2, 0.3], [-0.1, 0.5, -0.8]], dtype=DTYPE)
    log_temporal_scale = torch.tensor([-0.4, 0.25], dtype=DTYPE)

    covariance = covariance_from_block_cholesky(
        spatial_cholesky,
        tilt,
        log_temporal_scale,
    )
    assert torch.all(torch.linalg.eigvalsh(covariance) > 0.0)

    recovered = block_cholesky_from_covariance(covariance)
    torch.testing.assert_close(recovered.spatial_cholesky, spatial_cholesky)
    torch.testing.assert_close(recovered.space_time_tilt, tilt)
    torch.testing.assert_close(recovered.log_temporal_scale, log_temporal_scale)
    torch.testing.assert_close(recovered.covariance(), covariance)


def test_block_cholesky_round_trips_random_well_conditioned_spd4() -> None:
    generator = torch.Generator().manual_seed(20260727)
    factor = torch.randn((16, 4, 4), dtype=DTYPE, generator=generator)
    covariance = factor @ factor.transpose(-1, -2) + (
        0.35 * torch.eye(4, dtype=DTYPE)[None]
    )
    recovered = block_cholesky_from_covariance(covariance)
    torch.testing.assert_close(
        recovered.covariance(),
        covariance,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_space_time_cross_covariance_is_native_affine_motion_when_conditioned() -> None:
    spatial_cholesky = torch.tensor(
        [[[0.7, 0.0, 0.0], [0.1, 0.8, 0.0], [-0.2, 0.15, 0.6]]],
        dtype=DTYPE,
    )
    tilt = torch.tensor([[0.8, -0.35, 0.25]], dtype=DTYPE)
    temporal_log_scale = torch.tensor([-0.1], dtype=DTYPE)
    mean = torch.tensor([[1.2, -0.4, 2.1, 0.3]], dtype=DTYPE)
    covariance = covariance_from_block_cholesky(
        spatial_cholesky,
        tilt,
        temporal_log_scale,
    )
    temporal_variance = covariance[:, 3, 3]
    regression = covariance[:, :3, 3] / temporal_variance[:, None]
    conditional_covariance = covariance[:, :3, :3] - (
        covariance[:, :3, 3, None]
        * covariance[:, None, 3, :3]
        / temporal_variance[:, None, None]
    )
    torch.testing.assert_close(regression, tilt)
    torch.testing.assert_close(
        conditional_covariance,
        spatial_cholesky @ spatial_cholesky.transpose(-1, -2),
    )

    times = torch.tensor([-1.1, 0.3, 0.9, 2.0], dtype=DTYPE)
    conditioned_from_joint = mean[:, :3] + (
        regression[:, None, :] * (times[None, :, None] - mean[:, 3, None, None])
    )
    expected_worldline = mean[:, None, :3] + (
        tilt[:, None, :] * (times[None, :, None] - mean[:, 3, None, None])
    )
    torch.testing.assert_close(conditioned_from_joint, expected_worldline)


def test_affine_ray_bundle_and_inverse_gauge_are_exactly_equivalent() -> None:
    origin = torch.tensor([1.2, -0.7, 2.5, 0.3], dtype=DTYPE)
    uvt_basis = torch.tensor(
        [
            [2.0, 0.1, 0.2],
            [0.0, 1.5, -0.1],
            [0.2, -0.3, 0.4],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    )
    depth_direction = torch.tensor([0.1, -0.2, 3.0, 0.0], dtype=DTYPE)
    gauge = AffineRayGauge.from_ray_bundle(
        world_origin=origin,
        world_uvt_basis=uvt_basis,
        world_depth_direction=depth_direction,
    )
    uvt = torch.tensor([[0.2, -0.4, 0.7], [-1.0, 0.5, -0.2]], dtype=DTYPE)
    depth = torch.tensor([1.3, -0.8], dtype=DTYPE)

    expected_world = (
        origin[None, :]
        + uvt @ uvt_basis.T
        + depth[:, None] * depth_direction[None, :]
    )
    world = gauge.world_from_uvt_depth(uvt, depth)
    torch.testing.assert_close(world, expected_world)
    expected_gauge = torch.stack(
        (uvt[:, 0], uvt[:, 1], depth, uvt[:, 2]),
        dim=-1,
    )
    torch.testing.assert_close(gauge.to_gauge(world), expected_gauge)
    torch.testing.assert_close(gauge.to_world(expected_gauge), expected_world)
    torch.testing.assert_close(
        gauge.fiber_measure_scale,
        torch.linalg.vector_norm(depth_direction[:3]),
    )
    invalid_depth_direction = depth_direction.clone()
    invalid_depth_direction[3] = 0.1
    with torch.no_grad():
        try:
            AffineRayGauge.from_ray_bundle(
                world_origin=origin,
                world_uvt_basis=uvt_basis,
                world_depth_direction=invalid_depth_direction,
            )
        except ValueError as error:
            assert "fixed physical world time" in str(error)
        else:
            raise AssertionError("a time-slanted camera depth fiber was accepted")


def test_affine_pushforward_factorizes_the_joint_quadratic_exactly() -> None:
    atoms = WorldAtomBatch.from_block_cholesky(
        mean_xyzt=torch.tensor([[0.4, -0.2, 1.1, 0.3]], dtype=DTYPE),
        spatial_cholesky=torch.tensor(
            [[[0.7, 0.0, 0.0], [0.15, 0.9, 0.0], [-0.2, 0.1, 0.5]]],
            dtype=DTYPE,
        ),
        space_time_tilt=torch.tensor([[0.8, -0.3, 0.5]], dtype=DTYPE),
        log_temporal_scale=torch.tensor([-0.2], dtype=DTYPE),
        amplitude=torch.tensor([0.4], dtype=DTYPE),
        color=torch.tensor([[0.2, 0.5, 0.8]], dtype=DTYPE),
    )
    gauge = AffineRayGauge.from_ray_bundle(
        world_origin=torch.tensor([0.1, -0.2, 0.4, -0.3], dtype=DTYPE),
        world_uvt_basis=torch.tensor(
            [
                [1.4, 0.1, 0.2],
                [-0.2, 1.1, 0.1],
                [0.1, 0.3, -0.2],
                [0.0, 0.0, 1.0],
            ],
            dtype=DTYPE,
        ),
        world_depth_direction=torch.tensor([0.2, -0.1, 1.7, 0.0], dtype=DTYPE),
    )
    trace = pushforward_world_atoms(atoms, gauge)
    gauge_points = torch.tensor(
        [[0.1, -0.6, 1.0, 0.2], [-0.8, 0.4, 2.1, -0.3]],
        dtype=DTYPE,
    )
    world_points = gauge.to_world(gauge_points)
    world_delta = world_points - atoms.mean_xyzt
    direct = torch.einsum(
        "pi,ij,pj->p",
        world_delta,
        torch.linalg.inv(atoms.covariance_xyzt[0]),
        world_delta,
    )

    a = gauge_points[:, (0, 1, 3)]
    delta_a = a - trace.ma[0]
    marginal = torch.einsum(
        "pi,ij,pj->p",
        delta_a,
        trace.q_uvt_dense[0],
        delta_a,
    )
    conditional_mean = trace.depth0[0] + delta_a @ trace.depth_beta[0]
    conditional = (
        (gauge_points[:, 2] - conditional_mean).square()
        / trace.depth_variance[0]
    )
    torch.testing.assert_close(marginal + conditional, direct)
    torch.testing.assert_close(unpack_symmetric_3x3(trace.q_uvt), trace.q_uvt_dense)

    adapter = trace.to_uvt_tubes(opacity_mapping="peak_preserving")
    assert [tuple(value.shape) for value in adapter.as_tuple()] == [
        (1, 3),
        (1, 6),
        (1,),
        (1, 3),
        (1,),
        (1, 3),
    ]
    torch.testing.assert_close(adapter.opacity, trace.peak_density_amplitude)
    assert adapter.opacity_mapping == "peak_preserving"
    thin_adapter = trace.to_uvt_tubes(opacity_mapping="thin_fiber_optical_depth")
    torch.testing.assert_close(
        thin_adapter.opacity,
        trace.fiber_integrated_amplitude,
    )
    assert thin_adapter.opacity_mapping == "thin_fiber_optical_depth"
    assert all(
        value.dtype == torch.float32 and value.is_contiguous()
        for value in adapter.as_legacy_float32().as_tuple()
    )
    assert adapter.as_legacy_float32().opacity_mapping == "peak_preserving"


def test_peak_and_fiber_amplitudes_and_depth_reparameterization_are_explicit() -> None:
    covariance = torch.diag(torch.tensor([0.5, 0.8, 4.0, 0.3], dtype=DTYPE))[None]
    amplitude = torch.tensor([0.2], dtype=DTYPE)
    color = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    peak_atoms = WorldAtomBatch(
        mean_xyzt=torch.zeros((1, 4), dtype=DTYPE),
        covariance_xyzt=covariance,
        amplitude=amplitude,
        color=color,
        amplitude_convention="peak_density",
    )
    trace = pushforward_world_atoms(peak_atoms, _identity_gauge())
    expected_scale = torch.sqrt(torch.tensor(8.0 * torch.pi, dtype=DTYPE))
    torch.testing.assert_close(trace.peak_to_fiber_scale[0], expected_scale)
    torch.testing.assert_close(
        trace.fiber_integrated_amplitude,
        amplitude * expected_scale,
    )

    integrated_atoms = WorldAtomBatch(
        mean_xyzt=peak_atoms.mean_xyzt,
        covariance_xyzt=covariance,
        amplitude=trace.fiber_integrated_amplitude,
        color=color,
        amplitude_convention="fiber_integrated",
    )
    integrated_trace = pushforward_world_atoms(integrated_atoms, _identity_gauge())
    torch.testing.assert_close(integrated_trace.peak_density_amplitude, amplitude)

    depth_scale = torch.tensor(3.5, dtype=DTYPE)
    rescaled_gauge = AffineRayGauge(
        gauge_from_world=torch.diag(
            torch.stack(
                (
                    torch.ones((), dtype=DTYPE),
                    torch.ones((), dtype=DTYPE),
                    depth_scale,
                    torch.ones((), dtype=DTYPE),
                )
            )
        ),
        gauge_offset=torch.zeros(4, dtype=DTYPE),
        fiber_measure_scale=1.0 / depth_scale,
    )
    rescaled = pushforward_world_atoms(peak_atoms, rescaled_gauge)
    torch.testing.assert_close(
        rescaled.depth_variance,
        depth_scale.square() * trace.depth_variance,
    )
    torch.testing.assert_close(rescaled.peak_to_fiber_scale, trace.peak_to_fiber_scale)
    torch.testing.assert_close(
        rescaled.fiber_integrated_amplitude,
        trace.fiber_integrated_amplitude,
    )


def test_adapter_enters_existing_projective_atlas_without_geometry_loss() -> None:
    marginal_uvt = torch.tensor(
        [
            [4.0, 0.3, 0.5],
            [0.3, 2.0, -0.2],
            [0.5, -0.2, 0.8],
        ],
        dtype=DTYPE,
    )
    depth_beta = torch.tensor([0.1, -0.05, 0.2], dtype=DTYPE)
    covariance = _joint_covariance_from_depth_condition(
        marginal_uvt,
        depth_beta,
        torch.tensor(0.25, dtype=DTYPE),
    )[None]
    atoms = WorldAtomBatch(
        mean_xyzt=torch.tensor([[12.0, 10.0, 5.0, 0.0]], dtype=DTYPE),
        covariance_xyzt=covariance,
        amplitude=torch.tensor([0.2], dtype=DTYPE),
        color=torch.tensor([[0.2, 0.5, 0.8]], dtype=DTYPE),
    )
    trace = pushforward_world_atoms(atoms, _identity_gauge())
    adapter = trace.to_uvt_tubes(
        opacity_mapping="peak_preserving"
    ).as_legacy_float32()
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        *adapter.as_tuple(),
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32),
        sigma_px=1.0,
        image_width=32,
        image_height=24,
        tile_size=8,
        uv_padding=4.0,
        require_isotropic_spatial=False,
        allow_depth_affine_uv=True,
        temporal_mode="trace",
    )
    assert atlas.coeffs.shape == (1, 9)
    assert atlas.spatial_precision_uv is not None
    torch.testing.assert_close(
        atlas.spatial_precision_uv[0],
        trace.q_uvt_dense[0][(0, 0, 1), (0, 1, 1)].float(),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert atlas.depth_affine_uv is not None
    torch.testing.assert_close(
        atlas.depth_affine_uv[0, (0, 3)],
        trace.depth_beta[0, :2].float(),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    torch.testing.assert_close(atlas.opacity, trace.peak_density_amplitude.float())


def test_dense_retained_fiber_converges_to_analytic_infinite_fiber_integral() -> None:
    atoms = WorldAtomBatch(
        mean_xyzt=torch.tensor([[0.1, -0.2, 0.4, 0.3]], dtype=DTYPE),
        covariance_xyzt=torch.diag(
            torch.tensor([0.4, 0.7, 0.25, 0.5], dtype=DTYPE)
        )[None],
        amplitude=torch.tensor([0.35], dtype=DTYPE),
        color=torch.tensor([[0.25, 0.5, 0.75]], dtype=DTYPE),
    )
    trace = pushforward_world_atoms(atoms, _identity_gauge())
    query = torch.tensor([[0.1, -0.2, 0.3]], dtype=DTYPE)
    sigma = torch.sqrt(trace.depth_variance[0])
    edges = torch.linspace(
        trace.depth0[0] - 9.0 * sigma,
        trace.depth0[0] + 9.0 * sigma,
        4002,
        dtype=DTYPE,
    )
    rendered = dense_retained_fiber_render(trace, query, edges)
    analytic = analytic_fiber_optical_depth(trace, query).sum(dim=-1)
    torch.testing.assert_close(rendered.optical_depth, analytic, rtol=2.0e-6, atol=1.0e-9)
    torch.testing.assert_close(
        rendered.transmittance,
        torch.exp(-analytic),
        rtol=2.0e-6,
        atol=1.0e-9,
    )
    expected_rgb = (1.0 - torch.exp(-analytic))[:, None] * atoms.color
    torch.testing.assert_close(rendered.rgb, expected_rgb, rtol=2.0e-6, atol=1.0e-9)


def test_confidence_order_certificate_uses_exact_box_extrema_and_rejects_crossing() -> None:
    marginal = torch.diag(torch.tensor([0.3, 0.4, 0.5], dtype=DTYPE))
    beta_front = torch.tensor([0.4, 0.0, 0.0], dtype=DTYPE)
    beta_back = torch.tensor([-0.2, 0.0, 0.0], dtype=DTYPE)
    conditional_variance = torch.tensor(0.01, dtype=DTYPE)
    covariances = torch.stack(
        (
            _joint_covariance_from_depth_condition(
                marginal, beta_front, conditional_variance
            ),
            _joint_covariance_from_depth_condition(
                marginal, beta_back, conditional_variance
            ),
        )
    )
    atoms = WorldAtomBatch(
        mean_xyzt=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
            dtype=DTYPE,
        ),
        covariance_xyzt=covariances,
        amplitude=torch.ones(2, dtype=DTYPE),
        color=torch.eye(3, dtype=DTYPE)[:2],
    )
    trace = pushforward_world_atoms(atoms, _identity_gauge())
    lower = torch.tensor([-1.0, -0.5, -0.25], dtype=DTYPE)
    upper = torch.tensor([1.0, 0.5, 0.25], dtype=DTYPE)
    certificate = certify_confidence_band_order(
        trace,
        lower,
        upper,
        sigma_multiplier=2.0,
        proposed_order=torch.tensor([0, 1], dtype=torch.long),
    )
    assert certificate.certified_before[0, 1]
    assert certificate.proposed_order_certified

    corner_gaps = []
    radius = 2.0 * torch.sqrt(trace.depth_variance)
    for bits in itertools.product((0, 1), repeat=3):
        point = torch.where(
            torch.tensor(bits, dtype=torch.bool),
            upper,
            lower,
        )
        means = trace.depth0 + (
            trace.depth_beta * (point[None, :] - trace.ma)
        ).sum(dim=-1)
        corner_gaps.append(means[1] - radius[1] - means[0] - radius[0])
    torch.testing.assert_close(
        certificate.minimum_band_gap[0, 1],
        torch.stack(corner_gaps).min(),
    )

    crossing_atoms = WorldAtomBatch(
        mean_xyzt=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0]],
            dtype=DTYPE,
        ),
        covariance_xyzt=covariances,
        amplitude=atoms.amplitude,
        color=atoms.color,
    )
    crossing = certify_confidence_band_order(
        pushforward_world_atoms(crossing_atoms, _identity_gauge()),
        lower,
        upper,
        sigma_multiplier=2.0,
        proposed_order=torch.tensor([0, 1], dtype=torch.long),
    )
    assert crossing.ambiguous[0, 1]
    assert not crossing.proposed_order_certified


def test_thick_colored_overlap_rejects_hard_order_and_needs_retained_fiber() -> None:
    covariance = torch.diag(
        torch.tensor([0.08, 0.08, 0.8**2, 0.12], dtype=DTYPE)
    )
    atoms = WorldAtomBatch(
        mean_xyzt=torch.tensor(
            [[0.0, 0.0, -0.25, 0.0], [0.0, 0.0, 0.25, 0.0]],
            dtype=DTYPE,
        ),
        covariance_xyzt=covariance[None].expand(2, -1, -1).clone(),
        amplitude=torch.tensor([0.8, 0.8], dtype=DTYPE),
        color=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE),
    )
    trace = pushforward_world_atoms(atoms, _identity_gauge())
    box_lower = torch.tensor([-0.1, -0.1, -0.1], dtype=DTYPE)
    box_upper = torch.tensor([0.1, 0.1, 0.1], dtype=DTYPE)
    certificate = certify_confidence_band_order(
        trace,
        box_lower,
        box_upper,
        sigma_multiplier=2.0,
        proposed_order=torch.tensor([0, 1], dtype=torch.long),
    )
    assert certificate.ambiguous[0, 1]
    assert not certificate.proposed_order_certified

    query = torch.zeros((1, 3), dtype=DTYPE)
    edges = torch.linspace(-6.0, 6.0, 6002, dtype=DTYPE)
    dense = dense_retained_fiber_render(trace, query, edges)
    atom_optical_depth = analytic_fiber_optical_depth(trace, query)[0]
    atom_alpha = -torch.expm1(-atom_optical_depth)
    red_then_blue = (
        atom_alpha[0] * atoms.color[0]
        + (1.0 - atom_alpha[0]) * atom_alpha[1] * atoms.color[1]
    )
    blue_then_red = (
        atom_alpha[1] * atoms.color[1]
        + (1.0 - atom_alpha[1]) * atom_alpha[0] * atoms.color[0]
    )

    # All three integrate the same total extinction, but the retained profile
    # mixes colors continuously through the overlap instead of choosing either
    # atomic order.
    torch.testing.assert_close(
        dense.transmittance,
        torch.exp(-atom_optical_depth.sum())[None],
        rtol=3.0e-6,
        atol=1.0e-9,
    )
    assert torch.linalg.vector_norm(dense.rgb[0] - red_then_blue) > 0.05
    assert torch.linalg.vector_norm(dense.rgb[0] - blue_then_red) > 0.05


def test_compiler_is_float64_autograd_differentiable() -> None:
    mean = torch.tensor([[0.2, -0.1, 1.0, 0.3]], dtype=DTYPE, requires_grad=True)
    block_parameters = torch.tensor(
        [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.4, -0.3, 0.2, -0.15],
        dtype=DTYPE,
        requires_grad=True,
    )
    gauge_matrix = torch.tensor(
        [
            [1.1, 0.1, 0.0, 0.2],
            [-0.1, 0.9, 0.2, -0.1],
            [0.0, 0.1, 1.2, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )

    def compiled_scalar(
        mean_value: torch.Tensor,
        parameters: torch.Tensor,
        matrix: torch.Tensor,
    ) -> torch.Tensor:
        zero = parameters.new_zeros(())
        spatial = torch.stack(
            (
                torch.stack((torch.exp(parameters[0]), zero, zero)),
                torch.stack((parameters[1], torch.exp(parameters[2]), zero)),
                torch.stack(
                    (parameters[3], parameters[4], torch.exp(parameters[5]))
                ),
            )
        )[None]
        atoms = WorldAtomBatch.from_block_cholesky(
            mean_xyzt=mean_value,
            spatial_cholesky=spatial,
            space_time_tilt=parameters[6:9][None],
            log_temporal_scale=parameters[9:10],
            amplitude=parameters.new_tensor([0.4]),
            color=parameters.new_tensor([[0.2, 0.5, 0.7]]),
        )
        trace = pushforward_world_atoms(
            atoms,
            AffineRayGauge(
                gauge_from_world=matrix,
                gauge_offset=parameters.new_tensor([0.1, -0.2, 0.3, 0.0]),
                fiber_measure_scale=parameters.new_tensor(1.25),
            ),
        )
        return (
            0.11 * trace.ma.sum()
            + 0.07 * trace.q_uvt.sum()
            + 0.13 * trace.depth_beta.sum()
            + 0.17 * trace.depth_variance.sum()
            + 0.19 * trace.fiber_integrated_amplitude.sum()
        )

    assert torch.autograd.gradcheck(
        compiled_scalar,
        (mean, block_parameters, gauge_matrix),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )
