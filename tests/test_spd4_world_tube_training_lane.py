from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
STAR_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
TRAIN_SRC = ROOT / "src" / "train"
for path in (STAR_ROOT, TRAIN_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research_project.benchmarks.multicam_heldout_compare import (  # noqa: E402
    WorldTubeModel,
    compiled_projected_opacity,
    max_alpha_for_mode,
    projected_tile_load_proxy,
    project_world_tube_sequence_camera_mode,
    project_world_tube_sequence,
    render_projected_sequence,
    train_world_tubes,
)
from research_project.trainer_harness.spd4_world_atom import (  # noqa: E402
    SPD4WorldAtomModel,
    SPD4AffineGaugeBatch,
    compile_spd4_pinhole_motion_affine_gauges,
    project_spd4_world_atoms_affine_gauges,
    project_spd4_world_atoms_pinhole,
    project_spd4_world_atoms_pinhole_motion,
)
from research_project.trainer_harness.world_tube import (  # noqa: E402
    PinholeCamera,
    PinholeCameraMotion,
)
from research_experiments.spd4_world_tubes import (  # noqa: E402
    AffineRayGauge,
    WorldAtomBatch,
    pushforward_world_atoms,
)
from research_experiments.spd4_world_tubes.run_capacity_gate import (  # noqa: E402
    _camera_suite,
    _conditional_covariance_identifiability,
    _target_spd4,
)
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    UVTRenderConfig,
    brute_force_render_uvt_tubes,
    primitive_alpha,
)


def _model(
    *,
    count: int = 2,
    init_precision_z: float | None = None,
    alpha_mode: str = "peak_splat",
    amplitude_convention: str = "fiber_integrated",
) -> SPD4WorldAtomModel:
    return SPD4WorldAtomModel(
        init_x0=torch.tensor(
            [[-0.1, 0.0, 2.0], [0.1, 0.05, 2.2]][:count],
            dtype=torch.float32,
        ),
        init_color=torch.tensor(
            [[0.8, 0.2, 0.1], [0.1, 0.4, 0.9]][:count],
            dtype=torch.float32,
        ),
        init_t0=torch.zeros(count, dtype=torch.float32),
        frames=3,
        init_precision_xy=40.0,
        init_precision_z=init_precision_z,
        init_lambda_t=0.4,
        init_opacity=0.3,
        min_spatial_scale=1.0e-4,
        min_lambda_t=1.0e-5,
        tilt_reg_weight=1.0e-4,
        depth_tilt_reg_weight=0.0,
        position_reg_weight=1.0e-6,
        alpha_mode=alpha_mode,
        amplitude_convention=amplitude_convention,
    )


def test_spd4_chart_is_strictly_positive_and_derives_motion_from_covariance() -> None:
    model = _model()
    with torch.no_grad():
        model.space_time_tilt.copy_(
            torch.tensor([[0.2, -0.1, 0.05], [-0.03, 0.04, -0.02]])
        )
        model.spatial_cholesky_offdiag.copy_(
            torch.tensor([[0.02, -0.01, 0.03], [-0.02, 0.01, 0.01]])
        )
    batch = model.batch()

    eigenvalues = torch.linalg.eigvalsh(batch.covariance_xyzt)
    derived_tilt = (
        batch.covariance_xyzt[:, :3, 3]
        / batch.covariance_xyzt[:, 3, 3, None]
    )

    assert bool((eigenvalues > 0.0).all())
    torch.testing.assert_close(derived_tilt, batch.space_time_tilt)
    assert sum(parameter.numel() for parameter in model.parameters()) == 18 * 2
    assert model.representation_metadata()["geometry_dof_per_atom"] == 14


def test_peak_density_compiles_with_gauge_invariant_fiber_measure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(
        count=1,
        alpha_mode="beer_lambert",
        amplitude_convention="peak_density",
    )
    batch = model.batch()

    def unexpected_inverse(*_args, **_kwargs):
        raise AssertionError("fiber measure must use the reciprocal-frame formula")

    monkeypatch.setattr(torch.linalg, "inv", unexpected_inverse)
    identity = torch.eye(4, dtype=torch.float32)[None]
    base = project_spd4_world_atoms_affine_gauges(
        batch,
        SPD4AffineGaugeBatch(
            gauge_from_world=identity,
            gauge_offset=torch.zeros((1, 4), dtype=torch.float32),
            chart_time=None,
        ),
        screen_variance_floor=0.0,
    )
    depth_rescale = identity.clone()
    depth_rescale[:, 2, :] *= 3.0
    rescaled = project_spd4_world_atoms_affine_gauges(
        batch,
        SPD4AffineGaugeBatch(
            gauge_from_world=depth_rescale,
            gauge_offset=torch.zeros((1, 4), dtype=torch.float32),
            chart_time=None,
        ),
        screen_variance_floor=0.0,
    )

    torch.testing.assert_close(
        rescaled.depth_variance,
        9.0 * base.depth_variance,
    )
    torch.testing.assert_close(
        rescaled.peak_to_fiber_scale,
        base.peak_to_fiber_scale,
        rtol=2.0e-6,
        atol=2.0e-7,
    )

    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=3,
        alpha_mode="beer_lambert",
        max_alpha=1.0,
        amplitude_convention="peak_density",
    )
    projected = project_world_tube_sequence(
        model,
        torch.tensor(
            [[20.0, 0.0, 4.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        torch.eye(4, dtype=torch.float32),
        config,
    )
    assert projected.peak_to_fiber_scale is not None
    torch.testing.assert_close(
        compiled_projected_opacity(projected, config),
        projected.opacity * projected.peak_to_fiber_scale,
    )


def test_fiber_integrated_projection_skips_unused_peak_density_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(
        count=1,
        alpha_mode="beer_lambert",
        amplitude_convention="fiber_integrated",
    )

    def unexpected_inverse(*_args, **_kwargs):
        raise AssertionError("fiber-integrated projection must not invert the gauge")

    monkeypatch.setattr(torch.linalg, "inv", unexpected_inverse)
    projected = project_spd4_world_atoms_affine_gauges(
        model.batch(),
        SPD4AffineGaugeBatch(
            gauge_from_world=torch.eye(4, dtype=torch.float32)[None],
            gauge_offset=torch.zeros((1, 4), dtype=torch.float32),
            chart_time=None,
        ),
        screen_variance_floor=0.0,
    )

    assert projected.peak_to_fiber_scale is None
    assert float(projected.depth_variance[0].detach()) > 0.0


def test_spd4_depth_precision_can_match_a_near_planar_legacy_initialization() -> None:
    isotropic = _model(count=1)
    near_planar = _model(count=1, init_precision_z=1.0e6)

    isotropic_scales = torch.diagonal(
        isotropic.batch().conditional_spatial_cholesky[0]
    )
    near_planar_scales = torch.diagonal(
        near_planar.batch().conditional_spatial_cholesky[0]
    )

    torch.testing.assert_close(isotropic_scales[2], isotropic_scales[0])
    assert float(near_planar_scales[2].detach()) < (
        0.01 * float(near_planar_scales[0].detach())
    )


def test_capacity_camera_suite_identifies_covariance_outside_legacy_class() -> None:
    certificate = _conditional_covariance_identifiability(
        _target_spd4(), _camera_suite()
    )

    assert certificate["design_rank"] == 6
    assert certificate["best_legacy_observation_rmse"] > 1.0e-3
    assert certificate["full_spd_observation_rmse"] < 1.0e-10


def test_spd4_projection_retains_depth_variance_and_spatial_depth_coupling() -> None:
    model = _model(count=1)
    with torch.no_grad():
        model.spatial_cholesky_offdiag[0, 1] = 0.08
        model.space_time_tilt[0] = torch.tensor([0.05, -0.02, 0.03])
    camera = PinholeCamera(
        fx=24.0,
        fy=24.0,
        cx=8.0,
        cy=8.0,
        world_to_camera=torch.eye(4, dtype=torch.float32),
    )
    projected = project_spd4_world_atoms_pinhole(model.batch(), camera)

    assert tuple(projected.ma.shape) == (1, 3)
    assert tuple(projected.q_uvt.shape) == (1, 6)
    assert float(projected.depth_variance[0].detach()) > 0.0
    assert float(projected.depth_beta[0, 0].abs().detach()) > 1.0e-5
    loss = (
        projected.ma.square().mean()
        + 1.0e-6 * projected.q_uvt.square().mean()
        + projected.depth_beta.square().mean()
        + projected.depth_variance.mean()
        + projected.opacity.mean()
        + projected.color.mean()
    )
    loss.backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _tiny_bundle() -> SimpleNamespace:
    frames = torch.linspace(0.1, 0.9, 2 * 3 * 8 * 8, dtype=torch.float32).reshape(
        1, 2, 3, 8, 8
    )
    K = torch.tensor(
        [[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    w2c = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    return SimpleNamespace(
        train_frames=frames,
        train_K=K,
        train_w2c=w2c,
        train_lens_models=["pinhole"],
        train_distortions=None,
    )


def _train_one_step(
    world_representation: str,
    *,
    camera_sequence_mode: str = "static_view",
    alpha_mode: str = "peak_splat",
):
    return train_world_tubes(
        bundle=_tiny_bundle(),
        tube_count=4,
        train_seconds=30.0,
        max_steps=1,
        lr=0.01,
        lr_decay_step=0,
        lr_decay_factor=1.0,
        init_depth=2.0,
        init_views="first",
        init_sampling="grid",
        init_frames="all",
        init_precision_xy=30.0,
        init_lambda_t=0.4,
        init_opacity=0.3,
        min_precision_xy=1.0e-5,
        min_lambda_t=1.0e-5,
        velocity_reg_weight=1.0e-4,
        depth_velocity_reg_weight=0.0,
        position_reg_weight=1.0e-6,
        tile_load_reg_weight=0.0,
        tile_load_target=0.0,
        depth_slope_reg_weight=0.0,
        depth_margin_reg_weight=0.0,
        depth_margin=0.05,
        seed=17,
        backend="dense",
        camera_projection="legacy_pinhole",
        camera_sequence_mode=camera_sequence_mode,
        segment_frames=2,
        synthetic_pan_x=0.0,
        synthetic_pan_y=0.0,
        synthetic_dolly_z=0.0,
        synthetic_zoom=0.0,
        synthetic_principal_x=0.0,
        synthetic_principal_y=0.0,
        loss_scope="sampled_frame",
        window_frames=2,
        train_schedule="cycle",
        optimizer_train_views="all",
        validation_frame_stride=0,
        validation_frame_offset=1,
        sequence_consistency_every_steps=0,
        sequence_consistency_frames=0,
        sequence_consistency_weight=0.0,
        multiscale_loss_weight=0.0,
        multiscale_loss_factor=2,
        crop_loss_weight=0.0,
        crop_loss_size=4,
        checkpoint_every_steps=0,
        render_config=UVTRenderConfig(
            height=8,
            width=8,
            frames=2,
            tile_x=8,
            tile_y=8,
            tile_t=2,
            tile_capacity=128,
            max_alpha=max_alpha_for_mode(alpha_mode),
            alpha_mode=alpha_mode,
        ),
        world_representation=world_representation,
    )


@pytest.mark.parametrize(
    ("world_representation", "expected_type", "expected_dof"),
    (
        ("legacy_tube", WorldTubeModel, 10),
        ("full_spd4", SPD4WorldAtomModel, 14),
    ),
)
@pytest.mark.parametrize("alpha_mode", ("peak_splat", "beer_lambert"))
def test_actual_trainer_selects_parallel_representation_and_runs_one_step(
    world_representation: str,
    expected_type: type,
    expected_dof: int,
    alpha_mode: str,
) -> None:
    model, report, checkpoints = _train_one_step(
        world_representation,
        alpha_mode=alpha_mode,
    )

    assert isinstance(model, expected_type)
    assert report["steps"] == 1
    assert report["world_representation"] == world_representation
    assert report["geometry_dof_per_atom"] == expected_dof
    assert report["alpha_mode"] == alpha_mode
    assert report["opacity_semantics"] == (
        "peak_alpha_amplitude"
        if alpha_mode == "peak_splat"
        else "nonnegative_fiber_integrated_peak_optical_thickness"
    )
    assert report["logs"][0]["loss"] is not None
    assert checkpoints == []
    assert all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())


@pytest.mark.parametrize("alpha_mode", ("peak_splat", "beer_lambert"))
def test_both_world_models_initialize_the_same_center_alpha(alpha_mode: str) -> None:
    init_center_alpha = 0.3
    legacy = WorldTubeModel(
        init_x0=torch.tensor([[-0.1, 0.0, 2.0]], dtype=torch.float32),
        init_color=torch.tensor([[0.8, 0.2, 0.1]], dtype=torch.float32),
        init_t0=torch.zeros(1, dtype=torch.float32),
        frames=3,
        init_precision_xy=40.0,
        init_lambda_t=0.4,
        init_opacity=init_center_alpha,
        min_precision_xy=1.0e-5,
        min_lambda_t=1.0e-5,
        velocity_reg_weight=0.0,
        depth_velocity_reg_weight=0.0,
        position_reg_weight=0.0,
        alpha_mode=alpha_mode,
    )
    spd4 = _model(count=1, alpha_mode=alpha_mode)
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=3,
        max_alpha=max_alpha_for_mode(alpha_mode),
        alpha_mode=alpha_mode,
    )

    for model in (legacy, spd4):
        peak_field = model.batch().opacity
        center_alpha = primitive_alpha(
            peak_field,
            torch.zeros_like(peak_field),
            config,
        )
        torch.testing.assert_close(
            center_alpha,
            torch.full_like(center_alpha, init_center_alpha),
            atol=2.0e-7,
            rtol=2.0e-7,
        )
        assert model.representation_metadata()["alpha_mode"] == alpha_mode

    with torch.no_grad():
        legacy.raw_opacity.fill_(8.0)
        spd4.raw_opacity.fill_(8.0)
    if alpha_mode == "peak_splat":
        assert float(legacy.batch().opacity.detach().max()) < 0.99
        assert float(spd4.batch().opacity.detach().max()) < 0.99
    else:
        assert float(legacy.batch().opacity.detach().min()) > 1.0
        assert float(spd4.batch().opacity.detach().min()) > 1.0


def test_alpha_mode_selects_the_physical_compositing_cap() -> None:
    assert max_alpha_for_mode("peak_splat") == 0.99
    assert max_alpha_for_mode("beer_lambert") == 1.0
    with pytest.raises(ValueError, match="alpha_mode"):
        max_alpha_for_mode("unknown")


def test_beer_lambert_tile_load_proxy_uses_alpha_cutoff_in_optical_depth_space() -> None:
    alpha_threshold = 0.1
    center_alpha = 0.6
    peak_optical_thickness = -math.log1p(-center_alpha)
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=2,
        alpha_threshold=alpha_threshold,
        alpha_mode="beer_lambert",
    )
    q_uvt = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    actual = projected_tile_load_proxy(
        torch.zeros((1, 3), dtype=torch.float32),
        q_uvt,
        torch.tensor([peak_optical_thickness], dtype=torch.float32),
        config,
    )
    support_qv = -2.0 * math.log(
        -math.log1p(-alpha_threshold) / peak_optical_thickness
    )
    half_extent = math.sqrt(support_qv)
    expected = (
        (1.0 + 2.0 * half_extent / config.tile_x)
        * (1.0 + 2.0 * half_extent / config.tile_y)
        * (1.0 + 2.0 * half_extent / config.tile_t)
    )

    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=actual.dtype),
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_beer_lambert_metal_training_rejects_unvalidated_backward_before_dispatch() -> None:
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=3,
        alpha_mode="beer_lambert",
    )
    camera = PinholeCamera(
        fx=24.0,
        fy=24.0,
        cx=4.0,
        cy=4.0,
        world_to_camera=torch.eye(4, dtype=torch.float32),
    )
    projected = project_spd4_world_atoms_pinhole(_model().batch(), camera)

    with pytest.raises(ValueError, match="direct_atomic\\+index_add"):
        render_projected_sequence(
            projected,
            config,
            backend="metal_tile",
            reduction_mode="index_add",
            sample_emission_mode="atomic_append",
        )


def test_full_spd4_beer_lambert_cpu_optimizer_step_decreases_render_loss() -> None:
    model = _model(count=1, alpha_mode="beer_lambert")
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=3,
        alpha_mode="beer_lambert",
    )
    camera = PinholeCamera(
        fx=24.0,
        fy=24.0,
        cx=4.0,
        cy=4.0,
        world_to_camera=torch.eye(4, dtype=torch.float32),
    )
    optimizer = torch.optim.SGD(
        (model.raw_opacity, model.raw_color),
        lr=0.25,
    )

    def render_loss() -> torch.Tensor:
        projected = project_spd4_world_atoms_pinhole(model.batch(), camera)
        rendered = render_projected_sequence(
            projected,
            config,
            backend="dense",
        ).rgb
        return rendered.square().mean()

    optimizer.zero_grad(set_to_none=True)
    before = render_loss()
    before.backward()
    assert model.raw_opacity.grad is not None
    assert model.raw_color.grad is not None
    assert bool(torch.isfinite(model.raw_opacity.grad).all())
    assert bool(torch.isfinite(model.raw_color.grad).all())
    assert float(model.raw_opacity.grad.detach().abs().sum()) > 0.0
    assert float(model.raw_color.grad.detach().abs().sum()) > 0.0
    optimizer.step()
    with torch.no_grad():
        after = render_loss()

    assert float(after) < float(before.detach())


def _moving_camera() -> PinholeCameraMotion:
    world_to_camera = torch.eye(4, dtype=torch.float32)
    world_to_camera[:3, 3] = torch.tensor([0.1, -0.05, 0.2])
    angular_velocity = torch.tensor([0.015, -0.02, 0.01])
    wx, wy, wz = angular_velocity
    rotation_dot = torch.stack(
        (
            torch.stack((torch.zeros_like(wx), -wz, wy)),
            torch.stack((wz, torch.zeros_like(wx), -wx)),
            torch.stack((-wy, wx, torch.zeros_like(wx))),
        )
    )
    world_to_camera_dot = torch.zeros((4, 4), dtype=torch.float32)
    world_to_camera_dot[:3, :3] = rotation_dot
    world_to_camera_dot[:3, 3] = torch.tensor([0.03, -0.015, -0.02])
    return PinholeCameraMotion(
        fx=22.0,
        fy=19.0,
        cx=7.5,
        cy=8.5,
        fx_dot=0.4,
        fy_dot=-0.25,
        cx_dot=0.1,
        cy_dot=-0.08,
        world_to_camera=world_to_camera,
        world_to_camera_dot=world_to_camera_dot,
        chart_time=0.2,
    )


def test_moving_camera_affine_gauge_matches_float64_spd4_reference() -> None:
    model = _model(count=1)
    with torch.no_grad():
        model.space_time_tilt[0] = torch.tensor([0.18, -0.07, 0.04])
        model.spatial_cholesky_offdiag[0] = torch.tensor([0.02, -0.03, 0.01])
    batch = model.batch()
    camera = _moving_camera()
    gauges = compile_spd4_pinhole_motion_affine_gauges(batch, camera)
    projected = project_spd4_world_atoms_affine_gauges(
        batch,
        gauges,
        screen_variance_floor=0.0,
    )

    reference_covariance = batch.covariance_xyzt.to(torch.float64)
    reference_covariance = 0.5 * (
        reference_covariance + reference_covariance.transpose(-1, -2)
    )
    reference_atoms = WorldAtomBatch(
        mean_xyzt=batch.mean_xyzt.to(torch.float64),
        covariance_xyzt=reference_covariance,
        amplitude=batch.opacity.to(torch.float64),
        color=batch.color.to(torch.float64),
        amplitude_convention="fiber_integrated",
    )
    reference = pushforward_world_atoms(
        reference_atoms,
        AffineRayGauge(
            gauge_from_world=gauges.gauge_from_world[0].to(torch.float64),
            gauge_offset=gauges.gauge_offset[0].to(torch.float64),
            fiber_measure_scale=torch.ones((), dtype=torch.float64),
        ),
    )
    torch.testing.assert_close(
        projected.ma.to(torch.float64),
        reference.ma,
        rtol=3.0e-5,
        atol=3.0e-5,
    )
    torch.testing.assert_close(
        projected.q_uvt.to(torch.float64),
        reference.q_uvt,
        rtol=8.0e-5,
        atol=8.0e-5,
    )
    torch.testing.assert_close(
        projected.depth0.to(torch.float64),
        reference.depth0,
        rtol=3.0e-5,
        atol=3.0e-5,
    )
    torch.testing.assert_close(
        projected.depth_beta.to(torch.float64),
        reference.depth_beta,
        rtol=8.0e-5,
        atol=8.0e-5,
    )
    torch.testing.assert_close(
        projected.depth_variance.to(torch.float64),
        reference.depth_variance,
        rtol=8.0e-5,
        atol=8.0e-5,
    )


def test_moving_camera_trace_matches_camera_program_value_and_jacobian() -> None:
    model = _model(count=1)
    with torch.no_grad():
        model.space_time_tilt[0] = torch.tensor([0.18, -0.07, 0.04])
    batch = model.batch()
    camera = _moving_camera()
    projected = project_spd4_world_atoms_pinhole_motion(
        batch,
        camera,
        screen_variance_floor=0.0,
    )
    q = projected.q_uvt[0]
    screen_precision = torch.stack(
        (
            torch.stack((q[0], q[1])),
            torch.stack((q[1], q[3])),
        )
    )
    screen_time_precision = torch.stack((q[2], q[4]))
    trace_screen_slope = -torch.linalg.solve(
        screen_precision,
        screen_time_precision,
    )
    trace_depth_slope = (
        projected.depth_beta[0, 2]
        + projected.depth_beta[0, :2] @ trace_screen_slope
    )
    trace_time_delta = projected.ma.new_tensor(camera.chart_time) - projected.ma[0, 2]
    trace_screen_at_chart = (
        projected.ma[0, :2] + trace_screen_slope * trace_time_delta
    )
    trace_depth_at_chart = (
        projected.depth0[0] + trace_depth_slope * trace_time_delta
    )

    def exact_camera_program(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = time - time.new_tensor(camera.chart_time)
        world_center = batch.x0[0].to(torch.float64) + (
            batch.space_time_tilt[0].to(torch.float64)
            * (time - batch.t0[0].to(torch.float64))
        )
        homogeneous = torch.cat((world_center, time.new_ones(1)))
        world_to_camera = camera.world_to_camera.to(torch.float64) + (
            delta * camera.world_to_camera_dot.to(torch.float64)
        )
        intrinsic = torch.stack(
            (
                torch.stack(
                    (
                        time.new_tensor(camera.fx) + delta * time.new_tensor(camera.fx_dot),
                        time.new_zeros(()),
                        time.new_tensor(camera.cx) + delta * time.new_tensor(camera.cx_dot),
                    )
                ),
                torch.stack(
                    (
                        time.new_zeros(()),
                        time.new_tensor(camera.fy) + delta * time.new_tensor(camera.fy_dot),
                        time.new_tensor(camera.cy) + delta * time.new_tensor(camera.cy_dot),
                    )
                ),
                torch.stack((time.new_zeros(()), time.new_zeros(()), time.new_ones(()))),
            )
        )
        camera_point = world_to_camera[:3] @ homogeneous
        image = intrinsic @ camera_point
        return image[:2] / image[2], camera_point[2]

    chart_time = torch.tensor(camera.chart_time, dtype=torch.float64)
    epsilon = torch.tensor(1.0e-3, dtype=torch.float64)
    screen_chart, depth_chart = exact_camera_program(chart_time)
    screen_plus, depth_plus = exact_camera_program(chart_time + epsilon)
    screen_minus, depth_minus = exact_camera_program(chart_time - epsilon)
    finite_difference_screen = (screen_plus - screen_minus) / (2.0 * epsilon)
    finite_difference_depth = (depth_plus - depth_minus) / (2.0 * epsilon)

    torch.testing.assert_close(
        trace_screen_at_chart.to(torch.float64),
        screen_chart,
        rtol=3.0e-5,
        atol=3.0e-5,
    )
    torch.testing.assert_close(
        trace_depth_at_chart.to(torch.float64),
        depth_chart,
        rtol=3.0e-5,
        atol=3.0e-5,
    )
    torch.testing.assert_close(
        trace_screen_slope.to(torch.float64),
        finite_difference_screen,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    torch.testing.assert_close(
        trace_depth_slope.to(torch.float64),
        finite_difference_depth,
        rtol=2.0e-4,
        atol=2.0e-4,
    )


def test_full_spd4_dynamic_and_projective_modes_use_the_tested_affine_gauge() -> None:
    model = _model(count=1)
    with torch.no_grad():
        model.space_time_tilt[0] = torch.tensor([0.12, -0.04, 0.03])
    config = UVTRenderConfig(height=8, width=8, frames=3)
    frame_phase = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    K_sequence = torch.tensor(
        [[22.0, 0.0, 4.0], [0.0, 19.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )[None].repeat(3, 1, 1)
    K_sequence[:, 0, 0] += 0.3 * frame_phase
    K_sequence[:, 1, 1] -= 0.2 * frame_phase
    K_sequence[:, 0, 2] += 0.1 * frame_phase
    K_sequence.requires_grad_()
    world_to_camera_sequence = torch.eye(4, dtype=torch.float32)[None].repeat(
        3, 1, 1
    )
    world_to_camera_sequence[:, 0, 3] += 0.04 * frame_phase
    world_to_camera_sequence[:, 2, 3] -= 0.02 * frame_phase
    world_to_camera_sequence.requires_grad_()

    outputs = [
        project_world_tube_sequence_camera_mode(
            model=model,
            K_seq=K_sequence,
            w2c_seq=world_to_camera_sequence,
            config=config,
            full_frames=3,
            frame_start=0,
            camera_sequence_mode=mode,
            segment_frames=1,
        )
        for mode in ("dynamic_first_order", "projective_first_order")
    ]
    for field in (
        "ma",
        "q_uvt",
        "depth0",
        "depth_beta",
        "depth_variance",
        "opacity",
        "color",
    ):
        left = getattr(outputs[0], field)
        right = getattr(outputs[1], field)
        assert left is not None and right is not None
        torch.testing.assert_close(left, right)

    loss = (
        outputs[0].ma.square().mean()
        + 1.0e-6 * outputs[0].q_uvt.square().mean()
        + outputs[0].depth_beta.square().mean()
        + outputs[0].depth_variance.mean()
    )
    loss.backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter is not model.raw_opacity and parameter is not model.raw_color
    )
    assert K_sequence.grad is not None
    assert bool(torch.isfinite(K_sequence.grad).all())
    assert world_to_camera_sequence.grad is not None
    assert bool(torch.isfinite(world_to_camera_sequence.grad).all())


def test_full_spd4_dynamic_camera_runs_and_segmented_stays_fail_loud() -> None:
    model, report, checkpoints = _train_one_step(
        "full_spd4",
        camera_sequence_mode="dynamic_first_order",
    )
    assert isinstance(model, SPD4WorldAtomModel)
    assert report["steps"] == 1
    assert checkpoints == []
    with pytest.raises(ValueError, match="segmented compilation is not implemented"):
        _train_one_step("full_spd4", camera_sequence_mode="segmented")


def test_spd4_metal_forward_and_source_parameter_vjp_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    device = torch.device("mps")
    config = UVTRenderConfig(height=8, width=8, frames=2)
    model = SPD4WorldAtomModel(
        init_x0=torch.tensor(
            [[-0.10, 0.03, 2.00], [0.14, -0.04, 2.25]], device=device
        ),
        init_color=torch.tensor(
            [[0.8, 0.2, 0.1], [0.1, 0.4, 0.9]], device=device
        ),
        init_t0=torch.tensor([-0.2, 0.25], device=device),
        frames=2,
        init_precision_xy=35.0,
        init_precision_z=80.0,
        init_lambda_t=0.65,
        init_opacity=0.45,
        min_spatial_scale=1.0e-4,
        min_lambda_t=1.0e-5,
        tilt_reg_weight=0.0,
        depth_tilt_reg_weight=0.0,
        position_reg_weight=0.0,
    ).to(device)
    with torch.no_grad():
        model.spatial_cholesky_offdiag.copy_(
            torch.tensor(
                [[0.025, -0.015, 0.020], [-0.020, 0.010, 0.018]],
                device=device,
            )
        )
        model.space_time_tilt.copy_(
            torch.tensor(
                [[0.060, -0.035, 0.025], [-0.040, 0.050, -0.020]],
                device=device,
            )
        )
    angle = 0.35
    rotation = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        device=device,
    )
    world_to_camera = torch.eye(4, device=device)
    world_to_camera[:3, :3] = rotation
    intrinsics = torch.tensor(
        [[14.0, 0.0, 4.0], [0.0, 13.0, 4.0], [0.0, 0.0, 1.0]],
        device=device,
    )

    projected = project_world_tube_sequence(
        model, intrinsics, world_to_camera, config
    )
    projected.depth0.retain_grad()
    projected.depth_beta.retain_grad()
    metal = render_projected_sequence(
        projected, config, backend="metal_tile"
    ).rgb
    reference_inputs = [
        value.detach().cpu()
        for value in (
            projected.ma,
            projected.q_uvt,
            projected.depth0,
            projected.depth_beta,
            projected.opacity,
            projected.color,
        )
    ]
    reference = brute_force_render_uvt_tubes(*reference_inputs, config)

    torch.testing.assert_close(
        metal.detach().cpu(), reference, atol=1.0e-5, rtol=1.0e-5
    )
    image_weight = torch.linspace(
        0.7, 1.3, metal.numel(), device=device
    ).reshape_as(metal)
    (metal * image_weight).mean().backward()
    torch.mps.synchronize()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.norm().cpu()) > 0.0
    # Hard ordering is intentionally piecewise constant in the current Metal
    # VJP. Geometry still receives gradients through ma/q_uvt, while these two
    # ordering tensors receive explicit zero cotangents.
    assert projected.depth0.grad is not None
    assert projected.depth_beta.grad is not None
    assert int(torch.count_nonzero(projected.depth0.grad).cpu()) == 0
    assert int(torch.count_nonzero(projected.depth_beta.grad).cpu()) == 0
    assert projected.depth_variance is not None
    assert bool((projected.depth_variance > 0.0).all())
    torch.mps.empty_cache()


def test_spd4_hybrid_retained_fiber_training_vjp_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    device = torch.device("mps")
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=2,
        alpha_mode="beer_lambert",
        max_alpha=1.0,
        retained_depth_samples=24,
        retained_sigma_extent=4.0,
        order_certificate_sigma=4.0,
    )
    model = _model(
        count=2,
        init_precision_z=4.0,
        alpha_mode="beer_lambert",
    ).to(device)
    with torch.no_grad():
        model.x0.copy_(
            torch.tensor(
                [[0.0, 0.0, 2.0], [0.0, 0.0, 2.1]],
                dtype=torch.float32,
                device=device,
            )
        )
        model.raw_spatial_scale[1, 2].add_(0.35)
    intrinsics = torch.tensor(
        [[14.0, 0.0, 4.0], [0.0, 14.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    world_to_camera = torch.eye(4, dtype=torch.float32, device=device)
    projected = project_world_tube_sequence(
        model,
        intrinsics,
        world_to_camera,
        config,
    )
    projected.depth_variance.retain_grad()
    rendered = render_projected_sequence(
        projected,
        config,
        backend="hybrid_retained_fiber",
        reduction_mode="index_add",
        sample_emission_mode="direct_atomic",
    )
    image_weight = torch.tensor(
        [0.31, -0.17, 0.43],
        dtype=torch.float32,
        device=device,
    )
    loss = torch.dot(rendered.rgb[0, 4, 4], image_weight)
    loss.backward()
    torch.mps.synchronize()

    assert rendered.fallback_tiles is not None
    assert int(rendered.fallback_tiles.sum().cpu()) > 0
    assert rendered.fallback_reason_bits is not None
    assert bool(((rendered.fallback_reason_bits & 4) != 0).any().cpu())
    assert projected.depth_variance.grad is not None
    assert float(projected.depth_variance.grad.abs().sum().cpu()) > 0.0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    torch.mps.empty_cache()
