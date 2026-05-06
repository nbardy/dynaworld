from __future__ import annotations

from pathlib import Path

import pytest
import torch

from colorize import FeatureToColor
from powerfoam_direct import logit_clamped
from powerfoam_implicit_camera import PowerFoamImplicitCameraDecoder
from train_dynamic_powerfoam_metal import (
    DynamicMetalPowerFoamVideo,
    LOSS_DEFAULTS,
    TokenDynamicPowerFoamFeatures,
    init_colorizer_rgb_identity,
    make_raster_config,
    render_features_to_rgb,
    sample_background,
    temporal_motion_metrics,
)
from train_powerfoam_metal import make_pinhole_rays
from research_experiments.dynamic_foam.verify_dynamic_powerfoam_geometry_run import check_summary


def _make_camera_decoder() -> PowerFoamImplicitCameraDecoder:
    return PowerFoamImplicitCameraDecoder(
        frame_count=4,
        image_size=8,
        fov_degrees=55.0,
        base_radius=3.0,
        token_dim=8,
        hidden_dim=16,
        time_basis_count=3,
    )


def _make_model(
    dynamic_mode: str,
    *,
    camera_decoder: PowerFoamImplicitCameraDecoder | None = None,
) -> DynamicMetalPowerFoamVideo:
    frames = torch.rand(4, 3, 8, 8)
    return DynamicMetalPowerFoamVideo(
        frame_count=frames.shape[0],
        cell_count=16,
        render_size=8,
        fov_degrees=55.0,
        neighbor_count=4,
        adjacency_mode="knn",
        dynamic_mode=dynamic_mode,
        time_basis_count=3,
        time_basis_sigma_scale=0.75,
        temporal_init_mode="fit",
        dynamic_centers=True,
        dynamic_radii=True,
        dynamic_densities=True,
        dynamic_features=True,
        dynamic_normals=False,
        dynamic_texel_sites=False,
        xy_extent=1.25,
        z_min=1.0,
        z_max=3.25,
        radius_init=0.18,
        radius_min=0.03,
        radius_scale=0.72,
        density_init=16.0,
        normal_init_jitter=0.0,
        num_texel_sites=4,
        texel_site_scale=0.5,
        color_init_mode="image",
        seed=17,
        init_frames=frames,
        image_init_depth=2.0,
        image_init_jitter=0.1,
        raster_config=make_raster_config(
            {
                "near_plane": 0.05,
                "alpha_threshold": 0.0,
                "transmittance_threshold": 1.0e-4,
                "max_alpha": 0.99,
                "eps": 1.0e-6,
                "texel_temperature": 10.0,
            }
        ),
        camera_decoder=camera_decoder,
    )


def _make_token_model(
    *,
    camera_decoder: PowerFoamImplicitCameraDecoder | None = None,
) -> TokenDynamicPowerFoamFeatures:
    frames = torch.rand(4, 3, 8, 8)
    return TokenDynamicPowerFoamFeatures(
        frame_count=frames.shape[0],
        cell_count=16,
        render_size=8,
        fov_degrees=55.0,
        neighbor_count=4,
        adjacency_mode="knn",
        time_basis_count=3,
        time_basis_sigma_scale=0.75,
        temporal_init_mode="fit",
        dynamic_centers=True,
        dynamic_radii=True,
        dynamic_densities=True,
        dynamic_features=True,
        dynamic_normals=True,
        dynamic_texel_sites=True,
        feature_dim=8,
        feature_init_noise=0.01,
        feature_rgb_init="logit",
        token_dim=16,
        token_hidden_dim=32,
        token_hidden_layers=1,
        token_init_std=0.02,
        token_output_init_std=1.0e-4,
        token_point_residual_scale=0.08,
        token_z_residual_scale=0.08,
        token_radius_residual_scale=0.05,
        token_density_residual_scale=0.08,
        token_feature_residual_scale=0.25,
        token_normal_residual_scale=0.08,
        token_texel_site_residual_scale=0.08,
        token_temporal_residual_scale=0.2,
        xy_extent=1.25,
        z_min=1.0,
        z_max=3.25,
        radius_init=0.18,
        radius_min=0.03,
        radius_scale=0.72,
        density_init=16.0,
        normal_init_jitter=0.0,
        num_texel_sites=4,
        texel_site_scale=0.5,
        color_init_mode="image",
        seed=17,
        init_frames=frames,
        image_init_depth=2.0,
        image_init_jitter=0.1,
        raster_config=make_raster_config(
            {
                "near_plane": 0.05,
                "alpha_threshold": 0.0,
                "transmittance_threshold": 1.0e-4,
                "max_alpha": 0.99,
                "eps": 1.0e-6,
                "texel_temperature": 10.0,
            }
        ),
        camera_decoder=camera_decoder,
    )


def test_dynamic_powerfoam_rbf_decode_has_temporal_grads() -> None:
    model = _make_model("rbf")
    points, radii, densities, features, normals = model.decoded_parameters()
    assert points.shape == (4, 16, 3)
    assert radii.shape == (4, 16)
    assert densities.shape == (4, 16)
    assert features.shape == (4, 16, 4, 3)
    assert normals.shape == (4, 16, 3)

    reg_loss, terms = model.temporal_regularization(LOSS_DEFAULTS)
    loss = points.square().mean() + features.square().mean() + reg_loss
    loss.backward()
    assert model.raw_xy_coeff is not None
    assert model.raw_features_coeff is not None
    assert model.raw_xy_coeff.grad is not None
    assert model.raw_features_coeff.grad is not None
    assert terms["temporal_coeff_l2"].item() >= 0.0


def _zero_rbf_coeffs(model: DynamicMetalPowerFoamVideo) -> None:
    for name in (
        "raw_xy_coeff",
        "raw_z_coeff",
        "raw_radii_coeff",
        "raw_densities_coeff",
        "raw_features_coeff",
        "raw_normals_coeff",
        "raw_tangents_coeff",
        "raw_texel_sites_coeff",
    ):
        coeff = getattr(model, name, None)
        if coeff is not None:
            coeff.zero_()


def test_dynamic_powerfoam_zero_coefficients_reproduce_static_decode() -> None:
    model = _make_model("rbf")
    with torch.no_grad():
        _zero_rbf_coeffs(model)

    frame_indices = torch.tensor([0, 1])
    points, radii, densities, features, normals = model.decoded_parameters(frame_indices)
    texel_sites = model.decoded_texel_sites(frame_indices)

    assert torch.allclose(points[0], points[1], atol=1.0e-6)
    assert torch.allclose(radii[0], radii[1], atol=1.0e-6)
    assert torch.allclose(densities[0], densities[1], atol=1.0e-6)
    assert torch.allclose(features[0], features[1], atol=1.0e-6)
    assert torch.allclose(normals[0], normals[1], atol=1.0e-6)
    assert torch.allclose(texel_sites[0], texel_sites[1], atol=1.0e-6)


def test_dynamic_powerfoam_geometry_coefficients_do_not_repaint_features() -> None:
    model = _make_model("rbf")
    with torch.no_grad():
        _zero_rbf_coeffs(model)
        assert model.raw_xy_coeff is not None
        pattern = torch.linspace(-0.25, 0.25, model.raw_xy_coeff.shape[1])
        model.raw_xy_coeff[0, :, 0] = pattern
        model.raw_radii_coeff[0] = 0.1

    frame_indices = torch.tensor([0, 1])
    points, radii, _densities, features, _normals = model.decoded_parameters(frame_indices)

    assert not torch.allclose(points[0], points[1])
    assert not torch.allclose(radii[0], radii[1])
    assert torch.allclose(features[0], features[1], atol=1.0e-6)


def test_dynamic_powerfoam_feature_coefficients_do_not_move_geometry() -> None:
    model = _make_model("rbf")
    with torch.no_grad():
        _zero_rbf_coeffs(model)
        assert model.raw_features_coeff is not None
        model.raw_features_coeff[0] = 0.25

    frame_indices = torch.tensor([0, 1])
    points, radii, densities, features, _normals = model.decoded_parameters(frame_indices)

    assert torch.allclose(points[0], points[1], atol=1.0e-6)
    assert torch.allclose(radii[0], radii[1], atol=1.0e-6)
    assert torch.allclose(densities[0], densities[1], atol=1.0e-6)
    assert not torch.allclose(features[0], features[1])


def test_dynamic_powerfoam_per_frame_smooth_decode_has_temporal_loss() -> None:
    model = _make_model("per_frame_smooth")
    points, _radii, _densities, features, _normals = model.decoded_parameters()
    reg_loss, terms = model.temporal_regularization(LOSS_DEFAULTS)
    loss = points.square().mean() + features.square().mean() + reg_loss
    loss.backward()
    assert model.raw_xy.grad is not None
    assert model.raw_features.grad is not None
    assert terms["temporal_center_accel"].item() >= 0.0


def test_token_dynamic_powerfoam_features_decode_has_token_grads() -> None:
    model = _make_token_model()
    points, radii, densities, features, normals = model.decoded_parameters()
    assert points.shape == (4, 16, 3)
    assert radii.shape == (4, 16)
    assert densities.shape == (4, 16)
    assert features.shape == (4, 16, 4, 8)
    assert normals.shape == (4, 16, 3)

    reg_loss, terms = model.temporal_regularization(LOSS_DEFAULTS)
    loss = points.square().mean() + features.square().mean() + reg_loss
    loss.backward()
    assert model.tokens.grad is not None
    assert any(param.grad is not None for param in model.decoder.parameters())
    assert terms["temporal_coeff_l2"].item() >= 0.0


def test_token_dynamic_powerfoam_features_optimizer_groups_are_lean() -> None:
    model = _make_token_model()
    groups = model.optimizer_param_groups(
        {
            "lr": 0.01,
            "token_lr_multiplier": 1.0,
            "decoder_lr_multiplier": 0.5,
            "point_lr_multiplier": 0.1,
            "radius_lr_multiplier": 0.05,
            "density_lr_multiplier": 0.2,
            "feature_lr_multiplier": 0.01,
            "normal_lr_multiplier": 0.3,
            "texel_site_lr_multiplier": 2.0,
            "temporal_lr_multiplier": 0.25,
        }
    )
    lr_by_name = {str(group["name"]): float(group["lr"]) for group in groups}
    assert set(lr_by_name) == {"tokens", "decoder"}
    assert lr_by_name["tokens"] == 0.01
    assert lr_by_name["decoder"] == 0.005


def test_dynamic_powerfoam_default_rays_match_fixed_pinhole() -> None:
    frame_indices = torch.tensor([0, 1])
    expected = make_pinhole_rays(8, 8, 55.0, torch.device("cpu"))

    for model in (_make_model("rbf"), _make_token_model()):
        rays = model.decoded_camera_rays(frame_indices, dtype=torch.float32)
        assert rays.shape == expected.shape
        assert torch.allclose(rays, expected, atol=1.0e-6)


def test_dynamic_powerfoam_implicit_camera_rays_are_per_frame_and_trainable() -> None:
    decoder = _make_camera_decoder()
    model = _make_token_model(camera_decoder=decoder)
    groups = model.optimizer_param_groups(
        {
            "lr": 0.01,
            "token_lr_multiplier": 1.0,
            "decoder_lr_multiplier": 0.5,
            "camera_lr_multiplier": 0.25,
        }
    )
    assert {str(group["name"]) for group in groups} == {"tokens", "decoder", "implicit_camera"}

    frame_indices = torch.tensor([0, 1])
    zero_rays = model.decoded_camera_rays(frame_indices, dtype=torch.float32)
    assert zero_rays.shape == (2, 8, 8, 6)
    expected_origin = torch.tensor([0.0, 0.0, -3.0]).view(1, 1, 1, 3).expand_as(zero_rays[..., :3])
    assert torch.allclose(zero_rays[..., :3], expected_origin, atol=1.0e-6)

    with torch.no_grad():
        assert model.camera_decoder is not None
        model.camera_decoder.offset_head[-1].bias[0] = 0.5
        model.camera_decoder.offset_head[-1].bias[3] = 0.5

    moved_rays = model.decoded_camera_rays(frame_indices, dtype=torch.float32)
    assert not torch.allclose(moved_rays, zero_rays)
    reg_loss, terms = model.temporal_regularization(
        {
            **LOSS_DEFAULTS,
            "camera_motion_weight": 1.0,
            "camera_temporal_weight": 1.0,
            "camera_global_weight": 1.0,
        }
    )
    assert reg_loss.item() > 0.0
    assert terms["camera_rotation_l2"].item() > 0.0
    assert terms["camera_translation_l2"].item() > 0.0
    metrics = model.parameter_drift_metrics()
    assert metrics["state_camera_origin_delta_mean"] > 0.0


def test_feature_colorizer_identity_and_background_composition() -> None:
    colorizer = FeatureToColor(feature_dim=8, hidden_dim=None, activation="sigmoid", pre_norm=False)
    init_colorizer_rgb_identity(colorizer)
    rgb = torch.tensor([[[[0.2]], [[0.5]], [[0.8]]]], dtype=torch.float32)
    features = torch.zeros(1, 8, 1, 1)
    features[:, :3] = logit_clamped(rgb)
    alpha = torch.ones(1, 1, 1)
    rendered = render_features_to_rgb(
        features,
        alpha,
        colorizer,
        background=None,
        normalize_features_by_alpha=True,
        eps=1.0e-6,
    )
    assert torch.allclose(rendered, rgb, atol=1.0e-5)

    background = torch.tensor([[[[0.1]], [[0.3]], [[0.7]]]], dtype=torch.float32)
    empty = render_features_to_rgb(
        features,
        torch.zeros_like(alpha),
        colorizer,
        background=background,
        normalize_features_by_alpha=True,
        eps=1.0e-6,
    )
    assert torch.allclose(empty, background, atol=1.0e-6)


def test_rgb_direct_background_composition_uses_unpremultiplied_color() -> None:
    alpha = torch.full((1, 1, 1), 0.25)
    premultiplied_rgb = torch.tensor([[[[0.10]], [[0.15]], [[0.20]]]], dtype=torch.float32)
    background = torch.tensor([[[[0.5]], [[0.5]], [[0.5]]]], dtype=torch.float32)
    rendered = render_features_to_rgb(
        premultiplied_rgb,
        alpha,
        colorizer=None,
        background=background,
        normalize_features_by_alpha=True,
        eps=1.0e-6,
    )
    expected = premultiplied_rgb + (1.0 - alpha.unsqueeze(1)) * background
    assert torch.allclose(rendered, expected, atol=1.0e-6)


def test_random_background_sampler_bounds_and_shape() -> None:
    render_cfg = {
        "train_background_mode": "random_rgb",
        "eval_background_mode": "fixed_rgb",
        "background": [0.1, 0.2, 0.3],
        "random_background_min": 0.25,
        "random_background_max": 0.75,
    }
    bg = sample_background(render_cfg, phase="train", batch_size=5, device=torch.device("cpu"), dtype=torch.float32)
    assert bg is not None
    assert bg.shape == (5, 3, 1, 1)
    assert float(bg.min()) >= 0.25
    assert float(bg.max()) <= 0.75
    fixed = sample_background(render_cfg, phase="eval", batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    assert fixed is not None
    assert fixed.shape == (2, 3, 1, 1)
    assert torch.allclose(fixed[0, :, 0, 0], torch.tensor([0.1, 0.2, 0.3]))


def test_temporal_motion_metrics_report_screen_motion() -> None:
    points = torch.tensor(
        [
            [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]],
            [[0.1, 0.0, 2.0], [0.1, 0.2, 2.0]],
        ],
        dtype=torch.float32,
    )
    features = torch.zeros(2, 2, 1, 3)
    features[1] = 0.25
    metrics = temporal_motion_metrics(points, features, render_size=64, fov_degrees=55.0)
    assert metrics["state_mean_temporal_xy_delta"] > 0.0
    assert metrics["state_mean_temporal_screen_delta_px"] > 0.0
    assert metrics["state_mean_temporal_feature_abs_delta"] == 0.25


def test_token_dynamic_powerfoam_features_mps_raster_backward_smoke() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the dynamic PowerFoam raster smoke")
    try:
        model = _make_token_model().to("mps")
        colorizer = FeatureToColor(feature_dim=8, hidden_dim=None, activation="sigmoid", pre_norm=False).to("mps")
        init_colorizer_rgb_identity(colorizer)
        frame_indices = torch.tensor([0], device="mps")
        features, alpha = model(frame_indices)
        background = torch.rand(1, 3, 1, 1, device="mps")
        rendered = render_features_to_rgb(
            features,
            alpha,
            colorizer,
            background,
            normalize_features_by_alpha=True,
            eps=1.0e-6,
        )
        target = torch.rand_like(rendered)
        loss = torch.nn.functional.l1_loss(rendered, target) + 0.1 * torch.nn.functional.mse_loss(rendered, target)
        loss.backward()
    except (RuntimeError, ValueError) as exc:
        if "dynamic_powerfoam_metal custom ops not found" in str(exc):
            pytest.skip("dynamic_powerfoam_metal custom ops not found")
        raise
    assert model.tokens.grad is not None
    assert torch.isfinite(model.tokens.grad).all()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.decoder.parameters())
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in colorizer.parameters())


def test_token_dynamic_powerfoam_implicit_camera_pose_backprops_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the dynamic PowerFoam implicit-camera proof")
    try:
        model = _make_token_model(camera_decoder=_make_camera_decoder()).to("mps")
        frame_indices = torch.tensor([0], device="mps")
        features, alpha = model(frame_indices)
        loss = features.square().mean() + alpha.square().mean()
        loss.backward()
    except (RuntimeError, ValueError) as exc:
        if "dynamic_powerfoam_metal custom ops not found" in str(exc):
            pytest.skip("dynamic_powerfoam_metal custom ops not found")
        raise

    assert model.camera_decoder is not None
    offset_bias_grad = model.camera_decoder.offset_head[-1].bias.grad
    global_bias_grad = model.camera_decoder.global_head[-1].bias.grad
    assert offset_bias_grad is not None
    assert global_bias_grad is not None
    assert torch.isfinite(offset_bias_grad).all()
    assert torch.isfinite(global_bias_grad).all()
    assert float(offset_bias_grad.abs().sum().detach().cpu()) > 0.0
    assert float(global_bias_grad.abs().sum().detach().cpu()) > 0.0


def test_dynamic_powerfoam_geometry_motion_changes_alpha_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the dynamic PowerFoam alpha motion proof")
    try:
        model = _make_model("rbf").to("mps")
        with torch.no_grad():
            _zero_rbf_coeffs(model)
            frame_indices = torch.tensor([0, 1], device="mps")
            _static_features, static_alpha = model(frame_indices)
            assert torch.allclose(static_alpha[0], static_alpha[1], atol=1.0e-5)

            assert model.raw_xy_coeff is not None
            pattern = torch.linspace(-0.6, 0.6, model.raw_xy_coeff.shape[1], device="mps")
            model.raw_xy_coeff[0, :, 0] = pattern
            model.raw_radii_coeff[0] = 0.2
            _moving_features, moving_alpha = model(frame_indices)
    except (RuntimeError, ValueError) as exc:
        if "dynamic_powerfoam_metal custom ops not found" in str(exc):
            pytest.skip("dynamic_powerfoam_metal custom ops not found")
        raise

    assert not torch.allclose(moving_alpha[0], moving_alpha[1])
    assert float((moving_alpha[0] - moving_alpha[1]).abs().mean().detach().cpu()) > 1.0e-5


def test_dynamic_powerfoam_render_alpha_loss_backprops_to_geometry_coeffs_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the dynamic PowerFoam geometry gradient proof")
    try:
        model = _make_model("rbf").to("mps")
        for name in ("raw_features", "raw_features0", "raw_features_coeff"):
            param = getattr(model, name, None)
            if param is not None:
                param.requires_grad_(False)
        frame_indices = torch.tensor([0, 1], device="mps")
        _features, alpha = model(frame_indices)
        loss = alpha.mean()
        loss.backward()
    except (RuntimeError, ValueError) as exc:
        if "dynamic_powerfoam_metal custom ops not found" in str(exc):
            pytest.skip("dynamic_powerfoam_metal custom ops not found")
        raise

    assert model.raw_xy_coeff is not None
    assert model.raw_radii_coeff is not None
    assert model.raw_xy_coeff.grad is not None
    assert model.raw_radii_coeff.grad is not None
    assert float(model.raw_xy_coeff.grad.abs().sum().detach().cpu()) > 0.0
    assert float(model.raw_radii_coeff.grad.abs().sum().detach().cpu()) > 0.0


def test_dynamic_powerfoam_geometry_summary_verifier_contract(tmp_path: Path) -> None:
    summary = {
        "schema_version": "dynamic_powerfoam_geometry_summary_v1",
        "status": "ok",
        "config": {
            "dynamic_centers": True,
            "dynamic_radii": True,
            "dynamic_features": False,
        },
        "final_eval": {"eval_l1": 0.1},
        "motion_vs_repaint": {
            "state_mean_temporal_screen_delta_px": 0.25,
            "state_p95_temporal_screen_delta_px": 0.5,
            "state_mean_temporal_feature_abs_delta": 0.0,
            "eval_mean_temporal_alpha_delta": 0.01,
            "eval_mean_temporal_support_delta": 0.02,
        },
    }
    checks = check_summary(
        summary,
        require_geometry_motion=True,
        require_alpha_support_motion=True,
        require_appearance_freeze_control=True,
        min_screen_delta_px=1.0e-5,
        min_alpha_delta=1.0e-6,
        min_support_delta=0.0,
        max_feature_delta=1.0e-8,
    )
    assert all(check["passed"] for check in checks)

    summary["motion_vs_repaint"]["state_mean_temporal_feature_abs_delta"] = 0.1
    failed = check_summary(
        summary,
        require_geometry_motion=True,
        require_alpha_support_motion=True,
        require_appearance_freeze_control=True,
        min_screen_delta_px=1.0e-5,
        min_alpha_delta=1.0e-6,
        min_support_delta=0.0,
        max_feature_delta=1.0e-8,
    )
    assert not all(check["passed"] for check in failed)
