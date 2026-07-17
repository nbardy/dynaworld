from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

STAR_UVT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from star_uvt_projective_interval_backend import (  # noqa: E402
    make_projective_cell_interval_atlas_from_uvt_tubes,
    make_projective_cell_interval_live_atlas_from_uvt_tubes,
    make_projective_cell_interval_trainer_state_from_uvt_tubes,
)
from research_experiments.star_uvt_feature_tubes.projective_interval_cache_policy_benchmark import (  # noqa: E402
    _fmt as _cache_policy_fmt,
)
from research_project.trainer_harness import (  # noqa: E402
    render_projective_cell_interval_atlas_metal_backward,
)
from star_uvt_feature_overfit_trainer import (  # noqa: E402
    _ProjectiveIntervalFeatureRenderCache,
    _lock_projective_interval_spatial_precision,
    _projective_interval_cache_should_rebuild,
    _projective_interval_times,
    _render_projective_interval_feature_tubes_autograd,
)
import star_uvt_feature_overfit_trainer as feature_overfit_trainer  # noqa: E402
from star_uvt_feature_tube_model import (  # noqa: E402
    FeatureScreenTimeTubeModel,
    FeatureTubeRenderConfig,
    dense_render_feature_tubes,
)
from star_uvt_render_configs import star_uvt_render_configs_from_cfg  # noqa: E402
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    UVTRenderConfig,
    eval_projective_trace_cell_depth_at_uv_torch,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
    pack_projective_trace_tile_time_bins,
    projective_trace_cell_atlas_coverage_report,
    render_projective_trace_cell_atlas_reference,
    render_projective_trace_cell_interval_atlas_metal,
    uvt_tubes_to_projective_trace_cell_atlas,
)


def _affine_uvt_fixture(*, device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    dev = torch.device(device)
    sigma_px = 2.0
    inv_sigma2 = 1.0 / (sigma_px * sigma_px)
    ma = torch.tensor(
        [
            [4.5, 5.0, 0.0],
            [11.0, 8.5, 0.0],
        ],
        dtype=torch.float32,
        device=dev,
    )
    velocity = torch.tensor(
        [
            [0.50, 0.20],
            [-0.30, 0.15],
        ],
        dtype=torch.float32,
        device=dev,
    )
    q_uvt = torch.stack(
        (
            torch.full((2,), inv_sigma2, dtype=torch.float32, device=dev),
            torch.zeros((2,), dtype=torch.float32, device=dev),
            -inv_sigma2 * velocity[:, 0],
            torch.full((2,), inv_sigma2, dtype=torch.float32, device=dev),
            -inv_sigma2 * velocity[:, 1],
            inv_sigma2 * velocity[:, 0].square() + inv_sigma2 * velocity[:, 1].square(),
        ),
        dim=-1,
    )
    depth0 = torch.tensor([0.75, 1.50], dtype=torch.float32, device=dev)
    depth_beta = torch.zeros((2, 3), dtype=torch.float32, device=dev)
    opacity = torch.tensor([0.55, 0.40], dtype=torch.float32, device=dev)
    color = torch.tensor(
        [
            [0.90, 0.15, 0.05],
            [0.10, 0.35, 0.95],
        ],
        dtype=torch.float32,
        device=dev,
    )
    return ma, q_uvt.contiguous(), depth0, depth_beta, opacity, color


def _anisotropic_uvt_fixture(*, device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    ma, _q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture(device=device)
    dev = torch.device(device)
    velocity = torch.tensor(
        [
            [0.50, 0.20],
            [-0.30, 0.15],
        ],
        dtype=torch.float32,
        device=dev,
    )
    q_uu = torch.tensor([0.25, 0.18], dtype=torch.float32, device=dev)
    q_uv = torch.tensor([0.04, -0.03], dtype=torch.float32, device=dev)
    q_vv = torch.tensor([0.12, 0.30], dtype=torch.float32, device=dev)
    q_ut = -(q_uu * velocity[:, 0] + q_uv * velocity[:, 1])
    q_vt = -(q_uv * velocity[:, 0] + q_vv * velocity[:, 1])
    q_tt = (
        q_uu * velocity[:, 0].square()
        + 2.0 * q_uv * velocity[:, 0] * velocity[:, 1]
        + q_vv * velocity[:, 1].square()
    )
    q_uvt = torch.stack((q_uu, q_uv, q_ut, q_vv, q_vt, q_tt), dim=-1)
    return ma, q_uvt.contiguous(), depth0, depth_beta, opacity, color


def _uvt_center_velocity(q_uvt: torch.Tensor) -> torch.Tensor:
    q_uu = q_uvt[:, 0]
    q_uv = q_uvt[:, 1]
    q_ut = q_uvt[:, 2]
    q_vv = q_uvt[:, 3]
    q_vt = q_uvt[:, 4]
    det = q_uu * q_vv - q_uv.square()
    inv00 = q_vv / det
    inv01 = -q_uv / det
    inv11 = q_uu / det
    velocity_u = -(inv00 * q_ut + inv01 * q_vt)
    velocity_v = -(inv01 * q_ut + inv11 * q_vt)
    return torch.stack((velocity_u, velocity_v), dim=-1)


def test_feature_tube_model_has_spd_trainable_uv_cross_precision() -> None:
    config = FeatureTubeRenderConfig(frames=3, height=8, width=8, feature_dim=3)
    model = FeatureScreenTimeTubeModel(2, config, seed=1)
    with torch.no_grad():
        model.raw_spatial_correlation.fill_(0.5)
    ma, q_uvt, _depth0, _depth_beta, _opacity, _feature = model.tensors()

    det = q_uvt[:, 0] * q_uvt[:, 3] - q_uvt[:, 1].square()

    assert ma.shape == (2, 3)
    assert torch.all(q_uvt[:, 1].abs() > 0.0)
    assert torch.all(det > 0.0)
    torch.testing.assert_close(_uvt_center_velocity(q_uvt), model.velocity_uv.detach())


def test_projective_interval_measured_cache_policy_skips_cadence_rebuild() -> None:
    cache = _ProjectiveIntervalFeatureRenderCache(
        state=object(),
        last_rebuild_step=0,
    )

    assert _projective_interval_cache_should_rebuild(
        cache,
        step_index=3,
        refresh_every=2,
        refresh_policy="cadence",
    )
    assert not _projective_interval_cache_should_rebuild(
        cache,
        step_index=3,
        refresh_every=2,
        refresh_policy="measured",
    )

    with pytest.raises(ValueError, match="refresh_policy"):
        _projective_interval_cache_should_rebuild(
            cache,
            step_index=1,
            refresh_every=2,
            refresh_policy="banana",
        )


def test_projective_interval_cache_policy_report_preserves_tail_alpha_precision() -> None:
    assert _cache_policy_fmt(0.0003) == "0.0003"
    assert _cache_policy_fmt(0.00035) == "0.00035"
    assert _cache_policy_fmt(0.00032070223950928124) == "0.000320702"
    assert _cache_policy_fmt(0.0847767964) == "0.0848"


def test_projective_interval_cache_records_support_margin_slack() -> None:
    cache = _ProjectiveIntervalFeatureRenderCache()
    refresh = SimpleNamespace(
        support_margin_before=SimpleNamespace(
            missing_tile_pairs=2,
            min_boundary_slack_px=-0.25,
            max_boundary_overshoot_px=0.25,
        ),
        support_tail_alpha_bound_before=2.5e-4,
        rebinned=True,
        visibility_stratified=False,
        fallback_marked=False,
    )
    state = SimpleNamespace(refresh=lambda force=False: refresh)

    feature_overfit_trainer._refresh_projective_interval_cache_if_stale(cache, state)

    assert cache.staleness_check_count == 1
    assert cache.stale_refresh_count == 1
    assert cache.support_rebin_count == 1
    assert cache.last_support_margin_missing_tile_pairs == 2
    assert cache.last_support_margin_min_slack_px == pytest.approx(-0.25)
    assert cache.last_support_margin_max_overshoot_px == pytest.approx(0.25)
    assert cache.min_support_margin_min_slack_px == pytest.approx(-0.25)
    assert cache.max_support_margin_max_overshoot_px == pytest.approx(0.25)
    assert cache.last_support_tail_alpha_bound == pytest.approx(2.5e-4)
    assert cache.max_support_tail_alpha_bound == pytest.approx(2.5e-4)


def test_uvt_tubes_to_projective_cell_atlas_matches_dense_uvt_for_exact_affine_case() -> None:
    frames = 5
    height = 16
    width = 16
    sigma_px = 2.0
    alpha_threshold = 0.01
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()

    support_radius = math.sqrt(2.0 * sigma_px * sigma_px * math.log(float(opacity.max().item()) / alpha_threshold))
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=sigma_px,
        image_width=width,
        image_height=height,
        tile_size=8,
        uv_padding=math.ceil(support_radius) + 1.0,
        alpha_threshold=alpha_threshold,
    )

    atlas_image = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=width,
        image_height=height,
        tile_size=8,
        sigma_px=sigma_px,
        alpha_cutoff=alpha_threshold,
    )
    dense_image, _alpha = dense_render_feature_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        FeatureTubeRenderConfig(
            frames=frames,
            height=height,
            width=width,
            feature_dim=3,
            alpha_threshold=alpha_threshold,
            max_alpha=1.0,
        ),
    )

    assert atlas.source_primitive_ids == (0, 1)
    assert atlas.active_start == (0, 0)
    assert atlas.active_stop == (frames, frames)
    assert torch.allclose(atlas_image, dense_image.permute(0, 2, 3, 1), atol=1.0e-5, rtol=1.0e-5)


def test_uvt_tubes_to_projective_cell_atlas_matches_dense_uvt_with_temporal_envelope() -> None:
    frames = 9
    height = 16
    width = 16
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    q_uvt = q_uvt.clone()
    q_uvt[:, 5] += 1.0
    alpha_threshold = 0.20

    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=width,
        image_height=height,
        tile_size=8,
        uv_padding=8.0,
        alpha_threshold=alpha_threshold,
    )
    atlas_image = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=width,
        image_height=height,
        tile_size=8,
        sigma_px=2.0,
        alpha_cutoff=alpha_threshold,
    )
    dense_image, _alpha = dense_render_feature_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        FeatureTubeRenderConfig(
            frames=frames,
            height=height,
            width=width,
            feature_dim=3,
            alpha_threshold=alpha_threshold,
            max_alpha=1.0,
        ),
    )

    assert atlas.active_start == (3, 3)
    assert atlas.active_stop == (6, 6)
    assert atlas.opacity_time_coeffs is not None
    assert atlas.spatial_precision_uv is not None
    expected_spatial_precision = torch.stack((q_uvt[:, 0], q_uvt[:, 1], q_uvt[:, 3]), dim=-1)
    torch.testing.assert_close(atlas.spatial_precision_uv, expected_spatial_precision)
    assert torch.allclose(atlas_image, dense_image.permute(0, 2, 3, 1), atol=1.0e-5, rtol=1.0e-5)


def test_uvt_tubes_to_projective_cell_atlas_rejects_spatial_depth_by_default() -> None:
    times = (torch.arange(5, dtype=torch.float32) - 2.0).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    depth_beta = depth_beta.clone()
    depth_beta[:, 0] = torch.tensor([0.20, -0.10], dtype=torch.float32)
    depth_beta[:, 1] = torch.tensor([-0.05, 0.15], dtype=torch.float32)

    with pytest.raises(ValueError, match="depth_beta"):
        uvt_tubes_to_projective_trace_cell_atlas(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            color,
            times,
            sigma_px=2.0,
            image_width=16,
            image_height=16,
            tile_size=8,
            uv_padding=8.0,
            stratify_visibility=False,
        )


def test_uvt_tubes_to_projective_cell_atlas_lowers_spatial_depth_to_depth_affine_uv() -> None:
    times = (torch.arange(5, dtype=torch.float32) - 2.0).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    depth_beta = depth_beta.clone()
    depth_beta[:, 0] = torch.tensor([0.20, -0.10], dtype=torch.float32)
    depth_beta[:, 1] = torch.tensor([-0.05, 0.15], dtype=torch.float32)
    depth_beta[:, 2] = torch.tensor([0.30, -0.20], dtype=torch.float32)

    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=16,
        image_height=16,
        tile_size=8,
        uv_padding=8.0,
        alpha_threshold=0.0,
        allow_depth_affine_uv=True,
        stratify_visibility=False,
    )

    assert atlas.depth_affine_uv is not None
    zeros = torch.zeros_like(depth_beta[:, 0])
    expected_affine = torch.stack(
        (
            depth_beta[:, 0],
            zeros,
            zeros,
            depth_beta[:, 1],
            zeros,
            zeros,
        ),
        dim=-1,
    )
    torch.testing.assert_close(atlas.depth_affine_uv, expected_affine)

    velocity = _uvt_center_velocity(q_uvt)
    center_u = ma[:, 0:1] + velocity[:, 0:1] * (times.reshape(1, -1) - ma[:, 2:3])
    center_v = ma[:, 1:2] + velocity[:, 1:2] * (times.reshape(1, -1) - ma[:, 2:3])
    offset_u = torch.tensor([[1.25], [-0.75]], dtype=torch.float32)
    offset_v = torch.tensor([[-0.50], [0.90]], dtype=torch.float32)
    sample_u = center_u + offset_u
    sample_v = center_v + offset_v
    expected_depth = (
        depth0.reshape(-1, 1)
        + depth_beta[:, 0:1] * (sample_u - ma[:, 0:1])
        + depth_beta[:, 1:2] * (sample_v - ma[:, 1:2])
        + depth_beta[:, 2:3] * (times.reshape(1, -1) - ma[:, 2:3])
    )

    depth = eval_projective_trace_cell_depth_at_uv_torch(atlas, times, sample_u, sample_v)

    torch.testing.assert_close(depth, expected_depth, atol=1.0e-6, rtol=1.0e-6)


def test_uvt_tube_temporal_opacity_interval_metal_matches_reference_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("projective interval cell Metal op unavailable")

    frames = 5
    height = 16
    width = 16
    times = (torch.arange(frames, dtype=torch.float32, device="mps") - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture(device="mps")
    q_uvt = q_uvt.clone()
    q_uvt[:, 5] += 0.5
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=width,
        image_height=height,
        tile_size=8,
        uv_padding=8.0,
        alpha_threshold=0.01,
    )
    config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        alpha_threshold=0.01,
    )
    ref = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=width,
        image_height=height,
        tile_size=8,
        sigma_px=2.0,
        alpha_cutoff=0.01,
    )
    metal = render_projective_trace_cell_interval_atlas_metal(atlas, times, config, sigma_px=2.0)

    assert torch.allclose(metal.cpu(), ref.cpu(), atol=2.0e-4, rtol=2.0e-4)


def test_uvt_tube_temporal_opacity_reference_backprops_to_qtt() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    q_uvt = q_uvt.detach().clone()
    q_uvt[:, 5] += 0.75
    q_uvt.requires_grad_(True)

    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=16,
        image_height=16,
        tile_size=8,
        uv_padding=16.0,
        alpha_threshold=0.0,
    )
    image = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=16,
        image_height=16,
        tile_size=8,
        sigma_px=2.0,
    )

    image.sum().backward()

    assert q_uvt.grad is not None
    assert torch.isfinite(q_uvt.grad).all()
    assert torch.all(q_uvt.grad[:, 5].abs() > 0.0)


def test_uvt_tubes_to_projective_cell_atlas_rejects_non_isotropic_spatial_precision() -> None:
    times = torch.arange(3, dtype=torch.float32).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    q_uvt = q_uvt.clone()
    q_uvt[0, 3] *= 1.5

    with pytest.raises(ValueError, match="isotropic sigma_px"):
        uvt_tubes_to_projective_trace_cell_atlas(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            color,
            times,
            sigma_px=2.0,
            image_width=16,
            image_height=16,
            tile_size=8,
        )


def test_uvt_tubes_to_projective_cell_atlas_allows_anisotropic_spatial_precision_when_requested() -> None:
    frames = 5
    height = 16
    width = 16
    alpha_threshold = 0.01
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _anisotropic_uvt_fixture()

    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=width,
        image_height=height,
        tile_size=8,
        uv_padding=0.0,
        alpha_threshold=alpha_threshold,
        require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True,
    )
    atlas_image = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=width,
        image_height=height,
        tile_size=8,
        sigma_px=2.0,
        alpha_cutoff=alpha_threshold,
    )
    dense_image, _alpha = dense_render_feature_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        FeatureTubeRenderConfig(
            frames=frames,
            height=height,
            width=width,
            feature_dim=3,
            alpha_threshold=alpha_threshold,
            max_alpha=1.0,
        ),
    )

    assert atlas.spatial_precision_uv is not None
    expected_spatial_precision = torch.stack((q_uvt[:, 0], q_uvt[:, 1], q_uvt[:, 3]), dim=-1)
    torch.testing.assert_close(atlas.spatial_precision_uv, expected_spatial_precision)
    assert torch.allclose(atlas_image, dense_image.permute(0, 2, 3, 1), atol=1.0e-5, rtol=1.0e-5)


def test_uvt_tube_anisotropic_interval_metal_matches_reference_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("projective interval cell Metal op unavailable")

    frames = 4
    height = 16
    width = 16
    alpha_threshold = 0.01
    times = (torch.arange(frames, dtype=torch.float32, device="mps") - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _anisotropic_uvt_fixture(device="mps")
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=width,
        image_height=height,
        tile_size=8,
        uv_padding=0.0,
        alpha_threshold=alpha_threshold,
        require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True,
    )
    config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=128,
        alpha_threshold=alpha_threshold,
    )
    ref = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=width,
        image_height=height,
        tile_size=8,
        sigma_px=2.0,
        alpha_cutoff=alpha_threshold,
    )
    metal = render_projective_trace_cell_interval_atlas_metal(atlas, times, config, sigma_px=2.0)

    assert torch.allclose(metal.cpu(), ref.cpu(), atol=2.0e-4, rtol=2.0e-4)


def test_uvt_tube_anisotropic_interval_autograd_backprops_to_spatial_precision_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")

    frames = 1
    height = 8
    width = 8
    alpha_threshold = 1.0e-6
    times = torch.zeros((frames,), dtype=torch.float32, device="mps").contiguous()
    ma = torch.tensor([[3.5, 3.5, 0.0]], dtype=torch.float32, device="mps")
    q_uvt = torch.tensor([[0.25, 0.04, 0.0, 0.12, 0.0, 0.0]], dtype=torch.float32, device="mps")
    q_uvt.requires_grad_(True)
    depth0 = torch.tensor([1.0], dtype=torch.float32, device="mps")
    depth_beta = torch.zeros((1, 3), dtype=torch.float32, device="mps")
    opacity = torch.tensor([0.7], dtype=torch.float32, device="mps")
    color = torch.tensor([[1.0, 0.2, 0.1]], dtype=torch.float32, device="mps")

    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=width,
        image_height=height,
        tile_size=8,
        uv_padding=0.0,
        alpha_threshold=alpha_threshold,
        require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True,
    )
    config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=128,
        alpha_threshold=alpha_threshold,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )

    rendered = render_projective_cell_interval_atlas_metal_backward(atlas, times, config, sigma_px=2.0)
    rendered[..., 0].sum().backward()

    assert q_uvt.grad is not None
    assert float(q_uvt.grad[0, [0, 3]].abs().sum().detach().cpu().item()) > 0.0
    assert float(q_uvt.grad[0, [2, 4, 5]].abs().sum().detach().cpu().item()) == pytest.approx(0.0)


def test_projective_interval_backend_builds_atlas_from_compatible_uvt_tubes() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "depth_padding": 0.0,
                "check_visibility": True,
            },
        },
    }

    atlas = make_projective_cell_interval_atlas_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )

    assert atlas.coeffs.shape == (2, 9)
    assert atlas.source_primitive_ids == (0, 1)
    assert atlas.active_start == (0, 0)
    assert atlas.active_stop == (frames, frames)
    assert atlas.cells


def test_projective_interval_backend_allows_anisotropic_spatial_precision_with_config() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _anisotropic_uvt_fixture()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 0.0,
                "depth_padding": 0.0,
                "check_visibility": True,
                "allow_anisotropic_spatial_precision": True,
            },
        },
    }

    atlas = make_projective_cell_interval_atlas_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )

    assert atlas.spatial_precision_uv is not None
    expected_spatial_precision = torch.stack((q_uvt[:, 0], q_uvt[:, 1], q_uvt[:, 3]), dim=-1)
    torch.testing.assert_close(atlas.spatial_precision_uv, expected_spatial_precision)
    assert atlas.cells


def test_projective_interval_backend_builds_trainer_state_from_uvt_tubes() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "refresh_every": 2,
                "fallback_render_mode": "mixed",
                "max_fallback_fraction": 0.25,
            },
        },
    }

    state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
        primitive_ids=(11, 13),
    )
    ref = state.render_reference_with_fallback()

    assert state.atlas.source_primitive_ids == (11, 13)
    assert state.refresh_every == 2
    assert state.fallback_render_mode == "mixed"
    assert state.max_fallback_fraction == 0.25
    assert state.config.frames == frames
    assert state.config.tile_t == frames
    assert ref.shape == (frames, 16, 16, 3)
    assert float(ref.sum().item()) > 0.0


def test_projective_interval_backend_reports_anisotropic_auto_support_padding() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _anisotropic_uvt_fixture()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 0.0,
                "refresh_every": 2,
                "fallback_render_mode": "mixed",
                "allow_anisotropic_spatial_precision": True,
            },
        },
    }

    state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )

    assert state.uv_padding == 0.0
    assert state.support_uv_padding > 0.0
    assert state.atlas.spatial_precision_uv is not None


def test_projective_interval_live_atlas_reuses_cells_with_current_tensors() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
            },
        },
    }
    reference = make_projective_cell_interval_atlas_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )
    shifted_ma = ma.clone()
    shifted_ma[:, 0] += 0.25
    live = make_projective_cell_interval_live_atlas_from_uvt_tubes(
        shifted_ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        cfg,
        reference_atlas=reference,
    )
    rebuilt = make_projective_cell_interval_atlas_from_uvt_tubes(
        shifted_ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )

    assert live.cells is reference.cells
    assert live.source_primitive_ids == reference.source_primitive_ids
    assert torch.allclose(live.coeffs, rebuilt.coeffs)
    assert torch.allclose(live.opacity_time_coeffs, rebuilt.opacity_time_coeffs)
    torch.testing.assert_close(live.spatial_precision_uv, rebuilt.spatial_precision_uv)
    assert torch.allclose(live.color, color)


def test_projective_interval_live_atlas_allows_anisotropic_spatial_precision_with_config() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _anisotropic_uvt_fixture()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 0.0,
                "allow_anisotropic_spatial_precision": True,
            },
        },
    }
    reference = make_projective_cell_interval_atlas_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )
    shifted_ma = ma.clone()
    shifted_ma[:, 0] += 0.25
    live = make_projective_cell_interval_live_atlas_from_uvt_tubes(
        shifted_ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        cfg,
        reference_atlas=reference,
    )
    rebuilt = make_projective_cell_interval_atlas_from_uvt_tubes(
        shifted_ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )

    assert live.cells is reference.cells
    assert torch.allclose(live.coeffs, rebuilt.coeffs)
    torch.testing.assert_close(live.spatial_precision_uv, rebuilt.spatial_precision_uv)


def test_projective_interval_live_atlas_preserves_depth_affine_uv_from_reference() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    depth_beta = depth_beta.clone()
    depth_beta[:, 0] = torch.tensor([0.20, -0.10], dtype=torch.float32)
    depth_beta[:, 1] = torch.tensor([-0.05, 0.15], dtype=torch.float32)
    depth_beta[:, 2] = torch.tensor([0.30, -0.20], dtype=torch.float32)
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "allow_depth_affine_uv": True,
            },
        },
    }
    reference = make_projective_cell_interval_atlas_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )
    shifted_ma = ma.clone()
    shifted_ma[:, 0] += 0.25
    shifted_depth_beta = depth_beta.clone()
    shifted_depth_beta[:, 0] += torch.tensor([0.03, -0.02], dtype=torch.float32)
    live = make_projective_cell_interval_live_atlas_from_uvt_tubes(
        shifted_ma,
        q_uvt,
        depth0,
        shifted_depth_beta,
        opacity,
        color,
        cfg,
        reference_atlas=reference,
    )
    rebuilt = make_projective_cell_interval_atlas_from_uvt_tubes(
        shifted_ma,
        q_uvt,
        depth0,
        shifted_depth_beta,
        opacity,
        color,
        times,
        cfg,
    )

    assert live.cells is reference.cells
    assert live.depth_affine_uv is not None
    torch.testing.assert_close(live.depth_affine_uv, rebuilt.depth_affine_uv)
    torch.testing.assert_close(live.coeffs, rebuilt.coeffs)


def test_projective_interval_live_atlas_refresh_repairs_stale_support() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    ma = ma[:1].clone()
    q_uvt = q_uvt[:1].clone()
    depth0 = depth0[:1].clone()
    depth_beta = depth_beta[:1].clone()
    opacity = opacity[:1].clone()
    color = color[:1].clone()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 0.0,
                "refresh_every": 99,
            },
        },
    }
    state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )
    shifted_ma = ma.clone()
    shifted_ma[:, 0] += 9.0
    state.atlas = make_projective_cell_interval_live_atlas_from_uvt_tubes(
        shifted_ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        cfg,
        reference_atlas=state.atlas,
    )

    before = projective_trace_cell_atlas_coverage_report(
        state.atlas,
        times,
        image_width=16,
        image_height=16,
        tile_size=8,
        uv_padding=0.0,
    )
    refresh = state.refresh(force=False)

    assert before.stale is True
    assert refresh.rebinned is True
    assert refresh.after.stale is False


def test_projective_interval_support_guard_avoids_rebin_for_covered_motion() -> None:
    frames = 5
    times = (torch.arange(frames, dtype=torch.float32) - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture()
    ma = ma[:1].clone()
    q_uvt = q_uvt[:1].clone()
    depth0 = depth0[:1].clone()
    depth_beta = depth_beta[:1].clone()
    opacity = opacity[:1].clone()
    color = color[:1].clone()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": 16,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 0.0,
                "support_guard_padding": 16.0,
                "refresh_policy": "measured",
                "refresh_every": 99,
            },
        },
    }
    state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
    )
    shifted_ma = ma.clone()
    shifted_ma[:, 0] += 9.0
    state.atlas = make_projective_cell_interval_live_atlas_from_uvt_tubes(
        shifted_ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        cfg,
        reference_atlas=state.atlas,
    )

    before = projective_trace_cell_atlas_coverage_report(
        state.atlas,
        times,
        image_width=16,
        image_height=16,
        tile_size=8,
        uv_padding=0.0,
    )
    refresh = state.refresh(force=False)

    assert state.uv_padding == 0.0
    assert state.support_uv_padding == 16.0
    assert before.stale is False
    assert refresh.rebinned is False
    assert refresh.after.stale is False


def test_projective_interval_budgeted_support_guard_respects_tile_capacity() -> None:
    frames = 1
    sigma_px = 2.0
    inv_sigma2 = 1.0 / (sigma_px * sigma_px)
    per_side = 20
    times = torch.zeros((frames,), dtype=torch.float32).contiguous()
    ma = torch.cat(
        (
            torch.tensor([[4.0, 4.0, 0.0]], dtype=torch.float32).repeat(per_side, 1),
            torch.tensor([[12.0, 4.0, 0.0]], dtype=torch.float32).repeat(per_side, 1),
        ),
        dim=0,
    ).contiguous()
    tube_count = int(ma.shape[0])
    q_uvt = torch.zeros((tube_count, 6), dtype=torch.float32)
    q_uvt[:, 0] = inv_sigma2
    q_uvt[:, 3] = inv_sigma2
    depth0 = torch.arange(tube_count, dtype=torch.float32)
    depth_beta = torch.zeros((tube_count, 3), dtype=torch.float32)
    opacity = torch.full((tube_count,), 0.50, dtype=torch.float32)
    color = torch.stack(
        (
            torch.linspace(0.1, 0.9, tube_count),
            torch.linspace(0.9, 0.1, tube_count),
            torch.full((tube_count,), 0.25),
        ),
        dim=-1,
    ).contiguous()

    def cfg_for(policy: str) -> dict[str, Any]:
        return {
            "data": {
                "max_frames": frames,
                "target_size": 16,
            },
            "feature_uvt": {
                "feature_dim": 3,
                "tile_t": frames,
                "tile_capacity": 32,
                "alpha_threshold": 0.01,
                "max_alpha": 1.0,
                "projective_interval": {
                    "enabled": True,
                    "sigma_px": sigma_px,
                    "tile_size": 8,
                    "uv_padding": 0.0,
                    "support_guard_padding": 4.5,
                    "support_guard_policy": policy,
                    "support_guard_bisect_steps": 8,
                    "check_visibility": False,
                },
            },
        }

    fixed_atlas = make_projective_cell_interval_atlas_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg_for("fixed"),
    )
    fixed_bins = pack_projective_trace_tile_time_bins(
        fixed_atlas.cells,
        image_width=16,
        image_height=16,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=32,
        allow_fallback_cells=True,
    )
    state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg_for("budgeted"),
    )
    budgeted_bins = pack_projective_trace_tile_time_bins(
        state.atlas.cells,
        image_width=16,
        image_height=16,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=32,
        allow_fallback_cells=True,
    )

    assert int(fixed_bins.tile_overflow.sum().item()) > 0
    assert int(budgeted_bins.tile_overflow.sum().item()) == 0
    assert state.uv_padding == 0.0
    assert 0.0 < float(state.support_uv_padding) < 4.5


def test_projective_interval_local_budgeted_guard_preserves_headroom_tiles() -> None:
    frames = 1
    sigma_px = 2.0
    inv_sigma2 = 1.0 / (sigma_px * sigma_px)
    crowded_per_group = 20
    times = torch.zeros((frames,), dtype=torch.float32).contiguous()
    ma = torch.cat(
        (
            torch.tensor([[4.0, 4.0, 0.0]], dtype=torch.float32).repeat(crowded_per_group, 1),
            torch.tensor([[12.0, 4.0, 0.0]], dtype=torch.float32).repeat(crowded_per_group, 1),
            torch.tensor([[28.0, 4.0, 0.0]], dtype=torch.float32),
        ),
        dim=0,
    ).contiguous()
    tube_count = int(ma.shape[0])
    q_uvt = torch.zeros((tube_count, 6), dtype=torch.float32)
    q_uvt[:, 0] = inv_sigma2
    q_uvt[:, 3] = inv_sigma2
    depth0 = torch.arange(tube_count, dtype=torch.float32)
    depth_beta = torch.zeros((tube_count, 3), dtype=torch.float32)
    opacity = torch.full((tube_count,), 0.50, dtype=torch.float32)
    color = torch.ones((tube_count, 3), dtype=torch.float32).contiguous()

    def cfg_for(policy: str) -> dict[str, Any]:
        return {
            "data": {
                "max_frames": frames,
                "target_size": 32,
            },
            "feature_uvt": {
                "feature_dim": 3,
                "tile_t": frames,
                "tile_capacity": 32,
                "alpha_threshold": 0.01,
                "max_alpha": 1.0,
                "projective_interval": {
                    "enabled": True,
                    "sigma_px": sigma_px,
                    "tile_size": 8,
                    "uv_padding": 0.0,
                    "support_guard_padding": 4.5,
                    "support_guard_policy": policy,
                    "support_guard_bisect_steps": 8,
                    "check_visibility": False,
                },
            },
        }

    base_atlas = make_projective_cell_interval_atlas_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg_for("fixed") | {
            "feature_uvt": {
                **cfg_for("fixed")["feature_uvt"],
                "projective_interval": {
                    **cfg_for("fixed")["feature_uvt"]["projective_interval"],
                    "support_guard_padding": 0.0,
                },
            }
        },
    )
    local_state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg_for("local_budgeted"),
    )
    local_bins = pack_projective_trace_tile_time_bins(
        local_state.atlas.cells,
        image_width=32,
        image_height=32,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=32,
        allow_fallback_cells=True,
    )

    base_tile2_ids = {
        primitive_id
        for cell in base_atlas.cells
        if int(cell.tile_u) == 2
        for primitive_id in cell.ordered_primitive_ids
    }
    local_tile2_ids = {
        primitive_id
        for cell in local_state.atlas.cells
        if int(cell.tile_u) == 2
        for primitive_id in cell.ordered_primitive_ids
    }
    local_tile0_counts = [
        len(cell.ordered_primitive_ids)
        for cell in local_state.atlas.cells
        if int(cell.tile_u) == 0
    ]

    assert int(local_bins.tile_overflow.sum().item()) == 0
    assert local_state.support_uv_padding == 4.5
    assert base_tile2_ids == set()
    assert len(local_tile2_ids) > len(base_tile2_ids)
    assert crowded_per_group in local_tile2_ids
    assert (2 * crowded_per_group) in local_tile2_ids
    assert max(local_tile0_counts) == 20


def test_projective_interval_trace_budgeted_guard_spends_crowded_tile_headroom() -> None:
    frames = 1
    sigma_px = 2.0
    inv_sigma2 = 1.0 / (sigma_px * sigma_px)
    crowded_per_group = 20
    times = torch.zeros((frames,), dtype=torch.float32).contiguous()
    ma = torch.cat(
        (
            torch.tensor([[4.0, 4.0, 0.0]], dtype=torch.float32).repeat(crowded_per_group, 1),
            torch.tensor([[12.0, 4.0, 0.0]], dtype=torch.float32).repeat(crowded_per_group, 1),
            torch.tensor([[28.0, 4.0, 0.0]], dtype=torch.float32),
        ),
        dim=0,
    ).contiguous()
    tube_count = int(ma.shape[0])
    q_uvt = torch.zeros((tube_count, 6), dtype=torch.float32)
    q_uvt[:, 0] = inv_sigma2
    q_uvt[:, 3] = inv_sigma2
    depth0 = torch.arange(tube_count, dtype=torch.float32)
    depth_beta = torch.zeros((tube_count, 3), dtype=torch.float32)
    opacity = torch.full((tube_count,), 0.50, dtype=torch.float32)
    color = torch.ones((tube_count, 3), dtype=torch.float32).contiguous()

    def cfg_for(policy: str) -> dict[str, Any]:
        return {
            "data": {
                "max_frames": frames,
                "target_size": 32,
            },
            "feature_uvt": {
                "feature_dim": 3,
                "tile_t": frames,
                "tile_capacity": 32,
                "alpha_threshold": 0.01,
                "max_alpha": 1.0,
                "projective_interval": {
                    "enabled": True,
                    "sigma_px": sigma_px,
                    "tile_size": 8,
                    "uv_padding": 0.0,
                    "support_guard_padding": 4.5,
                    "support_guard_policy": policy,
                    "support_guard_bisect_steps": 8,
                    "check_visibility": False,
                },
            },
        }

    local_state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg_for("local_budgeted"),
    )
    trace_state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg_for("trace_budgeted"),
    )
    trace_bins = pack_projective_trace_tile_time_bins(
        trace_state.atlas.cells,
        image_width=32,
        image_height=32,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=32,
        allow_fallback_cells=True,
    )

    local_tile0_ids = {
        primitive_id
        for cell in local_state.atlas.cells
        if int(cell.tile_u) == 0
        for primitive_id in cell.ordered_primitive_ids
    }
    trace_tile0_ids = {
        primitive_id
        for cell in trace_state.atlas.cells
        if int(cell.tile_u) == 0
        for primitive_id in cell.ordered_primitive_ids
    }
    trace_tile0_counts = [
        len(cell.ordered_primitive_ids)
        for cell in trace_state.atlas.cells
        if int(cell.tile_u) == 0
    ]

    assert int(trace_bins.tile_overflow.sum().item()) == 0
    assert trace_state.support_uv_padding == 4.5
    assert len(local_tile0_ids) == crowded_per_group
    assert len(trace_tile0_ids) == 32
    assert set(range(crowded_per_group)).issubset(trace_tile0_ids)
    assert set(range(crowded_per_group, crowded_per_group + 12)).issubset(trace_tile0_ids)
    assert max(trace_tile0_counts) == 32


def test_projective_interval_slack_budgeted_guard_prefers_nearest_support_event() -> None:
    frames = 1
    sigma_px = 2.0
    inv_sigma2 = 1.0 / (sigma_px * sigma_px)
    base_count = 20
    far_extra_count = 12
    near_extra_count = 12
    times = torch.zeros((frames,), dtype=torch.float32).contiguous()
    ma = torch.cat(
        (
            torch.tensor([[4.0, 4.0, 0.0]], dtype=torch.float32).repeat(base_count, 1),
            torch.tensor([[12.0, 4.0, 0.0]], dtype=torch.float32).repeat(far_extra_count, 1),
            torch.tensor([[8.2, 4.0, 0.0]], dtype=torch.float32).repeat(near_extra_count, 1),
        ),
        dim=0,
    ).contiguous()
    tube_count = int(ma.shape[0])
    q_uvt = torch.zeros((tube_count, 6), dtype=torch.float32)
    q_uvt[:, 0] = inv_sigma2
    q_uvt[:, 3] = inv_sigma2
    depth0 = torch.arange(tube_count, dtype=torch.float32)
    depth_beta = torch.zeros((tube_count, 3), dtype=torch.float32)
    opacity = torch.full((tube_count,), 0.50, dtype=torch.float32)
    color = torch.ones((tube_count, 3), dtype=torch.float32).contiguous()

    def cfg_for(policy: str) -> dict[str, Any]:
        return {
            "data": {
                "max_frames": frames,
                "target_size": 32,
            },
            "feature_uvt": {
                "feature_dim": 3,
                "tile_t": frames,
                "tile_capacity": 32,
                "alpha_threshold": 0.01,
                "max_alpha": 1.0,
                "projective_interval": {
                    "enabled": True,
                    "sigma_px": sigma_px,
                    "tile_size": 8,
                    "uv_padding": 0.0,
                    "support_guard_padding": 4.5,
                    "support_guard_policy": policy,
                    "support_guard_bisect_steps": 8,
                    "check_visibility": False,
                },
            },
        }

    trace_state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg_for("trace_budgeted"),
    )
    slack_state = make_projective_cell_interval_trainer_state_from_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg_for("slack_budgeted"),
    )
    slack_bins = pack_projective_trace_tile_time_bins(
        slack_state.atlas.cells,
        image_width=32,
        image_height=32,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=32,
        allow_fallback_cells=True,
    )

    trace_tile0_ids = {
        primitive_id
        for cell in trace_state.atlas.cells
        if int(cell.tile_u) == 0 and int(cell.tile_v) == 0
        for primitive_id in cell.ordered_primitive_ids
    }
    slack_tile0_ids = {
        primitive_id
        for cell in slack_state.atlas.cells
        if int(cell.tile_u) == 0 and int(cell.tile_v) == 0
        for primitive_id in cell.ordered_primitive_ids
    }
    far_ids = set(range(base_count, base_count + far_extra_count))
    near_ids = set(range(base_count + far_extra_count, tube_count))

    assert int(slack_bins.tile_overflow.sum().item()) == 0
    assert slack_state.support_guard_policy == "slack_budgeted"
    assert set(range(base_count)).issubset(slack_tile0_ids)
    assert far_ids.issubset(trace_tile0_ids)
    assert near_ids.isdisjoint(trace_tile0_ids)
    assert near_ids.issubset(slack_tile0_ids)
    assert far_ids.isdisjoint(slack_tile0_ids)


def test_projective_interval_measured_cache_repairs_stale_support_in_render_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")

    frames = 5
    size = 16
    device = torch.device("mps")
    times = _projective_interval_times(frames, device)
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture(device=device)
    ma = ma[:1].clone()
    q_uvt = q_uvt[:1].clone()
    depth0 = depth0[:1].clone()
    depth_beta = depth_beta[:1].clone()
    opacity = opacity[:1].clone()
    color = color[:1].clone()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": size,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 0.0,
                "refresh_policy": "measured",
                "refresh_every": 99,
                "fallback_render_mode": "mixed",
            },
        },
    }
    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    cache = _ProjectiveIntervalFeatureRenderCache()

    first = _render_projective_interval_feature_tubes_autograd(
        ma=ma,
        q_uvt=q_uvt,
        depth0=depth0,
        depth_beta=depth_beta,
        opacity=opacity,
        feature=color,
        cfg=cfg,
        feature_config=feature_config,
        uvt_config=uvt_config,
        times=times,
        cache=cache,
        global_step=0,
        refresh_every=99,
        refresh_policy="measured",
    )
    shifted_ma = ma.clone()
    shifted_ma[:, 0] += 9.0
    second = _render_projective_interval_feature_tubes_autograd(
        ma=shifted_ma,
        q_uvt=q_uvt,
        depth0=depth0,
        depth_beta=depth_beta,
        opacity=opacity,
        feature=color,
        cfg=cfg,
        feature_config=feature_config,
        uvt_config=uvt_config,
        times=times,
        cache=cache,
        global_step=1,
        refresh_every=99,
        refresh_policy="measured",
    )

    assert first.feature_image.shape == (frames, 3, size, size)
    assert second.feature_image.shape == (frames, 3, size, size)
    assert cache.rebuild_count == 1
    assert cache.live_update_count == 1
    assert cache.staleness_check_count == 1
    assert cache.stale_refresh_count == 1
    assert cache.support_rebin_count == 1


def test_projective_interval_measured_cache_survives_optimizer_motion_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")

    frames = 5
    size = 16
    device = torch.device("mps")
    times = _projective_interval_times(frames, device)
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture(device=device)
    ma_param = torch.nn.Parameter(ma[:1].clone())
    q_uvt = q_uvt[:1].clone()
    depth0 = depth0[:1].clone()
    depth_beta = depth_beta[:1].clone()
    opacity = opacity[:1].clone()
    color = color[:1].clone()
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": size,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 0.0,
                "refresh_policy": "measured",
                "refresh_every": 99,
                "fallback_render_mode": "mixed",
            },
        },
    }
    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    optimizer = torch.optim.SGD([ma_param], lr=0.5)
    cache = _ProjectiveIntervalFeatureRenderCache()
    target_x = ma_param.detach()[0, 0] + 9.0

    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        render = _render_projective_interval_feature_tubes_autograd(
            ma=ma_param,
            q_uvt=q_uvt,
            depth0=depth0,
            depth_beta=depth_beta,
            opacity=opacity,
            feature=color,
            cfg=cfg,
            feature_config=feature_config,
            uvt_config=uvt_config,
            times=times,
            cache=cache,
            global_step=step,
            refresh_every=99,
            refresh_policy="measured",
        )
        loss = 0.001 * render.feature_image.square().mean() + (ma_param[0, 0] - target_x).square()
        loss.backward()
        optimizer.step()

    assert float((ma_param.detach()[0, 0] - target_x).abs().cpu()) < 0.05
    assert cache.rebuild_count == 1
    assert cache.live_update_count == 3
    assert cache.alpha_render_count == 4
    assert cache.staleness_check_count == 3
    assert cache.stale_refresh_count == 1
    assert cache.support_rebin_count == 1


def test_uvt_tube_produced_atlas_interval_metal_matches_reference_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("projective interval cell Metal op unavailable")

    frames = 4
    height = 16
    width = 16
    sigma_px = 2.0
    alpha_threshold = 0.01
    times = (torch.arange(frames, dtype=torch.float32, device="mps") - 0.5 * float(frames - 1)).contiguous()
    ma, q_uvt, depth0, depth_beta, opacity, color = _affine_uvt_fixture(device="mps")
    support_radius = math.sqrt(2.0 * sigma_px * sigma_px * math.log(float(opacity.max().cpu().item()) / alpha_threshold))
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=sigma_px,
        image_width=width,
        image_height=height,
        tile_size=8,
        uv_padding=math.ceil(support_radius) + 1.0,
        alpha_threshold=alpha_threshold,
    )
    config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=128,
        alpha_threshold=alpha_threshold,
    )

    ref = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=width,
        image_height=height,
        tile_size=8,
        sigma_px=sigma_px,
        alpha_cutoff=alpha_threshold,
    )
    metal = render_projective_trace_cell_interval_atlas_metal(
        atlas,
        times,
        config,
        sigma_px=sigma_px,
    )

    assert torch.allclose(metal.cpu(), ref.cpu(), atol=2.0e-4, rtol=2.0e-4)


def test_projective_interval_trainer_bridge_backprops_with_locked_spatial_precision_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")

    frames = 4
    size = 16
    feature_config = FeatureTubeRenderConfig(
        frames=frames,
        height=size,
        width=size,
        feature_dim=3,
        alpha_threshold=0.01,
        max_alpha=1.0,
    )
    model = FeatureScreenTimeTubeModel(3, feature_config, seed=7, device="mps")
    locked_precision = _lock_projective_interval_spatial_precision(model, sigma_px=2.0)
    times = _projective_interval_times(frames, torch.device("mps"))
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": size,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "fallback_render_mode": "mixed",
            },
        },
    }
    uvt_config = UVTRenderConfig(
        height=size,
        width=size,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=128,
        alpha_threshold=0.01,
        max_alpha=1.0,
    )
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
    assert float(q_uvt[:, 1].detach().abs().max().cpu()) == 0.0

    render = _render_projective_interval_feature_tubes_autograd(
        ma=ma,
        q_uvt=q_uvt,
        depth0=depth0.detach(),
        depth_beta=depth_beta.detach(),
        opacity=opacity,
        feature=feature,
        cfg=cfg,
        feature_config=feature_config,
        uvt_config=uvt_config,
        times=times,
    )
    loss = render.feature_image.square().mean() + 0.25 * render.alpha.mean()
    loss.backward()

    assert render.feature_image.shape == (frames, 3, size, size)
    assert render.alpha.shape == (frames, size, size)
    assert locked_precision == pytest.approx(0.25)
    assert model.center_uv.grad is not None and float(model.center_uv.grad.detach().abs().sum().cpu()) > 0.0
    assert model.center_t.grad is not None and float(model.center_t.grad.detach().abs().sum().cpu()) > 0.0
    assert model.velocity_uv.grad is not None and float(model.velocity_uv.grad.detach().abs().sum().cpu()) > 0.0
    assert model.raw_feature.grad is not None and float(model.raw_feature.grad.detach().abs().sum().cpu()) > 0.0
    assert model.raw_opacity.grad is not None and float(model.raw_opacity.grad.detach().abs().sum().cpu()) > 0.0
    assert model.raw_precision.grad is not None
    assert float(model.raw_precision.grad[:, 0:2].detach().abs().max().cpu()) == 0.0
    assert float(model.raw_precision.grad[:, 2].detach().abs().sum().cpu()) > 0.0
    if model.raw_spatial_correlation.grad is not None:
        assert float(model.raw_spatial_correlation.grad.detach().abs().max().cpu()) == 0.0


def test_projective_interval_trainer_bridge_can_train_anisotropic_spatial_precision_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")

    frames = 4
    size = 16
    feature_config = FeatureTubeRenderConfig(
        frames=frames,
        height=size,
        width=size,
        feature_dim=3,
        alpha_threshold=0.01,
        max_alpha=1.0,
    )
    model = FeatureScreenTimeTubeModel(3, feature_config, seed=7, device="mps")
    times = _projective_interval_times(frames, torch.device("mps"))
    cfg = {
        "data": {
            "max_frames": frames,
            "target_size": size,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": frames,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 1.0,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 0.0,
                "fallback_render_mode": "mixed",
                "allow_anisotropic_spatial_precision": True,
            },
        },
    }
    uvt_config = UVTRenderConfig(
        height=size,
        width=size,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=frames,
        tile_capacity=128,
        alpha_threshold=0.01,
        max_alpha=1.0,
    )
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()

    render = _render_projective_interval_feature_tubes_autograd(
        ma=ma,
        q_uvt=q_uvt,
        depth0=depth0.detach(),
        depth_beta=depth_beta.detach(),
        opacity=opacity,
        feature=feature,
        cfg=cfg,
        feature_config=feature_config,
        uvt_config=uvt_config,
        times=times,
    )
    x_weight = torch.linspace(-1.0, 1.0, size, device="mps").view(1, 1, 1, size)
    y_weight = torch.linspace(-0.7, 1.3, size, device="mps").view(1, 1, size, 1)
    skew_weight = x_weight * y_weight
    loss = (render.feature_image[:, :1] * skew_weight).sum() + render.feature_image.square().mean() + 0.25 * render.alpha.mean()
    loss.backward()

    assert render.feature_image.shape == (frames, 3, size, size)
    assert model.raw_precision.grad is not None
    assert float(model.raw_precision.grad[:, 0:2].detach().abs().sum().cpu()) > 0.0
    assert model.raw_spatial_correlation.grad is not None
    assert float(model.raw_spatial_correlation.grad.detach().abs().sum().cpu()) > 0.0


def test_feature_overfit_trainer_routes_projective_interval_producer_if_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")
    monkeypatch.setenv("STAR_UVT_TILE_X", "8")
    monkeypatch.setenv("STAR_UVT_TILE_Y", "8")
    monkeypatch.setenv("STAR_UVT_TILE_T", "2")
    monkeypatch.setenv("STAR_UVT_TILE_CAPACITY", "128")

    class _Sequence:
        def __init__(self, frames: torch.Tensor) -> None:
            self.frames = frames

    frames = 4
    size = 16
    target = torch.linspace(0.0, 1.0, frames * 3 * size * size, dtype=torch.float32).reshape(frames, 3, size, size)

    def _load_sequence(_cfg: dict[str, Any], device: torch.device) -> _Sequence:
        return _Sequence(target.to(device=device))

    monkeypatch.setattr(feature_overfit_trainer, "_load_training_sequence", _load_sequence)
    cfg: dict[str, Any] = {
        "data": {
            "video_path": str(tmp_path / "synthetic.mp4"),
            "start_seconds": None,
            "fps": None,
            "duration_seconds": None,
            "image_crop_mode": "center",
            "target_size": size,
            "max_frames": frames,
        },
        "train": {
            "steps": 3,
            "lr": 0.01,
            "device": "mps",
            "seed": 3,
            "frame_chunk_size": None,
            "require_loss_decrease": False,
            "require_gradient_flow": False,
            "require_no_tile_overflow": False,
        },
        "feature_uvt": {
            "tube_count": 4,
            "feature_dim": 3,
            "tile_t": 2,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 0.99,
            "render_mode": "feature_direct_atomic",
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "refresh_policy": "measured",
                "refresh_every": 2,
                "fallback_render_mode": "mixed",
            },
        },
        "colorize": {
            "hidden_dim": None,
            "activation": "sigmoid",
            "pre_norm": False,
            "weight_init": "kaiming",
            "weight_init_gain": 1.0,
        },
        "output": {
            "out_json": str(tmp_path / "row.json"),
            "contact_sheet": None,
            "contact_sheet_frames": frames,
            "contact_sheet_mode": "grid",
            "side_by_side_video": None,
            "side_by_side_fps": 8,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "unit",
            "wandb_run_name": "projective-interval-unit",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }

    row = feature_overfit_trainer.run_training(cfg)

    assert row["projective_interval_enabled"] is True
    assert row["projective_interval_runtime_enabled"] is True
    assert row["projective_interval_spatial_precision_locked"] is True
    assert row["projective_interval_locked_spatial_precision"] == pytest.approx(0.25)
    assert row["projective_interval_alpha_render_mode"] == "white_trace"
    assert row["projective_interval_refresh_policy"] == "measured"
    assert row["projective_interval_refresh_every"] == 2
    assert row["projective_interval_cache_rebuilds"] == 1
    assert row["projective_interval_cache_live_updates"] == 2
    assert row["projective_interval_cache_alpha_renders"] == 3
    assert row["projective_interval_cache_staleness_checks"] == 2
    assert row["projective_interval_cache_stale_refreshes"] == 0
    assert row["projective_interval_cache_support_rebins"] == 0
    assert row["projective_interval_cache_visibility_stratifications"] == 0
    assert row["projective_interval_cache_fallback_marks"] == 0
    assert row["raw_feature_grad_seen"] is True
    assert row["center_uv_grad_seen"] is True
    assert row["center_t_grad_seen"] is True
    assert row["velocity_uv_grad_seen"] is True
    assert row["raw_precision_grad_seen"] is True
    assert row["raw_opacity_grad_seen"] is True


def test_feature_overfit_trainer_measured_policy_reuses_atlas_vs_cadence_if_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")
    monkeypatch.setenv("STAR_UVT_TILE_X", "8")
    monkeypatch.setenv("STAR_UVT_TILE_Y", "8")
    monkeypatch.setenv("STAR_UVT_TILE_T", "2")
    monkeypatch.setenv("STAR_UVT_TILE_CAPACITY", "128")

    class _Sequence:
        def __init__(self, frames: torch.Tensor) -> None:
            self.frames = frames

    frames = 4
    size = 16
    target = torch.linspace(0.0, 1.0, frames * 3 * size * size, dtype=torch.float32).reshape(frames, 3, size, size)

    def _load_sequence(_cfg: dict[str, Any], device: torch.device) -> _Sequence:
        return _Sequence(target.to(device=device))

    monkeypatch.setattr(feature_overfit_trainer, "_load_training_sequence", _load_sequence)
    base_cfg: dict[str, Any] = {
        "data": {
            "video_path": str(tmp_path / "synthetic.mp4"),
            "start_seconds": None,
            "fps": None,
            "duration_seconds": None,
            "image_crop_mode": "center",
            "target_size": size,
            "max_frames": frames,
        },
        "train": {
            "steps": 4,
            "lr": 0.01,
            "device": "mps",
            "seed": 11,
            "frame_chunk_size": None,
            "require_loss_decrease": False,
            "require_gradient_flow": False,
            "require_no_tile_overflow": False,
        },
        "feature_uvt": {
            "tube_count": 4,
            "feature_dim": 3,
            "tile_t": 2,
            "tile_capacity": 128,
            "alpha_threshold": 0.01,
            "max_alpha": 0.99,
            "render_mode": "feature_direct_atomic",
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "refresh_every": 2,
                "fallback_render_mode": "mixed",
            },
        },
        "colorize": {
            "hidden_dim": None,
            "activation": "sigmoid",
            "pre_norm": False,
            "weight_init": "kaiming",
            "weight_init_gain": 1.0,
        },
        "output": {
            "out_json": str(tmp_path / "base.json"),
            "contact_sheet": None,
            "contact_sheet_frames": frames,
            "contact_sheet_mode": "grid",
            "side_by_side_video": None,
            "side_by_side_fps": 8,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "unit",
            "wandb_run_name": "projective-interval-cache-ab-unit",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }

    def _run(policy: str) -> dict[str, Any]:
        cfg = copy.deepcopy(base_cfg)
        cfg["feature_uvt"]["projective_interval"]["refresh_policy"] = policy
        cfg["output"]["out_json"] = str(tmp_path / f"{policy}.json")
        cfg["logging"]["wandb_run_name"] = f"projective-interval-cache-{policy}-unit"
        return feature_overfit_trainer.run_training(cfg)

    cadence = _run("cadence")
    measured = _run("measured")

    assert cadence["projective_interval_refresh_policy"] == "cadence"
    assert measured["projective_interval_refresh_policy"] == "measured"
    assert cadence["projective_interval_cache_rebuilds"] == 2
    assert cadence["projective_interval_cache_live_updates"] == 2
    assert cadence["projective_interval_cache_staleness_checks"] == 2
    assert measured["projective_interval_cache_rebuilds"] == 1
    assert measured["projective_interval_cache_live_updates"] == 3
    assert measured["projective_interval_cache_staleness_checks"] == 3
    assert measured["projective_interval_cache_stale_refreshes"] == 0
    assert measured["projective_interval_cache_alpha_renders"] == cadence["projective_interval_cache_alpha_renders"] == 4
    assert measured["end_loss"] == pytest.approx(cadence["end_loss"], abs=1.0e-5)
    assert measured["losses"] == pytest.approx(cadence["losses"], abs=1.0e-5)
