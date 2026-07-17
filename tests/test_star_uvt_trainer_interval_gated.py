from __future__ import annotations

import math

import torch
import pytest

from star_uvt_runtime import ensure_star_uvt_on_path
from star_uvt_projective_interval_backend import make_projective_cell_interval_trainer_state


ensure_star_uvt_on_path()

from research_project.benchmarks.video_fit_comparison import validate_uvt_backend_modes  # noqa: E402
from research_project.trainer_harness.tile_metal_autograd import (  # noqa: E402
    ProjectiveCellIntervalTrainerState,
    refresh_projective_cell_interval_atlas_if_stale,
    render_projective_cell_interval_atlas_metal_backward,
    render_uvt_tubes_metal_interval_gated_backward,
)
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    UVTRenderConfig,
    brute_force_render_uvt_tubes,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
    projective_trace_cell_atlas_fallback_stats,
    projective_trace_cell_visibility_event_report,
    projective_trace_windows_to_cell_trace_atlas,
    rebin_projective_trace_cell_atlas_support_events,
    render_projective_trace_cell_atlas_reference,
    render_projective_trace_cell_interval_atlas_metal,
    render_uvt_tubes,
    split_projective_trace_windows,
)


def _require_interval_gated_metal() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for STAR UVT interval-gated Metal trainer smoke")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "render_gated"):
        pytest.skip("STAR UVT native gated render op unavailable")
    if not hasattr(torch.ops.star_uvt_v0, "direct_atomic_backward_gated"):
        pytest.skip("STAR UVT native gated backward op unavailable")


def _require_projective_interval_cell_metal() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for STAR UVT projective interval-cell trainer smoke")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("STAR UVT projective interval cell render op unavailable")
    if not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("STAR UVT projective interval cell backward op unavailable")


def _set_default_tile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAR_UVT_TILE_X", "8")
    monkeypatch.setenv("STAR_UVT_TILE_Y", "8")
    monkeypatch.setenv("STAR_UVT_TILE_T", "2")
    monkeypatch.setenv("STAR_UVT_TILE_CAPACITY", "128")


def _mixed_fallback_interval_atlas(
    *,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
    all_fallback: bool = False,
) -> ProjectiveTraceCellTraceAtlas:
    coeffs = torch.tensor(
        [
            [3.4, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.2, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0005, 0.0, 0.0],
            [12.0, 0.0, 0.0, 3.5, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    ).contiguous()
    if requires_grad:
        coeffs.requires_grad_(True)
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=torch.tensor([0.65, 0.45, 0.55], dtype=torch.float32, device=device),
        color=torch.tensor(
            [[1.0, 0.1, 0.05], [0.05, 0.2, 1.0], [0.1, 1.0, 0.2]],
            dtype=torch.float32,
            device=device,
        ),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=4,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((1.0, 1.0), (1.0005, 1.0005)),
                fallback=True,
                fallback_reasons=("visibility_ambiguous_depth",),
            ),
            ProjectiveTraceTileTimeCell(
                tile_u=1,
                tile_v=0,
                start=0,
                stop=4,
                primitive_ids=(2,),
                ordered_primitive_ids=(2,),
                depth_intervals=((2.0, 2.0),),
                fallback=bool(all_fallback),
                fallback_reasons=("forced_test_fallback",) if all_fallback else (),
            ),
        ],
        source_window_indices=(0, 0, 0),
        source_primitive_ids=(0, 1, 2),
        active_start=(0, 0, 0),
        active_stop=(4, 4, 4),
    )


def test_interval_gated_backend_validation_selects_native_direct_gated_policy() -> None:
    validate_uvt_backend_modes(
        uvt_render_backend="metal_tile_interval_gated",
        uvt_reduction_mode="index_add",
        uvt_sample_emission_mode="direct_atomic",
        device=torch.device("mps"),
    )

    with pytest.raises(ValueError, match="direct_atomic_gated"):
        validate_uvt_backend_modes(
            uvt_render_backend="metal_tile_interval_gated",
            uvt_reduction_mode="index_add",
            uvt_sample_emission_mode="atomic_append",
            device=torch.device("mps"),
        )

    with pytest.raises(ValueError, match="requires --device=mps"):
        validate_uvt_backend_modes(
            uvt_render_backend="metal_tile_interval_gated",
            uvt_reduction_mode="index_add",
            uvt_sample_emission_mode="direct_atomic",
            device=torch.device("cpu"),
        )


def test_projective_interval_cell_trainer_state_builder_uses_production_config() -> None:
    cfg = {
        "data": {
            "max_frames": 4,
            "target_size": 8,
        },
        "feature_uvt": {
            "feature_dim": 3,
            "tile_t": 2,
            "tile_capacity": 32,
            "alpha_threshold": 0.01,
            "max_alpha": 0.9,
            "projective_interval": {
                "enabled": True,
                "sigma_px": 1.25,
                "tile_size": 8,
                "refresh_every": 3,
                "uv_padding": 1.5,
                "depth_padding": 0.25,
                "allow_ambiguous_fallback": True,
                "fallback_render_mode": "mixed",
                "enforce_complexity_budget": True,
                "max_interval_to_dense_trace_sample_ratio": 0.75,
                "max_fallback_fraction": 0.125,
                "max_cells_per_active_set_group": 4,
            },
        },
    }
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.0, 4.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        .contiguous()
        .requires_grad_(True),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.tensor([[1.0, 0.25, 0.125]], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=4,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.0),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(4,),
    )

    state = make_projective_cell_interval_trainer_state(atlas, torch.arange(4, dtype=torch.float32), cfg)

    assert isinstance(state, ProjectiveCellIntervalTrainerState)
    assert state.atlas is atlas
    assert state.sigma_px == 1.25
    assert state.tile_size == 8
    assert state.refresh_every == 3
    assert state.uv_padding == 1.5
    assert state.depth_padding == 0.25
    assert state.allow_ambiguous_fallback is True
    assert state.fallback_render_mode == "mixed"
    assert state.enforce_complexity_budget is True
    assert state.max_interval_to_dense_trace_sample_ratio == 0.75
    assert state.max_fallback_fraction == 0.125
    assert state.max_cells_per_active_set_group == 4
    assert state.config.frames == 4
    assert state.config.tile_t == 2
    assert state.config.tile_capacity == 32
    assert state.render_reference_with_fallback().shape == (4, 8, 8, 3)


def test_interval_gated_trainer_wrapper_prevents_off_window_leakage_and_trains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_interval_gated_metal()
    _set_default_tile_env(monkeypatch)
    config = UVTRenderConfig(height=8, width=8, frames=4)
    device = torch.device("mps")

    ma_cpu = torch.tensor(
        [
            [3.5, 3.5, -1.0],
            [3.5, 3.5, 1.0],
        ],
        dtype=torch.float32,
    )
    q_cpu = torch.tensor(
        [
            [0.35, 0.0, 0.0, 0.35, 0.0, 0.02],
            [0.35, 0.0, 0.0, 0.35, 0.0, 0.02],
        ],
        dtype=torch.float32,
    )
    depth0_cpu = torch.tensor([0.8, 1.2], dtype=torch.float32)
    depth_beta_cpu = torch.zeros((2, 3), dtype=torch.float32)
    opacity_cpu = torch.tensor([0.88, 0.88], dtype=torch.float32)
    target_color_cpu = torch.tensor(
        [
            [0.95, 0.05, 0.05],
            [0.05, 0.95, 0.05],
        ],
        dtype=torch.float32,
    )
    active_start_cpu = torch.tensor([0, 2], dtype=torch.int32)
    active_stop_cpu = torch.tensor([2, 4], dtype=torch.int32)

    target = brute_force_render_uvt_tubes(
        ma_cpu,
        q_cpu,
        depth0_cpu,
        depth_beta_cpu,
        opacity_cpu,
        target_color_cpu,
        config,
        active_start=active_start_cpu,
        active_stop=active_stop_cpu,
    ).to(device)

    ma = ma_cpu.to(device)
    q_uvt = q_cpu.to(device)
    depth0 = depth0_cpu.to(device)
    depth_beta = depth_beta_cpu.to(device)
    opacity = opacity_cpu.to(device)
    active_start = active_start_cpu.to(device)
    active_stop = active_stop_cpu.to(device)
    target_color = target_color_cpu.to(device)

    gated = render_uvt_tubes_metal_interval_gated_backward(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        target_color,
        active_start,
        active_stop,
        config,
    )
    torch.testing.assert_close(gated.cpu(), target.cpu(), atol=2.0e-5, rtol=2.0e-5)

    leaky = render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, target_color, config)
    assert float((leaky - target).abs().amax().detach().cpu().item()) > 1.0e-2

    train_color = torch.tensor(
        [
            [0.20, 0.20, 0.90],
            [0.90, 0.20, 0.20],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.SGD([train_color], lr=0.75)
    losses: list[float] = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        prediction = render_uvt_tubes_metal_interval_gated_backward(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            train_color,
            active_start,
            active_stop,
            config,
        )
        loss = (prediction - target).square().mean()
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        assert train_color.grad is not None
        assert float(train_color.grad.detach().abs().sum().cpu().item()) > 0.0
        optimizer.step()

    assert losses[-1] < losses[0]


def test_projective_interval_cell_trainer_wrapper_uses_split_windows_and_trains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_projective_interval_cell_metal()
    _set_default_tile_env(monkeypatch)

    source_coeffs = torch.tensor(
        [[4.0, 0.0, 0.35, 4.0, 0.45, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(8, dtype=torch.float32).sub_(3.5).contiguous()
    color = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    opacity = torch.tensor([0.6], dtype=torch.float32)
    windows = split_projective_trace_windows(
        source_coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert len(windows) > 1
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacity,
        color=color,
        image_width=8,
        image_height=8,
        tile_size=8,
        primitive_ids=[7],
        uv_padding=4.0,
    )
    assert len(set(zip(atlas.active_start, atlas.active_stop))) > 1
    assert any(start > 0 for start in atlas.active_start)
    assert any(stop < int(times.numel()) for stop in atlas.active_stop)

    target_coeffs = atlas.coeffs.clone()
    target_coeffs[:, 0] += 0.20
    target_coeffs[:, 3] -= 0.16
    target_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=target_coeffs,
        opacity=atlas.opacity,
        color=atlas.color,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=8,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    device = torch.device("mps")
    target_atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=target_atlas.coeffs.to(device),
        opacity=target_atlas.opacity.to(device),
        color=target_atlas.color.to(device),
        cells=target_atlas.cells,
        source_window_indices=target_atlas.source_window_indices,
        source_primitive_ids=target_atlas.source_primitive_ids,
        active_start=target_atlas.active_start,
        active_stop=target_atlas.active_stop,
    )
    target = render_projective_trace_cell_interval_atlas_metal(
        target_atlas_mps,
        times.to(device),
        config,
        sigma_px=1.6,
    ).detach()

    train_coeffs = atlas.coeffs.to(device).detach().clone().requires_grad_(True)
    train_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=train_coeffs,
        opacity=atlas.opacity.to(device),
        color=atlas.color.to(device),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    optimizer = torch.optim.SGD([train_coeffs], lr=8.0)
    losses: list[float] = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        prediction = render_projective_cell_interval_atlas_metal_backward(
            train_atlas,
            times.to(device),
            config,
            sigma_px=1.6,
        )
        loss = (prediction - target).square().mean()
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()
        assert train_coeffs.grad is not None
        assert float(train_coeffs.grad[:, :6].detach().abs().sum().cpu().item()) > 0.0
        optimizer.step()

    assert losses[-1] < losses[0]


def test_projective_interval_cell_lifecycle_rebins_after_optimizer_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_projective_interval_cell_metal()
    _set_default_tile_env(monkeypatch)

    coeffs = torch.tensor(
        [[4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    color = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    opacity = torch.tensor([0.6], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacity,
        color=color,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )
    assert all(cell.tile_u == 0 for cell in atlas.cells)

    device = torch.device("mps")
    train_coeffs = atlas.coeffs.to(device).detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([train_coeffs], lr=0.5)
    optimizer.zero_grad(set_to_none=True)
    coefficient_motion_loss = (train_coeffs[:, 0] - 12.5).square().mean()
    coefficient_motion_loss.backward()
    optimizer.step()

    stale_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=train_coeffs,
        opacity=atlas.opacity.to(device),
        color=atlas.color.to(device),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    refresh = refresh_projective_cell_interval_atlas_if_stale(
        stale_atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )
    assert refresh.rebinned
    assert refresh.before.stale
    assert refresh.before.missing_tile_pairs == 4
    assert not refresh.after.stale
    assert refresh.after.missing_tile_pairs == 0
    assert refresh.atlas.coeffs is train_coeffs
    assert any(cell.tile_u == 1 and cell.tile_v == 0 and 0 in cell.primitive_ids for cell in refresh.atlas.cells)

    optimizer.zero_grad(set_to_none=True)
    config = UVTRenderConfig(
        height=8,
        width=16,
        frames=4,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    rendered = render_projective_cell_interval_atlas_metal_backward(
        refresh.atlas,
        times.to(device),
        config,
        sigma_px=1.0,
    )
    left_energy = rendered[:, :, :8, :].sum()
    right_energy = rendered[:, :, 8:, :].sum()
    assert float(right_energy.detach().cpu().item()) > 0.0
    assert float(right_energy.detach().cpu().item()) > 100.0 * float(left_energy.detach().cpu().item())

    right_energy.backward()
    assert train_coeffs.grad is not None
    assert float(train_coeffs.grad[:, :6].detach().abs().sum().cpu().item()) > 0.0


def test_projective_interval_cell_refresh_rebins_depth_order_without_replacing_tensor() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    color = torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32)
    opacity = torch.tensor([0.6, 0.4], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacity,
        color=color,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )
    assert atlas.cells[0].ordered_primitive_ids == (0, 1)

    train_coeffs = atlas.coeffs.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([train_coeffs], lr=0.5)
    optimizer.zero_grad(set_to_none=True)
    depth_motion_loss = (train_coeffs[0, 6] - 3.0).square()
    depth_motion_loss.backward()
    optimizer.step()

    stale_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=train_coeffs,
        opacity=atlas.opacity,
        color=atlas.color,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    refresh = refresh_projective_cell_interval_atlas_if_stale(
        stale_atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )
    assert not refresh.before.stale
    assert refresh.visibility_before.stale
    assert refresh.visibility_before.order_mismatch_samples == 4
    assert refresh.rebinned
    assert not refresh.after.stale
    assert not refresh.visibility_after.stale
    assert refresh.visibility_after.order_mismatch_samples == 0
    assert refresh.atlas.coeffs is train_coeffs
    assert refresh.atlas.cells[0].ordered_primitive_ids == (1, 0)


def test_projective_interval_cell_trainer_state_owns_support_refresh_and_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_projective_interval_cell_metal()
    _set_default_tile_env(monkeypatch)

    coeffs = torch.tensor(
        [[4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    color = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    opacity = torch.tensor([0.6], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacity,
        color=color,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )

    device = torch.device("mps")
    train_coeffs = atlas.coeffs.to(device).detach().clone().requires_grad_(True)
    state = ProjectiveCellIntervalTrainerState(
        atlas=ProjectiveTraceCellTraceAtlas(
            coeffs=train_coeffs,
            opacity=atlas.opacity.to(device),
            color=atlas.color.to(device),
            cells=atlas.cells,
            source_window_indices=atlas.source_window_indices,
            source_primitive_ids=atlas.source_primitive_ids,
            active_start=atlas.active_start,
            active_stop=atlas.active_stop,
        ),
        times=times.to(device),
        config=UVTRenderConfig(
            height=8,
            width=16,
            frames=4,
            tile_x=8,
            tile_y=8,
            tile_t=2,
            tile_capacity=128,
            alpha_threshold=1.0e-6,
            transmittance_threshold=0.0,
            max_alpha=1.0,
        ),
        sigma_px=1.0,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
        refresh_every=1,
    )
    optimizer = torch.optim.SGD([train_coeffs], lr=0.5)
    optimizer.zero_grad(set_to_none=True)
    motion_loss = (train_coeffs[:, 0] - 12.5).square().mean()
    motion_loss.backward()
    optimizer.step()

    refresh = state.after_optimizer_step()
    assert refresh is not None
    assert refresh.rebinned
    assert state.last_refresh is refresh
    assert state.atlas is refresh.atlas
    assert state.atlas.coeffs is train_coeffs
    assert refresh.before.stale
    assert not refresh.after.stale

    optimizer.zero_grad(set_to_none=True)
    rendered = state.render()
    right_energy = rendered[:, :, 8:, :].sum()
    assert float(right_energy.detach().cpu().item()) > 0.0
    right_energy.backward()
    assert train_coeffs.grad is not None
    assert float(train_coeffs.grad[:, :6].detach().abs().sum().cpu().item()) > 0.0


def test_projective_interval_cell_trainer_state_owns_depth_order_refresh() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    color = torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32)
    opacity = torch.tensor([0.6, 0.4], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacity,
        color=color,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )
    train_coeffs = atlas.coeffs.detach().clone().requires_grad_(True)
    state = ProjectiveCellIntervalTrainerState(
        atlas=ProjectiveTraceCellTraceAtlas(
            coeffs=train_coeffs,
            opacity=atlas.opacity,
            color=atlas.color,
            cells=atlas.cells,
            source_window_indices=atlas.source_window_indices,
            source_primitive_ids=atlas.source_primitive_ids,
            active_start=atlas.active_start,
            active_stop=atlas.active_stop,
        ),
        times=times,
        config=UVTRenderConfig(height=8, width=8, frames=4, tile_x=8, tile_y=8, tile_t=2, tile_capacity=128),
        sigma_px=1.0,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
        refresh_every=2,
    )
    optimizer = torch.optim.SGD([train_coeffs], lr=0.5)
    optimizer.zero_grad(set_to_none=True)
    depth_motion_loss = (train_coeffs[0, 6] - 3.0).square()
    depth_motion_loss.backward()
    optimizer.step()

    assert state.after_optimizer_step() is None
    assert state.step_index == 1
    refresh = state.after_optimizer_step()
    assert refresh is not None
    assert state.step_index == 2
    assert not refresh.before.stale
    assert refresh.visibility_before.stale
    assert refresh.rebinned
    assert not refresh.visibility_after.stale
    assert state.atlas.coeffs is train_coeffs
    assert state.atlas.cells[0].ordered_primitive_ids == (1, 0)


def test_projective_interval_cell_trainer_state_marks_ambiguous_visibility_fallback() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0005, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    color = torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32)
    opacity = torch.tensor([0.6, 0.4], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacity,
        color=color,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )
    state = ProjectiveCellIntervalTrainerState(
        atlas=atlas,
        times=times,
        config=UVTRenderConfig(height=8, width=8, frames=4, tile_x=8, tile_y=8, tile_t=2, tile_capacity=128),
        sigma_px=1.0,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
        depth_epsilon=1.0e-3,
        allow_ambiguous_fallback=True,
    )

    refresh = state.refresh(force=True)
    assert refresh.rebinned
    assert refresh.fallback_marked
    assert refresh.visibility_before.stale
    assert refresh.visibility_before.ambiguous_depth_samples == 4
    assert not refresh.visibility_after.stale
    assert refresh.visibility_after.ambiguous_depth_samples == 4
    assert any(cell.fallback for cell in state.atlas.cells)
    assert state.atlas.cells[0].fallback_reasons == ("visibility_ambiguous_depth",)
    stats = state.fallback_stats()
    assert stats.fallback_cells == 1
    assert stats.fallback_tile_samples == 4
    assert stats.fallback_fraction == 1.0
    assert stats.fallback_reasons == ("visibility_ambiguous_depth",)
    with pytest.raises(RuntimeError, match="cannot execute fallback"):
        state.render()
    reference = state.render_reference_with_fallback()
    assert reference.shape == (4, 8, 8, 3)
    assert float(reference.sum().item()) > 0.0

    strict_state = ProjectiveCellIntervalTrainerState(
        atlas=atlas,
        times=times,
        config=UVTRenderConfig(height=8, width=8, frames=4, tile_x=8, tile_y=8, tile_t=2, tile_capacity=128),
        sigma_px=1.0,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
        depth_epsilon=1.0e-3,
        allow_ambiguous_fallback=False,
    )
    with pytest.raises(RuntimeError, match="visibility order"):
        strict_state.refresh(force=True)


def test_projective_interval_cell_trainer_state_mixed_fallback_keeps_reference_gradients() -> None:
    atlas = _mixed_fallback_interval_atlas(requires_grad=True, all_fallback=True)
    state = ProjectiveCellIntervalTrainerState(
        atlas=atlas,
        times=torch.arange(4, dtype=torch.float32).contiguous(),
        config=UVTRenderConfig(height=8, width=16, frames=4, tile_x=8, tile_y=8, tile_t=2, tile_capacity=128),
        sigma_px=1.7,
        image_width=16,
        image_height=8,
        tile_size=8,
        fallback_render_mode="mixed",
    )

    rendered = state.render()
    reference = state.render_reference_with_fallback()
    torch.testing.assert_close(rendered, reference, atol=1.0e-6, rtol=1.0e-6)
    rendered.sum().backward()

    assert atlas.coeffs.grad is not None
    assert float(atlas.coeffs.grad[:2, :6].abs().sum().item()) > 0.0


def test_projective_interval_cell_trainer_state_mixed_fallback_patches_fast_metal_if_available() -> None:
    _require_projective_interval_cell_metal()
    device = torch.device("mps")
    atlas = _mixed_fallback_interval_atlas(device=device, requires_grad=True)
    state = ProjectiveCellIntervalTrainerState(
        atlas=atlas,
        times=torch.arange(4, dtype=torch.float32, device=device).contiguous(),
        config=UVTRenderConfig(
            height=8,
            width=16,
            frames=4,
            tile_x=8,
            tile_y=8,
            tile_t=2,
            tile_capacity=128,
            alpha_threshold=1.0e-6,
            transmittance_threshold=0.0,
            max_alpha=1.0,
        ),
        sigma_px=1.7,
        image_width=16,
        image_height=8,
        tile_size=8,
        fallback_render_mode="mixed",
    )

    rendered = state.render()
    reference = state.render_reference_with_fallback()
    torch.testing.assert_close(rendered.cpu(), reference.detach().cpu(), atol=2.0e-4, rtol=2.0e-4)
    rendered.sum().backward()

    assert atlas.coeffs.grad is not None
    assert float(atlas.coeffs.grad[:2, :6].detach().abs().sum().cpu().item()) > 0.0
    assert float(atlas.coeffs.grad[2:, :6].detach().abs().sum().cpu().item()) > 0.0


def test_projective_interval_cell_refresh_stratifies_visibility_crossing_without_fallback() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, -0.2, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.detach().clone().requires_grad_(True),
        opacity=torch.tensor([0.6, 0.4], dtype=torch.float32),
        color=torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=4,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((1.0, 4.0), (2.4, 3.0)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0, 0),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(4, 4),
    )
    refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )

    assert refresh.rebinned
    assert refresh.visibility_stratified
    assert not refresh.fallback_marked
    assert refresh.budget_after.within_budget
    assert refresh.budget_after.stats.interval_to_dense_trace_sample_ratio == 0.5
    assert refresh.budget_after.stats.visibility_stratum_split_cells == 1
    assert refresh.visibility_before.stale
    assert refresh.visibility_before.order_mismatch_samples == 2
    assert not refresh.visibility_after.stale
    state = ProjectiveCellIntervalTrainerState(
        atlas=refresh.atlas,
        times=times,
        config=UVTRenderConfig(height=8, width=8, frames=4, tile_x=8, tile_y=8, tile_t=2, tile_capacity=128),
        sigma_px=1.0,
        image_width=8,
        image_height=8,
        tile_size=8,
    )
    stats = state.complexity_stats()
    budget = state.budget_report(
        max_interval_to_dense_trace_sample_ratio=0.60,
        max_fallback_fraction=0.0,
        max_cells_per_active_set_group=2,
    )
    assert stats.visibility_stratum_split_cells == 1
    assert stats.interval_to_dense_trace_sample_ratio == 0.5
    assert budget.within_budget
    assert [(cell.start, cell.stop, cell.ordered_primitive_ids) for cell in refresh.atlas.cells] == [
        (0, 2, (0, 1)),
        (2, 4, (1, 0)),
    ]
    assert refresh.atlas.coeffs is atlas.coeffs


def test_projective_interval_cell_refresh_can_tolerate_subpixel_support_overshoot() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[8.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.0),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
    )

    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        check_visibility=False,
    )
    tolerant_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        support_stale_overshoot_epsilon=0.10,
        check_visibility=False,
    )

    assert strict_refresh.rebinned
    assert tolerant_refresh.before.stale
    assert tolerant_refresh.support_margin_before.max_boundary_overshoot_px == pytest.approx(0.05, abs=1.0e-5)
    assert tolerant_refresh.support_margin_before.min_boundary_slack_px == pytest.approx(-0.05, abs=1.0e-5)
    assert not tolerant_refresh.rebinned
    assert tolerant_refresh.after is tolerant_refresh.before
    assert tolerant_refresh.atlas is atlas

    with pytest.raises(RuntimeError, match="interval_to_dense_trace_sample_ratio"):
        refresh_projective_cell_interval_atlas_if_stale(
            atlas,
            times,
            image_width=8,
            image_height=8,
            tile_size=8,
            uv_padding=2.0,
            enforce_complexity_budget=True,
            max_interval_to_dense_trace_sample_ratio=0.40,
        )


def test_projective_interval_subpixel_support_debounce_has_bounded_tail_error() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[4.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.0),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
    )

    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        check_visibility=False,
    )
    tolerant_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        support_stale_overshoot_epsilon=0.10,
        check_visibility=False,
    )

    strict_image = render_projective_trace_cell_atlas_reference(
        strict_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    tolerant_image = render_projective_trace_cell_atlas_reference(
        tolerant_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    diff = (strict_image - tolerant_image).abs()

    assert strict_refresh.rebinned
    assert tolerant_refresh.support_margin_before.max_boundary_overshoot_px == pytest.approx(0.05, abs=1.0e-5)
    assert not tolerant_refresh.rebinned
    assert float(diff.max().item()) > 0.0
    assert float(diff.max().item()) < 1.0e-4
    assert float(diff.mean().item()) < 1.0e-6


def test_projective_interval_support_tail_alpha_certificate_debounces_subpixel_sliver() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[4.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.0),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
    )

    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        support_stale_tail_alpha_epsilon=1.0e-4,
        sigma_px=1.0,
        check_visibility=False,
    )
    certified_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        support_stale_tail_alpha_epsilon=3.0e-4,
        sigma_px=1.0,
        check_visibility=False,
    )

    assert strict_refresh.rebinned
    assert certified_refresh.before.stale
    assert certified_refresh.support_margin_before.max_boundary_overshoot_px == pytest.approx(0.05, abs=1.0e-5)
    assert 1.0e-4 < certified_refresh.support_tail_alpha_bound_before < 3.0e-4
    assert certified_refresh.support_tail_alpha_bound_after == certified_refresh.support_tail_alpha_bound_before
    assert not certified_refresh.rebinned
    assert certified_refresh.atlas is atlas


def test_projective_interval_support_tail_alpha_certificate_uses_spatial_precision_uv() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[4.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.0),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
        spatial_precision_uv=torch.tensor([[4.0, 0.0, 0.25]], dtype=torch.float32).contiguous(),
    )

    refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        support_stale_tail_alpha_epsilon=1.0e-6,
        sigma_px=1.0,
        check_visibility=False,
    )

    isotropic_tail = 0.5 * math.exp(-0.5 * (3.95**2))
    assert refresh.before.stale
    assert refresh.support_margin_before.max_boundary_overshoot_px == pytest.approx(0.05, abs=1.0e-5)
    assert refresh.support_tail_alpha_bound_before < 1.0e-6
    assert refresh.support_tail_alpha_bound_before < isotropic_tail * 1.0e-6
    assert refresh.support_tail_alpha_bound_after == refresh.support_tail_alpha_bound_before
    assert not refresh.rebinned
    assert refresh.atlas is atlas


def test_projective_interval_support_tail_alpha_certificate_aggregates_overlapping_tails() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    trace_count = 16
    coeffs = torch.tensor(
        [[4.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0 + 0.001 * index, 0.0, 0.0] for index in range(trace_count)],
        dtype=torch.float32,
    ).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=torch.full((trace_count,), 0.5, dtype=torch.float32),
        color=torch.ones((trace_count, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=tuple(range(trace_count)),
                ordered_primitive_ids=tuple(range(trace_count)),
                depth_intervals=tuple((1.0 + 0.001 * index, 1.0 + 0.001 * index) for index in range(trace_count)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=tuple(range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(2 for _ in range(trace_count)),
    )

    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        sigma_px=1.0,
        check_visibility=False,
    )
    bounded_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        support_stale_tail_alpha_epsilon=1.0e-3,
        sigma_px=1.0,
        check_visibility=False,
    )
    loose_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        support_stale_tail_alpha_epsilon=4.0e-3,
        sigma_px=1.0,
        check_visibility=False,
    )
    strict_image = render_projective_trace_cell_atlas_reference(
        strict_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    loose_image = render_projective_trace_cell_atlas_reference(
        loose_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    diff = (strict_image - loose_image).abs()

    per_trace_tail = 0.5 * math.exp(-0.5 * (3.95**2))
    assert strict_refresh.rebinned
    assert bounded_refresh.before.stale
    assert bounded_refresh.support_tail_alpha_bound_before == pytest.approx(trace_count * per_trace_tail)
    assert bounded_refresh.support_tail_alpha_bound_before > 1.0e-3
    assert bounded_refresh.rebinned
    assert loose_refresh.support_tail_alpha_bound_before == pytest.approx(bounded_refresh.support_tail_alpha_bound_before)
    assert not loose_refresh.rebinned
    assert float(diff.max().item()) > 1.0e-4
    assert float(diff.max().item()) < loose_refresh.support_tail_alpha_bound_before


def test_projective_interval_support_tail_alpha_certificate_rejects_core_loss() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[8.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.0),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
    )

    refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=0.0,
        support_stale_tail_alpha_epsilon=0.1,
        sigma_px=1.0,
        check_visibility=False,
    )

    assert refresh.support_tail_alpha_bound_before == pytest.approx(0.5)
    assert refresh.rebinned


def test_projective_interval_spatial_precision_metadata_survives_support_rebin() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    spatial_precision_uv = torch.tensor([[1.0, 0.2, 1.5]], dtype=torch.float32).contiguous()
    depth_affine_uv = torch.tensor([[0.1, 0.01, 0.0, -0.2, 0.0, 0.0]], dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[4.0, 0.25, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
        spatial_precision_uv=spatial_precision_uv,
        depth_affine_uv=depth_affine_uv,
    )

    rebinned = rebin_projective_trace_cell_atlas_support_events(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
    )
    refresh = refresh_projective_cell_interval_atlas_if_stale(
        rebinned,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        force=True,
        check_visibility=False,
    )

    assert rebinned.spatial_precision_uv is spatial_precision_uv
    assert rebinned.depth_affine_uv is depth_affine_uv
    assert refresh.atlas.spatial_precision_uv is spatial_precision_uv
    assert refresh.atlas.depth_affine_uv is depth_affine_uv
    torch.testing.assert_close(refresh.atlas.spatial_precision_uv, spatial_precision_uv)
    torch.testing.assert_close(refresh.atlas.depth_affine_uv, depth_affine_uv)


def test_projective_interval_spatial_precision_metadata_rejects_non_spd() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
        spatial_precision_uv=torch.tensor([[1.0, 1.1, 1.0]], dtype=torch.float32).contiguous(),
    )

    with pytest.raises(ValueError, match="positive definite"):
        rebin_projective_trace_cell_atlas_support_events(
            atlas,
            times,
            image_width=16,
            image_height=8,
            tile_size=8,
            uv_padding=4.0,
        )


def test_projective_interval_orbit_support_debounce_has_bounded_tail_error() -> None:
    theta = torch.linspace(-math.radians(3.0), math.radians(3.0), 4, dtype=torch.float32)
    times = torch.tan(0.5 * theta).contiguous()
    point_x = 0.05
    base_depth = 2.6
    vertical = 0.02
    center_u = 11.92
    center_v = 8.0
    scale = 1.0
    raw_u = torch.tensor([point_x, 2.0, -point_x], dtype=torch.float32)
    raw_v = torch.tensor([vertical, 0.0, vertical], dtype=torch.float32)
    depth = torch.tensor([base_depth + 0.25, 2.0 * point_x, base_depth - 0.25], dtype=torch.float32)
    pixel_u = center_u * depth + scale * raw_u
    pixel_v = center_v * depth + scale * raw_v
    coeffs = torch.tensor([[*pixel_u.tolist(), *pixel_v.tolist(), *depth.tolist()]], dtype=torch.float32).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=0.01,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert all(window.accepted for window in windows)
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
    )
    assert [(cell.tile_u, cell.tile_v, cell.start, cell.stop) for cell in atlas.cells] == [(0, 0, 0, 4)]

    moved_coeffs = atlas.coeffs.clone()
    moved_coeffs[:, 0] += 0.10
    moved_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=moved_coeffs,
        opacity=atlas.opacity,
        color=atlas.color,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )

    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        moved_atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
        check_visibility=False,
    )
    tolerant_refresh = refresh_projective_cell_interval_atlas_if_stale(
        moved_atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
        support_stale_overshoot_epsilon=0.075,
        check_visibility=False,
    )

    strict_image = render_projective_trace_cell_atlas_reference(
        strict_refresh.atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        sigma_px=1.0,
    )
    tolerant_image = render_projective_trace_cell_atlas_reference(
        tolerant_refresh.atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        sigma_px=1.0,
    )
    diff = (strict_image - tolerant_image).abs()

    assert strict_refresh.rebinned
    assert not tolerant_refresh.rebinned
    assert 0.04 < tolerant_refresh.support_margin_before.max_boundary_overshoot_px < 0.075
    assert tolerant_refresh.support_margin_before.missing_tile_pairs == 4
    assert float(diff.max().item()) > 0.0
    assert float(diff.max().item()) < 1.0e-4
    assert float(diff.mean().item()) < 1.0e-6


def test_projective_interval_orbit_support_debounce_still_repairs_visibility_order() -> None:
    theta = torch.linspace(-math.radians(3.0), math.radians(3.0), 4, dtype=torch.float32)
    times = torch.tan(0.5 * theta).contiguous()
    point_x = 0.05
    base_depth = 2.6
    vertical = 0.02
    center_u = 11.92
    center_v = 8.0
    scale = 1.0
    raw_u = torch.tensor([point_x, 2.0, -point_x], dtype=torch.float32)
    raw_v = torch.tensor([vertical, 0.0, vertical], dtype=torch.float32)
    depth_far = torch.tensor([base_depth + 0.25, 2.0 * point_x, base_depth - 0.25], dtype=torch.float32)
    depth_near = depth_far - 1.2
    coeffs = torch.tensor(
        [
            [
                *(center_u * depth_far + scale * raw_u).tolist(),
                *(center_v * depth_far + scale * raw_v).tolist(),
                *depth_far.tolist(),
            ],
            [
                *(center_u * depth_near + scale * raw_u).tolist(),
                *(center_v * depth_near + scale * raw_v).tolist(),
                *depth_near.tolist(),
            ],
        ],
        dtype=torch.float32,
    ).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=0.01,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert all(window.accepted for window in windows)
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=torch.tensor([0.5, 0.5], dtype=torch.float32),
        color=torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.2, 0.9]], dtype=torch.float32),
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
    )
    assert [(cell.tile_u, cell.tile_v, cell.start, cell.stop) for cell in atlas.cells] == [(0, 0, 0, 4)]

    moved_coeffs = atlas.coeffs.clone()
    moved_coeffs[:, 0] += 0.10
    stale_cells = [
        ProjectiveTraceTileTimeCell(
            tile_u=cell.tile_u,
            tile_v=cell.tile_v,
            start=cell.start,
            stop=cell.stop,
            primitive_ids=cell.primitive_ids,
            ordered_primitive_ids=(0, 1),
            depth_intervals=((1.0, 1.0), (2.0, 2.0)),
            fallback=False,
            fallback_reasons=(),
        )
        for cell in atlas.cells
    ]
    moved_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=moved_coeffs,
        opacity=atlas.opacity,
        color=atlas.color,
        cells=stale_cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )

    support_only_refresh = refresh_projective_cell_interval_atlas_if_stale(
        moved_atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
        support_stale_overshoot_epsilon=0.10,
        check_visibility=False,
    )
    refresh = refresh_projective_cell_interval_atlas_if_stale(
        moved_atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
        support_stale_overshoot_epsilon=0.10,
    )

    assert support_only_refresh.before.stale
    assert 0.05 < support_only_refresh.support_margin_before.max_boundary_overshoot_px < 0.10
    assert not support_only_refresh.rebinned
    assert refresh.before.stale
    assert refresh.support_margin_before.max_boundary_overshoot_px == pytest.approx(
        support_only_refresh.support_margin_before.max_boundary_overshoot_px
    )
    assert refresh.visibility_before.stale
    assert refresh.visibility_before.order_mismatch_samples == 4
    assert refresh.rebinned
    assert not refresh.after.stale
    assert not refresh.visibility_after.stale
    assert refresh.visibility_after.order_mismatch_samples == 0
    assert {cell.ordered_primitive_ids for cell in refresh.atlas.cells} == {(1, 0)}


def test_projective_interval_orbit_visibility_crossing_splits_into_stable_strata() -> None:
    theta = torch.linspace(-math.radians(3.0), math.radians(3.0), 4, dtype=torch.float32)
    times = torch.tan(0.5 * theta).contiguous()
    center_u = 11.0
    center_v = 8.0
    raw_u = torch.tensor([0.02, 0.0, -0.02], dtype=torch.float32)
    raw_v = torch.tensor([0.02, 0.0, 0.02], dtype=torch.float32)
    depth_rising = torch.tensor([2.0, 4.0, 0.0], dtype=torch.float32)
    depth_falling = torch.tensor([2.0, -4.0, 0.0], dtype=torch.float32)
    coeffs = torch.tensor(
        [
            [
                *(center_u * depth_rising + raw_u).tolist(),
                *(center_v * depth_rising + raw_v).tolist(),
                *depth_rising.tolist(),
            ],
            [
                *(center_u * depth_falling + raw_u).tolist(),
                *(center_v * depth_falling + raw_v).tolist(),
                *depth_falling.tolist(),
            ],
        ],
        dtype=torch.float32,
    ).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=0.01,
        max_depth_residual=0.01,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert all(window.accepted for window in windows)
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=torch.tensor([0.6, 0.4], dtype=torch.float32),
        color=torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32),
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
    )
    assert [(cell.tile_u, cell.tile_v, cell.start, cell.stop) for cell in atlas.cells] == [(0, 0, 0, 4)]

    event_report = projective_trace_cell_visibility_event_report(atlas, times)
    assert len(event_report.events) == 1
    assert event_report.events[0].time == pytest.approx(0.0, abs=1.0e-7)
    assert event_report.split_times == pytest.approx((0.0,), abs=1.0e-7)

    refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
    )

    assert not refresh.before.stale
    assert refresh.visibility_before.stale
    assert refresh.visibility_before.order_mismatch_samples == 2
    assert refresh.rebinned
    assert refresh.visibility_stratified
    assert not refresh.fallback_marked
    assert not refresh.visibility_after.stale
    assert refresh.visibility_after.order_mismatch_samples == 0
    assert refresh.budget_after.stats.visibility_stratum_split_cells == 1
    assert refresh.budget_after.stats.interval_to_dense_trace_sample_ratio == 0.5
    assert [(cell.start, cell.stop, cell.ordered_primitive_ids) for cell in refresh.atlas.cells] == [
        (0, 2, (0, 1)),
        (2, 4, (1, 0)),
    ]


def test_projective_interval_orbit_tail_visibility_and_slack_guard_share_one_refresh() -> None:
    base_count = 20
    far_extra_count = 12
    near_extra_count = 12
    tile_capacity = 32
    theta = torch.linspace(-math.radians(3.0), math.radians(3.0), 4, dtype=torch.float32)
    times = torch.tan(0.5 * theta).contiguous()
    raw_u = torch.tensor([0.02, 0.0, -0.02], dtype=torch.float32)
    raw_v = torch.tensor([0.02, 0.0, 0.02], dtype=torch.float32)
    centers_u = (
        [3.95 if index == 0 else 3.85 + 0.01 * (index % 3) for index in range(base_count)]
        + [16.0 + 0.01 * (index % 3) for index in range(far_extra_count)]
        + [12.2 + 0.01 * (index % 3) for index in range(near_extra_count)]
    )
    coeff_rows: list[list[float]] = []
    for trace_index, center_u in enumerate(centers_u):
        if trace_index == 0:
            depth = torch.tensor([2.0, 4.0, 0.0], dtype=torch.float32)
        elif trace_index == 1:
            depth = torch.tensor([2.0, -4.0, 0.0], dtype=torch.float32)
        else:
            depth = torch.tensor([3.0 + 0.01 * trace_index, 0.0, 0.0], dtype=torch.float32)
        coeff_rows.append(
            [
                *(center_u * depth + raw_u).tolist(),
                *(4.0 * depth + raw_v).tolist(),
                *depth.tolist(),
            ]
        )

    coeffs = torch.tensor(coeff_rows, dtype=torch.float32).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=0.02,
        max_depth_residual=0.02,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert all(window.accepted for window in windows)
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=torch.full((len(centers_u),), 0.5, dtype=torch.float32),
        color=torch.ones((len(centers_u), 3), dtype=torch.float32),
        image_width=32,
        image_height=16,
        tile_size=8,
        uv_padding=4.0,
    )
    moved_coeffs = atlas.coeffs.clone()
    moved_coeffs[0, 0] += 0.10
    moved_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=moved_coeffs,
        opacity=atlas.opacity,
        color=atlas.color,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )

    event_report = projective_trace_cell_visibility_event_report(moved_atlas, times)
    assert event_report.split_times == pytest.approx((0.0,), abs=1.0e-7)

    def refresh_for(policy: str):
        return refresh_projective_cell_interval_atlas_if_stale(
            moved_atlas,
            times,
            image_width=32,
            image_height=16,
            tile_size=8,
            uv_padding=4.0,
            support_uv_padding=8.5,
            tile_capacity=tile_capacity,
            support_guard_policy=policy,
            support_stale_tail_alpha_epsilon=3.0e-4,
            sigma_px=1.0,
        )

    trace_refresh = refresh_for("trace_budgeted")
    slack_refresh = refresh_for("slack_budgeted")
    far_ids = set(range(base_count, base_count + far_extra_count))
    near_ids = set(range(base_count + far_extra_count, len(centers_u)))

    def tile0_ids(refresh) -> set[int]:
        return {
            primitive_id
            for cell in refresh.atlas.cells
            if int(cell.tile_u) == 0 and int(cell.tile_v) == 0
            for primitive_id in cell.ordered_primitive_ids
        }

    assert slack_refresh.before.stale
    assert 1.0e-4 < slack_refresh.support_tail_alpha_bound_before < 3.0e-4
    assert 0.04 < slack_refresh.support_margin_before.max_boundary_overshoot_px < 0.08
    assert slack_refresh.visibility_before.stale
    assert slack_refresh.visibility_before.order_mismatch_samples == 4
    assert slack_refresh.rebinned
    assert slack_refresh.visibility_stratified
    assert not slack_refresh.fallback_marked
    assert not slack_refresh.after.stale
    assert slack_refresh.support_tail_alpha_bound_after == 0.0
    assert not slack_refresh.visibility_after.stale
    assert slack_refresh.visibility_after.order_mismatch_samples == 0
    assert slack_refresh.budget_after.within_budget
    assert max(len(cell.ordered_primitive_ids) for cell in slack_refresh.atlas.cells) == tile_capacity
    assert max(len(cell.ordered_primitive_ids) for cell in trace_refresh.atlas.cells) == tile_capacity
    assert set(range(base_count)).issubset(tile0_ids(slack_refresh))
    assert far_ids.issubset(tile0_ids(trace_refresh))
    assert near_ids.isdisjoint(tile0_ids(trace_refresh))
    assert near_ids.issubset(tile0_ids(slack_refresh))
    assert far_ids.isdisjoint(tile0_ids(slack_refresh))


def test_projective_interval_subpixel_support_debounce_rejects_underspecified_support_assumption() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[8.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.0),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
    )

    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=0.0,
        check_visibility=False,
    )
    tolerant_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=0.0,
        support_stale_overshoot_epsilon=0.10,
        check_visibility=False,
    )

    strict_image = render_projective_trace_cell_atlas_reference(
        strict_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    tolerant_image = render_projective_trace_cell_atlas_reference(
        tolerant_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )

    assert strict_refresh.rebinned
    assert not tolerant_refresh.rebinned
    assert float((strict_image - tolerant_image).abs().max().item()) > 0.35


def test_projective_interval_support_debounce_still_repairs_visibility_order() -> None:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [
                [4.05, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                [4.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.5, 0.5], dtype=torch.float32),
        color=torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.2, 0.9]], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((1.0, 1.0), (2.0, 2.0)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0, 0),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(2, 2),
    )

    refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        support_stale_overshoot_epsilon=0.10,
    )

    assert refresh.before.stale
    assert refresh.support_margin_before.max_boundary_overshoot_px == pytest.approx(0.05, abs=1.0e-5)
    assert refresh.visibility_before.stale
    assert refresh.visibility_before.order_mismatch_samples == 2
    assert refresh.rebinned
    assert not refresh.after.stale
    assert not refresh.visibility_after.stale
    assert refresh.visibility_after.order_mismatch_samples == 0
    assert {cell.ordered_primitive_ids for cell in refresh.atlas.cells} == {(1, 0)}


def test_projective_interval_cell_refresh_uses_event_roots_to_localize_fallback() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.detach().clone().requires_grad_(True),
        opacity=torch.tensor([0.6, 0.4], dtype=torch.float32),
        color=torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=4,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((0.0, 3.0), (1.0, 1.0)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0, 0),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(4, 4),
    )

    refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
        allow_ambiguous_fallback=True,
    )

    assert refresh.rebinned
    assert refresh.visibility_stratified
    assert refresh.fallback_marked
    assert not refresh.visibility_after.stale
    assert refresh.visibility_after.order_mismatch_samples == 0
    assert refresh.visibility_after.ambiguous_depth_samples == 1
    assert [(cell.start, cell.stop, cell.ordered_primitive_ids, cell.fallback) for cell in refresh.atlas.cells] == [
        (0, 1, (0, 1), False),
        (1, 2, (0, 1), True),
        (2, 4, (1, 0), False),
    ]
    stats = projective_trace_cell_atlas_fallback_stats(refresh.atlas)
    assert stats.fallback_tile_samples == 1
    assert stats.fallback_fraction == 0.25
    assert refresh.atlas.coeffs is atlas.coeffs


def test_projective_interval_cell_refresh_uses_support_events_for_tile_runs() -> None:
    coeffs = torch.tensor(
        [[8.0, 8.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.detach().clone().requires_grad_(True),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=4,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.0),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(4,),
    )

    refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=64,
        image_height=64,
        tile_size=16,
        check_visibility=False,
    )

    assert refresh.rebinned
    assert not refresh.after.stale
    assert [(cell.tile_u, cell.start, cell.stop) for cell in refresh.atlas.cells] == [
        (0, 0, 1),
        (1, 1, 3),
        (2, 3, 4),
    ]
    assert refresh.atlas.coeffs is atlas.coeffs
