from __future__ import annotations

from dataclasses import replace
import math
import sys
from pathlib import Path

import pytest
import torch

STAR_UVT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceCellSensorTimeQuadrature,
    ProjectiveTraceCellSensorTimeQuadratureSample,
    ProjectiveTraceTileTimeCell,
    UVTRenderConfig,
    assemble_projective_trace_tile_time_atlas,
    bin_projective_trace_support_bounds,
    bound_projective_trace_windows,
    brute_force_render_uvt_tubes,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    direct_backward_projective_trace_tile_time_atlas_metal,
    direct_atomic_backward,
    direct_atomic_backward_gated,
    direct_backward_projective_trace_uvt_bridge_metal_gated,
    eval_projective_trace_cell_depth_at_uv_torch,
    eval_projective_trace_cell_torch,
    eval_projective_trace_torch,
    has_projective_trace_cell_metal,
    has_projective_trace_cell_interval_metal,
    has_projective_trace_cell_interval_rows_metal,
    has_projective_trace_cell_interval_backward_metal,
    lower_projective_trace_cell_atlas_quadrature,
    lower_projective_trace_cell_atlas_rolling_quadrature,
    mark_projective_trace_cell_visibility_fallbacks,
    pack_projective_trace_tile_time_bins,
    projective_trace_cell_atlas_budget_report,
    projective_trace_cell_atlas_complexity_stats,
    projective_trace_cell_atlas_coverage_report,
    projective_trace_cell_atlas_fallback_stats,
    projective_trace_cell_atlas_fallback_tile_sample_mask,
    projective_trace_cell_atlas_support_margin_report,
    projective_trace_cell_atlas_visibility_report,
    projective_trace_cell_sensor_time_event_partition,
    projective_trace_cell_sensor_time_partition_quadrature,
    projective_trace_cell_sensor_time_partition_rolling_quadrature,
    projective_trace_windows_to_cell_trace_atlas,
    projective_trace_windows_to_uvt_tubes,
    projective_trace_uvt_bridge_active_spans,
    render_projective_trace_cell_atlas_metal,
    render_projective_trace_cell_interval_atlas_metal,
    render_projective_trace_cell_atlas_reference,
    render_projective_trace_cell_atlas_quadrature_reference,
    render_projective_trace_cell_atlas_quadrature_interval_metal,
    render_projective_trace_cell_atlas_quadrature_interval_mixed_metal,
    render_projective_trace_cell_atlas_rolling_quadrature_reference,
    render_projective_trace_cell_atlas_rolling_quadrature_batched_reference,
    render_projective_trace_cell_atlas_rolling_quadrature_interval_metal,
    render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal,
    render_projective_trace_tile_time_atlas_metal,
    render_projective_trace_tile_time_atlas_reference,
    render_projective_trace_uvt_bridge_metal_gated,
    render_projective_trace_uvt_bridge_reference,
    render_uvt_tubes,
    render_uvt_tubes_gated,
    rebin_projective_trace_cell_atlas,
    slice_projective_trace_cell_atlas_frames,
    split_projective_trace_cell_atlas_fallback_cells,
    split_projective_trace_windows,
    stratify_projective_trace_cell_atlas_visibility,
)


def _tan_half_angle_times(theta_min: float, theta_max: float, count: int) -> torch.Tensor:
    theta = torch.linspace(theta_min, theta_max, count, dtype=torch.float32)
    return torch.tan(0.5 * theta).contiguous()


def _pixel_orbit_coeffs(
    *,
    point_x: float,
    base_depth: float,
    vertical: float,
    center_u: float,
    center_v: float,
    scale: float,
) -> list[float]:
    raw_u = torch.tensor([point_x, 2.0, -point_x], dtype=torch.float32)
    raw_v = torch.tensor([vertical, 0.0, vertical], dtype=torch.float32)
    depth = torch.tensor([base_depth + 0.25, 2.0 * point_x, base_depth - 0.25], dtype=torch.float32)
    pixel_u = float(center_u) * depth + float(scale) * raw_u
    pixel_v = float(center_v) * depth + float(scale) * raw_v
    return [*pixel_u.tolist(), *pixel_v.tolist(), *depth.tolist()]


def _cells_covering(cells, *, tile_u: int, tile_v: int, sample_index: int):
    return [
        cell
        for cell in cells
        if cell.tile_u == tile_u
        and cell.tile_v == tile_v
        and cell.start <= sample_index < cell.stop
    ]


def test_projective_atlas_covers_dense_orbit_projection_reference() -> None:
    coeffs = torch.tensor(
        [
            _pixel_orbit_coeffs(point_x=0.25, base_depth=2.5, vertical=0.1, center_u=48.0, center_v=40.0, scale=18.0),
            _pixel_orbit_coeffs(point_x=-0.20, base_depth=2.8, vertical=-0.1, center_u=72.0, center_v=58.0, scale=16.0),
        ],
        dtype=torch.float32,
    ).contiguous()
    times = _tan_half_angle_times(-math.radians(60.0), math.radians(60.0), 64)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=0.75,
        min_denominator_abs=1.0e-3,
    )

    assert all(window.accepted for window in windows)
    bounds = bound_projective_trace_windows(windows)
    records = bin_projective_trace_support_bounds(
        bounds,
        image_width=128,
        image_height=128,
        tile_size=16,
        primitive_ids=[11, 22],
    )
    cells = assemble_projective_trace_tile_time_atlas(records)
    dense = eval_projective_trace_torch(coeffs, times)

    checked = 0
    for primitive_index, primitive_id in enumerate((11, 22)):
        for sample_index in range(times.numel()):
            u, v, _depth, valid_sign = dense[primitive_index, sample_index]
            if valid_sign == 0.0 or not (0.0 <= u < 128.0 and 0.0 <= v < 128.0):
                continue
            tile_u = int(torch.floor(u / 16.0).item())
            tile_v = int(torch.floor(v / 16.0).item())
            covering = _cells_covering(cells, tile_u=tile_u, tile_v=tile_v, sample_index=sample_index)
            assert any(primitive_id in cell.primitive_ids for cell in covering)
            checked += 1

    assert checked == 2 * times.numel()


def test_projective_atlas_depth_order_matches_dense_stable_reference() -> None:
    coeffs = torch.tensor(
        [
            [32.0, 0.0, 0.0, 32.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [64.0, 0.0, 0.0, 64.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=2.0e-5,
        min_denominator_abs=1.0e-3,
    )
    bounds = bound_projective_trace_windows(windows)
    records = bin_projective_trace_support_bounds(
        bounds,
        image_width=64,
        image_height=64,
        tile_size=16,
        primitive_ids=[101, 202],
    )
    cells = assemble_projective_trace_tile_time_atlas(records)
    dense = eval_projective_trace_torch(coeffs, times)

    for sample_index in range(times.numel()):
        dense_order = tuple(
            primitive_id
            for _depth, primitive_id in sorted(
                (float(dense[primitive_index, sample_index, 2].item()), primitive_id)
                for primitive_index, primitive_id in enumerate((101, 202))
            )
        )
        covering = _cells_covering(cells, tile_u=2, tile_v=2, sample_index=sample_index)
        assert len(covering) == 1
        assert covering[0].ordered_primitive_ids == dense_order


def _render_dense_projective_reference(
    coeffs: torch.Tensor,
    times: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    sigma_px: float,
) -> torch.Tensor:
    dense = eval_projective_trace_torch(coeffs, times)
    out = torch.zeros((times.numel(), image_height, image_width, colors.shape[1]), dtype=torch.float32)
    pixel_u = torch.arange(image_width, dtype=torch.float32) + 0.5
    pixel_v = torch.arange(image_height, dtype=torch.float32) + 0.5
    du = pixel_u.reshape(1, -1)
    dv = pixel_v.reshape(-1, 1)

    for sample_index in range(times.numel()):
        order = sorted(
            range(coeffs.shape[0]),
            key=lambda primitive_index: float(dense[primitive_index, sample_index, 2].item()),
        )
        transmittance = torch.ones((image_height, image_width), dtype=torch.float32)
        for primitive_index in order:
            center_u = dense[primitive_index, sample_index, 0]
            center_v = dense[primitive_index, sample_index, 1]
            valid_sign = dense[primitive_index, sample_index, 3]
            if valid_sign == 0.0:
                continue
            radius2 = (du - center_u).square() + (dv - center_v).square()
            alpha = opacities[primitive_index] * torch.exp(-0.5 * radius2 / float(sigma_px * sigma_px))
            alpha = alpha.clamp(0.0, 1.0)
            out[sample_index] += transmittance.unsqueeze(-1) * alpha.unsqueeze(-1) * colors[primitive_index]
            transmittance = transmittance * (1.0 - alpha)

    return out


def _render_dense_projective_cell_reference(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    sigma_px: float,
) -> torch.Tensor:
    dense = eval_projective_trace_cell_torch(atlas.coeffs, times)
    out = torch.zeros((times.numel(), image_height, image_width, atlas.color.shape[1]), dtype=torch.float32)
    pixel_u = torch.arange(image_width, dtype=torch.float32) + 0.5
    pixel_v = torch.arange(image_height, dtype=torch.float32) + 0.5
    du = pixel_u.reshape(1, -1)
    dv = pixel_v.reshape(-1, 1)

    for sample_index in range(times.numel()):
        active = [
            trace_id
            for trace_id in range(atlas.coeffs.shape[0])
            if int(atlas.active_start[trace_id]) <= sample_index < int(atlas.active_stop[trace_id])
            and float(dense[trace_id, sample_index, 3].item()) != 0.0
        ]
        order = sorted(active, key=lambda trace_id: float(dense[trace_id, sample_index, 2].item()))
        transmittance = torch.ones((image_height, image_width), dtype=torch.float32)
        for trace_id in order:
            center_u = dense[trace_id, sample_index, 0]
            center_v = dense[trace_id, sample_index, 1]
            radius2 = (du - center_u).square() + (dv - center_v).square()
            alpha = atlas.opacity[trace_id] * torch.exp(-0.5 * radius2 / float(sigma_px * sigma_px))
            alpha = alpha.clamp(0.0, 1.0)
            out[sample_index] += transmittance.unsqueeze(-1) * alpha.unsqueeze(-1) * atlas.color[trace_id]
            transmittance = transmittance * (1.0 - alpha)

    return out


def _direct_continuous_cell_atlas(
    *,
    coeffs: torch.Tensor | None = None,
    colors: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
) -> ProjectiveTraceCellTraceAtlas:
    if coeffs is None:
        coeffs = torch.tensor(
            [
                [3.5, 0.25, 0.0, 3.5, 0.08, 0.0, 1.0, 0.10, 0.0],
                [4.6, -0.18, 0.0, 3.2, 0.12, 0.0, 1.8, -0.06, 0.0],
            ],
            dtype=torch.float32,
        ).contiguous()
    if colors is None:
        colors = torch.tensor([[1.0, 0.1, 0.05], [0.05, 0.25, 1.0]], dtype=torch.float32)
    if opacities is None:
        opacities = torch.tensor([0.65, 0.45], dtype=torch.float32)
    trace_count = int(coeffs.shape[0])
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=opacities,
        color=colors,
        cells=[],
        source_window_indices=tuple(0 for _ in range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(1 for _ in range(trace_count)),
    )


def test_projective_cell_atlas_frame_slices_preserve_reference_forward_and_vjp() -> None:
    coeffs = torch.tensor(
        [
            [3.0, 0.1, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.0, -0.1, 0.0, 4.0, 0.0, 0.0, 1.2, 0.0, 0.0],
            [5.0, 0.0, 0.0, 3.5, 0.1, 0.0, 1.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    opacity = torch.tensor(
        [0.55, 0.45, 0.35],
        dtype=torch.float32,
        requires_grad=True,
    )
    color = torch.tensor(
        [[0.9, 0.1, 0.05], [0.05, 0.8, 0.2], [0.1, 0.2, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=opacity,
        color=color,
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0, 2),
                ordered_primitive_ids=(0, 2),
                depth_intervals=((0.9, 1.1), (1.7, 1.9)),
                fallback=False,
                fallback_reasons=(),
            ),
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=2,
                stop=4,
                primitive_ids=(1, 2),
                ordered_primitive_ids=(1, 2),
                depth_intervals=((1.1, 1.3), (1.7, 1.9)),
                fallback=False,
                fallback_reasons=(),
            ),
        ],
        source_window_indices=(0, 1, 2),
        source_primitive_ids=(10, 11, 12),
        active_start=(0, 2, 0),
        active_stop=(2, 4, 4),
    )
    times = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32)
    render_args = {
        "image_width": 8,
        "image_height": 8,
        "tile_size": 8,
        "sigma_px": 1.0,
    }
    full = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        **render_args,
    )
    chunks = [
        render_projective_trace_cell_atlas_reference(
            slice_projective_trace_cell_atlas_frames(
                atlas,
                start=start,
                stop=stop,
            ),
            times[start:stop],
            **render_args,
        )
        for start, stop in ((0, 2), (2, 4))
    ]
    concatenated = torch.cat(chunks, dim=0)

    torch.testing.assert_close(concatenated, full, rtol=0.0, atol=1.0e-7)
    first_slice = slice_projective_trace_cell_atlas_frames(
        atlas,
        start=0,
        stop=2,
    )
    second_slice = slice_projective_trace_cell_atlas_frames(
        atlas,
        start=2,
        stop=4,
    )
    assert first_slice.source_primitive_ids == (10, 12)
    assert second_slice.source_primitive_ids == (11, 12)
    assert all(
        active_start < active_stop
        for sliced in (first_slice, second_slice)
        for active_start, active_stop in zip(
            sliced.active_start,
            sliced.active_stop,
        )
    )

    weights = torch.linspace(0.1, 1.0, full.numel()).reshape_as(full)
    full_grads = torch.autograd.grad(
        (full * weights).sum(),
        (coeffs, opacity, color),
        retain_graph=True,
    )
    chunk_grads = torch.autograd.grad(
        (concatenated * weights).sum(),
        (coeffs, opacity, color),
    )
    for full_grad, chunk_grad in zip(full_grads, chunk_grads):
        torch.testing.assert_close(
            chunk_grad,
            full_grad,
            rtol=1.0e-6,
            atol=1.0e-7,
        )


def test_projective_cell_atlas_frame_slice_can_be_temporally_empty() -> None:
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[3.5, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.7], dtype=torch.float32),
        color=torch.tensor([[0.8, 0.2, 0.1]], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=2,
                stop=4,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.2),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(7,),
        active_start=(2,),
        active_stop=(4,),
    )

    empty = slice_projective_trace_cell_atlas_frames(
        atlas,
        start=0,
        stop=2,
    )

    assert empty.coeffs.shape == (0, 9)
    assert empty.opacity.shape == (0,)
    assert empty.color.shape == (0, 3)
    assert empty.cells == []
    assert empty.source_window_indices == ()
    assert empty.source_primitive_ids == ()
    assert empty.active_start == ()
    assert empty.active_stop == ()
    rendered = render_projective_trace_cell_atlas_reference(
        empty,
        torch.tensor([-1.5, -0.5], dtype=torch.float32),
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    torch.testing.assert_close(rendered, torch.zeros_like(rendered))


def _mixed_fallback_cell_atlas() -> ProjectiveTraceCellTraceAtlas:
    coeffs = torch.tensor(
        [
            [3.4, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.4, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [12.0, 0.0, 0.0, 3.5, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    colors = torch.tensor(
        [[1.0, 0.1, 0.05], [0.05, 0.2, 1.0], [0.1, 1.0, 0.2]],
        dtype=torch.float32,
    )
    opacities = torch.tensor([0.65, 0.45, 0.55], dtype=torch.float32)
    return _direct_continuous_cell_atlas(coeffs=coeffs, colors=colors, opacities=opacities)


def test_projective_cell_atlas_reference_uses_spatial_precision_uv() -> None:
    times = torch.zeros((1,), dtype=torch.float32).contiguous()
    cell = ProjectiveTraceTileTimeCell(
        tile_u=0,
        tile_v=0,
        start=0,
        stop=1,
        primitive_ids=(0,),
        ordered_primitive_ids=(0,),
        depth_intervals=((1.0, 1.0),),
        fallback=False,
        fallback_reasons=(),
    )
    scalar_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 1), dtype=torch.float32),
        cells=[cell],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(1,),
    )
    anisotropic_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=scalar_atlas.coeffs,
        opacity=scalar_atlas.opacity,
        color=scalar_atlas.color,
        cells=scalar_atlas.cells,
        source_window_indices=scalar_atlas.source_window_indices,
        source_primitive_ids=scalar_atlas.source_primitive_ids,
        active_start=scalar_atlas.active_start,
        active_stop=scalar_atlas.active_stop,
        spatial_precision_uv=torch.tensor([[4.0, 0.0, 0.25]], dtype=torch.float32).contiguous(),
    )

    scalar = render_projective_trace_cell_atlas_reference(
        scalar_atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    anisotropic = render_projective_trace_cell_atlas_reference(
        anisotropic_atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )

    q_radius2 = 4.0 * (0.5**2) + 0.25 * (0.5**2)
    expected = 0.5 * math.exp(-0.5 * q_radius2)
    assert anisotropic[0, 4, 4, 0].item() == pytest.approx(expected, abs=1.0e-7)
    assert anisotropic[0, 4, 4, 0].item() != pytest.approx(scalar[0, 4, 4, 0].item(), abs=1.0e-4)


def test_projective_cell_depth_at_uv_uses_depth_affine_uv() -> None:
    times = torch.tensor([0.0, 1.0], dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.0, 1.0, 0.0, 6.0, 0.0, 0.0, 2.0, 0.5, 0.0]], dtype=torch.float32).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 1), dtype=torch.float32),
        cells=[],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
        depth_affine_uv=torch.tensor([[0.1, 0.01, 0.0, -0.2, 0.0, 0.0]], dtype=torch.float32).contiguous(),
    )

    depth = eval_projective_trace_cell_depth_at_uv_torch(
        atlas,
        times,
        u=torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        v=torch.tensor([[5.0, 5.0]], dtype=torch.float32),
    )
    expected = torch.tensor([[2.3, 2.81]], dtype=torch.float32)

    torch.testing.assert_close(depth, expected, atol=1.0e-7, rtol=0.0)


def test_projective_cell_depth_at_uv_falls_back_to_center_depth() -> None:
    times = torch.tensor([0.0, 1.0], dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.0, 1.0, 0.0, 6.0, 0.0, 0.0, 2.0, 0.5, 0.0]], dtype=torch.float32).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 1), dtype=torch.float32),
        cells=[],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
    )

    depth = eval_projective_trace_cell_depth_at_uv_torch(atlas, times, u=123.0, v=-7.0)

    torch.testing.assert_close(depth, torch.tensor([[2.0, 2.5]], dtype=torch.float32), atol=1.0e-7, rtol=0.0)


def test_projective_cell_depth_at_uv_rejects_bad_depth_affine_metadata() -> None:
    times = torch.tensor([0.0], dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.0, 0.0, 0.0, 6.0, 0.0, 0.0, 2.0, 0.0, 0.0]], dtype=torch.float32).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 1), dtype=torch.float32),
        cells=[],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(1,),
        depth_affine_uv=torch.zeros((1, 5), dtype=torch.float32).contiguous(),
    )

    with pytest.raises(ValueError, match="depth_affine_uv"):
        eval_projective_trace_cell_depth_at_uv_torch(atlas, times, u=4.0, v=6.0)


def test_projective_cell_quadrature_reference_uses_spatial_precision_uv() -> None:
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 1), dtype=torch.float32),
        cells=[],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(1,),
        spatial_precision_uv=torch.tensor([[4.0, 0.0, 0.25]], dtype=torch.float32).contiguous(),
    )
    quadrature = ProjectiveTraceCellSensorTimeQuadrature(
        samples=(
            ProjectiveTraceCellSensorTimeQuadratureSample(
                interval_index=0,
                row_index=0,
                start_time=0.0,
                stop_time=1.0,
                time=0.0,
                weight=1.0,
            ),
        ),
        total_weight=1.0,
    )

    image = render_projective_trace_cell_atlas_quadrature_reference(
        atlas,
        quadrature,
        image_width=8,
        image_height=8,
        sigma_px=1.0,
    )

    q_radius2 = 4.0 * (0.5**2) + 0.25 * (0.5**2)
    expected = 0.5 * math.exp(-0.5 * q_radius2)
    assert image[4, 4, 0].item() == pytest.approx(expected, abs=1.0e-7)


def test_projective_cell_quadrature_reference_matches_explicit_weighted_samples() -> None:
    atlas = _direct_continuous_cell_atlas()
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=8,
        image_height=8,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    quadrature = projective_trace_cell_sensor_time_partition_quadrature(
        partition,
        exposure_start=0.25,
        exposure_stop=2.75,
        samples_per_interval=2,
    )

    rendered = render_projective_trace_cell_atlas_quadrature_reference(
        atlas,
        quadrature,
        image_width=8,
        image_height=8,
        sigma_px=1.7,
    )
    sample_times = torch.tensor([sample.time for sample in quadrature.samples], dtype=torch.float32).contiguous()
    sample_weights = torch.tensor([sample.weight for sample in quadrature.samples], dtype=torch.float32).reshape(-1, 1, 1, 1)
    dense_atlas = _direct_continuous_cell_atlas(
        coeffs=atlas.coeffs,
        colors=atlas.color,
        opacities=atlas.opacity,
    )
    dense_atlas = replace(
        dense_atlas,
        active_stop=tuple(int(sample_times.numel()) for _ in range(atlas.coeffs.shape[0])),
    )
    expected = (
        _render_dense_projective_cell_reference(
            dense_atlas,
            sample_times,
            image_width=8,
            image_height=8,
            sigma_px=1.7,
        )
        * sample_weights
    ).sum(dim=0)

    assert quadrature.total_weight == pytest.approx(1.0)
    torch.testing.assert_close(rendered, expected, atol=1.0e-6, rtol=1.0e-6)


def test_projective_cell_quadrature_lowering_builds_sample_indexed_interval_atlas() -> None:
    depth_affine_uv = torch.tensor(
        [[0.1, 0.01, 0.0, -0.2, 0.0, 0.0], [0.05, 0.0, 0.0, 0.1, -0.01, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    atlas = replace(_direct_continuous_cell_atlas(), depth_affine_uv=depth_affine_uv)
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=8,
        image_height=8,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    quadrature = projective_trace_cell_sensor_time_partition_quadrature(
        partition,
        exposure_start=0.25,
        exposure_stop=2.75,
        samples_per_interval=2,
    )

    lowering = lower_projective_trace_cell_atlas_quadrature(
        atlas,
        quadrature,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
    )
    reference = render_projective_trace_cell_atlas_quadrature_reference(
        atlas,
        quadrature,
        image_width=8,
        image_height=8,
        sigma_px=1.7,
    )
    interval_samples = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.7,
    )
    lowered = (interval_samples * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0)

    assert lowering.source_trace_indices == (0, 1)
    assert lowering.times.tolist() == sorted(sample.time for sample in quadrature.samples)
    assert lowering.weights.sum().item() == pytest.approx(1.0)
    assert all(cell.start >= 0 and cell.stop <= int(lowering.times.numel()) for cell in lowering.atlas.cells)
    torch.testing.assert_close(lowering.atlas.depth_affine_uv, depth_affine_uv)
    torch.testing.assert_close(lowered, reference, atol=1.0e-6, rtol=1.0e-6)


def test_projective_cell_quadrature_lowering_respects_domain_time_activity() -> None:
    coeffs = torch.tensor(
        [
            [2.0, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=torch.tensor([0.7, 0.7], dtype=torch.float32),
        color=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        cells=[],
        source_window_indices=(0, 1),
        source_primitive_ids=(10, 10),
        active_start=(0, 2),
        active_stop=(2, 4),
    )
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=8,
        image_height=8,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    quadrature = projective_trace_cell_sensor_time_partition_quadrature(
        partition,
        exposure_start=0.25,
        exposure_stop=2.75,
        samples_per_interval=1,
    )

    lowering = lower_projective_trace_cell_atlas_quadrature(
        atlas,
        quadrature,
        image_width=8,
        image_height=8,
        tile_size=8,
        domain_times=domain_times,
        uv_padding=4.0,
    )

    assert lowering.source_trace_indices == (0, 1)
    assert lowering.atlas.active_start == (0, 2)
    assert lowering.atlas.active_stop == (2, 3)
    assert all(0 <= cell.start < cell.stop <= int(lowering.times.numel()) for cell in lowering.atlas.cells)


def test_projective_cell_quadrature_reference_backprops_to_trace_params() -> None:
    coeffs = torch.tensor(
        [
            [3.5, 0.25, 0.0, 3.5, 0.08, 0.0, 1.0, 0.10, 0.0],
            [4.6, -0.18, 0.0, 3.2, 0.12, 0.0, 1.8, -0.06, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    ).contiguous()
    colors = torch.tensor([[1.0, 0.1, 0.05], [0.05, 0.25, 1.0]], dtype=torch.float32, requires_grad=True)
    opacities = torch.tensor([0.65, 0.45], dtype=torch.float32, requires_grad=True)
    atlas = _direct_continuous_cell_atlas(coeffs=coeffs, colors=colors, opacities=opacities)
    quadrature = ProjectiveTraceCellSensorTimeQuadrature(
        samples=tuple(
            projective_trace_cell_sensor_time_partition_quadrature(
                projective_trace_cell_sensor_time_event_partition(
                    atlas,
                    torch.arange(4, dtype=torch.float32).contiguous(),
                    image_width=8,
                    image_height=8,
                    tile_size=8,
                    include_support=False,
                    include_visibility=False,
                    extra_split_times=[1.5],
                ),
                exposure_start=0.25,
                exposure_stop=2.75,
                samples_per_interval=1,
            ).samples
        ),
        total_weight=1.0,
    )

    rendered = render_projective_trace_cell_atlas_quadrature_reference(
        atlas,
        quadrature,
        image_width=8,
        image_height=8,
        sigma_px=1.7,
    )
    rendered.square().sum().backward()

    assert coeffs.grad is not None
    assert colors.grad is not None
    assert opacities.grad is not None
    assert float(coeffs.grad[:, :6].abs().sum().item()) > 0.0
    assert float(colors.grad.abs().sum().item()) > 0.0
    assert float(opacities.grad.abs().sum().item()) > 0.0


def test_projective_cell_rolling_quadrature_reference_uses_per_row_schedules() -> None:
    atlas = _direct_continuous_cell_atlas()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        torch.arange(4, dtype=torch.float32).contiguous(),
        image_width=8,
        image_height=3,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    row_quadrature = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=3,
        frame_time=0.0,
        exposure_duration=0.75,
        readout_duration=1.0,
        samples_per_interval=2,
    )

    rolling = render_projective_trace_cell_atlas_rolling_quadrature_reference(
        atlas,
        row_quadrature,
        image_width=8,
        image_height=3,
        sigma_px=1.7,
    )
    expected_rows = []
    for row_index, quadrature in enumerate(row_quadrature):
        full_frame = render_projective_trace_cell_atlas_quadrature_reference(
            atlas,
            quadrature,
            image_width=8,
            image_height=3,
            sigma_px=1.7,
        )
        expected_rows.append(full_frame[row_index])

    assert [sample.row_index for sample in row_quadrature[2].samples] == [2, 2]
    torch.testing.assert_close(rolling, torch.stack(expected_rows, dim=0), atol=1.0e-6, rtol=1.0e-6)
    assert float((rolling[0] - rolling[2]).abs().sum().item()) > 1.0e-4


def test_projective_cell_rolling_quadrature_batched_lowering_reuses_sample_times() -> None:
    atlas = _direct_continuous_cell_atlas()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        torch.arange(4, dtype=torch.float32).contiguous(),
        image_width=8,
        image_height=3,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    row_quadrature = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=3,
        frame_time=0.0,
        exposure_duration=0.75,
        readout_duration=1.0,
        samples_per_interval=2,
    )

    lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
        atlas,
        row_quadrature,
        image_width=8,
        image_height=3,
        tile_size=8,
        uv_padding=4.0,
    )
    total_row_samples = sum(len(quadrature.samples) for quadrature in row_quadrature)

    assert lowering.row_weights.shape == (int(lowering.times.numel()), 3)
    assert int(lowering.times.numel()) < total_row_samples
    torch.testing.assert_close(
        lowering.row_weights.sum(dim=0),
        torch.ones(3, dtype=torch.float32),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_projective_cell_rolling_quadrature_batched_reference_matches_rowwise_reference() -> None:
    atlas = _direct_continuous_cell_atlas()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        torch.arange(4, dtype=torch.float32).contiguous(),
        image_width=8,
        image_height=3,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    row_quadrature = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=3,
        frame_time=0.0,
        exposure_duration=0.75,
        readout_duration=1.0,
        samples_per_interval=2,
    )

    rowwise = render_projective_trace_cell_atlas_rolling_quadrature_reference(
        atlas,
        row_quadrature,
        image_width=8,
        image_height=3,
        sigma_px=1.7,
    )
    batched = render_projective_trace_cell_atlas_rolling_quadrature_batched_reference(
        atlas,
        row_quadrature,
        image_width=8,
        image_height=3,
        tile_size=8,
        sigma_px=1.7,
        uv_padding=4.0,
    )

    torch.testing.assert_close(batched, rowwise, atol=1.0e-6, rtol=1.0e-6)


def test_projective_cell_quadrature_interval_metal_matches_reference_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal render op unavailable")

    atlas = _direct_continuous_cell_atlas()
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=8,
        image_height=8,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    quadrature = projective_trace_cell_sensor_time_partition_quadrature(
        partition,
        exposure_start=0.25,
        exposure_stop=2.75,
        samples_per_interval=2,
    )
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=1,
        tile_x=8,
        tile_y=8,
        tile_t=1,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    ref = render_projective_trace_cell_atlas_quadrature_reference(
        atlas,
        quadrature,
        image_width=8,
        image_height=8,
        sigma_px=1.7,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=[],
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    metal = render_projective_trace_cell_atlas_quadrature_interval_metal(
        atlas_mps,
        quadrature,
        config,
        sigma_px=1.7,
        uv_padding=4.0,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)


def test_projective_cell_rolling_quadrature_interval_metal_matches_reference_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_rows_metal():
        pytest.skip("STAR UVT projective interval row Metal render op unavailable")

    atlas = _direct_continuous_cell_atlas()
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=8,
        image_height=3,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    row_quadrature = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=3,
        frame_time=0.0,
        exposure_duration=0.75,
        readout_duration=1.0,
        samples_per_interval=2,
    )
    config = UVTRenderConfig(
        height=3,
        width=8,
        frames=1,
        tile_x=8,
        tile_y=8,
        tile_t=1,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    ref = render_projective_trace_cell_atlas_rolling_quadrature_reference(
        atlas,
        row_quadrature,
        image_width=8,
        image_height=3,
        sigma_px=1.7,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=[],
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    metal = render_projective_trace_cell_atlas_rolling_quadrature_interval_metal(
        atlas_mps,
        row_quadrature,
        config,
        sigma_px=1.7,
        uv_padding=4.0,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)


def test_projective_cell_quadrature_interval_mixed_metal_patches_fallback_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal render op unavailable")

    atlas = _mixed_fallback_cell_atlas()
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=16,
        image_height=8,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    quadrature = projective_trace_cell_sensor_time_partition_quadrature(
        partition,
        exposure_start=0.25,
        exposure_stop=2.75,
        samples_per_interval=2,
    )
    config = UVTRenderConfig(
        height=8,
        width=16,
        frames=1,
        tile_x=8,
        tile_y=8,
        tile_t=1,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    lowering = lower_projective_trace_cell_atlas_quadrature(
        atlas,
        quadrature,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )
    lowering = replace(
        lowering,
        atlas=mark_projective_trace_cell_visibility_fallbacks(
            lowering.atlas,
            lowering.times,
            depth_epsilon=1.0e-6,
        ),
    )
    stats = projective_trace_cell_atlas_fallback_stats(lowering.atlas)
    assert stats.fallback_cells > 0
    assert stats.fallback_cells < stats.total_cells
    interval_ref = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.7,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
        allow_fallback_cells=True,
    )
    expected = (interval_ref * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0)
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=[],
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    mixed = render_projective_trace_cell_atlas_quadrature_interval_mixed_metal(
        atlas_mps,
        quadrature,
        config,
        sigma_px=1.7,
        uv_padding=2.0,
        depth_epsilon=1.0e-6,
    )

    torch.testing.assert_close(mixed.cpu(), expected, atol=2.0e-4, rtol=2.0e-4)


def test_projective_cell_rolling_quadrature_interval_mixed_metal_patches_fallback_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal render op unavailable")

    atlas = _mixed_fallback_cell_atlas()
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=16,
        image_height=4,
        tile_size=8,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    row_quadrature = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=4,
        frame_time=0.0,
        exposure_duration=0.75,
        readout_duration=1.0,
        samples_per_interval=2,
    )
    config = UVTRenderConfig(
        height=4,
        width=16,
        frames=1,
        tile_x=8,
        tile_y=8,
        tile_t=1,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
        atlas,
        row_quadrature,
        image_width=16,
        image_height=4,
        tile_size=8,
        uv_padding=2.0,
    )
    lowering = replace(
        lowering,
        atlas=mark_projective_trace_cell_visibility_fallbacks(
            lowering.atlas,
            lowering.times,
            depth_epsilon=1.0e-6,
        ),
    )
    stats = projective_trace_cell_atlas_fallback_stats(lowering.atlas)
    assert stats.fallback_cells > 0
    assert stats.fallback_cells < stats.total_cells
    interval_ref = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=16,
        image_height=4,
        tile_size=8,
        sigma_px=1.7,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
        allow_fallback_cells=True,
    )
    expected = (interval_ref * lowering.row_weights.reshape(-1, 4, 1, 1)).sum(dim=0)
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=[],
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    mixed = render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal(
        atlas_mps,
        row_quadrature,
        config,
        sigma_px=1.7,
        uv_padding=2.0,
        depth_epsilon=1.0e-6,
    )

    torch.testing.assert_close(mixed.cpu(), expected, atol=2.0e-4, rtol=2.0e-4)


def test_projective_atlas_reference_renderer_matches_dense_per_frame_compositing() -> None:
    coeffs = torch.tensor(
        [
            [3.5, 0.25, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.5, -0.25, 0.0, 3.5, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.linspace(-1.0, 1.0, 5, dtype=torch.float32).contiguous()
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    opacities = torch.tensor([0.55, 0.35], dtype=torch.float32)
    sigma_px = 2.0

    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
    )
    bounds = bound_projective_trace_windows(windows, uv_padding=6.0)
    records = bin_projective_trace_support_bounds(
        bounds,
        image_width=8,
        image_height=8,
        tile_size=8,
        primitive_ids=[10, 20],
    )
    cells = assemble_projective_trace_tile_time_atlas(records)

    atlas_render = render_projective_trace_tile_time_atlas_reference(
        cells,
        coeffs,
        times,
        colors,
        opacities,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=sigma_px,
        primitive_ids=[10, 20],
    )
    dense_render = _render_dense_projective_reference(
        coeffs,
        times,
        colors,
        opacities,
        image_width=8,
        image_height=8,
        sigma_px=sigma_px,
    )

    torch.testing.assert_close(atlas_render, dense_render, atol=1.0e-6, rtol=1.0e-6)


def test_projective_tile_time_bins_preserve_split_window_intervals() -> None:
    coeffs = torch.tensor(
        [[4.0, 0.0, 0.35, 4.0, 0.45, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(8, dtype=torch.float32).sub_(3.5).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert len(windows) > 1
    bounds = bound_projective_trace_windows(windows, uv_padding=4.0)
    records = bin_projective_trace_support_bounds(
        bounds,
        image_width=8,
        image_height=8,
        tile_size=8,
    )
    cells = assemble_projective_trace_tile_time_atlas(records)
    bins = pack_projective_trace_tile_time_bins(
        cells,
        image_width=8,
        image_height=8,
        frames=8,
        tile_x=8,
        tile_y=8,
        tile_t=4,
        tile_capacity=128,
    )

    starts = bins.tile_active_start[bins.tile_primitive_ids >= 0].tolist()
    stops = bins.tile_active_stop[bins.tile_primitive_ids >= 0].tolist()
    assert (0, 8) not in set(zip(starts, stops))
    assert len(set(zip(starts, stops))) == len(windows)
    assert int(bins.tile_overflow.sum().item()) == 0


def test_projective_split_windows_lower_to_cell_trace_atlas_reference() -> None:
    coeffs = torch.tensor(
        [[4.0, 0.0, 0.35, 4.0, 0.45, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(8, dtype=torch.float32).sub_(3.5).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    opacities = torch.tensor([0.6], dtype=torch.float32)
    sigma_px = 1.6
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert len(windows) > 1

    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacities,
        color=colors,
        image_width=8,
        image_height=8,
        tile_size=8,
        primitive_ids=[7],
        uv_padding=4.0,
    )
    assert atlas.coeffs.shape == (len(windows), 9)
    assert atlas.source_window_indices == tuple(range(len(windows)))
    assert atlas.source_primitive_ids == (7,) * len(windows)
    assert atlas.active_start[0] == windows[0].start
    assert atlas.active_stop[-1] == windows[-1].stop

    cell_samples = eval_projective_trace_cell_torch(atlas.coeffs, times)
    assert torch.all(cell_samples[:, :, 3] == 1.0)
    cell_render = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=sigma_px,
    )
    dense_render = _render_dense_projective_reference(
        coeffs,
        times,
        colors,
        opacities,
        image_width=8,
        image_height=8,
        sigma_px=sigma_px,
    )

    torch.testing.assert_close(cell_render, dense_render, atol=1.0e-6, rtol=1.0e-6)


def test_projective_cell_atlas_coverage_report_detects_motion_and_rebin_repairs() -> None:
    coeffs = torch.tensor(
        [[4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    opacities = torch.tensor([0.6], dtype=torch.float32)
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
        opacity=opacities,
        color=colors,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )

    fresh = projective_trace_cell_atlas_coverage_report(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )
    assert not fresh.stale
    assert fresh.checked_tile_pairs == 4
    assert fresh.missing_tile_pairs == 0
    fresh_margin = projective_trace_cell_atlas_support_margin_report(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )
    assert not fresh_margin.stale
    assert fresh_margin.max_boundary_overshoot_px == 0.0
    assert fresh_margin.min_boundary_slack_px >= 0.0

    moved_coeffs = atlas.coeffs.clone()
    moved_coeffs[:, 0] += 8.5
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
    stale = projective_trace_cell_atlas_coverage_report(
        moved_atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )
    assert stale.stale
    assert stale.missing_tile_pairs == 4
    assert stale.missing_examples[0] == (0, 0, 1, 0)
    stale_margin = projective_trace_cell_atlas_support_margin_report(
        moved_atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )
    assert stale_margin.stale
    assert stale_margin.missing_tile_pairs == 4
    assert stale_margin.max_boundary_overshoot_px == pytest.approx(5.5)
    assert stale_margin.mean_boundary_overshoot_px == pytest.approx(5.5)
    assert stale_margin.min_boundary_slack_px == pytest.approx(-5.5)
    assert stale_margin.mean_boundary_slack_px == pytest.approx(-5.5)
    assert stale_margin.missing_examples[0][:4] == (0, 0, 1, 0)
    assert stale_margin.missing_examples[0][4] == pytest.approx(5.5)

    rebinned = rebin_projective_trace_cell_atlas(
        moved_atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )
    repaired = projective_trace_cell_atlas_coverage_report(
        rebinned,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=1.0,
    )
    assert not repaired.stale
    assert repaired.missing_tile_pairs == 0
    assert any(cell.tile_u == 1 and cell.tile_v == 0 and 0 in cell.primitive_ids for cell in rebinned.cells)


def test_projective_cell_atlas_visibility_report_detects_depth_order_flip_and_rebin_repairs() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32)
    opacities = torch.tensor([0.6, 0.4], dtype=torch.float32)
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
        opacity=opacities,
        color=colors,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )
    fresh = projective_trace_cell_atlas_visibility_report(atlas, times)
    assert not fresh.stale
    assert fresh.checked_tile_samples == 4
    assert fresh.order_mismatch_samples == 0
    assert atlas.cells[0].ordered_primitive_ids == (0, 1)

    moved_coeffs = atlas.coeffs.clone()
    moved_coeffs[0, 6] = 3.0
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
    stale = projective_trace_cell_atlas_visibility_report(moved_atlas, times)
    assert stale.stale
    assert stale.order_mismatch_samples == 4
    assert stale.mismatch_examples[0] == (0, 0, 0, 0, 1)

    rebinned = rebin_projective_trace_cell_atlas(
        moved_atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )
    repaired = projective_trace_cell_atlas_visibility_report(rebinned, times)
    assert not repaired.stale
    assert repaired.order_mismatch_samples == 0
    assert rebinned.cells[0].ordered_primitive_ids == (1, 0)
    assert rebinned.coeffs is moved_atlas.coeffs


def test_projective_cell_atlas_visibility_ambiguity_marks_fallback_cells() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0005, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32)
    opacities = torch.tensor([0.6, 0.4], dtype=torch.float32)
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
        opacity=opacities,
        color=colors,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )

    ambiguous = projective_trace_cell_atlas_visibility_report(atlas, times, depth_epsilon=1.0e-3)
    assert ambiguous.stale
    assert ambiguous.order_mismatch_samples == 0
    assert ambiguous.ambiguous_depth_samples == 4
    assert ambiguous.ambiguous_examples[0] == (0, 0, 0, 0, 1)

    fallback_atlas = mark_projective_trace_cell_visibility_fallbacks(
        atlas,
        times,
        depth_epsilon=1.0e-3,
    )
    stats = projective_trace_cell_atlas_fallback_stats(fallback_atlas)
    assert any(cell.fallback for cell in fallback_atlas.cells)
    assert fallback_atlas.cells[0].fallback_reasons == ("visibility_ambiguous_depth",)
    assert stats.fallback_cells == 1
    assert stats.fallback_tile_samples == 4
    assert stats.fallback_trace_samples == 8
    assert stats.fallback_fraction == 1.0
    assert stats.fallback_reasons == ("visibility_ambiguous_depth",)
    with pytest.raises(ValueError, match="fallback"):
        pack_projective_trace_tile_time_bins(
            fallback_atlas.cells,
            image_width=8,
            image_height=8,
            frames=4,
            tile_x=8,
            tile_y=8,
            tile_t=2,
            tile_capacity=128,
        )
    bins = pack_projective_trace_tile_time_bins(
        fallback_atlas.cells,
        image_width=8,
        image_height=8,
        frames=4,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        allow_fallback_cells=True,
    )
    assert int(bins.tile_counts.sum().item()) > 0


def test_projective_cell_atlas_visibility_stratifies_depth_crossing_without_fallback() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, -0.2, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
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
    stale = projective_trace_cell_atlas_visibility_report(atlas, times)
    assert stale.stale
    assert stale.order_mismatch_samples == 2

    stratified = stratify_projective_trace_cell_atlas_visibility(atlas, times)
    repaired = projective_trace_cell_atlas_visibility_report(stratified, times)
    stats = projective_trace_cell_atlas_complexity_stats(stratified)
    budget = projective_trace_cell_atlas_budget_report(
        stratified,
        max_interval_to_dense_trace_sample_ratio=0.60,
        max_fallback_fraction=0.0,
        max_cells_per_active_set_group=2,
    )
    assert not repaired.stale
    assert repaired.order_mismatch_samples == 0
    assert stats.fallback_cells == 0
    assert stats.tile_active_set_groups == 1
    assert stats.visibility_stratum_split_cells == 1
    assert stats.max_cells_per_active_set_group == 2
    assert stats.interval_trace_entries == 4
    assert stats.dense_trace_samples == 8
    assert stats.interval_to_dense_trace_sample_ratio == 0.5
    assert budget.within_budget
    assert budget.failures == ()
    assert [(cell.start, cell.stop, cell.ordered_primitive_ids) for cell in stratified.cells] == [
        (0, 2, (0, 1)),
        (2, 4, (1, 0)),
    ]

    tight_budget = projective_trace_cell_atlas_budget_report(
        stratified,
        max_interval_to_dense_trace_sample_ratio=0.40,
        max_fallback_fraction=0.0,
        max_cells_per_active_set_group=1,
    )
    assert not tight_budget.within_budget
    assert tight_budget.failures == (
        "interval_to_dense_trace_sample_ratio",
        "max_cells_per_active_set_group",
    )


def test_projective_cell_atlas_reference_fallback_sorts_live_depth() -> None:
    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.2, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(4, dtype=torch.float32).contiguous()
    colors = torch.tensor([[1.0, 0.05, 0.05], [0.05, 0.05, 1.0]], dtype=torch.float32)
    opacities = torch.tensor([0.8, 0.7], dtype=torch.float32)
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
        opacity=opacities,
        color=colors,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=2.0,
    )
    assert atlas.cells[0].ordered_primitive_ids == (0, 1)

    moved_coeffs = atlas.coeffs.clone()
    moved_coeffs[0, 6] = 3.0
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
    fallback_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=moved_coeffs,
        opacity=atlas.opacity,
        color=atlas.color,
        cells=[
            replace(
                cell,
                fallback=True,
                fallback_reasons=("visibility_order_live_sort",),
            )
            for cell in atlas.cells
        ],
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )

    static_render = render_projective_trace_cell_atlas_reference(
        moved_atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=2.0,
    )
    live_fallback_render = render_projective_trace_cell_atlas_reference(
        fallback_atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=2.0,
        allow_fallback_cells=True,
    )
    dense_cell_render = _render_dense_projective_cell_reference(
        fallback_atlas,
        times,
        image_width=8,
        image_height=8,
        sigma_px=2.0,
    )

    assert float((static_render - dense_cell_render).abs().amax().item()) > 1.0e-2
    torch.testing.assert_close(live_fallback_render, dense_cell_render, atol=1.0e-6, rtol=1.0e-6)


def test_projective_cell_fallback_split_and_mask_tracks_whole_tile_samples() -> None:
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.zeros((2, 9), dtype=torch.float32).contiguous(),
        opacity=torch.ones((2,), dtype=torch.float32),
        color=torch.ones((2, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((1.0, 1.1),),
                fallback=False,
                fallback_reasons=(),
            ),
            ProjectiveTraceTileTimeCell(
                tile_u=1,
                tile_v=0,
                start=1,
                stop=3,
                primitive_ids=(1,),
                ordered_primitive_ids=(1,),
                depth_intervals=((1.0, 1.1),),
                fallback=True,
                fallback_reasons=("visibility_ambiguous_depth",),
            ),
        ],
        source_window_indices=(0, 1),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(3, 3),
    )

    fast_atlas, fallback_atlas = split_projective_trace_cell_atlas_fallback_cells(atlas)
    mask = projective_trace_cell_atlas_fallback_tile_sample_mask(
        atlas,
        frames=3,
        image_width=16,
        image_height=8,
        tile_size=8,
    )

    assert len(fast_atlas.cells) == 1
    assert len(fallback_atlas.cells) == 1
    assert mask[:, 0, 0].tolist() == [False, False, False]
    assert mask[:, 0, 1].tolist() == [False, True, True]


def test_projective_cell_trace_atlas_renders_in_metal_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_metal():
        pytest.skip("STAR UVT projective cell atlas Metal render op unavailable")

    coeffs = torch.tensor(
        [[4.0, 0.0, 0.35, 4.0, 0.45, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(8, dtype=torch.float32).sub_(3.5).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    opacities = torch.tensor([0.6], dtype=torch.float32)
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
        opacity=opacities,
        color=colors,
        image_width=8,
        image_height=8,
        tile_size=8,
        primitive_ids=[7],
        uv_padding=4.0,
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
    ref = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.6,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    metal = render_projective_trace_cell_atlas_metal(
        atlas_mps,
        times.to("mps"),
        config,
        sigma_px=1.6,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)


def test_projective_cell_trace_interval_atlas_renders_in_metal_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal render op unavailable")

    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.10, 4.0, 0.12, 0.0, 1.0, 0.0, 0.0],
            [4.6, 0.0, -0.08, 4.2, -0.10, 0.0, 1.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32)
    opacities = torch.tensor([0.6, 0.4], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=2,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=3,
    )
    assert len(windows) == 1
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacities,
        color=colors,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
    )
    opacity_time_coeffs = torch.tensor(
        [[0.08, -0.03, 0.05], [0.02, 0.04, 0.07]],
        dtype=torch.float32,
    ).contiguous()
    atlas = replace(atlas, opacity_time_coeffs=opacity_time_coeffs)
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=16,
        tile_x=8,
        tile_y=8,
        tile_t=4,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    ref = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.6,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        opacity_time_coeffs=atlas.opacity_time_coeffs.to("mps"),
    )
    metal = render_projective_trace_cell_interval_atlas_metal(
        atlas_mps,
        times.to("mps"),
        config,
        sigma_px=1.6,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)


def test_projective_cell_trace_interval_atlas_forward_uses_spatial_precision_uv_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal op unavailable")

    times = torch.zeros((1,), dtype=torch.float32).contiguous()
    cell = ProjectiveTraceTileTimeCell(
        tile_u=0,
        tile_v=0,
        start=0,
        stop=1,
        primitive_ids=(0,),
        ordered_primitive_ids=(0,),
        depth_intervals=((1.0, 1.0),),
        fallback=False,
        fallback_reasons=(),
    )
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.tensor([[1.0, 0.25, 0.1]], dtype=torch.float32),
        cells=[cell],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(1,),
        spatial_precision_uv=torch.tensor([[4.0, 0.0, 0.25]], dtype=torch.float32).contiguous(),
    )
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=1,
        tile_x=8,
        tile_y=8,
        tile_t=1,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    ref = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        spatial_precision_uv=atlas.spatial_precision_uv.to("mps"),
    )
    metal = render_projective_trace_cell_interval_atlas_metal(
        atlas_mps,
        times.to("mps"),
        config,
        sigma_px=1.0,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)


def test_projective_cell_trace_interval_atlas_forward_uses_depth_affine_uv_for_order_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal op unavailable")

    times = torch.zeros((1,), dtype=torch.float32).contiguous()
    cell = ProjectiveTraceTileTimeCell(
        tile_u=0,
        tile_v=0,
        start=0,
        stop=1,
        primitive_ids=(0, 1),
        ordered_primitive_ids=(0, 1),
        depth_intervals=((0.0, 3.0), (-1.0, 2.0)),
        fallback=False,
        fallback_reasons=(),
    )
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [
                [3.5, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
                [3.5, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([0.7, 0.7], dtype=torch.float32),
        color=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        cells=[cell],
        source_window_indices=(0, 0),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(1, 1),
        depth_affine_uv=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
    )
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=1,
        tile_x=8,
        tile_y=8,
        tile_t=1,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        depth_affine_uv=atlas.depth_affine_uv.to("mps"),
    )
    metal = render_projective_trace_cell_interval_atlas_metal(
        atlas_mps,
        times.to("mps"),
        config,
        sigma_px=2.0,
    ).cpu()

    assert metal[0, 3, 2, 0].item() > metal[0, 3, 2, 2].item()
    assert metal[0, 3, 5, 2].item() > metal[0, 3, 5, 0].item()


def test_projective_cell_trace_interval_atlas_backward_matches_torch_autograd_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal backward op unavailable")

    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.10, 4.0, 0.12, 0.0, 1.0, 0.0, 0.0],
            [4.6, 0.0, -0.08, 4.2, -0.10, 0.0, 1.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32)
    opacities = torch.tensor([0.6, 0.4], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=2,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=3,
    )
    assert len(windows) == 1
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacities,
        color=colors,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
    )
    opacity_time_coeffs = torch.tensor(
        [[0.08, -0.03, 0.05], [0.02, 0.04, 0.07]],
        dtype=torch.float32,
    ).contiguous()
    atlas = replace(atlas, opacity_time_coeffs=opacity_time_coeffs)
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=16,
        tile_x=8,
        tile_y=8,
        tile_t=4,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    grad_image = torch.linspace(-0.2, 0.35, steps=config.frames * config.height * config.width * 3, dtype=torch.float32)
    grad_image = grad_image.reshape(config.frames, config.height, config.width, 3).contiguous()

    ref_coeffs = atlas.coeffs.clone().detach().requires_grad_(True)
    ref_colors = atlas.color.clone().detach().requires_grad_(True)
    ref_opacities = atlas.opacity.clone().detach().requires_grad_(True)
    ref_opacity_time_coeffs = atlas.opacity_time_coeffs.clone().detach().requires_grad_(True)
    ref_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=ref_coeffs,
        opacity=ref_opacities,
        color=ref_colors,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        opacity_time_coeffs=ref_opacity_time_coeffs,
    )
    ref = render_projective_trace_cell_atlas_reference(
        ref_atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.6,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    (ref * grad_image).sum().backward()

    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        opacity_time_coeffs=atlas.opacity_time_coeffs.to("mps"),
    )
    grads = direct_backward_projective_trace_cell_interval_atlas_metal(
        atlas_mps,
        times.to("mps"),
        grad_image.to("mps"),
        config,
        sigma_px=1.6,
    )

    torch.testing.assert_close(grads.grad_color.cpu(), ref_colors.grad, atol=5.0e-4, rtol=5.0e-4)
    torch.testing.assert_close(grads.grad_opacity.cpu(), ref_opacities.grad, atol=5.0e-4, rtol=5.0e-4)
    assert grads.grad_opacity_time_coeffs is not None
    torch.testing.assert_close(grads.grad_opacity_time_coeffs.cpu(), ref_opacity_time_coeffs.grad, atol=5.0e-4, rtol=5.0e-4)
    torch.testing.assert_close(grads.grad_coeffs.cpu(), ref_coeffs.grad, atol=8.0e-4, rtol=8.0e-4)
    assert float(grads.grad_coeffs[:, :6].cpu().abs().sum().item()) > 0.0
    assert float(grads.grad_opacity_time_coeffs.cpu().abs().sum().item()) > 0.0


def test_projective_cell_trace_interval_atlas_backward_uses_spatial_precision_uv_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal backward op unavailable")

    times = torch.tensor([-0.5, 0.5], dtype=torch.float32).contiguous()
    cell = ProjectiveTraceTileTimeCell(
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
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor([[4.0, 0.1, 0.0, 4.0, -0.08, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32).contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.tensor([[1.0, 0.25, 0.1]], dtype=torch.float32),
        cells=[cell],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(2,),
        opacity_time_coeffs=torch.tensor([[0.04, -0.02, 0.03]], dtype=torch.float32).contiguous(),
        spatial_precision_uv=torch.tensor([[4.0, 0.2, 0.25]], dtype=torch.float32).contiguous(),
    )
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=2,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    grad_image = torch.linspace(-0.2, 0.35, steps=config.frames * config.height * config.width * 3, dtype=torch.float32)
    grad_image = grad_image.reshape(config.frames, config.height, config.width, 3).contiguous()

    ref_coeffs = atlas.coeffs.clone().detach().requires_grad_(True)
    ref_colors = atlas.color.clone().detach().requires_grad_(True)
    ref_opacities = atlas.opacity.clone().detach().requires_grad_(True)
    ref_opacity_time_coeffs = atlas.opacity_time_coeffs.clone().detach().requires_grad_(True)
    ref_spatial_precision_uv = atlas.spatial_precision_uv.clone().detach().requires_grad_(True)
    ref_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=ref_coeffs,
        opacity=ref_opacities,
        color=ref_colors,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        opacity_time_coeffs=ref_opacity_time_coeffs,
        spatial_precision_uv=ref_spatial_precision_uv,
    )
    ref = render_projective_trace_cell_atlas_reference(
        ref_atlas,
        times,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    (ref * grad_image).sum().backward()

    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        opacity_time_coeffs=atlas.opacity_time_coeffs.to("mps"),
        spatial_precision_uv=atlas.spatial_precision_uv.to("mps"),
    )
    grads = direct_backward_projective_trace_cell_interval_atlas_metal(
        atlas_mps,
        times.to("mps"),
        grad_image.to("mps"),
        config,
        sigma_px=1.0,
    )

    torch.testing.assert_close(grads.grad_color.cpu(), ref_colors.grad, atol=5.0e-4, rtol=5.0e-4)
    torch.testing.assert_close(grads.grad_opacity.cpu(), ref_opacities.grad, atol=5.0e-4, rtol=5.0e-4)
    assert grads.grad_opacity_time_coeffs is not None
    torch.testing.assert_close(grads.grad_opacity_time_coeffs.cpu(), ref_opacity_time_coeffs.grad, atol=5.0e-4, rtol=5.0e-4)
    assert grads.grad_spatial_precision_uv is not None
    torch.testing.assert_close(grads.grad_spatial_precision_uv.cpu(), ref_spatial_precision_uv.grad, atol=5.0e-4, rtol=5.0e-4)
    torch.testing.assert_close(grads.grad_coeffs.cpu(), ref_coeffs.grad, atol=8.0e-4, rtol=8.0e-4)
    assert float(grads.grad_coeffs[:, :6].cpu().abs().sum().item()) > 0.0
    assert float(grads.grad_spatial_precision_uv.cpu().abs().sum().item()) > 0.0


def test_projective_cell_trace_interval_atlas_one_step_coeff_training_smoke_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("STAR UVT projective interval cell atlas Metal backward op unavailable")

    start_coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.10, 4.0, 0.12, 0.0, 1.0, 0.0, 0.0],
            [4.6, 0.0, -0.08, 4.2, -0.10, 0.0, 1.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    target_coeffs = start_coeffs.clone()
    target_coeffs[:, 0] += torch.tensor([0.35, -0.20], dtype=torch.float32)
    target_coeffs[:, 3] += torch.tensor([-0.25, 0.18], dtype=torch.float32)
    target_coeffs[:, 4] += torch.tensor([0.02, -0.03], dtype=torch.float32)
    times = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1], [0.1, 0.35, 0.85]], dtype=torch.float32)
    opacities = torch.tensor([0.6, 0.4], dtype=torch.float32)
    windows = split_projective_trace_windows(
        start_coeffs,
        times,
        degree=2,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=3,
    )
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacities,
        color=colors,
        image_width=8,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
    )
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
        frames=16,
        tile_x=8,
        tile_y=8,
        tile_t=4,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )

    device = torch.device("mps")
    train_atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to(device),
        opacity=atlas.opacity.to(device),
        color=atlas.color.to(device),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
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
    before = render_projective_trace_cell_interval_atlas_metal(
        train_atlas_mps,
        times.to(device),
        config,
        sigma_px=1.6,
    )
    before_loss = (before - target).square().mean()
    grad_image = (2.0 / float(before.numel())) * (before - target)
    grads = direct_backward_projective_trace_cell_interval_atlas_metal(
        train_atlas_mps,
        times.to(device),
        grad_image.contiguous(),
        config,
        sigma_px=1.6,
    )

    candidate_losses = []
    for lr in (16.0, 64.0, 128.0, 256.0, 512.0):
        stepped_atlas = ProjectiveTraceCellTraceAtlas(
            coeffs=(train_atlas_mps.coeffs - lr * grads.grad_coeffs).detach(),
            opacity=train_atlas_mps.opacity,
            color=train_atlas_mps.color,
            cells=train_atlas_mps.cells,
            source_window_indices=train_atlas_mps.source_window_indices,
            source_primitive_ids=train_atlas_mps.source_primitive_ids,
            active_start=train_atlas_mps.active_start,
            active_stop=train_atlas_mps.active_stop,
        )
        after = render_projective_trace_cell_interval_atlas_metal(
            stepped_atlas,
            times.to(device),
            config,
            sigma_px=1.6,
        )
        candidate_losses.append((after - target).square().mean())

    best_loss = torch.stack(candidate_losses).min()
    assert float(before_loss.cpu().item()) > 1.0e-6
    assert float(grads.grad_coeffs[:, :6].cpu().abs().sum().item()) > 0.0
    assert float(best_loss.cpu().item()) < 0.99 * float(before_loss.cpu().item())


def test_projective_quadratic_atlas_cells_render_in_metal_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "render_projective_trace_tiles"):
        pytest.skip("STAR UVT projective atlas Metal render op unavailable")

    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.35, 4.0, 0.45, 0.0, 1.0, 0.0, 0.0],
            [4.5, 0.0, -0.20, 4.1, -0.35, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(8, dtype=torch.float32).sub_(3.5).contiguous()
    colors = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.3, 0.9]], dtype=torch.float32)
    opacities = torch.tensor([0.55, 0.45], dtype=torch.float32)
    sigma_px = 1.6
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=2,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=3,
    )
    assert len(windows) == 1
    assert windows[0].fit.degree == 2
    bounds = bound_projective_trace_windows(windows, uv_padding=5.0)
    records = bin_projective_trace_support_bounds(
        bounds,
        image_width=8,
        image_height=8,
        tile_size=8,
    )
    cells = assemble_projective_trace_tile_time_atlas(records)
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
    ref = render_projective_trace_tile_time_atlas_reference(
        cells,
        coeffs,
        times,
        colors,
        opacities,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=sigma_px,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    metal = render_projective_trace_tile_time_atlas_metal(
        cells,
        coeffs.to("mps"),
        times.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        config,
        sigma_px=sigma_px,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)


def test_projective_quadratic_atlas_cell_backward_matches_torch_autograd_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "direct_projective_trace_backward"):
        pytest.skip("STAR UVT projective atlas Metal backward op unavailable")

    coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.25, 4.0, 0.35, 0.0, 1.25, 0.0, 0.0],
            [4.6, 0.0, -0.15, 4.2, -0.25, 0.0, 2.10, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(6, dtype=torch.float32).sub_(2.5).contiguous()
    colors = torch.tensor([[0.8, 0.2, 0.15], [0.1, 0.35, 0.75]], dtype=torch.float32)
    opacities = torch.tensor([0.42, 0.36], dtype=torch.float32)
    sigma_px = 1.7
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=2,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=3,
    )
    bounds = bound_projective_trace_windows(windows, uv_padding=5.0)
    records = bin_projective_trace_support_bounds(
        bounds,
        image_width=8,
        image_height=8,
        tile_size=8,
    )
    cells = assemble_projective_trace_tile_time_atlas(records)
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=6,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=0.0,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    grad_image = torch.linspace(-0.3, 0.4, steps=config.frames * config.height * config.width * 3, dtype=torch.float32)
    grad_image = grad_image.reshape(config.frames, config.height, config.width, 3).contiguous()

    ref_coeffs = coeffs.clone().detach().requires_grad_(True)
    ref_colors = colors.clone().detach().requires_grad_(True)
    ref_opacities = opacities.clone().detach().requires_grad_(True)
    ref = render_projective_trace_tile_time_atlas_reference(
        cells,
        ref_coeffs,
        times,
        ref_colors,
        ref_opacities,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=sigma_px,
        alpha_cutoff=config.alpha_threshold,
        transmittance_cutoff=config.transmittance_threshold,
    )
    (ref * grad_image).sum().backward()

    grads = direct_backward_projective_trace_tile_time_atlas_metal(
        cells,
        coeffs.to("mps"),
        times.to("mps"),
        colors.to("mps"),
        opacities.to("mps"),
        grad_image.to("mps"),
        config,
        sigma_px=sigma_px,
    )

    torch.testing.assert_close(grads.grad_color.cpu(), ref_colors.grad, atol=4.0e-4, rtol=4.0e-4)
    torch.testing.assert_close(grads.grad_opacity.cpu(), ref_opacities.grad, atol=4.0e-4, rtol=4.0e-4)
    torch.testing.assert_close(grads.grad_coeffs.cpu(), ref_coeffs.grad, atol=8.0e-4, rtol=8.0e-4)


def test_projective_quadratic_atlas_cell_one_step_coeff_training_smoke_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "direct_projective_trace_backward"):
        pytest.skip("STAR UVT projective atlas Metal backward op unavailable")

    start_coeffs = torch.tensor(
        [
            [4.0, 0.0, 0.25, 4.0, 0.35, 0.0, 1.25, 0.0, 0.0],
            [4.6, 0.0, -0.15, 4.2, -0.25, 0.0, 2.10, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    target_coeffs = start_coeffs.clone()
    target_coeffs[:, 0] += torch.tensor([0.40, -0.25], dtype=torch.float32)
    target_coeffs[:, 1] += torch.tensor([0.06, -0.04], dtype=torch.float32)
    target_coeffs[:, 3] += torch.tensor([-0.30, 0.22], dtype=torch.float32)
    times = torch.arange(6, dtype=torch.float32).sub_(2.5).contiguous()
    colors = torch.tensor([[0.8, 0.2, 0.15], [0.1, 0.35, 0.75]], dtype=torch.float32)
    opacities = torch.tensor([0.42, 0.36], dtype=torch.float32)
    sigma_px = 1.7
    windows = split_projective_trace_windows(
        start_coeffs,
        times,
        degree=2,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=3,
    )
    bounds = bound_projective_trace_windows(windows, uv_padding=7.0)
    records = bin_projective_trace_support_bounds(
        bounds,
        image_width=8,
        image_height=8,
        tile_size=8,
    )
    cells = assemble_projective_trace_tile_time_atlas(records)
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=6,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=0.0,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )

    device = torch.device("mps")
    target = render_projective_trace_tile_time_atlas_metal(
        cells,
        target_coeffs.to(device),
        times.to(device),
        colors.to(device),
        opacities.to(device),
        config,
        sigma_px=sigma_px,
    ).detach()
    before = render_projective_trace_tile_time_atlas_metal(
        cells,
        start_coeffs.to(device),
        times.to(device),
        colors.to(device),
        opacities.to(device),
        config,
        sigma_px=sigma_px,
    )
    before_loss = (before - target).square().mean()
    grad_image = (2.0 / float(before.numel())) * (before - target)
    grads = direct_backward_projective_trace_tile_time_atlas_metal(
        cells,
        start_coeffs.to(device),
        times.to(device),
        colors.to(device),
        opacities.to(device),
        grad_image.contiguous(),
        config,
        sigma_px=sigma_px,
    )

    candidate_losses = []
    for lr in (0.25, 0.5, 1.0, 2.0):
        stepped_coeffs = (start_coeffs.to(device) - lr * grads.grad_coeffs).detach()
        after = render_projective_trace_tile_time_atlas_metal(
            cells,
            stepped_coeffs,
            times.to(device),
            colors.to(device),
            opacities.to(device),
            config,
            sigma_px=sigma_px,
        )
        candidate_losses.append((after - target).square().mean())

    best_loss = torch.stack(candidate_losses).min()
    assert float(before_loss.cpu().item()) > 1.0e-6
    assert float(grads.grad_coeffs.cpu().abs().sum().item()) > 0.0
    assert float(best_loss.cpu().item()) < 0.99 * float(before_loss.cpu().item())


def test_projective_affine_charts_lower_to_existing_q_uvt_renderer_contract() -> None:
    coeffs = torch.tensor(
        [
            [3.5, 0.25, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.5, -0.25, 0.0, 3.5, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(5, dtype=torch.float32).sub_(2.0).contiguous()
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    opacities = torch.tensor([0.55, 0.35], dtype=torch.float32)
    sigma_px = 2.0

    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
    )
    bridge = projective_trace_windows_to_uvt_tubes(
        windows,
        sigma_px=sigma_px,
        opacity=opacities,
        color=colors,
        primitive_ids=[10, 20],
    )
    bounds = bound_projective_trace_windows(windows, uv_padding=6.0)
    records = bin_projective_trace_support_bounds(
        bounds,
        image_width=8,
        image_height=8,
        tile_size=8,
        primitive_ids=[10, 20],
    )
    cells = assemble_projective_trace_tile_time_atlas(records)

    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=5,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=0.0,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    q_uvt_render = brute_force_render_uvt_tubes(
        bridge.ma,
        bridge.q_uvt,
        bridge.depth0,
        bridge.depth_beta,
        bridge.opacity,
        bridge.color,
        config,
    )
    atlas_render = render_projective_trace_tile_time_atlas_reference(
        cells,
        coeffs,
        times,
        colors,
        opacities,
        image_width=8,
        image_height=8,
        tile_size=8,
        sigma_px=sigma_px,
        primitive_ids=[10, 20],
    )

    assert bridge.source_window_indices == (0, 0)
    assert bridge.source_primitive_ids == (10, 20)
    assert bridge.active_start == (0, 0)
    assert bridge.active_stop == (5, 5)
    torch.testing.assert_close(q_uvt_render, atlas_render, atol=1.0e-6, rtol=1.0e-6)


def test_projective_affine_q_uvt_bridge_matches_metal_renderer_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "render"):
        pytest.skip("STAR UVT Metal render op unavailable")

    coeffs = torch.tensor(
        [
            [3.5, 0.25, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.5, -0.25, 0.0, 3.5, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(5, dtype=torch.float32).sub_(2.0).contiguous()
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    opacities = torch.tensor([0.55, 0.35], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
    )
    bridge = projective_trace_windows_to_uvt_tubes(
        windows,
        sigma_px=2.0,
        opacity=opacities,
        color=colors,
        primitive_ids=[10, 20],
    )
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=5,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    ref = brute_force_render_uvt_tubes(
        bridge.ma,
        bridge.q_uvt,
        bridge.depth0,
        bridge.depth_beta,
        bridge.opacity,
        bridge.color,
        config,
    )
    metal = render_uvt_tubes(
        bridge.ma.to("mps"),
        bridge.q_uvt.to("mps"),
        bridge.depth0.to("mps"),
        bridge.depth_beta.to("mps"),
        bridge.opacity.to("mps"),
        bridge.color.to("mps"),
        config,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)


def test_q_uvt_native_interval_gates_match_cpu_reference_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "render_gated"):
        pytest.skip("STAR UVT native gated Metal render op unavailable")

    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=6,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    ma = torch.tensor(
        [
            [3.5, 3.5, -1.5],
            [4.5, 3.5, 1.5],
        ],
        dtype=torch.float32,
    )
    q_uvt = torch.tensor(
        [
            [0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    depth0 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    depth_beta = torch.zeros((2, 3), dtype=torch.float32)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    opacities = torch.tensor([0.55, 0.45], dtype=torch.float32)
    active_start = torch.tensor([0, 3], dtype=torch.int32)
    active_stop = torch.tensor([3, 6], dtype=torch.int32)

    ref = brute_force_render_uvt_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacities,
        colors,
        config,
        active_start=active_start,
        active_stop=active_stop,
    )
    metal = render_uvt_tubes_gated(
        ma.to("mps"),
        q_uvt.to("mps"),
        depth0.to("mps"),
        depth_beta.to("mps"),
        opacities.to("mps"),
        colors.to("mps"),
        active_start.to("mps"),
        active_stop.to("mps"),
        config,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)
    assert float(ref[:3, :, :, 0].abs().amax().item()) > 0.0
    assert float(ref[3:, :, :, 2].abs().amax().item()) > 0.0
    assert float(ref[:3, :, :, 2].abs().amax().item()) < 1.0e-6
    assert float(ref[3:, :, :, 0].abs().amax().item()) < 1.0e-6


def test_q_uvt_native_interval_gated_backward_matches_single_tube_references_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "direct_atomic_backward_gated"):
        pytest.skip("STAR UVT native gated backward op unavailable")

    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=6,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )
    ma = torch.tensor(
        [
            [3.5, 3.5, -1.5],
            [4.5, 3.5, 1.5],
        ],
        dtype=torch.float32,
    )
    q_uvt = torch.tensor(
        [
            [0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.20, 0.0, 0.0, 0.20, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    depth0 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    depth_beta = torch.zeros((2, 3), dtype=torch.float32)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    opacities = torch.tensor([0.55, 0.45], dtype=torch.float32)
    active_start = torch.tensor([0, 3], dtype=torch.int32)
    active_stop = torch.tensor([3, 6], dtype=torch.int32)
    grad_image = torch.linspace(
        0.05,
        0.95,
        config.frames * config.height * config.width * 3,
        dtype=torch.float32,
    ).reshape(config.frames, config.height, config.width, 3)

    gated = direct_atomic_backward_gated(
        ma.to("mps"),
        q_uvt.to("mps"),
        depth0.to("mps"),
        depth_beta.to("mps"),
        opacities.to("mps"),
        colors.to("mps"),
        grad_image.to("mps"),
        active_start.to("mps"),
        active_stop.to("mps"),
        config,
    )

    refs = []
    for tube_index in range(2):
        masked_grad = grad_image.clone()
        mask = torch.zeros((config.frames, 1, 1, 1), dtype=torch.float32)
        mask[int(active_start[tube_index]) : int(active_stop[tube_index])] = 1.0
        masked_grad = masked_grad * mask
        refs.append(
            direct_atomic_backward(
                ma[tube_index : tube_index + 1].to("mps"),
                q_uvt[tube_index : tube_index + 1].to("mps"),
                depth0[tube_index : tube_index + 1].to("mps"),
                depth_beta[tube_index : tube_index + 1].to("mps"),
                opacities[tube_index : tube_index + 1].to("mps"),
                colors[tube_index : tube_index + 1].to("mps"),
                masked_grad.to("mps"),
                config,
            )
        )

    expected = (
        torch.cat([refs[0][0].cpu(), refs[1][0].cpu()], dim=0),
        torch.cat([refs[0][1].cpu(), refs[1][1].cpu()], dim=0),
        torch.cat([refs[0][2].cpu(), refs[1][2].cpu()], dim=0),
        torch.cat([refs[0][3].cpu(), refs[1][3].cpu()], dim=0),
    )

    for actual, ref in zip(gated[:4], expected, strict=True):
        torch.testing.assert_close(actual.cpu(), ref, atol=3.0e-4, rtol=3.0e-4)
    assert float(gated[0].cpu().abs().sum().item()) > 0.0
    assert float(gated[1].cpu().abs().sum().item()) > 0.0
    assert not bool(torch.any(gated[4].cpu() != 0).item())


def test_projective_split_q_uvt_bridge_window_gates_prevent_segment_leakage() -> None:
    coeffs = torch.tensor(
        [[4.0, 0.0, 0.35, 4.0, 0.45, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(8, dtype=torch.float32).sub_(3.5).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    opacities = torch.tensor([0.6], dtype=torch.float32)
    sigma_px = 1.6
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert len(windows) > 1

    bridge = projective_trace_windows_to_uvt_tubes(
        windows,
        sigma_px=sigma_px,
        opacity=opacities,
        color=colors,
        primitive_ids=[7],
    )
    dense = _render_dense_projective_reference(
        coeffs,
        times,
        colors,
        opacities,
        image_width=8,
        image_height=8,
        sigma_px=sigma_px,
    )
    gated = render_projective_trace_uvt_bridge_reference(
        bridge,
        image_width=8,
        image_height=8,
        frame_times=times,
        max_alpha=1.0,
        use_window_gates=True,
    )
    ungated = render_projective_trace_uvt_bridge_reference(
        bridge,
        image_width=8,
        image_height=8,
        frame_times=times,
        max_alpha=1.0,
        use_window_gates=False,
    )

    torch.testing.assert_close(gated, dense, atol=1.0e-6, rtol=1.0e-6)
    assert float((ungated - dense).abs().amax().item()) > 0.05
    assert len(projective_trace_uvt_bridge_active_spans(bridge, frames=times.numel())) < times.numel()
    assert bridge.active_start[0] == windows[0].start
    assert bridge.active_stop[-1] == windows[-1].stop


def test_projective_split_q_uvt_bridge_interval_gates_reach_metal_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "render_gated"):
        pytest.skip("STAR UVT native gated Metal render op unavailable")

    coeffs = torch.tensor(
        [[4.0, 0.0, 0.35, 4.0, 0.45, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(8, dtype=torch.float32).sub_(3.5).contiguous()
    colors = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    opacities = torch.tensor([0.6], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    bridge = projective_trace_windows_to_uvt_tubes(
        windows,
        sigma_px=1.6,
        opacity=opacities,
        color=colors,
        primitive_ids=[7],
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
    ref = render_projective_trace_uvt_bridge_reference(
        bridge,
        image_width=8,
        image_height=8,
        frame_times=times,
        alpha_threshold=config.alpha_threshold,
        transmittance_threshold=config.transmittance_threshold,
        background=config.background,
        max_alpha=config.max_alpha,
    )
    metal = render_projective_trace_uvt_bridge_metal_gated(
        bridge=type(bridge)(
            ma=bridge.ma.to("mps"),
            q_uvt=bridge.q_uvt.to("mps"),
            depth0=bridge.depth0.to("mps"),
            depth_beta=bridge.depth_beta.to("mps"),
            opacity=bridge.opacity.to("mps"),
            color=bridge.color.to("mps"),
            source_window_indices=bridge.source_window_indices,
            source_primitive_ids=bridge.source_primitive_ids,
            active_start=bridge.active_start,
            active_stop=bridge.active_stop,
        ),
        config=config,
    )

    torch.testing.assert_close(metal.cpu(), ref, atol=2.0e-4, rtol=2.0e-4)


def test_projective_interval_gated_bridge_one_step_color_training_smoke_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not hasattr(torch.ops, "star_uvt_v0") or not hasattr(torch.ops.star_uvt_v0, "direct_atomic_backward_gated"):
        pytest.skip("STAR UVT native gated backward op unavailable")

    coeffs = torch.tensor(
        [[4.0, 0.0, 0.35, 4.0, 0.45, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.arange(8, dtype=torch.float32).sub_(3.5).contiguous()
    target_color = torch.tensor([[0.9, 0.25, 0.1]], dtype=torch.float32)
    start_color = torch.tensor([[0.2, 0.05, 0.02]], dtype=torch.float32)
    opacities = torch.tensor([0.6], dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    assert len(windows) > 1

    target_bridge = projective_trace_windows_to_uvt_tubes(
        windows,
        sigma_px=1.6,
        opacity=opacities,
        color=target_color,
        primitive_ids=[7],
    )
    train_bridge = projective_trace_windows_to_uvt_tubes(
        windows,
        sigma_px=1.6,
        opacity=opacities,
        color=start_color,
        primitive_ids=[7],
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

    def to_mps_bridge(bridge, *, color: torch.Tensor | None = None):
        return type(bridge)(
            ma=bridge.ma.to("mps"),
            q_uvt=bridge.q_uvt.to("mps"),
            depth0=bridge.depth0.to("mps"),
            depth_beta=bridge.depth_beta.to("mps"),
            opacity=bridge.opacity.to("mps"),
            color=(bridge.color if color is None else color).to("mps"),
            source_window_indices=bridge.source_window_indices,
            source_primitive_ids=bridge.source_primitive_ids,
            active_start=bridge.active_start,
            active_stop=bridge.active_stop,
        )

    target = render_projective_trace_uvt_bridge_metal_gated(to_mps_bridge(target_bridge), config).detach()
    current_bridge = to_mps_bridge(train_bridge)
    before = render_projective_trace_uvt_bridge_metal_gated(current_bridge, config)
    before_loss = (before - target).square().mean()
    grad_image = (2.0 / float(before.numel())) * (before - target)
    grads = direct_backward_projective_trace_uvt_bridge_metal_gated(
        current_bridge,
        grad_image.contiguous(),
        config,
    )
    updated_color = current_bridge.color - 20.0 * grads.grad_color
    after_bridge = to_mps_bridge(train_bridge, color=updated_color.detach().cpu())
    after = render_projective_trace_uvt_bridge_metal_gated(after_bridge, config)
    after_loss = (after - target).square().mean()

    assert float(before_loss.cpu().item()) > 1.0e-5
    assert float(grads.grad_color.cpu().abs().sum().item()) > 0.0
    assert not bool(torch.any(grads.tile_unstable.cpu() != 0).item())
    assert float(after_loss.cpu().item()) < 0.85 * float(before_loss.cpu().item())
