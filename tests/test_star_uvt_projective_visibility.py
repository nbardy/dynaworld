from __future__ import annotations

import sys
from pathlib import Path

import torch

STAR_UVT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    adapt_projective_trace_cell_atlas_uv_visibility_events,
    bound_projective_trace_visible_swap_cost,
    compare_projective_trace_depth_order,
    make_projective_trace_appearance_sidecar,
    make_projective_trace_visibility_sidecar,
    mark_projective_trace_cell_visibility_fallbacks,
    projective_trace_cell_atlas_fallback_stats,
    projective_trace_cell_atlas_visibility_report,
    projective_trace_cell_uv_visibility_event_report,
    projective_trace_cell_visibility_event_report,
    render_projective_trace_cell_atlas_reference,
    split_projective_trace_cell_atlas_uv_visibility_events,
    split_projective_trace_windows,
    stratify_projective_trace_cell_atlas_visibility_events,
)


def _constant_screen_depth_trace(depth0: float, depth1: float) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, depth0, depth1, 0.0]],
        dtype=torch.float32,
    ).contiguous()


def _single_window(coeffs: torch.Tensor, times: torch.Tensor):
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-5,
        max_depth_residual=1.0e-5,
        min_denominator_abs=1.0e-3,
    )
    assert len(windows) == 1
    assert windows[0].accepted
    return windows[0]


def _cell_atlas_from_depth_coeffs(depth_coeffs: torch.Tensor, *, frame_count: int) -> ProjectiveTraceCellTraceAtlas:
    trace_count = int(depth_coeffs.shape[0])
    coeffs = torch.zeros((trace_count, 9), dtype=torch.float32)
    coeffs[:, 0] = torch.arange(trace_count, dtype=torch.float32) * 0.25 + 4.0
    coeffs[:, 3] = 4.0
    coeffs[:, 6:9] = depth_coeffs
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.contiguous(),
        opacity=torch.full((trace_count,), 0.5, dtype=torch.float32),
        color=torch.ones((trace_count, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=frame_count,
                primitive_ids=tuple(range(trace_count)),
                ordered_primitive_ids=tuple(range(trace_count)),
                depth_intervals=tuple((0.0, 1.0) for _ in range(trace_count)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=tuple(0 for _ in range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(frame_count for _ in range(trace_count)),
    )


def test_projective_visibility_sidecar_records_depth_monotonicity() -> None:
    times = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).contiguous()
    window = _single_window(_constant_screen_depth_trace(2.0, 0.25), times)

    sidecar = make_projective_trace_visibility_sidecar(window, chart_gauge_id=7)

    assert sidecar.chart_gauge_id == 7
    assert torch.isclose(sidecar.depth_min[0], torch.tensor(1.75), atol=1.0e-5)
    assert torch.isclose(sidecar.depth_max[0], torch.tensor(2.25), atol=1.0e-5)
    assert sidecar.depth_slope_min[0] > 0.0
    assert sidecar.depth_slope_max[0] > 0.0
    assert sidecar.depth_monotonic_sign[0] == 1
    assert not sidecar.denominator_has_root[0]


def test_projective_depth_order_detects_stable_front_back_relation() -> None:
    times = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).contiguous()
    front = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(1.0, 0.1), times)
    )
    back = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, 0.1), times)
    )

    order = compare_projective_trace_depth_order(front, back)

    assert order.a_before_b[0]
    assert not order.b_before_a[0]
    assert not order.crosses[0]
    assert not order.ambiguous[0]


def test_projective_depth_order_detects_crossing_visibility_stratum() -> None:
    times = torch.linspace(-0.5, 0.5, 9, dtype=torch.float32).contiguous()
    rising = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, 1.0), times)
    )
    falling = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, -1.0), times)
    )

    order = compare_projective_trace_depth_order(rising, falling)

    assert not order.a_before_b[0]
    assert not order.b_before_a[0]
    assert order.crosses[0]
    assert order.ambiguous[0]


def test_projective_cell_visibility_event_report_finds_linear_crossing_time() -> None:
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = _cell_atlas_from_depth_coeffs(
        torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [3.0, -0.2, 0.0],
            ],
            dtype=torch.float32,
        ),
        frame_count=int(times.numel()),
    )

    report = projective_trace_cell_visibility_event_report(atlas, times)

    assert len(report.events) == 1
    event = report.events[0]
    assert (event.cell_index, event.tile_u, event.tile_v, event.trace_a, event.trace_b) == (0, 0, 0, 0, 1)
    assert abs(event.time - (5.0 / 3.0)) < 1.0e-5
    assert len(report.split_times) == 1
    assert abs(report.split_times[0] - (5.0 / 3.0)) < 1.0e-5


def test_projective_cell_visibility_event_report_finds_quadratic_crossings() -> None:
    times = torch.linspace(-2.0, 2.0, 5, dtype=torch.float32).contiguous()
    atlas = _cell_atlas_from_depth_coeffs(
        torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        frame_count=int(times.numel()),
    )

    report = projective_trace_cell_visibility_event_report(atlas, times)

    assert len(report.events) == 2
    assert len(report.split_times) == 2
    assert abs(report.split_times[0] + 1.0) < 1.0e-6
    assert abs(report.split_times[1] - 1.0) < 1.0e-6


def test_projective_cell_visibility_event_report_returns_empty_for_stable_depths() -> None:
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = _cell_atlas_from_depth_coeffs(
        torch.tensor(
            [
                [1.0, 0.1, 0.0],
                [3.0, 0.1, 0.0],
            ],
            dtype=torch.float32,
        ),
        frame_count=int(times.numel()),
    )

    report = projective_trace_cell_visibility_event_report(atlas, times)

    assert report.events == ()
    assert report.split_times == ()


def test_projective_cell_visibility_event_stratifier_isolates_exact_root_sample() -> None:
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = _cell_atlas_from_depth_coeffs(
        torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        frame_count=int(times.numel()),
    )

    stratified = stratify_projective_trace_cell_atlas_visibility_events(atlas, times)

    assert [(cell.start, cell.stop, cell.ordered_primitive_ids) for cell in stratified.cells] == [
        (0, 1, (0, 1)),
        (1, 2, (0, 1)),
        (2, 4, (1, 0)),
    ]
    report = projective_trace_cell_atlas_visibility_report(stratified, times)
    assert report.order_mismatch_samples == 0
    assert report.ambiguous_depth_samples == 1

    fallback = mark_projective_trace_cell_visibility_fallbacks(stratified, times)
    stats = projective_trace_cell_atlas_fallback_stats(fallback)
    assert [(cell.start, cell.stop, cell.fallback) for cell in fallback.cells] == [
        (0, 1, False),
        (1, 2, True),
        (2, 4, False),
    ]
    assert stats.fallback_tile_samples == 1


def _depth_affine_order_flip_atlas(
    *,
    fallback: bool = False,
    back_center_depth: float = 1.1,
) -> ProjectiveTraceCellTraceAtlas:
    return ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.5, 0.0, 0.0, back_center_depth, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([1.0, 1.0], dtype=torch.float32),
        color=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=1,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((1.0, 1.0), (1.1, 1.1)),
                fallback=fallback,
                fallback_reasons=("visibility_ambiguous_depth",) if fallback else (),
            )
        ],
        source_window_indices=(0, 1),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(1, 1),
        depth_affine_uv=torch.tensor(
            [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
    )


def test_projective_cell_visibility_report_uses_depth_affine_uv_tile_range() -> None:
    times = torch.tensor([0.0], dtype=torch.float32).contiguous()
    atlas = _depth_affine_order_flip_atlas()

    center_only = projective_trace_cell_atlas_visibility_report(atlas, times)
    tile_range = projective_trace_cell_atlas_visibility_report(
        atlas,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
    )
    fallback = mark_projective_trace_cell_visibility_fallbacks(
        atlas,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
    )
    repaired = projective_trace_cell_atlas_visibility_report(
        fallback,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
        mark_ambiguous_stale=False,
    )

    assert center_only.ambiguous_depth_samples == 0
    assert tile_range.ambiguous_depth_samples == 1
    assert tile_range.stale
    assert fallback.cells[0].fallback
    assert "visibility_ambiguous_depth" in fallback.cells[0].fallback_reasons
    assert "visibility_uv_depth_line" in fallback.cells[0].fallback_reasons
    assert not repaired.stale


def test_projective_cell_uv_visibility_event_report_finds_in_tile_depth_line() -> None:
    times = torch.tensor([0.0], dtype=torch.float32).contiguous()
    atlas = _depth_affine_order_flip_atlas()

    report = projective_trace_cell_uv_visibility_event_report(
        atlas,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
    )

    assert report.event_tile_samples == 1
    assert len(report.events) == 1
    event = report.events[0]
    assert (event.cell_index, event.tile_u, event.tile_v, event.sample_index) == (0, 0, 0, 0)
    assert (event.trace_a, event.trace_b) == (0, 1)
    assert event.min_delta < 0.0
    assert event.max_delta > 0.0
    assert abs(event.line_u - 0.4) < 1.0e-6
    assert abs(event.line_v) < 1.0e-6
    assert abs(event.line_0 + 0.9) < 1.0e-6


def test_projective_cell_uv_visibility_event_report_ignores_stable_depth_plane() -> None:
    times = torch.tensor([0.0], dtype=torch.float32).contiguous()
    atlas = _depth_affine_order_flip_atlas()
    stable = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs,
        opacity=atlas.opacity,
        color=atlas.color,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        depth_affine_uv=torch.tensor(
            [[0.02, 0.0, 0.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
    )

    report = projective_trace_cell_uv_visibility_event_report(
        stable,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
    )

    assert report.event_tile_samples == 0
    assert report.events == ()


def test_projective_cell_uv_visibility_spatial_split_retiles_to_stable_child_orders() -> None:
    times = torch.tensor([0.0], dtype=torch.float32).contiguous()
    atlas = _depth_affine_order_flip_atlas()

    split = split_projective_trace_cell_atlas_uv_visibility_events(
        atlas,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
        child_tile_size=2,
    )
    event_report = projective_trace_cell_uv_visibility_event_report(
        split,
        times,
        image_width=4,
        image_height=1,
        tile_size=2,
    )
    fallback = mark_projective_trace_cell_visibility_fallbacks(
        split,
        times,
        image_width=4,
        image_height=1,
        tile_size=2,
    )
    visibility = projective_trace_cell_atlas_visibility_report(
        fallback,
        times,
        image_width=4,
        image_height=1,
        tile_size=2,
    )
    image = render_projective_trace_cell_atlas_reference(
        fallback,
        times,
        image_width=4,
        image_height=1,
        tile_size=2,
        sigma_px=16.0,
    )

    assert [(cell.tile_u, cell.tile_v, cell.ordered_primitive_ids) for cell in split.cells] == [
        (0, 0, (0, 1)),
        (1, 0, (1, 0)),
    ]
    assert event_report.events == ()
    assert not any(cell.fallback for cell in fallback.cells)
    assert not visibility.stale
    assert visibility.ambiguous_depth_samples == 0
    assert image[0, 0, 0, 0] > image[0, 0, 0, 2]
    assert image[0, 0, 3, 2] > image[0, 0, 3, 0]


def test_projective_cell_uv_visibility_adaptive_split_chooses_coarsest_stable_child_grid() -> None:
    times = torch.tensor([0.0], dtype=torch.float32).contiguous()
    atlas = _depth_affine_order_flip_atlas()

    report = adapt_projective_trace_cell_atlas_uv_visibility_events(
        atlas,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
        min_child_tile_size=1,
    )

    assert report.split_attempted
    assert report.accepted
    assert report.input_tile_size == 4
    assert report.output_tile_size == 2
    assert report.candidate_tile_sizes == (2, 1)
    assert report.parent_uv_event_tile_samples == 1
    assert report.residual_uv_event_tile_samples == 0
    assert report.fallback_cells == 0
    assert [(cell.tile_u, cell.tile_v, cell.ordered_primitive_ids) for cell in report.atlas.cells] == [
        (0, 0, (0, 1)),
        (1, 0, (1, 0)),
    ]


def test_projective_cell_uv_visibility_adaptive_split_keeps_fallback_when_min_child_still_crosses() -> None:
    times = torch.tensor([0.0], dtype=torch.float32).contiguous()
    atlas = _depth_affine_order_flip_atlas(back_center_depth=0.6)

    report = adapt_projective_trace_cell_atlas_uv_visibility_events(
        atlas,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
        min_child_tile_size=2,
    )

    assert report.split_attempted
    assert not report.accepted
    assert report.output_tile_size == 2
    assert report.parent_uv_event_tile_samples == 1
    assert report.residual_uv_event_tile_samples == 1
    assert report.fallback_cells == 1
    fallback_cells = [cell for cell in report.atlas.cells if cell.fallback]
    assert len(fallback_cells) == 1
    assert fallback_cells[0].tile_u == 0
    assert "visibility_uv_depth_line" in fallback_cells[0].fallback_reasons


def test_projective_cell_uv_visibility_adaptive_split_measures_high_motion_fallback_reduction() -> None:
    times = torch.tensor([0.0, 1.0], dtype=torch.float32).contiguous()
    atlas = _depth_affine_order_flip_atlas()
    moving = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
                [
                    [2.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.1, 0.8, 0.0],
            ],
            dtype=torch.float32,
        ).contiguous(),
        opacity=atlas.opacity,
        color=atlas.color,
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=2,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((1.0, 1.0), (1.1, 1.9)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=(0, 0),
        active_stop=(2, 2),
        depth_affine_uv=atlas.depth_affine_uv,
    )

    report = adapt_projective_trace_cell_atlas_uv_visibility_events(
        moving,
        times,
        image_width=8,
        image_height=1,
        tile_size=8,
        min_child_tile_size=1,
    )

    assert report.accepted
    assert report.output_tile_size == 2
    assert report.candidate_tile_sizes == (4, 2, 1)
    assert report.parent_cells == 1
    assert report.parent_uv_event_tile_samples == 2
    assert report.parent_fallback_cells == 1
    assert report.parent_fallback_fraction == 1.0
    assert report.residual_uv_event_tile_samples == 0
    assert report.fallback_cells == 0
    assert report.fallback_fraction == 0.0
    assert report.output_cells == 4


def test_projective_cell_reference_fallback_sorts_depth_affine_uv_per_pixel() -> None:
    times = torch.tensor([0.0], dtype=torch.float32).contiguous()
    atlas = _depth_affine_order_flip_atlas(fallback=True)

    image = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=4,
        image_height=1,
        tile_size=4,
        sigma_px=16.0,
        allow_fallback_cells=True,
    )

    left = image[0, 0, 0]
    right = image[0, 0, 3]
    assert left[0] > left[2]
    assert right[2] > right[0]


def test_projective_swap_bound_accepts_visually_negligible_crossing() -> None:
    times = torch.linspace(-0.5, 0.5, 9, dtype=torch.float32).contiguous()
    rising = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, 1.0), times)
    )
    falling = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, -1.0), times)
    )
    order = compare_projective_trace_depth_order(rising, falling)
    appearance_a = make_projective_trace_appearance_sidecar(
        torch.tensor([0.1], dtype=torch.float32),
        torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float32),
    )
    appearance_b = make_projective_trace_appearance_sidecar(
        torch.tensor([0.1], dtype=torch.float32),
        torch.tensor([[0.8, 0.8, 0.8]], dtype=torch.float32),
    )

    swap = bound_projective_trace_visible_swap_cost(
        order,
        appearance_a,
        appearance_b,
        threshold=0.01,
    )

    assert torch.isclose(swap.swap_bound[0], torch.tensor(0.006), atol=1.0e-6)
    assert swap.safely_commutable[0]
    assert not swap.needs_fallback[0]


def test_projective_swap_bound_flags_visible_crossing_for_fallback() -> None:
    times = torch.linspace(-0.5, 0.5, 9, dtype=torch.float32).contiguous()
    rising = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, 1.0), times)
    )
    falling = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, -1.0), times)
    )
    order = compare_projective_trace_depth_order(rising, falling)
    appearance_a = make_projective_trace_appearance_sidecar(
        torch.tensor([0.8], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    appearance_b = make_projective_trace_appearance_sidecar(
        torch.tensor([0.8], dtype=torch.float32),
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
    )

    swap = bound_projective_trace_visible_swap_cost(
        order,
        appearance_a,
        appearance_b,
        threshold=0.01,
    )

    assert torch.isclose(swap.swap_bound[0], torch.tensor(0.64), atol=1.0e-6)
    assert not swap.safely_commutable[0]
    assert swap.needs_fallback[0]


def test_projective_swap_bound_includes_color_interval_uncertainty() -> None:
    times = torch.linspace(-0.5, 0.5, 9, dtype=torch.float32).contiguous()
    rising = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, 1.0), times)
    )
    falling = make_projective_trace_visibility_sidecar(
        _single_window(_constant_screen_depth_trace(2.0, -1.0), times)
    )
    order = compare_projective_trace_depth_order(rising, falling)
    appearance_a = make_projective_trace_appearance_sidecar(
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
        color_radius=0.1,
    )
    appearance_b = make_projective_trace_appearance_sidecar(
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
        color_radius=0.1,
    )

    swap = bound_projective_trace_visible_swap_cost(
        order,
        appearance_a,
        appearance_b,
        threshold=0.01,
    )

    assert torch.isclose(swap.swap_bound[0], torch.tensor(0.05), atol=1.0e-6)
    assert swap.needs_fallback[0]
