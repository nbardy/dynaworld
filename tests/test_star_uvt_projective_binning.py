from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

STAR_UVT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeRecord,
    ProjectiveTraceTileTimeCell,
    assemble_projective_trace_tile_time_atlas,
    bin_projective_trace_support_bounds,
    bound_projective_trace_windows,
    count_projective_trace_dense_per_frame_tile_pairs,
    pack_projective_trace_tile_time_bins,
    projective_trace_cell_atlas_coverage_report,
    projective_trace_cell_sensor_time_event_partition,
    projective_trace_cell_sensor_time_partition_quadrature,
    projective_trace_cell_sensor_time_partition_rolling_quadrature,
    projective_trace_cell_support_event_report,
    projective_trace_windows_to_cell_trace_atlas,
    rebin_projective_trace_cell_atlas,
    rebin_projective_trace_cell_atlas_support_events,
    split_projective_trace_windows,
)


def _screen_linear_coeffs() -> torch.Tensor:
    return torch.tensor(
        [
            [24.0, 16.0, 0.0, 16.0, 6.0, 0.0, 1.0, 0.0, 0.0],
            [80.0, 8.0, 0.0, 80.0, 8.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()


def _bounds_for_linear_fixture():
    coeffs = _screen_linear_coeffs()
    times = torch.linspace(-1.0, 1.0, 9, dtype=torch.float32).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=2.0e-5,
        min_denominator_abs=1.0e-3,
    )
    assert len(windows) == 1
    assert windows[0].accepted
    return bound_projective_trace_windows(windows)


def test_projective_tile_time_binning_emits_compressed_visible_records() -> None:
    records = bin_projective_trace_support_bounds(
        _bounds_for_linear_fixture(),
        image_width=64,
        image_height=64,
        tile_size=16,
    )

    assert len(records) == 1
    record = records[0]
    assert record.primitive_id == 0
    assert record.window_index == 0
    assert record.start == 0
    assert record.stop == 9
    assert record.tile_u_min == 0
    assert record.tile_u_max == 3
    assert record.tile_v_min == 0
    assert record.tile_v_max == 2
    assert abs(record.depth_min - 1.0) < 1.0e-5
    assert abs(record.depth_max - 1.0) < 1.0e-5
    assert not record.fallback


def test_projective_tile_time_binning_preserves_custom_ids_and_fallback_mask() -> None:
    records = bin_projective_trace_support_bounds(
        _bounds_for_linear_fixture(),
        image_width=128,
        image_height=128,
        tile_size=16,
        primitive_ids=[101, 202],
        fallback_mask=[False, True],
        fallback_reason="visibility",
    )

    assert [record.primitive_id for record in records] == [101, 202]
    assert not records[0].fallback
    assert records[0].fallback_reason == ""
    assert records[1].fallback
    assert records[1].fallback_reason == "visibility"
    assert records[1].tile_u_min == 4
    assert records[1].tile_u_max == 6
    assert records[1].tile_v_min == 4
    assert records[1].tile_v_max == 6


def test_projective_tile_time_atlas_assembly_groups_active_sets_and_depth_order() -> None:
    records = [
        ProjectiveTraceTileTimeRecord(2, 0, 0, 8, 1, 2, 1, 2, 2.0, 2.2, False, ""),
        ProjectiveTraceTileTimeRecord(1, 0, 0, 8, 1, 2, 1, 2, 0.8, 1.0, False, ""),
        ProjectiveTraceTileTimeRecord(3, 0, 0, 8, 1, 2, 1, 2, 1.2, 1.4, True, "visibility"),
    ]

    cells = assemble_projective_trace_tile_time_atlas(records)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.tile_u == 1
    assert cell.tile_v == 1
    assert cell.start == 0
    assert cell.stop == 8
    assert cell.primitive_ids == (1, 2, 3)
    assert cell.ordered_primitive_ids == (1, 3, 2)
    assert cell.depth_intervals == ((0.8, 1.0), (1.2, 1.4), (2.0, 2.2))
    assert cell.fallback
    assert cell.fallback_reasons == ("visibility",)


def test_projective_tile_time_atlas_assembly_expands_tile_rectangles() -> None:
    records = [
        ProjectiveTraceTileTimeRecord(4, 0, 2, 5, 0, 2, 1, 3, 1.0, 1.0, False, ""),
    ]

    cells = assemble_projective_trace_tile_time_atlas(records)

    assert [(cell.tile_u, cell.tile_v) for cell in cells] == [(0, 1), (1, 1), (0, 2), (1, 2)]
    assert all(cell.primitive_ids == (4,) for cell in cells)
    assert all(cell.ordered_primitive_ids == (4,) for cell in cells)
    assert all(not cell.fallback for cell in cells)


def _single_trace_cell_atlas(coeffs: torch.Tensor, *, frame_count: int) -> ProjectiveTraceCellTraceAtlas:
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.contiguous(),
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=frame_count,
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
        active_stop=(frame_count,),
    )


def test_projective_cell_support_event_report_finds_tile_boundary_times() -> None:
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = _single_trace_cell_atlas(
        torch.tensor([[8.0, 8.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        frame_count=int(times.numel()),
    )

    report = projective_trace_cell_support_event_report(
        atlas,
        times,
        image_width=64,
        image_height=64,
        tile_size=16,
    )

    assert [round(event.time, 6) for event in report.events] == [1.0, 3.0]
    assert [event.boundary for event in report.events] == [16.0, 32.0]
    assert report.split_times == (1.0, 3.0)


def test_projective_cell_support_event_rebin_splits_tile_runs() -> None:
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = _single_trace_cell_atlas(
        torch.tensor([[8.0, 8.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        frame_count=int(times.numel()),
    )

    sampled = rebin_projective_trace_cell_atlas(
        atlas,
        times,
        image_width=64,
        image_height=64,
        tile_size=16,
    )
    event_rebinned = rebin_projective_trace_cell_atlas_support_events(
        atlas,
        times,
        image_width=64,
        image_height=64,
        tile_size=16,
    )

    assert [(cell.tile_u, cell.start, cell.stop) for cell in sampled.cells] == [
        (0, 0, 4),
        (1, 0, 4),
        (2, 0, 4),
    ]
    assert [(cell.tile_u, cell.start, cell.stop) for cell in event_rebinned.cells] == [
        (0, 0, 1),
        (1, 1, 3),
        (2, 3, 4),
    ]
    coverage = projective_trace_cell_atlas_coverage_report(
        event_rebinned,
        times,
        image_width=64,
        image_height=64,
        tile_size=16,
    )
    assert not coverage.stale
    assert event_rebinned.coeffs is atlas.coeffs


def test_projective_cell_sensor_time_partition_merges_support_visibility_and_exposure_events() -> None:
    times = torch.arange(4, dtype=torch.float32).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [
                [8.0, 8.0, 0.0, 8.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [48.0, 0.0, 0.0, 8.0, 0.0, 0.0, 2.0, -0.25, 0.0],
            ],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.full((2,), 0.5, dtype=torch.float32),
        color=torch.ones((2, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=4,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((0.0, 3.0), (1.25, 2.0)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0, 0),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(4, 4),
    )

    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        times,
        image_width=64,
        image_height=64,
        tile_size=16,
        extra_split_times=(0.5, 2.5),
    )

    assert partition.split_times == (0.0, 0.5, 1.0, 1.6, 2.5, 3.0)
    assert [(interval.start_time, interval.stop_time) for interval in partition.intervals] == [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 1.6),
        (1.6, 2.5),
        (2.5, 3.0),
    ]
    assert [event.time for event in partition.support_events] == [1.0, 3.0]
    assert len(partition.visibility_events) == 1
    assert abs(partition.visibility_events[0].time - 1.6) < 1.0e-6


def test_projective_cell_sensor_time_partition_quadrature_clips_exposure_to_event_cells() -> None:
    partition = projective_trace_cell_sensor_time_event_partition(
        _single_trace_cell_atlas(
            torch.tensor([[8.0, 8.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            frame_count=4,
        ),
        torch.arange(4, dtype=torch.float32).contiguous(),
        image_width=64,
        image_height=64,
        tile_size=16,
        extra_split_times=(0.5, 2.5),
    )

    quadrature = projective_trace_cell_sensor_time_partition_quadrature(
        partition,
        exposure_start=0.25,
        exposure_stop=2.75,
        samples_per_interval=1,
    )

    assert [(round(sample.start_time, 6), round(sample.stop_time, 6), round(sample.time, 6)) for sample in quadrature.samples] == [
        (0.25, 0.5, 0.375),
        (0.5, 1.0, 0.75),
        (1.0, 2.5, 1.75),
        (2.5, 2.75, 2.625),
    ]
    assert [sample.interval_index for sample in quadrature.samples] == [0, 1, 2, 3]
    assert abs(sum(sample.weight for sample in quadrature.samples) - 1.0) < 1.0e-6
    assert abs(quadrature.total_weight - 1.0) < 1.0e-6


def test_projective_cell_sensor_time_rolling_quadrature_offsets_rows() -> None:
    partition = projective_trace_cell_sensor_time_event_partition(
        _single_trace_cell_atlas(
            torch.tensor([[8.0, 8.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
            frame_count=4,
        ),
        torch.arange(4, dtype=torch.float32).contiguous(),
        image_width=64,
        image_height=64,
        tile_size=16,
        extra_split_times=(0.5, 2.5),
    )

    rows = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=3,
        frame_time=0.0,
        exposure_duration=1.0,
        readout_duration=1.0,
    )

    assert [
        [(sample.row_index, round(sample.start_time, 6), round(sample.stop_time, 6), round(sample.time, 6)) for sample in row.samples]
        for row in rows
    ] == [
        [(0, 0.0, 0.5, 0.25), (0, 0.5, 1.0, 0.75)],
        [(1, 0.5, 1.0, 0.75), (1, 1.0, 1.5, 1.25)],
        [(2, 1.0, 2.0, 1.5)],
    ]
    assert all(abs(row.total_weight - 1.0) < 1.0e-6 for row in rows)


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


def _orbit_times(frame_count: int) -> torch.Tensor:
    theta = torch.linspace(-math.radians(45.0), math.radians(45.0), frame_count, dtype=torch.float32)
    return torch.tan(0.5 * theta).contiguous()


def test_projective_interval_packing_scales_sublinearly_over_frame_count() -> None:
    coeffs = torch.tensor(
        [
            _pixel_orbit_coeffs(point_x=0.25, base_depth=2.5, vertical=0.1, center_u=48.0, center_v=40.0, scale=18.0),
            _pixel_orbit_coeffs(point_x=-0.20, base_depth=2.8, vertical=-0.1, center_u=72.0, center_v=58.0, scale=16.0),
            _pixel_orbit_coeffs(point_x=0.10, base_depth=3.2, vertical=0.0, center_u=38.0, center_v=82.0, scale=14.0),
            _pixel_orbit_coeffs(point_x=-0.30, base_depth=2.7, vertical=0.2, center_u=92.0, center_v=36.0, scale=12.0),
        ],
        dtype=torch.float32,
    ).contiguous()
    colors = torch.ones((coeffs.shape[0], 3), dtype=torch.float32)
    opacities = torch.full((coeffs.shape[0],), 0.5, dtype=torch.float32)

    dense_counts: list[int] = []
    interval_counts: list[int] = []
    metal_slab_counts: list[int] = []
    for frame_count in (4, 8, 16, 32, 64):
        times = _orbit_times(frame_count)
        windows = split_projective_trace_windows(
            coeffs,
            times,
            degree=2,
            max_residual_uv=0.75,
            min_denominator_abs=1.0e-3,
            min_samples=3,
        )
        assert len(windows) == 1
        assert windows[0].accepted
        atlas = projective_trace_windows_to_cell_trace_atlas(
            windows,
            opacity=opacities,
            color=colors,
            image_width=128,
            image_height=128,
            tile_size=16,
            uv_padding=5.0,
        )
        dense_counts.append(
            count_projective_trace_dense_per_frame_tile_pairs(
                coeffs,
                times,
                image_width=128,
                image_height=128,
                tile_size=16,
                uv_padding=5.0,
            )
        )
        interval_bins = pack_projective_trace_tile_time_bins(
            atlas.cells,
            image_width=128,
            image_height=128,
            frames=frame_count,
            tile_x=16,
            tile_y=16,
            tile_t=frame_count,
            tile_capacity=1024,
        )
        metal_slab_bins = pack_projective_trace_tile_time_bins(
            atlas.cells,
            image_width=128,
            image_height=128,
            frames=frame_count,
            tile_x=16,
            tile_y=16,
            tile_t=4,
            tile_capacity=1024,
        )
        interval_counts.append(int(interval_bins.tile_counts.sum().item()))
        metal_slab_counts.append(int(metal_slab_bins.tile_counts.sum().item()))

    assert dense_counts[-1] > 12 * dense_counts[0]
    assert max(interval_counts) == min(interval_counts)
    assert interval_counts[-1] / dense_counts[-1] < 0.1 * (interval_counts[0] / dense_counts[0])
    assert metal_slab_counts[-1] == 16 * interval_counts[-1]
