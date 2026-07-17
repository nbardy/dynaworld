from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

STAR_UVT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))
STAR_UVT_HARNESS_ROOT = STAR_UVT_ROOT / "research_project" / "trainer_harness"
if str(STAR_UVT_HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_HARNESS_ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    UVTRenderConfig,
    adapt_projective_trace_cell_atlas_uv_visibility_events,
    bound_projective_trace_window,
    bound_projective_trace_windows,
    brute_force_render_uvt_tubes,
    eval_projective_trace_torch,
    fit_projective_trace_polynomial,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
    projective_trace_cell_atlas_complexity_stats,
    projective_trace_cell_atlas_fallback_stats,
    render_projective_trace_cell_atlas_reference,
    render_projective_trace_cell_interval_atlas_metal,
    split_projective_trace_windows,
    uvt_tubes_to_projective_trace_cell_atlas,
)
from tile_metal_autograd import render_projective_cell_interval_atlas_metal_backward  # noqa: E402
from variable_camera_segments import project_piecewise_camera_time_segments  # noqa: E402
from world_tube import WorldTubeBatch  # noqa: E402


def _tan_half_angle_times(theta_min: float, theta_max: float, count: int) -> torch.Tensor:
    theta = torch.linspace(theta_min, theta_max, count, dtype=torch.float32)
    return torch.tan(0.5 * theta).contiguous()


def _orbit_trace_coeffs(*, point_x: float = 0.35, base_depth: float = 2.5, vertical: float = 0.2) -> torch.Tensor:
    # q = tan(theta / 2), so cos(theta) = (1 - q^2)/(1 + q^2)
    # and sin(theta) = 2q/(1 + q^2). Multiplying projection numerator and
    # denominator by (1 + q^2) gives the quadratic homogeneous trace.
    #
    # h_u = cos(theta) * point_x + sin(theta)
    # h_v = vertical * (1 + q^2)
    # h_z = base_depth + sin(theta) * point_x + 0.25 cos(theta)
    return torch.tensor(
        [
            [
                point_x,
                2.0,
                -point_x,
                vertical,
                0.0,
                vertical,
                base_depth + 0.25,
                2.0 * point_x,
                base_depth - 0.25,
            ]
        ],
        dtype=torch.float32,
    ).contiguous()


def _look_at_w2c(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1.0e-8)
    up_hint = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    right = torch.cross(up_hint, forward, dim=0)
    right = right / right.norm().clamp_min(1.0e-8)
    up = torch.cross(forward, right, dim=0)
    up = up / up.norm().clamp_min(1.0e-8)
    rotation = torch.stack((right, up, forward), dim=0)
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -(rotation @ eye)
    return w2c


def _elevated_orbit_camera_sequence(frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    k_seq = torch.eye(3, dtype=torch.float32).repeat(frames, 1, 1)
    k_seq[:, 0, 0] = 60.0
    k_seq[:, 1, 1] = 58.0
    k_seq[:, 0, 2] = 16.0
    k_seq[:, 1, 2] = 16.0
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    w2c = []
    for theta in torch.linspace(-math.radians(60.0), math.radians(60.0), frames):
        eye = torch.tensor(
            [
                2.5 * math.sin(float(theta)),
                0.7,
                -2.5 * math.cos(float(theta)),
            ],
            dtype=torch.float32,
        )
        w2c.append(_look_at_w2c(eye, target))
    return k_seq.contiguous(), torch.stack(w2c, dim=0).contiguous()


def _orbit_world_tube_batch() -> WorldTubeBatch:
    return WorldTubeBatch(
        x0=torch.tensor([[0.15, 0.08, 0.0], [-0.10, 0.05, 0.04]], dtype=torch.float32),
        velocity=torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=torch.float32),
        t0=torch.zeros(2, dtype=torch.float32),
        precision_xy=torch.tensor([[40.0, 160.0], [120.0, 50.0]], dtype=torch.float32),
        lambda_t=torch.tensor([0.2, 0.3], dtype=torch.float32),
        opacity=torch.tensor([0.5, 0.45], dtype=torch.float32),
        color=torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.6, 0.9]], dtype=torch.float32),
    )


def _accepted_window_count(coeffs: torch.Tensor, times: torch.Tensor, *, max_residual_uv: float) -> int:
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=max_residual_uv,
        min_denominator_abs=1.0e-3,
    )
    assert all(window.accepted for window in windows)
    return len(windows)


def _render_orbit_segments(
    *,
    frames: int,
    frames_per_segment: int,
    batch: WorldTubeBatch,
    k_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
) -> tuple[object, torch.Tensor]:
    config = UVTRenderConfig(
        height=32,
        width=32,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=0.01,
        max_alpha=1.0,
    )
    projected = project_piecewise_camera_time_segments(
        batch,
        k_seq,
        w2c_seq,
        config,
        full_frames=frames,
        frames_per_segment=frames_per_segment,
    )
    image = brute_force_render_uvt_tubes(
        projected.ma,
        projected.q_uvt,
        projected.depth0,
        projected.depth_beta,
        projected.opacity,
        projected.color,
        config,
    )
    return projected, image


def _orbit_times(frames: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    return (torch.arange(frames, dtype=torch.float32, device=device) - 0.5 * float(frames - 1)).contiguous()


def _orbit_config(frames: int) -> UVTRenderConfig:
    return UVTRenderConfig(
        height=32,
        width=32,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=0.01,
        max_alpha=1.0,
    )


def _compile_orbit_interval_atlas(projected: object, times: torch.Tensor):
    return uvt_tubes_to_projective_trace_cell_atlas(
        projected.ma.to(device=times.device),
        projected.q_uvt.to(device=times.device),
        projected.depth0.to(device=times.device),
        projected.depth_beta.to(device=times.device),
        projected.opacity.to(device=times.device),
        projected.color.to(device=times.device),
        times,
        sigma_px=2.0,
        image_width=32,
        image_height=32,
        tile_size=8,
        uv_padding=0.0,
        alpha_threshold=0.01,
        require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True,
    )


def _render_differentiable_orbit_interval_metal(*, frames: int, frames_per_segment: int):
    batch = _orbit_world_tube_batch()
    k_seq, w2c_seq = _elevated_orbit_camera_sequence(frames)
    projected, _dense = _render_orbit_segments(
        frames=frames,
        frames_per_segment=frames_per_segment,
        batch=batch,
        k_seq=k_seq,
        w2c_seq=w2c_seq,
    )
    times = _orbit_times(frames, device="mps")
    ma = projected.ma.to("mps").detach().requires_grad_(True)
    q_uvt = projected.q_uvt.to("mps").detach().requires_grad_(True)
    depth0 = projected.depth0.to("mps").detach()
    depth_beta = projected.depth_beta.to("mps").detach()
    opacity = projected.opacity.to("mps").detach().requires_grad_(True)
    color = projected.color.to("mps").detach().requires_grad_(True)
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=32,
        image_height=32,
        tile_size=8,
        uv_padding=0.0,
        alpha_threshold=0.01,
        require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True,
    )
    image = render_projective_cell_interval_atlas_metal_backward(
        atlas,
        times,
        _orbit_config(frames),
        sigma_px=2.0,
    )
    return projected, atlas, image, ma, q_uvt, opacity, color


def test_revolving_camera_segments_carry_rotated_spd_uv_fiber_metric() -> None:
    frames = 16
    config = UVTRenderConfig(
        height=32,
        width=32,
        frames=frames,
        tile_x=8,
        tile_y=8,
        tile_t=2,
        tile_capacity=128,
        alpha_threshold=0.01,
        max_alpha=1.0,
    )
    batch = _orbit_world_tube_batch()
    k_seq, w2c_seq = _elevated_orbit_camera_sequence(frames)

    charted = project_piecewise_camera_time_segments(
        batch,
        k_seq,
        w2c_seq,
        config,
        full_frames=frames,
        frames_per_segment=4,
    )
    per_frame = project_piecewise_camera_time_segments(
        batch,
        k_seq,
        w2c_seq,
        config,
        full_frames=frames,
        frames_per_segment=1,
    )
    image = brute_force_render_uvt_tubes(
        charted.ma,
        charted.q_uvt,
        charted.depth0,
        charted.depth_beta,
        charted.opacity,
        charted.color,
        config,
    )

    q_uu = charted.q_uvt[:, 0]
    q_uv = charted.q_uvt[:, 1]
    q_vv = charted.q_uvt[:, 3]
    det = q_uu * q_vv - q_uv.square()
    tube0_q_uv = q_uv[0:: batch.x0.shape[0]]

    assert charted.diagnostics.temporal_chunk_count == 4
    assert per_frame.diagnostics.temporal_chunk_count == frames
    assert charted.diagnostics.segment_count < per_frame.diagnostics.segment_count
    assert torch.all(q_uu > 0.0)
    assert torch.all(q_vv > 0.0)
    assert torch.all(det > 0.0)
    assert float(q_uv.abs().max().item()) > 1.0e-3
    assert float(tube0_q_uv.amin().item()) < 0.0 < float(tube0_q_uv.amax().item())
    assert image.shape == (frames, 32, 32, 3)
    assert torch.isfinite(image).all()
    assert float(image.abs().sum().item()) > 0.0


def test_revolving_camera_chart_size_sweep_quantifies_error_vs_framewise_reference() -> None:
    frames = 8
    batch = _orbit_world_tube_batch()
    k_seq, w2c_seq = _elevated_orbit_camera_sequence(frames)
    reference, reference_image = _render_orbit_segments(
        frames=frames,
        frames_per_segment=1,
        batch=batch,
        k_seq=k_seq,
        w2c_seq=w2c_seq,
    )
    rows = []
    for frames_per_segment in (1, 2, 4, 8):
        projected, image = _render_orbit_segments(
            frames=frames,
            frames_per_segment=frames_per_segment,
            batch=batch,
            k_seq=k_seq,
            w2c_seq=w2c_seq,
        )
        delta = image - reference_image
        rows.append(
            {
                "frames_per_segment": frames_per_segment,
                "segment_ratio": projected.diagnostics.segment_count / reference.diagnostics.segment_count,
                "mean_abs": float(delta.abs().mean().item()),
                "max_abs": float(delta.abs().max().item()),
                "mse": float(delta.square().mean().item()),
            }
        )

    assert [row["segment_ratio"] for row in rows] == [1.0, 0.5, 0.25, 0.125]
    assert rows[0]["mean_abs"] == 0.0
    assert rows[0]["mse"] == 0.0
    for row in rows[1:]:
        assert row["mean_abs"] < 0.009
        assert row["mse"] < 0.0011
        assert row["max_abs"] < 0.40


def test_revolving_camera_interval_atlas_sweep_reports_compression_and_zero_fallback() -> None:
    frames = 8
    batch = _orbit_world_tube_batch()
    k_seq, w2c_seq = _elevated_orbit_camera_sequence(frames)
    times = _orbit_times(frames)
    rows = []
    for frames_per_segment in (1, 2, 4, 8):
        projected, dense = _render_orbit_segments(
            frames=frames,
            frames_per_segment=frames_per_segment,
            batch=batch,
            k_seq=k_seq,
            w2c_seq=w2c_seq,
        )
        atlas = _compile_orbit_interval_atlas(projected, times)
        atlas_image = render_projective_trace_cell_atlas_reference(
            atlas,
            times,
            image_width=32,
            image_height=32,
            tile_size=8,
            sigma_px=2.0,
            alpha_cutoff=0.01,
        )
        stats = projective_trace_cell_atlas_complexity_stats(atlas)
        fallback = projective_trace_cell_atlas_fallback_stats(atlas)
        delta = atlas_image - dense
        rows.append(
            {
                "frames_per_segment": frames_per_segment,
                "trace_count": int(atlas.coeffs.shape[0]),
                "fallback_fraction": fallback.fallback_fraction,
                "interval_ratio": stats.interval_to_dense_trace_sample_ratio,
                "mean_abs": float(delta.abs().mean().item()),
                "max_abs": float(delta.abs().max().item()),
            }
        )

    assert [row["trace_count"] for row in rows] == [16, 8, 4, 2]
    assert [row["fallback_fraction"] for row in rows] == [0.0, 0.0, 0.0, 0.0]
    assert [row["interval_ratio"] for row in rows] == sorted(
        (row["interval_ratio"] for row in rows),
        reverse=True,
    )
    assert rows[0]["interval_ratio"] == 1.0
    assert rows[-1]["interval_ratio"] < 0.35
    for row in rows:
        assert row["mean_abs"] < 3.0e-5
        assert row["max_abs"] < 0.02


def test_revolving_camera_fixed_chart_count_keeps_world_side_work_sublinear_with_frame_growth() -> None:
    batch = _orbit_world_tube_batch()
    rows = []
    for frames in (8, 16, 32):
        k_seq, w2c_seq = _elevated_orbit_camera_sequence(frames)
        per_frame, _ = _render_orbit_segments(
            frames=frames,
            frames_per_segment=1,
            batch=batch,
            k_seq=k_seq,
            w2c_seq=w2c_seq,
        )
        charted, _ = _render_orbit_segments(
            frames=frames,
            frames_per_segment=frames // 4,
            batch=batch,
            k_seq=k_seq,
            w2c_seq=w2c_seq,
        )
        atlas = _compile_orbit_interval_atlas(charted, _orbit_times(frames))
        stats = projective_trace_cell_atlas_complexity_stats(atlas)
        fallback = projective_trace_cell_atlas_fallback_stats(atlas)
        rows.append(
            {
                "frames": frames,
                "per_frame_segments": per_frame.diagnostics.segment_count,
                "charted_segments": charted.diagnostics.segment_count,
                "trace_count": int(atlas.coeffs.shape[0]),
                "interval_trace_entries": stats.interval_trace_entries,
                "dense_trace_samples": stats.dense_trace_samples,
                "interval_ratio": stats.interval_to_dense_trace_sample_ratio,
                "fallback_fraction": fallback.fallback_fraction,
            }
        )

    assert [row["per_frame_segments"] for row in rows] == [16, 32, 64]
    assert [row["charted_segments"] for row in rows] == [8, 8, 8]
    assert [row["trace_count"] for row in rows] == [8, 8, 8]
    assert [row["fallback_fraction"] for row in rows] == [0.0, 0.0, 0.0]
    assert rows[-1]["dense_trace_samples"] > 5 * rows[0]["dense_trace_samples"]
    assert rows[-1]["interval_trace_entries"] < 2 * rows[0]["interval_trace_entries"]
    assert [row["interval_ratio"] for row in rows] == sorted(
        (row["interval_ratio"] for row in rows),
        reverse=True,
    )
    assert rows[-1]["interval_ratio"] < 0.35 * rows[0]["interval_ratio"]


def test_revolving_camera_interval_metal_matches_reference_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal():
        pytest.skip("projective interval cell Metal op unavailable")

    frames = 4
    batch = _orbit_world_tube_batch()
    k_seq, w2c_seq = _elevated_orbit_camera_sequence(frames)
    projected, _dense = _render_orbit_segments(
        frames=frames,
        frames_per_segment=2,
        batch=batch,
        k_seq=k_seq,
        w2c_seq=w2c_seq,
    )
    times = _orbit_times(frames, device="mps")
    atlas = _compile_orbit_interval_atlas(projected, times)
    config = _orbit_config(frames)
    ref = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=32,
        image_height=32,
        tile_size=8,
        sigma_px=2.0,
        alpha_cutoff=0.01,
    )
    metal = render_projective_trace_cell_interval_atlas_metal(
        atlas,
        times,
        config,
        sigma_px=2.0,
    )

    torch.testing.assert_close(metal.cpu(), ref.cpu(), atol=5.0e-4, rtol=5.0e-4)


def test_revolving_camera_interval_backward_reaches_orbit_uvt_trace_params_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")

    frames = 4
    _projected, _atlas, image, ma, q_uvt, opacity, color = _render_differentiable_orbit_interval_metal(
        frames=frames,
        frames_per_segment=2,
    )
    x_weight = torch.linspace(-1.0, 1.0, 32, device="mps").view(1, 1, 32, 1)
    y_weight = torch.linspace(-0.7, 1.3, 32, device="mps").view(1, 32, 1, 1)
    loss = (image * x_weight * y_weight).sum() + 0.1 * image.square().mean()
    loss.backward()

    assert ma.grad is not None and float(ma.grad.detach().abs().sum().cpu()) > 0.0
    assert q_uvt.grad is not None
    assert float(q_uvt.grad[:, [0, 1, 3]].detach().abs().sum().cpu()) > 0.0
    assert float(q_uvt.grad[:, 1].detach().abs().sum().cpu()) > 0.0
    assert float(q_uvt.grad[:, [2, 4, 5]].detach().abs().sum().cpu()) > 0.0
    assert opacity.grad is not None and float(opacity.grad.detach().abs().sum().cpu()) > 0.0
    assert color.grad is not None and float(color.grad.detach().abs().sum().cpu()) > 0.0


def test_revolving_camera_interval_backward_keeps_fixed_chart_params_when_frames_grow_if_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        pytest.skip("projective interval cell Metal forward/backward ops unavailable")

    rows = []
    for frames in (4, 8):
        projected, atlas, image, ma, q_uvt, opacity, color = _render_differentiable_orbit_interval_metal(
            frames=frames,
            frames_per_segment=frames // 2,
        )
        x_weight = torch.linspace(-1.0, 1.0, 32, device="mps").view(1, 1, 32, 1)
        y_weight = torch.linspace(-0.7, 1.3, 32, device="mps").view(1, 32, 1, 1)
        t_weight = torch.linspace(0.8, 1.2, frames, device="mps").view(frames, 1, 1, 1)
        loss = (image * x_weight * y_weight * t_weight).sum() + 0.1 * image.square().mean()
        loss.backward()
        assert ma.grad is not None and float(ma.grad.detach().abs().sum().cpu()) > 0.0
        assert q_uvt.grad is not None
        assert float(q_uvt.grad[:, [0, 1, 3]].detach().abs().sum().cpu()) > 0.0
        assert float(q_uvt.grad[:, 1].detach().abs().sum().cpu()) > 0.0
        assert float(q_uvt.grad[:, [2, 4, 5]].detach().abs().sum().cpu()) > 0.0
        assert opacity.grad is not None and float(opacity.grad.detach().abs().sum().cpu()) > 0.0
        assert color.grad is not None and float(color.grad.detach().abs().sum().cpu()) > 0.0
        rows.append(
            {
                "segments": projected.diagnostics.segment_count,
                "trace_count": int(atlas.coeffs.shape[0]),
            }
        )

    assert [row["segments"] for row in rows] == [4, 4]
    assert [row["trace_count"] for row in rows] == [4, 4]


def test_orbit_derived_uv_visibility_split_report_reduces_fallback() -> None:
    times = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32).contiguous()
    orbit_depth = _orbit_trace_coeffs(point_x=0.15, base_depth=1.4, vertical=0.2)[0, 6:9]
    coeffs = torch.zeros((2, 9), dtype=torch.float32)
    coeffs[:, 0] = 2.0
    coeffs[:, 3] = 0.5
    coeffs[0, 6:9] = orbit_depth
    coeffs[1, 6:9] = orbit_depth + torch.tensor([0.9, 1.6, 0.0], dtype=torch.float32)
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.contiguous(),
        opacity=torch.tensor([1.0, 1.0], dtype=torch.float32),
        color=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=3,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((0.0, 3.0), (0.0, 4.0)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0, 1),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(3, 3),
        depth_affine_uv=torch.tensor(
            [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
    )

    report = adapt_projective_trace_cell_atlas_uv_visibility_events(
        atlas,
        times,
        image_width=8,
        image_height=1,
        tile_size=8,
        min_child_tile_size=1,
    )

    assert report.accepted
    assert report.candidate_tile_sizes == (4, 2, 1)
    assert report.output_tile_size == 2
    assert report.parent_uv_event_tile_samples == 3
    assert report.parent_fallback_fraction == 1.0
    assert report.residual_uv_event_tile_samples == 0
    assert report.fallback_fraction == 0.0
    assert report.output_cells == 4


def test_orbit_windows_do_not_grow_with_frame_density() -> None:
    coeffs = _orbit_trace_coeffs()
    counts = []

    for frames in (16, 32, 64, 128, 256):
        times = _tan_half_angle_times(-math.radians(75.0), math.radians(75.0), frames)
        counts.append(_accepted_window_count(coeffs, times, max_residual_uv=0.015))

    assert counts[0] >= 1
    assert max(counts) <= 2 * counts[0]
    assert max(counts) < 0.10 * 256


def test_orbit_windows_increase_with_orbit_span_complexity() -> None:
    coeffs = _orbit_trace_coeffs()
    counts = []

    for half_span_degrees in (15.0, 30.0, 60.0, 90.0, 120.0):
        times = _tan_half_angle_times(
            -math.radians(half_span_degrees),
            math.radians(half_span_degrees),
            128,
        )
        counts.append(_accepted_window_count(coeffs, times, max_residual_uv=0.015))

    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_orbit_windows_increase_when_residual_contract_tightens() -> None:
    coeffs = _orbit_trace_coeffs()
    times = _tan_half_angle_times(-math.radians(90.0), math.radians(90.0), 96)

    loose = _accepted_window_count(coeffs, times, max_residual_uv=0.08)
    tight = _accepted_window_count(coeffs, times, max_residual_uv=0.005)

    assert loose >= 1
    assert tight > loose


def test_orbit_windows_mark_denominator_crossing_unresolved() -> None:
    coeffs = _orbit_trace_coeffs(point_x=0.0, base_depth=0.0, vertical=0.1)
    times = _tan_half_angle_times(-math.radians(120.0), math.radians(120.0), 17)

    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=0.02,
        min_denominator_abs=1.0e-3,
    )

    assert any(not window.accepted for window in windows)
    assert any(
        "denominator" in window.reason or "invalid_samples" in window.reason
        for window in windows
    )


def test_orbit_window_denominator_root_between_samples_is_a_chart_boundary() -> None:
    coeffs = torch.tensor(
        [[1.0, 0.0, 0.0, -2.0, 0.0, 0.0, -0.3, 1.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.tensor([0.0, 0.6], dtype=torch.float32).contiguous()

    fit = fit_projective_trace_polynomial(coeffs, times, degree=1, eps=1.0e-4)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0,
        min_denominator_abs=1.0e-3,
    )

    assert fit.valid_fraction[0] == 1.0
    assert fit.denominator_has_root[0]
    assert len(windows) == 1
    assert not windows[0].accepted
    assert "denominator_boundary" in windows[0].reason


def test_projective_window_rejects_hidden_near_zero_denominator_minimum() -> None:
    # h_z(t) = (t - 0.5)^2 + 1e-5.  Both frame samples are healthy, but the
    # continuous minimum is below the requested chart margin.
    coeffs = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25001, -1.0, 1.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.tensor([0.0, 1.0], dtype=torch.float32).contiguous()

    fit = fit_projective_trace_polynomial(coeffs, times, degree=1)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e-4,
        min_denominator_abs=1.0e-3,
    )

    assert not fit.denominator_has_root[0]
    assert fit.denominator_min_abs[0] < 1.0e-3
    assert not windows[0].accepted
    assert "denominator" in windows[0].reason


def test_projective_window_detects_small_quadratic_root_on_large_time_interval() -> None:
    # h_z(t) = 1e-7 t^2 - 1e-5 has roots at t = +/-10.  Its raw quadratic
    # coefficient is smaller than the evaluator epsilon, so this catches the
    # old raw-coefficient degree classifier.
    coeffs = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0e-5, 0.0, 1.0e-7]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.tensor([-20.0, 20.0], dtype=torch.float32).contiguous()

    fit = fit_projective_trace_polynomial(coeffs, times, degree=1)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0,
        min_denominator_abs=1.0e-6,
    )

    assert fit.denominator_has_root[0]
    assert fit.denominator_min_abs[0] == 0.0
    assert not windows[0].accepted
    assert "denominator_boundary" in windows[0].reason


def test_split_window_stores_the_domain_root_certificate() -> None:
    coeffs = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 1.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.tensor([0.0, 0.2, 0.4, 0.6], dtype=torch.float32).contiguous()

    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0e6,
        min_denominator_abs=1.0e-3,
    )
    boundary = next(window for window in windows if "denominator_boundary" in window.reason)

    assert not boundary.accepted
    assert boundary.fit.denominator_has_root[0]
    assert boundary.fit.denominator_min_abs[0] == 0.0


def test_projective_support_bounds_cover_accepted_orbit_window_samples() -> None:
    coeffs = _orbit_trace_coeffs()
    times = _tan_half_angle_times(-math.radians(90.0), math.radians(90.0), 96)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=0.015,
        min_denominator_abs=1.0e-3,
    )
    bounds = bound_projective_trace_windows(windows)

    assert len(bounds) == len(windows)
    for window, bound in zip(windows, bounds, strict=True):
        samples = eval_projective_trace_torch(coeffs, times[window.start : window.stop].contiguous())[:, :, :3]
        uv = samples[:, :, :2]
        depth = samples[:, :, 2]

        assert torch.all(uv >= bound.uv_min[:, None, :] - 1.0e-5)
        assert torch.all(uv <= bound.uv_max[:, None, :] + 1.0e-5)
        assert torch.all(depth >= bound.depth_min[:, None] - 1.0e-5)
        assert torch.all(depth <= bound.depth_max[:, None] + 1.0e-5)


def test_projective_support_bounds_refuse_unresolved_windows_by_default() -> None:
    coeffs = torch.tensor(
        [[1.0, 0.0, 0.0, -2.0, 0.0, 0.0, -0.3, 1.0, 0.0]],
        dtype=torch.float32,
    ).contiguous()
    times = torch.tensor([0.0, 0.6], dtype=torch.float32).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=1.0,
        min_denominator_abs=1.0e-3,
    )

    assert not windows[0].accepted
    try:
        bound_projective_trace_window(windows[0])
    except ValueError as exc:
        assert "unresolved" in str(exc)
    else:
        raise AssertionError("unresolved projective window should not produce default support bounds")
