from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import ROOT, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, write_report_json

from torch_gsplat_bridge_star_uvt import (
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    adapt_projective_trace_cell_atlas_uv_visibility_events,
)


SCHEMA_VERSION = "projective_uv_visibility_split_report_v1"
HIGH_MOTION_PROXY_VIDEO = ROOT / "data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4"


@dataclass(frozen=True)
class UVVisibilitySplitCase:
    name: str
    source: str
    reference_video_path: str | None
    reference_video_exists: bool
    accepted: bool
    split_attempted: bool
    input_tile_size: int
    output_tile_size: int
    min_child_tile_size: int
    candidate_tile_sizes: tuple[int, ...]
    parent_cells: int
    output_cells: int
    cell_growth: float
    parent_uv_events: int
    parent_uv_event_tile_samples: int
    residual_uv_events: int
    residual_uv_event_tile_samples: int
    parent_fallback_cells: int
    fallback_cells: int
    parent_fallback_fraction: float
    fallback_fraction: float
    fallback_fraction_reduction: float
    needs_oblique_halfspace: bool
    extraction: dict[str, Any] | None = None


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local video deps.
        raise ImportError("OpenCV is required to extract high-motion UV visibility diagnostics.") from exc
    return cv2


def _read_video_gray_frames(path: Path, *, target_size: int, max_frames: int) -> tuple[list[torch.Tensor], dict[str, Any]]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    metadata = {
        "source_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "source_fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "source_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "source_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "target_size": int(target_size),
        "max_frames": int(max_frames),
    }
    frames: list[torch.Tensor] = []
    try:
        while len(frames) < int(max_frames):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame_bgr, (int(target_size), int(target_size)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
            frames.append(torch.from_numpy(gray.copy()))
    finally:
        capture.release()
    metadata["frames_read"] = len(frames)
    return frames, metadata


def _fit_quadratic_coeffs(times: torch.Tensor, values: torch.Tensor) -> tuple[float, float, float]:
    if times.ndim != 1 or values.ndim != 1 or int(times.numel()) != int(values.numel()):
        raise ValueError("times and values must be matching 1D tensors")
    if int(times.numel()) < 2:
        raise ValueError("Need at least two samples to fit a trace diagnostic")
    design = torch.stack([torch.ones_like(times), times, times.square()], dim=1)
    coeffs = torch.linalg.lstsq(design, values.reshape(-1, 1)).solution.reshape(-1)
    return tuple(float(value) for value in coeffs[:3])


def _high_motion_video_centroid_pair_atlas(
    *,
    video_path: Path,
    image_width: int,
    image_height: int,
    target_size: int = 64,
    max_frames: int = 16,
    sample_count: int = 3,
) -> tuple[ProjectiveTraceCellTraceAtlas, torch.Tensor, dict[str, Any]]:
    frames, metadata = _read_video_gray_frames(video_path, target_size=target_size, max_frames=max_frames)
    if len(frames) < 2:
        raise ValueError(f"Need at least two decoded frames in {video_path}")

    pair_diffs = [(right - left).abs() for left, right in zip(frames[:-1], frames[1:])]
    pair_scores = torch.tensor([float(diff.sum().item()) for diff in pair_diffs], dtype=torch.float32)
    selected_count = min(int(sample_count), int(pair_scores.numel()))
    if selected_count < 2:
        raise ValueError("Need at least two motion pairs for the UV visibility diagnostic")
    selected_pair_indices = sorted(int(index) for index in torch.topk(pair_scores, k=selected_count).indices.tolist())

    u_axis = torch.linspace(0.5, float(image_width) - 0.5, int(target_size), dtype=torch.float32)
    v_axis = torch.linspace(0.5, float(image_height) - 0.5, int(target_size), dtype=torch.float32)
    centroid_u: list[float] = []
    centroid_v: list[float] = []
    selected_scores: list[float] = []
    for pair_index in selected_pair_indices:
        diff = pair_diffs[pair_index]
        energy = float(diff.sum().item())
        selected_scores.append(energy)
        if energy <= 1.0e-8:
            centroid_u.append(0.5 * float(image_width))
            centroid_v.append(0.5 * float(image_height))
            continue
        col_energy = diff.sum(dim=0)
        row_energy = diff.sum(dim=1)
        centroid_u.append(float((col_energy * u_axis).sum().item() / energy))
        centroid_v.append(float((row_energy * v_axis).sum().item() / energy))

    roots = torch.tensor(centroid_u, dtype=torch.float32).clamp(0.5, float(image_width) - 0.5)
    times = torch.arange(int(roots.numel()), dtype=torch.float32).contiguous()
    # With the report's symmetric +/-0.2 depth slopes and centers at u=2,
    # front_depth - back_depth = 0 crosses at u=(z_back-0.2)/0.4.
    back_depths = 0.4 * roots + 0.2
    back_depth_coeffs = _fit_quadratic_coeffs(times, back_depths)
    atlas = _base_pair_atlas(
        times=times,
        back_depth_coeffs=back_depth_coeffs,
        image_width=image_width,
    )
    extraction = {
        **metadata,
        "pair_indices": tuple(selected_pair_indices),
        "sample_count": int(selected_count),
        "motion_scores": tuple(float(value) for value in selected_scores),
        "centroid_u": tuple(float(value) for value in centroid_u),
        "centroid_v": tuple(float(value) for value in centroid_v),
        "root_positions_u": tuple(float(value) for value in roots.tolist()),
        "back_depth_coeffs": tuple(float(value) for value in back_depth_coeffs),
    }
    return atlas, times, extraction


def _base_pair_atlas(
    *,
    times: torch.Tensor,
    back_depth_coeffs: tuple[float, float, float],
    image_width: int,
) -> ProjectiveTraceCellTraceAtlas:
    del image_width
    frames = int(times.numel())
    return ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.5, 0.0, 0.0, *back_depth_coeffs],
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
                stop=frames,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((0.0, 4.0), (0.0, 5.0)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0, 1),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(frames, frames),
        depth_affine_uv=torch.tensor(
            [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
    )


def _orbit_depth_coeffs() -> torch.Tensor:
    # Matches the q = tan(theta/2) yaw-orbit depth polynomial used in
    # tests/test_star_uvt_projective_orbit_windows.py.
    point_x = 0.15
    base_depth = 1.4
    return torch.tensor(
        [base_depth + 0.25, 2.0 * point_x, base_depth - 0.25],
        dtype=torch.float32,
    )


def _orbit_pair_atlas(times: torch.Tensor) -> ProjectiveTraceCellTraceAtlas:
    frames = int(times.numel())
    orbit_depth = _orbit_depth_coeffs()
    coeffs = torch.zeros((2, 9), dtype=torch.float32)
    coeffs[:, 0] = 2.0
    coeffs[:, 3] = 0.5
    coeffs[0, 6:9] = orbit_depth
    coeffs[1, 6:9] = orbit_depth + torch.tensor([0.9, 1.6, 0.0], dtype=torch.float32)
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.contiguous(),
        opacity=torch.tensor([1.0, 1.0], dtype=torch.float32),
        color=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=frames,
                primitive_ids=(0, 1),
                ordered_primitive_ids=(0, 1),
                depth_intervals=((0.0, 4.0), (0.0, 5.0)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0, 1),
        source_primitive_ids=(0, 1),
        active_start=(0, 0),
        active_stop=(frames, frames),
        depth_affine_uv=torch.tensor(
            [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
    )


def _measure_case(
    *,
    name: str,
    source: str,
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    image_width: int,
    image_height: int = 1,
    tile_size: int = 8,
    min_child_tile_size: int = 1,
    reference_video_path: Path | None = None,
    extraction: dict[str, Any] | None = None,
) -> UVVisibilitySplitCase:
    report = adapt_projective_trace_cell_atlas_uv_visibility_events(
        atlas,
        times,
        image_width=int(image_width),
        image_height=int(image_height),
        tile_size=int(tile_size),
        min_child_tile_size=int(min_child_tile_size),
    )
    parent_cells = max(1, int(report.parent_cells))
    cell_growth = float(report.output_cells) / float(parent_cells)
    fallback_reduction = float(report.parent_fallback_fraction) - float(report.fallback_fraction)
    return UVVisibilitySplitCase(
        name=name,
        source=source,
        reference_video_path=None if reference_video_path is None else str(reference_video_path),
        reference_video_exists=bool(reference_video_path is not None and reference_video_path.exists()),
        accepted=bool(report.accepted),
        split_attempted=bool(report.split_attempted),
        input_tile_size=int(report.input_tile_size),
        output_tile_size=int(report.output_tile_size),
        min_child_tile_size=int(min_child_tile_size),
        candidate_tile_sizes=tuple(int(value) for value in report.candidate_tile_sizes),
        parent_cells=int(report.parent_cells),
        output_cells=int(report.output_cells),
        cell_growth=cell_growth,
        parent_uv_events=int(report.parent_uv_events),
        parent_uv_event_tile_samples=int(report.parent_uv_event_tile_samples),
        residual_uv_events=int(report.residual_uv_events),
        residual_uv_event_tile_samples=int(report.residual_uv_event_tile_samples),
        parent_fallback_cells=int(report.parent_fallback_cells),
        fallback_cells=int(report.fallback_cells),
        parent_fallback_fraction=float(report.parent_fallback_fraction),
        fallback_fraction=float(report.fallback_fraction),
        fallback_fraction_reduction=fallback_reduction,
        needs_oblique_halfspace=bool((not report.accepted) or report.residual_uv_event_tile_samples > 0),
        extraction=extraction,
    )


def build_uv_visibility_split_report() -> dict[str, Any]:
    high_motion_atlas, high_motion_times, high_motion_extraction = _high_motion_video_centroid_pair_atlas(
        video_path=HIGH_MOTION_PROXY_VIDEO,
        image_width=8,
        image_height=1,
    )
    orbit_times = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32).contiguous()
    cases = [
        _measure_case(
            name="high_motion_video_centroid_line_sweep",
            source="extracted_video_motion_centroid",
            reference_video_path=HIGH_MOTION_PROXY_VIDEO,
            atlas=high_motion_atlas,
            times=high_motion_times,
            image_width=8,
            extraction=high_motion_extraction,
        ),
        _measure_case(
            name="orbit_parameterized_line_sweep",
            source="synthetic_orbit_q_tan_half_angle",
            reference_video_path=None,
            atlas=_orbit_pair_atlas(orbit_times),
            times=orbit_times,
            image_width=8,
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if all(case.accepted for case in cases) else "needs_oblique_halfspace",
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
        "summary": {
            "max_parent_fallback_fraction": max(case.parent_fallback_fraction for case in cases),
            "max_output_fallback_fraction": max(case.fallback_fraction for case in cases),
            "max_cell_growth": max(case.cell_growth for case in cases),
            "any_needs_oblique_halfspace": any(case.needs_oblique_halfspace for case in cases),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write UV visibility adaptive split-vs-fallback report JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/projective_uv_visibility_split_report.json"),
    )
    args = parser.parse_args()
    output = write_report_json(args.output, build_uv_visibility_split_report())
    print(output)


if __name__ == "__main__":
    main()
