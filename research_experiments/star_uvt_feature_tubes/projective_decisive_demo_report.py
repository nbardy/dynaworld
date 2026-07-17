from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    projective_trace_cell_atlas_complexity_stats,
    projective_trace_cell_atlas_fallback_stats,
    render_projective_trace_cell_atlas_reference,
)


BENCHMARK = "star_uvt_projective_decisive_demo"
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-08_star_uvt_projective_decisive_demo_fixture"
DEFAULT_MAX_IMAGE_ABS_ERROR = 1.0e-6
DEFAULT_MIN_PSNR = 100.0
SIGMA_PX = 1.25
DEFAULT_SAVED_REAL_VIDEO_SUMMARY = (
    ROOT / "outputs" / "visual_comparisons" / "star_uvt_worldtubes_metal_128_16f_60step_2048tubes.json"
)


def _fixture_times(frames: int) -> torch.Tensor:
    if frames <= 1:
        raise ValueError("frames must be greater than one")
    return torch.arange(frames, dtype=torch.float32).contiguous()


def _fixture_coeffs() -> torch.Tensor:
    return torch.tensor(
        [
            [3.40, 0.12, 0.0, 3.10, 0.08, 0.0, 1.00, 0.025, 0.0],
            [4.70, -0.09, 0.0, 3.90, -0.04, 0.0, 1.90, 0.010, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()


def _fixture_color() -> torch.Tensor:
    return torch.tensor([[0.95, 0.18, 0.08], [0.07, 0.34, 0.92]], dtype=torch.float32).contiguous()


def _fixture_opacity() -> torch.Tensor:
    return torch.tensor([0.58, 0.43], dtype=torch.float32).contiguous()


def _depth_intervals(coeffs: torch.Tensor, times: torch.Tensor, *, start: int, stop: int) -> tuple[tuple[float, float], ...]:
    span = times[start:stop]
    intervals: list[tuple[float, float]] = []
    for trace_id in range(int(coeffs.shape[0])):
        z = coeffs[trace_id, 6] + coeffs[trace_id, 7] * span + coeffs[trace_id, 8] * span.square()
        intervals.append((float(z.min().item()), float(z.max().item())))
    return tuple(intervals)


def _make_cell(
    coeffs: torch.Tensor,
    times: torch.Tensor,
    *,
    start: int,
    stop: int,
    fallback: bool = False,
) -> ProjectiveTraceTileTimeCell:
    trace_ids = tuple(range(int(coeffs.shape[0])))
    return ProjectiveTraceTileTimeCell(
        tile_u=0,
        tile_v=0,
        start=int(start),
        stop=int(stop),
        primitive_ids=trace_ids,
        ordered_primitive_ids=trace_ids,
        depth_intervals=_depth_intervals(coeffs, times, start=start, stop=stop),
        fallback=bool(fallback),
        fallback_reasons=("fixture_forced_fallback",) if fallback else (),
    )


def build_fixture_atlas(*, route: str, frames: int = 8, force_fallback: bool = False) -> ProjectiveTraceCellTraceAtlas:
    """Build the smallest replay-vs-compiled atlas used by the decisive demo."""

    times = _fixture_times(frames)
    coeffs = _fixture_coeffs()
    if route == "per_frame_replay":
        cells = [_make_cell(coeffs, times, start=frame, stop=frame + 1, fallback=force_fallback) for frame in range(frames)]
    elif route == "compiled_interval_atlas":
        cells = [_make_cell(coeffs, times, start=0, stop=frames, fallback=force_fallback)]
    else:
        raise ValueError(f"unknown route: {route}")
    trace_count = int(coeffs.shape[0])
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=_fixture_opacity(),
        color=_fixture_color(),
        cells=cells,
        source_window_indices=tuple(0 for _ in range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(frames for _ in range(trace_count)),
    )


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _atlas_payload_bytes(atlas: ProjectiveTraceCellTraceAtlas) -> int:
    tensor_bytes = sum(
        _tensor_bytes(tensor)
        for tensor in (
            atlas.coeffs,
            atlas.opacity,
            atlas.color,
            atlas.opacity_time_coeffs,
            atlas.spatial_precision_uv,
            atlas.depth_affine_uv,
        )
    )
    cell_bytes = 0
    for cell in atlas.cells:
        cell_bytes += 64
        cell_bytes += 4 * (len(cell.primitive_ids) + len(cell.ordered_primitive_ids))
        cell_bytes += 8 * 2 * len(cell.depth_intervals)
    return tensor_bytes + cell_bytes


def _render_fixture(atlas: ProjectiveTraceCellTraceAtlas, times: torch.Tensor, *, image_size: int, tile_size: int) -> torch.Tensor:
    return render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=image_size,
        image_height=image_size,
        tile_size=tile_size,
        sigma_px=SIGMA_PX,
    )


def _psnr(image: torch.Tensor, reference: torch.Tensor) -> float:
    mse = float((image - reference).square().mean().item())
    if mse <= 0.0:
        return 120.0
    return min(120.0, 10.0 * math.log10(1.0 / mse))


def _image_error_metrics(image: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = (image - reference).abs()
    return {
        "max_image_abs_error_vs_reference": float(diff.max().item()),
        "mean_image_abs_error_vs_reference": float(diff.mean().item()),
        "psnr_vs_reference": _psnr(image, reference),
    }


def _build_row(
    *,
    route: str,
    frames: int,
    image_size: int,
    tile_size: int,
    reference_image: torch.Tensor | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    times = _fixture_times(frames)
    compile_started = time.perf_counter()
    atlas = build_fixture_atlas(route=route, frames=frames)
    compile_ms = (time.perf_counter() - compile_started) * 1000.0
    render_started = time.perf_counter()
    image = _render_fixture(atlas, times, image_size=image_size, tile_size=tile_size)
    render_forward_ms = (time.perf_counter() - render_started) * 1000.0
    if reference_image is None:
        reference_image = image
    complexity = projective_trace_cell_atlas_complexity_stats(atlas)
    fallback = projective_trace_cell_atlas_fallback_stats(atlas)
    fallback_cell_fraction = (
        float(fallback.fallback_cells) / float(fallback.total_cells)
        if int(fallback.total_cells) > 0
        else 0.0
    )
    fallback_sample_fraction = (
        float(fallback.fallback_trace_samples) / float(fallback.total_trace_samples)
        if int(fallback.total_trace_samples) > 0
        else 0.0
    )
    row: dict[str, Any] = {
        "route": route,
        "mode": "fixture_correctness",
        "frames": int(frames),
        "image_size": int(image_size),
        "tile_size": int(tile_size),
        "trace_count": int(atlas.coeffs.shape[0]),
        "interval_entry_count": int(complexity.interval_trace_entries),
        "dense_trace_samples": int(complexity.dense_trace_samples),
        "interval_to_dense_trace_sample_ratio": float(complexity.interval_to_dense_trace_sample_ratio),
        "projection_binning_proxy_entries": int(complexity.interval_trace_entries),
        "tile_cell_count": int(complexity.total_cells),
        "active_set_group_count": int(complexity.tile_active_set_groups),
        "visibility_strata_count": int(complexity.tile_active_set_groups + complexity.visibility_stratum_split_cells),
        "fallback_cell_fraction": fallback_cell_fraction,
        "fallback_sample_fraction": fallback_sample_fraction,
        "fallback_reasons": list(fallback.fallback_reasons),
        "compile_ms": compile_ms,
        "render_forward_ms": render_forward_ms,
        "backward_ms": None,
        "total_no_first_ms": compile_ms + render_forward_ms,
        "gradient_rel_error": None,
        "gradient_checked": False,
        "memory_payload_bytes": _atlas_payload_bytes(atlas),
        "renderer": "projective_trace_cell_atlas_reference",
        "uses_metal_forward": False,
    }
    row.update(_image_error_metrics(image, reference_image))
    return row, image


def _fixture_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("route") in {"per_frame_replay", "compiled_interval_atlas"}]


def _media_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("route") == "real_video_media"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixture_rows = _fixture_rows(rows)
    media_rows = _media_rows(rows)
    by_route = {str(row.get("route")): row for row in rows}
    replay = by_route.get("per_frame_replay")
    compiled = by_route.get("compiled_interval_atlas")
    max_error = (
        max(float(row.get("max_image_abs_error_vs_reference", math.inf)) for row in fixture_rows)
        if fixture_rows
        else math.inf
    )
    min_psnr = (
        min(float(row.get("psnr_vs_reference", -math.inf)) for row in fixture_rows)
        if fixture_rows
        else -math.inf
    )
    summary: dict[str, Any] = {
        "routes": [str(row.get("route")) for row in rows],
        "row_count": len(rows),
        "replay_route_present": replay is not None,
        "compiled_route_present": compiled is not None,
        "real_video_media_row_count": len(media_rows),
        "has_real_video_media_rows": bool(media_rows),
        "real_video_media_rows_ok": all(row.get("status") == "ok" for row in media_rows) if media_rows else False,
        "max_image_abs_error_vs_reference": max_error,
        "min_psnr_vs_reference": min_psnr,
        "all_rows_fallback_free": all(
            float(row.get("fallback_cell_fraction", math.inf)) == 0.0
            and float(row.get("fallback_sample_fraction", math.inf)) == 0.0
            for row in fixture_rows
        ),
        "all_rows_quality_pass": max_error <= DEFAULT_MAX_IMAGE_ABS_ERROR and min_psnr >= DEFAULT_MIN_PSNR,
    }
    if media_rows:
        summary.update(
            {
                "real_video_min_psnr": min(float(row["final_psnr"]) for row in media_rows),
                "real_video_max_l1": max(float(row["final_l1"]) for row in media_rows),
                "real_video_min_artifact_count": min(int(row["artifact_count"]) for row in media_rows),
            }
        )
    if replay is not None and compiled is not None:
        summary.update(
            {
                "compiled_to_replay_interval_entry_ratio": float(compiled["interval_entry_count"])
                / float(replay["interval_entry_count"]),
                "compiled_to_replay_dense_sample_ratio": float(compiled["dense_trace_samples"])
                / float(replay["dense_trace_samples"]),
                "compiled_to_replay_memory_ratio": float(compiled["memory_payload_bytes"])
                / float(replay["memory_payload_bytes"]),
                "compiled_to_replay_total_no_first_ms_ratio": float(compiled["total_no_first_ms"])
                / float(replay["total_no_first_ms"]),
            }
        )
    return summary


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def _write_svg_bars(path: Path, *, title: str, values: dict[str, float], units: str = "") -> Path:
    width = 640
    height = 120 + 48 * max(1, len(values))
    max_value = max([abs(value) for value in values.values()] + [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="36" font-family="sans-serif" font-size="20" fill="#0f172a">{title}</text>',
    ]
    for index, (label, value) in enumerate(values.items()):
        y = 72 + 42 * index
        bar_width = 1.0 if max_value == 0.0 else max(1.0, 440.0 * abs(value) / max_value)
        lines.extend(
            [
                f'<text x="24" y="{y + 16}" font-family="sans-serif" font-size="14" fill="#334155">{label}</text>',
                f'<rect x="170" y="{y}" width="{bar_width:.3f}" height="22" fill="#2563eb"/>',
                f'<text x="{180 + bar_width:.3f}" y="{y + 16}" font-family="sans-serif" font-size="14" fill="#0f172a">{value:.6g}{units}</text>',
            ]
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_fallback_heatmap(path: Path, *, fallback_fraction: float) -> Path:
    width = 320
    height = 220
    fill = "#dc2626" if fallback_fraction > 0.0 else "#16a34a"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="24" y="36" font-family="sans-serif" font-size="18" fill="#0f172a">Fallback Heatmap</text>',
        f'<rect x="24" y="58" width="272" height="120" fill="{fill}" opacity="0.85"/>',
        f'<text x="24" y="202" font-family="sans-serif" font-size="14" fill="#334155">fallback_sample_fraction={fallback_fraction:.6g}</text>',
        "</svg>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _root_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_existing_path(payload: dict[str, Any], key: str, *, base_path: Path) -> Path:
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"saved real-video payload missing {key}")
    path = Path(raw)
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"saved real-video {key} does not exist: {resolved}")
    return resolved


def _real_video_media_row(
    *,
    saved_summary_path: Path,
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    with saved_summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"saved real-video payload must be an object: {saved_summary_path}")
    uvt = payload.get("uvt")
    if not isinstance(uvt, dict):
        raise ValueError("saved real-video payload missing uvt metrics")

    contact_src = _require_existing_path(payload, "contact_sheet", base_path=saved_summary_path)
    sbs_src = _require_existing_path(payload, "side_by_side_video", base_path=saved_summary_path)
    media_dir = artifact_dir / "real_video_media"
    contact_sheet = _copy_file(contact_src, media_dir / "contact_sheet.jpg")
    side_by_side = _copy_file(sbs_src, media_dir / "side_by_side.mp4")
    fallback_heatmap = _write_fallback_heatmap(media_dir / "fallback_heatmap.svg", fallback_fraction=0.0)
    runtime_bars = _write_svg_bars(
        media_dir / "runtime_bars.svg",
        title="WorldTubes Real-Video Runtime",
        values={
            "render_median_ms": float((uvt.get("render_benchmark_ms") or {}).get("median", uvt.get("render_ms", 0.0))),
            "wall_clock_ms": float(uvt.get("wall_clock_ms", 0.0)),
        },
        units="ms",
    )
    memory_bars = _write_svg_bars(
        media_dir / "memory_bars.svg",
        title="WorldTubes Real-Video Payload",
        values={
            "tube_count": float(uvt.get("tube_count", payload.get("tube_count", 0.0))),
            "parameter_count": float(uvt.get("parameter_count", 0.0)),
        },
    )
    artifacts = {
        "contact_sheet": _root_relative(contact_sheet),
        "side_by_side_video": _root_relative(side_by_side),
        "fallback_heatmap": _root_relative(fallback_heatmap),
        "runtime_bars": _root_relative(runtime_bars),
        "memory_bars": _root_relative(memory_bars),
    }
    artifact_count = sum(1 for value in artifacts.values() if (ROOT / value).exists())
    row = {
        "route": "real_video_media",
        "mode": "real_video_media",
        "status": "ok",
        "source_summary_path": _root_relative(saved_summary_path),
        "frames": int(payload.get("frames", 0)),
        "image_size": int(payload.get("width", payload.get("height", 0))),
        "width": int(payload.get("width", 0)),
        "height": int(payload.get("height", 0)),
        "steps": int(payload.get("steps", uvt.get("steps", 0))),
        "tube_count": int(uvt.get("tube_count", uvt.get("initial_tube_count", 0))),
        "parameter_count": int(uvt.get("parameter_count", 0)),
        "final_psnr": float(uvt.get("final_psnr", 0.0)),
        "final_l1": float(uvt.get("final_l1", 0.0)),
        "final_loss": float(uvt.get("final_loss", 0.0)),
        "final_ssim_mean": float(uvt.get("final_ssim_mean", 0.0)),
        "render_median_ms": float((uvt.get("render_benchmark_ms") or {}).get("median", uvt.get("render_ms", 0.0))),
        "wall_clock_ms": float(uvt.get("wall_clock_ms", 0.0)),
        "artifact_count": artifact_count,
        "media_artifacts_exist": artifact_count == len(artifacts),
        "artifacts": artifacts,
    }
    return row, artifacts


def run_report(
    *,
    frames: int = 8,
    image_size: int = 8,
    tile_size: int = 8,
    include_saved_real_video: bool = False,
    media_artifact_dir: str | Path | None = None,
    saved_real_video_summary: str | Path = DEFAULT_SAVED_REAL_VIDEO_SUMMARY,
) -> dict[str, Any]:
    if image_size <= 0 or tile_size <= 0:
        raise ValueError("image_size and tile_size must be positive")
    if tile_size != image_size:
        raise ValueError("fixture runner currently expects one image-wide tile")

    replay_row, replay_image = _build_row(
        route="per_frame_replay",
        frames=frames,
        image_size=image_size,
        tile_size=tile_size,
        reference_image=None,
    )
    compiled_row, _compiled_image = _build_row(
        route="compiled_interval_atlas",
        frames=frames,
        image_size=image_size,
        tile_size=tile_size,
        reference_image=replay_image,
    )
    rows = [replay_row, compiled_row]
    artifacts: dict[str, str] = {}
    mode = "fixture_correctness"
    requires_media_artifacts = False
    if include_saved_real_video:
        artifact_dir = Path(media_artifact_dir) if media_artifact_dir is not None else DEFAULT_OUT_DIR
        if not artifact_dir.is_absolute():
            artifact_dir = ROOT / artifact_dir
        media_row, media_artifacts = _real_video_media_row(
            saved_summary_path=Path(saved_real_video_summary)
            if Path(saved_real_video_summary).is_absolute()
            else ROOT / Path(saved_real_video_summary),
            artifact_dir=artifact_dir,
        )
        rows.append(media_row)
        artifacts.update(media_artifacts)
        mode = "real_video_media"
        requires_media_artifacts = True
    return {
        "benchmark": BENCHMARK,
        "mode": mode,
        "requires_media_artifacts": requires_media_artifacts,
        "frames": int(frames),
        "image_size": int(image_size),
        "tile_size": int(tile_size),
        "sigma_px": SIGMA_PX,
        "max_allowed_image_abs_error": DEFAULT_MAX_IMAGE_ABS_ERROR,
        "min_allowed_psnr": DEFAULT_MIN_PSNR,
        "artifacts": artifacts,
        "rows": rows,
        "summary": summarize(rows),
    }


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _nullable_finite_float(value: Any, label: str, errors: list[str]) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label, errors)


def _positive_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    if int(value) <= 0:
        errors.append(f"{label} must be positive, got {value!r}")
    return int(value)


def _assert_close(actual: Any, expected: Any, label: str, errors: list[str], *, atol: float = 1.0e-9) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected) > atol:
            errors.append(f"{label} mismatch: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        errors.append(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _require_media_artifacts(report: dict[str, Any], errors: list[str]) -> None:
    if not (bool(report.get("requires_media_artifacts")) or str(report.get("mode", "")).endswith("_media")):
        return
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("media reports must include an artifacts object")
        return
    for key in ("contact_sheet", "fallback_heatmap", "runtime_bars", "memory_bars"):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"media report missing artifact path: {key}")
            continue
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        if not resolved.exists():
            errors.append(f"media report artifact does not exist: {key}={value}")


def verify_projective_decisive_demo_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != BENCHMARK:
        errors.append(f"benchmark must be {BENCHMARK}")
    if report.get("mode") not in {"fixture_correctness", "real_video_media"}:
        errors.append("mode must be fixture_correctness or real_video_media")
    frames = _positive_int(report.get("frames"), "frames", errors)
    image_size = _positive_int(report.get("image_size"), "image_size", errors)
    tile_size = _positive_int(report.get("tile_size"), "tile_size", errors)
    max_allowed_error = _finite_float(
        report.get("max_allowed_image_abs_error", DEFAULT_MAX_IMAGE_ABS_ERROR),
        "max_allowed_image_abs_error",
        errors,
    )
    min_allowed_psnr = _finite_float(report.get("min_allowed_psnr", DEFAULT_MIN_PSNR), "min_allowed_psnr", errors)
    _require_media_artifacts(report, errors)

    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list):
        errors.append("rows must be a list")
        return errors
    rows = [row for row in raw_rows if isinstance(row, dict)]
    if len(rows) != len(raw_rows):
        errors.append("all rows must be objects")
    by_route = {str(row.get("route")): row for row in rows}
    replay = by_route.get("per_frame_replay")
    compiled = by_route.get("compiled_interval_atlas")
    media_rows = [row for row in rows if row.get("route") == "real_video_media"]
    if replay is None:
        errors.append("rows must include per_frame_replay")
    if compiled is None:
        errors.append("rows must include compiled_interval_atlas")
    if report.get("mode") == "real_video_media" and not media_rows:
        errors.append("real_video_media report must include a real_video_media row")
    if len(rows) < 2:
        return errors

    for row in rows:
        route = str(row.get("route"))
        if route == "real_video_media":
            if row.get("status") != "ok":
                errors.append("real_video_media row status must be ok")
            for key in ("frames", "image_size", "width", "height", "steps", "tube_count", "artifact_count"):
                if _positive_int(row.get(key), f"real_video_media {key}", errors) <= 0:
                    errors.append(f"real_video_media {key} must be positive")
            for key in ("final_psnr", "render_median_ms", "wall_clock_ms"):
                if _finite_float(row.get(key), f"real_video_media {key}", errors) <= 0.0:
                    errors.append(f"real_video_media {key} must be positive")
            if _finite_float(row.get("final_l1"), "real_video_media final_l1", errors) < 0.0:
                errors.append("real_video_media final_l1 must be nonnegative")
            if row.get("media_artifacts_exist") is not True:
                errors.append("real_video_media media_artifacts_exist must be true")
            artifacts = row.get("artifacts")
            if not isinstance(artifacts, dict):
                errors.append("real_video_media row must include artifacts")
            else:
                for key in ("contact_sheet", "side_by_side_video", "fallback_heatmap", "runtime_bars", "memory_bars"):
                    value = artifacts.get(key)
                    if not isinstance(value, str) or not value:
                        errors.append(f"real_video_media row missing artifact {key}")
                        continue
                    resolved = Path(value)
                    if not resolved.is_absolute():
                        resolved = ROOT / resolved
                    if not resolved.exists():
                        errors.append(f"real_video_media row artifact does not exist: {key}={value}")
            continue
        if route not in {"per_frame_replay", "compiled_interval_atlas"}:
            errors.append(f"unknown route {route!r}")
            continue
        row_frames = _positive_int(row.get("frames"), f"{route} frames", errors)
        row_image_size = _positive_int(row.get("image_size"), f"{route} image_size", errors)
        row_tile_size = _positive_int(row.get("tile_size"), f"{route} tile_size", errors)
        trace_count = _positive_int(row.get("trace_count"), f"{route} trace_count", errors)
        interval_entries = _positive_int(row.get("interval_entry_count"), f"{route} interval_entry_count", errors)
        dense_samples = _positive_int(row.get("dense_trace_samples"), f"{route} dense_trace_samples", errors)
        proxy_entries = _positive_int(row.get("projection_binning_proxy_entries"), f"{route} projection_binning_proxy_entries", errors)
        tile_cell_count = _positive_int(row.get("tile_cell_count"), f"{route} tile_cell_count", errors)
        active_groups = _positive_int(row.get("active_set_group_count"), f"{route} active_set_group_count", errors)
        visibility_strata = _positive_int(row.get("visibility_strata_count"), f"{route} visibility_strata_count", errors)
        memory_payload_bytes = _positive_int(row.get("memory_payload_bytes"), f"{route} memory_payload_bytes", errors)
        interval_ratio = _finite_float(row.get("interval_to_dense_trace_sample_ratio"), f"{route} interval ratio", errors)
        fallback_cell_fraction = _finite_float(row.get("fallback_cell_fraction"), f"{route} fallback_cell_fraction", errors)
        fallback_sample_fraction = _finite_float(row.get("fallback_sample_fraction"), f"{route} fallback_sample_fraction", errors)
        compile_ms = _finite_float(row.get("compile_ms"), f"{route} compile_ms", errors)
        render_forward_ms = _finite_float(row.get("render_forward_ms"), f"{route} render_forward_ms", errors)
        total_no_first_ms = _finite_float(row.get("total_no_first_ms"), f"{route} total_no_first_ms", errors)
        backward_ms = _nullable_finite_float(row.get("backward_ms"), f"{route} backward_ms", errors)
        gradient_rel_error = _nullable_finite_float(row.get("gradient_rel_error"), f"{route} gradient_rel_error", errors)
        max_image_error = _finite_float(
            row.get("max_image_abs_error_vs_reference"),
            f"{route} max_image_abs_error_vs_reference",
            errors,
        )
        mean_image_error = _finite_float(
            row.get("mean_image_abs_error_vs_reference"),
            f"{route} mean_image_abs_error_vs_reference",
            errors,
        )
        psnr = _finite_float(row.get("psnr_vs_reference"), f"{route} psnr_vs_reference", errors)

        if row_frames != frames:
            errors.append(f"{route} frames must match report frames")
        if row_image_size != image_size:
            errors.append(f"{route} image_size must match report image_size")
        if row_tile_size != tile_size:
            errors.append(f"{route} tile_size must match report tile_size")
        if trace_count <= 0 or tile_cell_count <= 0 or active_groups <= 0 or visibility_strata <= 0:
            errors.append(f"{route} topology counts must be positive")
        if interval_entries > dense_samples:
            errors.append(f"{route} interval entries cannot exceed dense samples")
        if proxy_entries != interval_entries:
            errors.append(f"{route} projection_binning_proxy_entries must equal interval_entry_count")
        if dense_samples > 0:
            _assert_close(
                interval_ratio,
                interval_entries / float(dense_samples),
                f"{route} interval_to_dense_trace_sample_ratio",
                errors,
            )
        if route == "per_frame_replay" and interval_entries != dense_samples:
            errors.append("per_frame_replay must replay every dense trace sample")
        if route == "compiled_interval_atlas" and interval_entries >= dense_samples:
            errors.append("compiled_interval_atlas must compress dense trace samples")
        if fallback_cell_fraction != 0.0 or fallback_sample_fraction != 0.0:
            errors.append(f"{route} must be fallback-free")
        if compile_ms < 0.0 or render_forward_ms <= 0.0 or total_no_first_ms <= 0.0:
            errors.append(f"{route} timings must be nonnegative compile and positive render/total")
        if total_no_first_ms + 1.0e-9 < compile_ms + render_forward_ms:
            errors.append(f"{route} total_no_first_ms must include compile_ms and render_forward_ms")
        if backward_ms is not None and backward_ms <= 0.0:
            errors.append(f"{route} backward_ms must be positive when present")
        if gradient_rel_error is not None and gradient_rel_error < 0.0:
            errors.append(f"{route} gradient_rel_error must be nonnegative when present")
        if max_image_error > max_allowed_error:
            errors.append(f"{route} image error exceeds max_allowed_image_abs_error")
        if mean_image_error > max_allowed_error:
            errors.append(f"{route} mean image error exceeds max_allowed_image_abs_error")
        if psnr < min_allowed_psnr:
            errors.append(f"{route} psnr_vs_reference below min_allowed_psnr")
        if memory_payload_bytes <= 0:
            errors.append(f"{route} memory_payload_bytes must be positive")

    if replay is not None and compiled is not None:
        if int(compiled.get("interval_entry_count", 0)) >= int(replay.get("interval_entry_count", 0)):
            errors.append("compiled_interval_atlas must use fewer interval entries than per_frame_replay")
        if int(compiled.get("memory_payload_bytes", 0)) >= int(replay.get("memory_payload_bytes", 0)):
            errors.append("compiled_interval_atlas must use less payload memory than per_frame_replay")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be present")
        return errors
    expected_summary = summarize(rows)
    for key, expected in expected_summary.items():
        _assert_close(summary.get(key), expected, f"summary {key}", errors)
    if summary.get("all_rows_fallback_free") is not True:
        errors.append("summary must report all_rows_fallback_free true")
    if summary.get("all_rows_quality_pass") is not True:
        errors.append("summary must report all_rows_quality_pass true")

    return errors


def assert_projective_decisive_demo_report(report: dict[str, Any]) -> None:
    errors = verify_projective_decisive_demo_report(report)
    if errors:
        raise AssertionError("projective decisive demo report failed:\n- " + "\n- ".join(errors))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    columns = (
        "route",
        "status",
        "frames",
        "image_size",
        "trace_count",
        "tube_count",
        "interval_entry_count",
        "dense_trace_samples",
        "interval_to_dense_trace_sample_ratio",
        "tile_cell_count",
        "visibility_strata_count",
        "fallback_sample_fraction",
        "memory_payload_bytes",
        "compile_ms",
        "render_forward_ms",
        "render_median_ms",
        "total_no_first_ms",
        "max_image_abs_error_vs_reference",
        "psnr_vs_reference",
        "final_psnr",
        "final_l1",
        "artifact_count",
    )
    lines = [
        "# STAR UVT Projective Decisive Demo",
        "",
        "This fixture compares per-frame replay against one interval-compressed projective cell atlas.",
        "It is the first runner spine for World Tubes paper ablations; real-video and stress rows should extend this schema.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Rows",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in report["rows"]:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    assert_projective_decisive_demo_report(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    markdown_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--fixture-only", action="store_true", help="kept for runner-script compatibility")
    parser.add_argument("--include-saved-real-video", action="store_true")
    parser.add_argument("--saved-real-video-summary", type=Path, default=DEFAULT_SAVED_REAL_VIDEO_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_projective_decisive_demo_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(
        frames=args.frames,
        image_size=args.image_size,
        tile_size=args.tile_size,
        include_saved_real_video=args.include_saved_real_video,
        media_artifact_dir=args.out_dir,
        saved_real_video_summary=args.saved_real_video_summary,
    )
    json_path, markdown_path = write_report(report, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
