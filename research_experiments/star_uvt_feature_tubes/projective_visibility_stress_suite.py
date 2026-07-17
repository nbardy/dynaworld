from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
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
    projective_trace_cell_atlas_visibility_report,
    render_projective_trace_cell_atlas_reference,
    stratify_projective_trace_cell_atlas_visibility,
)


BENCHMARK = "star_uvt_projective_visibility_stress_suite"
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-08_star_uvt_projective_visibility_stress_suite"
SIGMA_PX = 1.2
QUALITY_ERROR_THRESHOLD = 1.0e-5
FALLBACK_COLLAPSE_THRESHOLD = 0.40
RUNTIME_COLLAPSE_THRESHOLD = 1.0
REQUIRED_CASE_IDS = (
    "clean_orbit_ordered",
    "crossing_raw_interval",
    "crossing_stratified",
    "forced_fallback_ambiguous",
)


def _times(frames: int) -> torch.Tensor:
    if frames <= 1:
        raise ValueError("frames must be greater than one")
    return torch.arange(frames, dtype=torch.float32).contiguous()


def _depth_values(coeffs: torch.Tensor, times: torch.Tensor, trace_id: int, start: int, stop: int) -> torch.Tensor:
    span = times[start:stop]
    return coeffs[trace_id, 6] + coeffs[trace_id, 7] * span + coeffs[trace_id, 8] * span.square()


def _depth_intervals(
    coeffs: torch.Tensor,
    times: torch.Tensor,
    *,
    ordered_ids: tuple[int, ...],
    start: int,
    stop: int,
) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for trace_id in ordered_ids:
        depth = _depth_values(coeffs, times, trace_id, start, stop)
        intervals.append((float(depth.min().item()), float(depth.max().item())))
    return tuple(intervals)


def _ordered_ids_at_midpoint(coeffs: torch.Tensor, times: torch.Tensor, start: int, stop: int) -> tuple[int, ...]:
    mid_index = max(start, min(stop - 1, (start + stop - 1) // 2))
    depths = [
        (
            float(_depth_values(coeffs, times, trace_id, mid_index, mid_index + 1).item()),
            trace_id,
        )
        for trace_id in range(int(coeffs.shape[0]))
    ]
    return tuple(trace_id for _depth, trace_id in sorted(depths))


def _cell(
    coeffs: torch.Tensor,
    times: torch.Tensor,
    *,
    start: int,
    stop: int,
    ordered_ids: tuple[int, ...] | None = None,
    fallback: bool = False,
    fallback_reasons: tuple[str, ...] = (),
) -> ProjectiveTraceTileTimeCell:
    if ordered_ids is None:
        ordered_ids = _ordered_ids_at_midpoint(coeffs, times, start, stop)
    return ProjectiveTraceTileTimeCell(
        tile_u=0,
        tile_v=0,
        start=int(start),
        stop=int(stop),
        primitive_ids=tuple(sorted(ordered_ids)),
        ordered_primitive_ids=ordered_ids,
        depth_intervals=_depth_intervals(coeffs, times, ordered_ids=ordered_ids, start=start, stop=stop),
        fallback=bool(fallback),
        fallback_reasons=fallback_reasons if fallback else (),
    )


def _atlas(
    *,
    coeffs: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    cells: list[ProjectiveTraceTileTimeCell],
    frames: int,
) -> ProjectiveTraceCellTraceAtlas:
    trace_count = int(coeffs.shape[0])
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs.contiguous(),
        opacity=opacity.contiguous(),
        color=color.contiguous(),
        cells=cells,
        source_window_indices=tuple(0 for _ in range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(frames for _ in range(trace_count)),
    )


def _clean_case(frames: int) -> ProjectiveTraceCellTraceAtlas:
    times = _times(frames)
    coeffs = torch.tensor(
        [
            [3.40, 0.02, 0.0, 3.50, 0.01, 0.0, 1.00, 0.02, 0.0],
            [4.70, -0.01, 0.0, 3.60, 0.02, 0.0, 2.35, 0.01, 0.0],
        ],
        dtype=torch.float32,
    )
    opacity = torch.tensor([0.55, 0.42], dtype=torch.float32)
    color = torch.tensor([[0.95, 0.2, 0.08], [0.08, 0.32, 0.9]], dtype=torch.float32)
    return _atlas(
        coeffs=coeffs,
        opacity=opacity,
        color=color,
        frames=frames,
        cells=[_cell(coeffs, times, start=0, stop=frames, ordered_ids=(0, 1))],
    )


def _crossing_raw_case(frames: int, *, force_fallback: bool = False) -> ProjectiveTraceCellTraceAtlas:
    times = _times(frames)
    coeffs = torch.tensor(
        [
            [4.00, 0.00, 0.0, 4.00, 0.00, 0.0, 1.00, 1.00, 0.0],
            [4.20, 0.00, 0.0, 4.00, 0.00, 0.0, 3.00, -0.20, 0.0],
        ],
        dtype=torch.float32,
    )
    opacity = torch.tensor([0.62, 0.48], dtype=torch.float32)
    color = torch.tensor([[0.92, 0.22, 0.08], [0.08, 0.35, 0.90]], dtype=torch.float32)
    return _atlas(
        coeffs=coeffs,
        opacity=opacity,
        color=color,
        frames=frames,
        cells=[
            _cell(
                coeffs,
                times,
                start=0,
                stop=frames,
                ordered_ids=(0, 1),
                fallback=force_fallback,
                fallback_reasons=("visibility_ambiguous_depth",),
            )
        ],
    )


def _per_frame_live_atlas(atlas: ProjectiveTraceCellTraceAtlas, times: torch.Tensor) -> ProjectiveTraceCellTraceAtlas:
    cells = [
        _cell(atlas.coeffs, times, start=frame, stop=frame + 1)
        for frame in range(int(times.numel()))
    ]
    return replace(atlas, cells=cells)


def _render(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    *,
    image_size: int,
    tile_size: int,
    allow_fallback_cells: bool = False,
) -> torch.Tensor:
    return render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=image_size,
        image_height=image_size,
        tile_size=tile_size,
        sigma_px=SIGMA_PX,
        allow_fallback_cells=allow_fallback_cells,
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


def _max_cells_per_trace(atlas: ProjectiveTraceCellTraceAtlas) -> int:
    counts = [0 for _ in range(int(atlas.coeffs.shape[0]))]
    for cell in atlas.cells:
        for trace_id in cell.primitive_ids:
            counts[int(trace_id)] += 1
    return max(counts, default=0)


def _quality_error(image: torch.Tensor, reference: torch.Tensor) -> float:
    return float((image - reference).abs().max().item())


def _collapse_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if float(row["fallback_cell_fraction"]) > FALLBACK_COLLAPSE_THRESHOLD:
        reasons.append("fallback_cell_fraction")
    if float(row["fallback_sample_fraction"]) > FALLBACK_COLLAPSE_THRESHOLD:
        reasons.append("fallback_sample_fraction")
    if float(row["runtime_ratio"]) >= RUNTIME_COLLAPSE_THRESHOLD:
        reasons.append("runtime_ratio")
    if not bool(row["fallback_enabled"]) and float(row["quality_error"]) > QUALITY_ERROR_THRESHOLD:
        reasons.append("quality_error_without_fallback")
    return reasons


def _row_for_case(
    *,
    case_id: str,
    scene_family: str,
    policy: str,
    atlas: ProjectiveTraceCellTraceAtlas,
    reference_atlas: ProjectiveTraceCellTraceAtlas,
    frames: int,
    image_size: int,
    tile_size: int,
    fallback_enabled: bool = False,
) -> dict[str, Any]:
    times = _times(frames)
    baseline_atlas = _per_frame_live_atlas(reference_atlas, times)
    baseline_start = time.perf_counter()
    baseline_image = _render(baseline_atlas, times, image_size=image_size, tile_size=tile_size)
    baseline_ms = max((time.perf_counter() - baseline_start) * 1000.0, 1.0e-9)
    render_start = time.perf_counter()
    image = _render(
        atlas,
        times,
        image_size=image_size,
        tile_size=tile_size,
        allow_fallback_cells=fallback_enabled,
    )
    render_ms = max((time.perf_counter() - render_start) * 1000.0, 1.0e-9)

    visibility = projective_trace_cell_atlas_visibility_report(atlas, times)
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
    checked = int(visibility.checked_tile_samples)
    ambiguous = int(visibility.ambiguous_depth_samples)
    quality_error = _quality_error(image, baseline_image)
    runtime_ratio = float(complexity.interval_to_dense_trace_sample_ratio)
    row: dict[str, Any] = {
        "case_id": case_id,
        "scene_family": scene_family,
        "policy": policy,
        "frames": int(frames),
        "image_size": int(image_size),
        "tile_size": int(tile_size),
        "trace_count": int(atlas.coeffs.shape[0]),
        "cell_count": int(len(atlas.cells)),
        "interval_entry_count": int(complexity.interval_trace_entries),
        "dense_trace_samples": int(complexity.dense_trace_samples),
        "interval_to_dense_trace_sample_ratio": float(complexity.interval_to_dense_trace_sample_ratio),
        "fallback_enabled": bool(fallback_enabled),
        "fallback_cell_fraction": fallback_cell_fraction,
        "fallback_sample_fraction": fallback_sample_fraction,
        "fallback_reasons": list(fallback.fallback_reasons),
        "visibility_stale": bool(visibility.stale),
        "checked_tile_samples": checked,
        "order_flip_surface_count": int(visibility.order_mismatch_samples),
        "ambiguous_pair_count": ambiguous,
        "commutable_pair_count": max(0, checked - ambiguous - int(visibility.order_mismatch_samples)),
        "depth_interval_overlap_rate": ambiguous / float(checked) if checked > 0 else 0.0,
        "visibility_strata_count": int(complexity.tile_active_set_groups + complexity.visibility_stratum_split_cells),
        "max_cells_per_trace": _max_cells_per_trace(atlas),
        "max_active_set_group_count": int(complexity.max_cells_per_active_set_group),
        "quality_error": quality_error,
        "runtime_ms": render_ms,
        "baseline_runtime_ms": baseline_ms,
        "measured_runtime_ratio": render_ms / baseline_ms,
        "runtime_ratio": runtime_ratio,
        "memory_payload_bytes": _atlas_payload_bytes(atlas),
        "baseline_memory_payload_bytes": _atlas_payload_bytes(baseline_atlas),
        "memory_ratio": _atlas_payload_bytes(atlas) / float(max(1, _atlas_payload_bytes(baseline_atlas))),
    }
    row["collapse_reasons"] = _collapse_reasons(row)
    row["collapse"] = bool(row["collapse_reasons"])
    return row


def run_report(*, frames: int = 4, image_size: int = 8, tile_size: int = 8) -> dict[str, Any]:
    if image_size <= 0 or tile_size <= 0:
        raise ValueError("image_size and tile_size must be positive")
    if tile_size != image_size:
        raise ValueError("fixture stress suite currently expects one image-wide tile")
    if frames < 4:
        raise ValueError("frames must be at least 4 so the crossing fixture flips order")

    clean = _clean_case(frames)
    crossing_raw = _crossing_raw_case(frames)
    crossing_stratified = stratify_projective_trace_cell_atlas_visibility(crossing_raw, _times(frames))
    forced_fallback = _crossing_raw_case(frames, force_fallback=True)

    rows = [
        _row_for_case(
            case_id="clean_orbit_ordered",
            scene_family="clean_orbit_ordered",
            policy="compiled_interval",
            atlas=clean,
            reference_atlas=clean,
            frames=frames,
            image_size=image_size,
            tile_size=tile_size,
        ),
        _row_for_case(
            case_id="crossing_raw_interval",
            scene_family="crossing_translucent_planes",
            policy="raw_interval",
            atlas=crossing_raw,
            reference_atlas=crossing_raw,
            frames=frames,
            image_size=image_size,
            tile_size=tile_size,
        ),
        _row_for_case(
            case_id="crossing_stratified",
            scene_family="crossing_translucent_planes",
            policy="visibility_stratified",
            atlas=crossing_stratified,
            reference_atlas=crossing_raw,
            frames=frames,
            image_size=image_size,
            tile_size=tile_size,
        ),
        _row_for_case(
            case_id="forced_fallback_ambiguous",
            scene_family="dense_alpha_cloud",
            policy="fallback_live_depth_sort",
            atlas=forced_fallback,
            reference_atlas=crossing_raw,
            frames=frames,
            image_size=image_size,
            tile_size=tile_size,
            fallback_enabled=True,
        ),
    ]
    return {
        "benchmark": BENCHMARK,
        "mode": "fixture_visibility_stress",
        "frames": int(frames),
        "image_size": int(image_size),
        "tile_size": int(tile_size),
        "sigma_px": SIGMA_PX,
        "collapse_thresholds": {
            "fallback_cell_fraction": FALLBACK_COLLAPSE_THRESHOLD,
            "fallback_sample_fraction": FALLBACK_COLLAPSE_THRESHOLD,
            "runtime_ratio": RUNTIME_COLLAPSE_THRESHOLD,
            "quality_error_without_fallback": QUALITY_ERROR_THRESHOLD,
        },
        "rows": rows,
        "summary": summarize(rows),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    collapsed = [row for row in rows if bool(row.get("collapse"))]
    noncollapsed = [row for row in rows if not bool(row.get("collapse"))]
    case_ids = [str(row.get("case_id")) for row in rows]
    return {
        "row_count": len(rows),
        "case_ids": case_ids,
        "required_case_ids_present": all(case_id in case_ids for case_id in REQUIRED_CASE_IDS),
        "collapsed_case_count": len(collapsed),
        "collapsed_case_ids": [str(row.get("case_id")) for row in collapsed],
        "noncollapsed_case_ids": [str(row.get("case_id")) for row in noncollapsed],
        "max_fallback_sample_fraction": max((float(row.get("fallback_sample_fraction", 0.0)) for row in rows), default=0.0),
        "max_quality_error": max((float(row.get("quality_error", 0.0)) for row in rows), default=0.0),
        "max_depth_interval_overlap_rate": max((float(row.get("depth_interval_overlap_rate", 0.0)) for row in rows), default=0.0),
        "has_clean_case": "clean_orbit_ordered" in case_ids,
        "has_ambiguous_case": any(
            int(row.get("ambiguous_pair_count", 0)) > 0
            or int(row.get("order_flip_surface_count", 0)) > 0
            or bool(row.get("visibility_stale"))
            for row in rows
        ),
        "has_repaired_crossing_case": any(
            row.get("case_id") == "crossing_stratified"
            and not bool(row.get("visibility_stale"))
            and not bool(row.get("collapse"))
            for row in rows
        ),
        "has_collapse_boundary": len(collapsed) > 0,
    }


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _assert_summary_close(
    summary: dict[str, Any],
    expected: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    actual = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > atol:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    elif actual != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")


def verify_projective_visibility_stress_suite(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != BENCHMARK:
        errors.append(f"benchmark must be {BENCHMARK}")
    if report.get("mode") != "fixture_visibility_stress":
        errors.append("mode must be fixture_visibility_stress")
    frames = _finite_int(report.get("frames"), "frames", errors)
    image_size = _finite_int(report.get("image_size"), "image_size", errors)
    tile_size = _finite_int(report.get("tile_size"), "tile_size", errors)
    if frames < 4:
        errors.append("frames must be at least 4")
    if image_size <= 0 or tile_size <= 0:
        errors.append("image_size and tile_size must be positive")
    thresholds = report.get("collapse_thresholds")
    if not isinstance(thresholds, dict):
        errors.append("collapse_thresholds must be present")
        thresholds = {}
    fallback_sample_threshold = _finite_float(
        thresholds.get("fallback_sample_fraction", FALLBACK_COLLAPSE_THRESHOLD),
        "collapse_thresholds.fallback_sample_fraction",
        errors,
    )
    fallback_cell_threshold = _finite_float(
        thresholds.get("fallback_cell_fraction", FALLBACK_COLLAPSE_THRESHOLD),
        "collapse_thresholds.fallback_cell_fraction",
        errors,
    )
    quality_threshold = _finite_float(
        thresholds.get("quality_error_without_fallback", QUALITY_ERROR_THRESHOLD),
        "collapse_thresholds.quality_error_without_fallback",
        errors,
    )

    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list):
        errors.append("rows must be a list")
        return errors
    rows = [row for row in raw_rows if isinstance(row, dict)]
    if len(rows) != len(raw_rows):
        errors.append("all rows must be objects")
    by_case = {str(row.get("case_id")): row for row in rows}
    for case_id in REQUIRED_CASE_IDS:
        if case_id not in by_case:
            errors.append(f"missing required case_id {case_id}")

    for row in rows:
        case_id = str(row.get("case_id"))
        row_frames = _finite_int(row.get("frames"), f"{case_id} frames", errors)
        row_image_size = _finite_int(row.get("image_size"), f"{case_id} image_size", errors)
        row_tile_size = _finite_int(row.get("tile_size"), f"{case_id} tile_size", errors)
        trace_count = _finite_int(row.get("trace_count"), f"{case_id} trace_count", errors)
        cell_count = _finite_int(row.get("cell_count"), f"{case_id} cell_count", errors)
        interval_entries = _finite_int(row.get("interval_entry_count"), f"{case_id} interval_entry_count", errors)
        dense_samples = _finite_int(row.get("dense_trace_samples"), f"{case_id} dense_trace_samples", errors)
        checked_tile_samples = _finite_int(row.get("checked_tile_samples"), f"{case_id} checked_tile_samples", errors)
        ambiguous_pairs = _finite_int(row.get("ambiguous_pair_count"), f"{case_id} ambiguous_pair_count", errors)
        order_flips = _finite_int(row.get("order_flip_surface_count"), f"{case_id} order_flip_surface_count", errors)
        strata = _finite_int(row.get("visibility_strata_count"), f"{case_id} visibility_strata_count", errors)
        max_cells_per_trace = _finite_int(row.get("max_cells_per_trace"), f"{case_id} max_cells_per_trace", errors)
        max_active_set_group_count = _finite_int(
            row.get("max_active_set_group_count"),
            f"{case_id} max_active_set_group_count",
            errors,
        )
        fallback_cell_fraction = _finite_float(row.get("fallback_cell_fraction"), f"{case_id} fallback_cell_fraction", errors)
        fallback_sample_fraction = _finite_float(
            row.get("fallback_sample_fraction"),
            f"{case_id} fallback_sample_fraction",
            errors,
        )
        interval_ratio = _finite_float(
            row.get("interval_to_dense_trace_sample_ratio"),
            f"{case_id} interval_to_dense_trace_sample_ratio",
            errors,
        )
        overlap_rate = _finite_float(row.get("depth_interval_overlap_rate"), f"{case_id} depth_interval_overlap_rate", errors)
        quality_error = _finite_float(row.get("quality_error"), f"{case_id} quality_error", errors)
        runtime_ratio = _finite_float(row.get("runtime_ratio"), f"{case_id} runtime_ratio", errors)
        memory_ratio = _finite_float(row.get("memory_ratio"), f"{case_id} memory_ratio", errors)
        collapse = bool(row.get("collapse"))
        collapse_reasons = row.get("collapse_reasons")
        if not isinstance(collapse_reasons, list):
            errors.append(f"{case_id} collapse_reasons must be a list")
            collapse_reasons = []

        if row_frames != frames or row_image_size != image_size or row_tile_size != tile_size:
            errors.append(f"{case_id} dimensions must match report dimensions")
        if trace_count <= 0 or cell_count <= 0 or interval_entries <= 0 or dense_samples <= 0:
            errors.append(f"{case_id} topology counts must be positive")
        if interval_entries > dense_samples:
            errors.append(f"{case_id} interval entries cannot exceed dense samples")
        if dense_samples > 0 and abs(interval_ratio - interval_entries / float(dense_samples)) > 1.0e-6:
            errors.append(f"{case_id} interval ratio mismatch")
        if not 0.0 <= fallback_cell_fraction <= 1.0:
            errors.append(f"{case_id} fallback_cell_fraction must be in [0,1]")
        if not 0.0 <= fallback_sample_fraction <= 1.0:
            errors.append(f"{case_id} fallback_sample_fraction must be in [0,1]")
        if checked_tile_samples > 0 and abs(overlap_rate - ambiguous_pairs / float(checked_tile_samples)) > 1.0e-6:
            errors.append(f"{case_id} depth_interval_overlap_rate mismatch")
        if quality_error < 0.0 or runtime_ratio <= 0.0 or memory_ratio <= 0.0:
            errors.append(f"{case_id} quality/runtime/memory metrics must be positive or zero quality")
        if strata <= 0 or max_cells_per_trace <= 0 or max_active_set_group_count <= 0:
            errors.append(f"{case_id} visibility complexity metrics must be positive")
        if collapse and not collapse_reasons:
            errors.append(f"{case_id} collapsed row must list collapse_reasons")
        if not collapse and collapse_reasons:
            errors.append(f"{case_id} noncollapsed row must not list collapse_reasons")
        if fallback_sample_fraction > fallback_sample_threshold and "fallback_sample_fraction" not in collapse_reasons:
            errors.append(f"{case_id} high fallback_sample_fraction must explain collapse")
        if fallback_cell_fraction > fallback_cell_threshold and "fallback_cell_fraction" not in collapse_reasons:
            errors.append(f"{case_id} high fallback_cell_fraction must explain collapse")
        if (
            not bool(row.get("fallback_enabled"))
            and quality_error > quality_threshold
            and "quality_error_without_fallback" not in collapse_reasons
        ):
            errors.append(f"{case_id} high quality_error without fallback must explain collapse")
        if int(order_flips) > 0 and not bool(row.get("visibility_stale")):
            errors.append(f"{case_id} order flips must mark visibility_stale")

    clean = by_case.get("clean_orbit_ordered")
    if clean is not None:
        if bool(clean.get("collapse")):
            errors.append("clean_orbit_ordered must not collapse")
        if bool(clean.get("visibility_stale")):
            errors.append("clean_orbit_ordered must not be visibility_stale")

    raw = by_case.get("crossing_raw_interval")
    if raw is not None:
        if not bool(raw.get("visibility_stale")):
            errors.append("crossing_raw_interval must be visibility_stale")
        if not bool(raw.get("collapse")):
            errors.append("crossing_raw_interval must expose a collapse boundary")

    stratified = by_case.get("crossing_stratified")
    if stratified is not None:
        if bool(stratified.get("visibility_stale")):
            errors.append("crossing_stratified must repair visibility_stale")
        if bool(stratified.get("collapse")):
            errors.append("crossing_stratified must not collapse")
        if float(stratified.get("quality_error", math.inf)) > quality_threshold:
            errors.append("crossing_stratified quality_error must stay below threshold")

    fallback = by_case.get("forced_fallback_ambiguous")
    if fallback is not None:
        if not bool(fallback.get("collapse")):
            errors.append("forced_fallback_ambiguous must collapse because fallback dominates")
        if float(fallback.get("fallback_sample_fraction", 0.0)) <= fallback_sample_threshold:
            errors.append("forced_fallback_ambiguous must exceed fallback collapse threshold")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be present")
        return errors
    expected = summarize(rows)
    for key in expected:
        _assert_summary_close(summary, expected, key, errors)
    if summary.get("required_case_ids_present") is not True:
        errors.append("summary must report required_case_ids_present true")
    if summary.get("has_collapse_boundary") is not True:
        errors.append("visibility stress suite must include at least one collapsed stress boundary")
    if summary.get("has_clean_case") is not True:
        errors.append("summary must report has_clean_case true")
    if summary.get("has_ambiguous_case") is not True:
        errors.append("summary must report has_ambiguous_case true")
    if summary.get("has_repaired_crossing_case") is not True:
        errors.append("summary must report has_repaired_crossing_case true")
    return errors


def assert_projective_visibility_stress_suite(report: dict[str, Any]) -> None:
    errors = verify_projective_visibility_stress_suite(report)
    if errors:
        raise AssertionError("projective visibility stress suite failed:\n- " + "\n- ".join(errors))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    columns = (
        "case_id",
        "policy",
        "collapse",
        "collapse_reasons",
        "visibility_stale",
        "order_flip_surface_count",
        "ambiguous_pair_count",
        "depth_interval_overlap_rate",
        "fallback_sample_fraction",
        "visibility_strata_count",
        "quality_error",
        "runtime_ratio",
        "memory_ratio",
    )
    lines = [
        "# STAR UVT Projective Visibility Stress Suite",
        "",
        "This fixture records where a projective interval atlas remains stable, where raw depth order fails, and where fallback dominates.",
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
    assert_projective_visibility_stress_suite(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    markdown_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_projective_visibility_stress_suite(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(frames=args.frames, image_size=args.image_size, tile_size=args.tile_size)
    json_path, markdown_path = write_report(report, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
