from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    has_projective_trace_cell_interval_backward_metal,
    lower_projective_trace_cell_atlas_quadrature,
    lower_projective_trace_cell_atlas_rolling_quadrature,
    projective_trace_cell_sensor_time_event_partition,
    projective_trace_cell_sensor_time_partition_quadrature,
    projective_trace_cell_sensor_time_partition_rolling_quadrature,
    render_projective_trace_cell_atlas_reference,
)
from research_experiments.star_uvt_feature_tubes.projective_exposure_rolling_quadrature_report import (  # noqa: E402
    _atlas_to_mps,
    _config,
    _direct_continuous_cell_atlas,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_exposure_rolling_backward"
)


def _grad_image(*, height: int, width: int, channels: int = 3) -> torch.Tensor:
    values = torch.linspace(-0.35, 0.42, steps=height * width * channels, dtype=torch.float32)
    return values.reshape(height, width, channels).contiguous()


def _reference_grads(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    sample_adjoint: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    sigma_px: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    ref_coeffs = atlas.coeffs.clone().detach().requires_grad_(True)
    ref_colors = atlas.color.clone().detach().requires_grad_(True)
    ref_opacities = atlas.opacity.clone().detach().requires_grad_(True)
    ref_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=ref_coeffs,
        opacity=ref_opacities,
        color=ref_colors,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        opacity_time_coeffs=atlas.opacity_time_coeffs,
        spatial_precision_uv=atlas.spatial_precision_uv,
        depth_affine_uv=atlas.depth_affine_uv,
    )
    samples = render_projective_trace_cell_atlas_reference(
        ref_atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        alpha_cutoff=1.0e-6,
        transmittance_cutoff=0.0,
    )
    (samples * sample_adjoint).sum().backward()
    assert ref_coeffs.grad is not None
    assert ref_opacities.grad is not None
    assert ref_colors.grad is not None
    return (
        {
            "coeffs": ref_coeffs.grad.detach(),
            "opacity": ref_opacities.grad.detach(),
            "color": ref_colors.grad.detach(),
        },
        samples.detach(),
    )


def _max_rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.abs().max().clamp_min(1.0e-8)
    return float(((actual - expected).abs().max() / denom).item())


def _grad_compare_row(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    max_abs = 0.0
    max_rel = 0.0
    for key in ("coeffs", "opacity", "color"):
        abs_error = float((actual[key] - expected[key]).abs().max().item())
        rel_error = _max_rel_error(actual[key], expected[key])
        grad_norm = float(expected[key].norm().item())
        rows[key] = {
            "max_abs_error": abs_error,
            "max_rel_error": rel_error,
            "reference_grad_norm": grad_norm,
        }
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
    rows["max_abs_error"] = max_abs
    rows["max_rel_error"] = max_rel
    return rows


def _finite_backward_case(*, run_metal: bool) -> dict[str, Any]:
    atlas = _direct_continuous_cell_atlas()
    width = 8
    height = 8
    tile_size = 8
    sigma_px = 1.7
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        torch.arange(4, dtype=torch.float32).contiguous(),
        image_width=width,
        image_height=height,
        tile_size=tile_size,
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
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        uv_padding=4.0,
    )
    final_adjoint = _grad_image(height=height, width=width)
    sample_adjoint = lowering.weights.reshape(-1, 1, 1, 1) * final_adjoint.reshape(1, height, width, 3)
    reference, samples = _reference_grads(
        lowering.atlas,
        lowering.times,
        sample_adjoint,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
    )
    metal_compare: dict[str, Any] | None = None
    if run_metal and torch.backends.mps.is_available() and has_projective_trace_cell_interval_backward_metal():
        sample_config = replace(_config(width=width, height=height, tile_size=tile_size), frames=int(lowering.times.numel()))
        grads = direct_backward_projective_trace_cell_interval_atlas_metal(
            _atlas_to_mps(lowering.atlas),
            lowering.times.to("mps"),
            sample_adjoint.to("mps").contiguous(),
            sample_config,
            sigma_px=sigma_px,
        )
        metal_compare = _grad_compare_row(
            {
                "coeffs": grads.grad_coeffs.cpu(),
                "opacity": grads.grad_opacity.cpu(),
                "color": grads.grad_color.cpu(),
            },
            reference,
        )
    return {
        "quadrature_sample_count": len(quadrature.samples),
        "lowered_unique_sample_count": int(lowering.times.numel()),
        "weight_sum": float(lowering.weights.sum().item()),
        "sample_adjoint_abs_sum": float(sample_adjoint.abs().sum().item()),
        "sample_image_abs_sum": float(samples.abs().sum().item()),
        "reference_grad_norms": {key: float(value.norm().item()) for key, value in reference.items()},
        "metal_compare": metal_compare,
    }


def _rolling_backward_case(*, run_metal: bool) -> dict[str, Any]:
    atlas = _direct_continuous_cell_atlas()
    width = 8
    height = 3
    tile_size = 8
    sigma_px = 1.7
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        torch.arange(4, dtype=torch.float32).contiguous(),
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    row_quadrature = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=height,
        frame_time=0.0,
        exposure_duration=0.75,
        readout_duration=1.0,
        samples_per_interval=2,
    )
    lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
        atlas,
        row_quadrature,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        uv_padding=4.0,
    )
    final_adjoint = _grad_image(height=height, width=width)
    sample_adjoint = lowering.row_weights.reshape(-1, height, 1, 1) * final_adjoint.reshape(1, height, width, 3)
    reference, samples = _reference_grads(
        lowering.atlas,
        lowering.times,
        sample_adjoint,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
    )
    total_row_samples = sum(len(quadrature.samples) for quadrature in row_quadrature)
    metal_compare: dict[str, Any] | None = None
    if run_metal and torch.backends.mps.is_available() and has_projective_trace_cell_interval_backward_metal():
        sample_config = replace(_config(width=width, height=height, tile_size=tile_size), frames=int(lowering.times.numel()))
        grads = direct_backward_projective_trace_cell_interval_atlas_metal(
            _atlas_to_mps(lowering.atlas),
            lowering.times.to("mps"),
            sample_adjoint.to("mps").contiguous(),
            sample_config,
            sigma_px=sigma_px,
        )
        metal_compare = _grad_compare_row(
            {
                "coeffs": grads.grad_coeffs.cpu(),
                "opacity": grads.grad_opacity.cpu(),
                "color": grads.grad_color.cpu(),
            },
            reference,
        )
    return {
        "row_count": height,
        "total_row_sample_count": int(total_row_samples),
        "lowered_unique_sample_count": int(lowering.times.numel()),
        "unique_to_row_sample_ratio": float(lowering.times.numel()) / float(max(total_row_samples, 1)),
        "row_weight_sums": [float(value) for value in lowering.row_weights.sum(dim=0).detach().cpu().tolist()],
        "sample_adjoint_abs_sum": float(sample_adjoint.abs().sum().item()),
        "sample_image_abs_sum": float(samples.abs().sum().item()),
        "reference_grad_norms": {key: float(value.norm().item()) for key, value in reference.items()},
        "metal_compare": metal_compare,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    finite_compare = report["finite_exposure_backward"].get("metal_compare")
    rolling_compare = report["rolling_shutter_backward"].get("metal_compare")
    compare_rows = [row for row in (finite_compare, rolling_compare) if row is not None]
    return {
        "finite_has_metal_backward": finite_compare is not None,
        "rolling_has_metal_backward": rolling_compare is not None,
        "rolling_unique_to_row_sample_ratio": report["rolling_shutter_backward"]["unique_to_row_sample_ratio"],
        "max_metal_grad_abs_error": max((row["max_abs_error"] for row in compare_rows), default=None),
        "max_metal_grad_rel_error": max((row["max_rel_error"] for row in compare_rows), default=None),
        "metal_backward_case_count": len(compare_rows),
    }


def run_report(*, run_metal: bool | None = None) -> dict[str, Any]:
    should_run_metal = bool(torch.backends.mps.is_available()) if run_metal is None else bool(run_metal)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_exposure_rolling_backward",
        "theory_contract": (
            "Finite-exposure and rolling-shutter backward passes reuse the lowered interval atlas. "
            "Final-image adjoints are pushed to sample adjoints by quadrature weights or row_weights, "
            "then one interval-cell VJP accumulates trace gradients."
        ),
        "device": {
            "mps_available": bool(torch.backends.mps.is_available()),
            "interval_backward_metal_available": bool(has_projective_trace_cell_interval_backward_metal()),
            "requested_metal": should_run_metal,
        },
        "finite_exposure_backward": _finite_backward_case(run_metal=should_run_metal),
        "rolling_shutter_backward": _rolling_backward_case(run_metal=should_run_metal),
    }
    report["summary"] = summarize(report)
    errors = verify_exposure_rolling_backward_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{label} must be a nonnegative integer, got {value!r}")
        return 0
    return int(value)


def _require_close(value: Any, expected: float, label: str, errors: list[str], *, atol: float) -> None:
    actual = _finite_float(value, label, errors)
    if abs(actual - float(expected)) > float(atol):
        errors.append(f"{label} must be within {atol:g} of {expected:g}, got {actual:g}")


def _verify_grad_norms(row: Any, label: str, errors: list[str]) -> None:
    if not isinstance(row, dict):
        errors.append(f"{label} reference_grad_norms must be an object")
        return
    for key in ("coeffs", "opacity", "color"):
        norm = _finite_float(row.get(key), f"{label} {key} grad norm", errors)
        if norm <= 0.0:
            errors.append(f"{label} {key} grad norm must be positive")


def _verify_compare(row: Any, label: str, errors: list[str]) -> None:
    if row is None:
        return
    if not isinstance(row, dict):
        errors.append(f"{label} metal_compare must be an object or null")
        return
    max_abs = _finite_float(row.get("max_abs_error"), f"{label} max_abs_error", errors)
    max_rel = _finite_float(row.get("max_rel_error"), f"{label} max_rel_error", errors)
    if max_abs > 1.0e-3:
        errors.append(f"{label} max_abs_error must be <= 1e-3, got {max_abs}")
    if max_rel > 5.0e-3:
        errors.append(f"{label} max_rel_error must be <= 5e-3, got {max_rel}")
    sub_abs_errors: list[float] = []
    sub_rel_errors: list[float] = []
    for key in ("coeffs", "opacity", "color"):
        subrow = row.get(key)
        if not isinstance(subrow, dict):
            errors.append(f"{label} {key} compare row must be an object")
            continue
        sub_abs_errors.append(_finite_float(subrow.get("max_abs_error"), f"{label} {key} max_abs_error", errors))
        sub_rel_errors.append(_finite_float(subrow.get("max_rel_error"), f"{label} {key} max_rel_error", errors))
        grad_norm = _finite_float(subrow.get("reference_grad_norm"), f"{label} {key} reference_grad_norm", errors)
        if grad_norm <= 0.0:
            errors.append(f"{label} {key} reference_grad_norm must be positive")
    if sub_abs_errors:
        _require_close(max_abs, max(sub_abs_errors), f"{label} max_abs_error", errors, atol=1.0e-12)
    if sub_rel_errors:
        _require_close(max_rel, max(sub_rel_errors), f"{label} max_rel_error", errors, atol=1.0e-12)


def _verify_summary(report: dict[str, Any], summary: dict[str, Any], errors: list[str]) -> None:
    expected = summarize(report)
    for key, expected_value in expected.items():
        actual_value = summary.get(key)
        if isinstance(expected_value, bool):
            if actual_value is not expected_value:
                errors.append(f"summary {key} must be {expected_value}, got {actual_value!r}")
        elif expected_value is None:
            if actual_value is not None:
                errors.append(f"summary {key} must be None, got {actual_value!r}")
        elif isinstance(expected_value, int):
            if actual_value != expected_value:
                errors.append(f"summary {key} must be {expected_value}, got {actual_value!r}")
        else:
            _require_close(actual_value, float(expected_value), f"summary {key}", errors, atol=1.0e-12)


def verify_exposure_rolling_backward_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_exposure_rolling_backward":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if not isinstance(report.get("theory_contract"), str) or "sample adjoints" not in report.get("theory_contract", ""):
        errors.append("theory_contract must state the sample-adjoint backward contract")
    if not isinstance(report.get("device"), dict):
        errors.append("device must be an object")
    for key in ("finite_exposure_backward", "rolling_shutter_backward", "summary"):
        if not isinstance(report.get(key), dict):
            errors.append(f"{key} must be an object")
    if errors:
        return errors

    finite = report["finite_exposure_backward"]
    rolling = report["rolling_shutter_backward"]
    summary = report["summary"]
    device = report["device"]
    for key in ("mps_available", "interval_backward_metal_available", "requested_metal"):
        if not isinstance(device.get(key), bool):
            errors.append(f"device {key} must be boolean")

    finite_samples = _finite_int(finite.get("quadrature_sample_count"), "finite quadrature_sample_count", errors)
    finite_unique_samples = _finite_int(
        finite.get("lowered_unique_sample_count"),
        "finite lowered_unique_sample_count",
        errors,
    )
    if finite_samples <= 0:
        errors.append("finite quadrature_sample_count must be positive")
    if finite_samples != finite_unique_samples:
        errors.append("finite backward must keep one lowered sample per quadrature sample")
    weight_sum = _finite_float(finite.get("weight_sum"), "finite weight_sum", errors)
    if abs(weight_sum - 1.0) > 1.0e-6:
        errors.append(f"finite weight_sum must equal 1, got {weight_sum}")
    if _finite_float(finite.get("sample_adjoint_abs_sum"), "finite sample_adjoint_abs_sum", errors) <= 0.0:
        errors.append("finite sample adjoint must be nonzero")
    if _finite_float(finite.get("sample_image_abs_sum"), "finite sample_image_abs_sum", errors) <= 0.0:
        errors.append("finite sample image must be nonzero")
    _verify_grad_norms(finite.get("reference_grad_norms"), "finite", errors)
    _verify_compare(finite.get("metal_compare"), "finite", errors)

    row_count = _finite_int(rolling.get("row_count"), "rolling row_count", errors)
    total_row_samples = _finite_int(rolling.get("total_row_sample_count"), "rolling total_row_sample_count", errors)
    unique_row_samples = _finite_int(
        rolling.get("lowered_unique_sample_count"),
        "rolling lowered_unique_sample_count",
        errors,
    )
    if row_count <= 0:
        errors.append("rolling row_count must be positive")
    if not 0 < unique_row_samples < total_row_samples:
        errors.append("rolling backward must deduplicate row quadrature samples")
    ratio = _finite_float(rolling.get("unique_to_row_sample_ratio"), "rolling unique_to_row_sample_ratio", errors)
    if not 0.0 < ratio < 1.0:
        errors.append(f"rolling unique_to_row_sample_ratio must be in (0,1), got {ratio}")
    elif total_row_samples > 0:
        _require_close(
            ratio,
            float(unique_row_samples) / float(total_row_samples),
            "rolling unique_to_row_sample_ratio",
            errors,
            atol=1.0e-12,
        )
    row_weight_sums = rolling.get("row_weight_sums")
    if not isinstance(row_weight_sums, list) or len(row_weight_sums) != row_count:
        errors.append("rolling row_weight_sums must contain one entry per row")
    else:
        for row_index, value in enumerate(row_weight_sums):
            row_sum = _finite_float(value, f"rolling row {row_index} weight sum", errors)
            if abs(row_sum - 1.0) > 1.0e-6:
                errors.append(f"rolling row {row_index} weight sum must equal 1, got {row_sum}")
    if _finite_float(rolling.get("sample_adjoint_abs_sum"), "rolling sample_adjoint_abs_sum", errors) <= 0.0:
        errors.append("rolling sample adjoint must be nonzero")
    if _finite_float(rolling.get("sample_image_abs_sum"), "rolling sample_image_abs_sum", errors) <= 0.0:
        errors.append("rolling sample image must be nonzero")
    _verify_grad_norms(rolling.get("reference_grad_norms"), "rolling", errors)
    _verify_compare(rolling.get("metal_compare"), "rolling", errors)

    _verify_summary(report, summary, errors)
    return errors


def assert_exposure_rolling_backward_report(report: dict[str, Any]) -> None:
    errors = verify_exposure_rolling_backward_report(report)
    if errors:
        raise AssertionError("exposure/rolling backward report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Exposure And Rolling Backward",
        "",
        "This report pins the adjoint contract for finite exposure and rolling shutter:",
        "",
        "```text",
        "dL/d sample_image[q,row] = weight[q,row] * dL/d final_image[row]",
        "```",
        "",
        "Then one shared interval-cell VJP accumulates trace gradients.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Device",
        "",
        "```json",
        json.dumps(report["device"], indent=2, sort_keys=True),
        "```",
        "",
        "## Cases",
        "",
        "```json",
        json.dumps(
            {
                "finite_exposure_backward": report["finite_exposure_backward"],
                "rolling_shutter_backward": report["rolling_shutter_backward"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, default=None)
    parser.add_argument("--skip-metal", action="store_true")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_exposure_rolling_backward_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(run_metal=not args.skip_metal)
    assert_exposure_rolling_backward_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
