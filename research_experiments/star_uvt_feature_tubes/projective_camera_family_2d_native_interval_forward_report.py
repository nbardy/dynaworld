from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    has_projective_trace_family_interval_metal,
    has_projective_trace_cell_interval_metal,
    render_projective_trace_cell_interval_atlas_metal,
    render_projective_trace_family_interval_atlas_metal,
)

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_materialized_batch_report import (  # noqa: E402
    _batched_atlas_from_family,
    _trace_color,
    _trace_opacity,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_chain_rule_report import (  # noqa: E402
    _q_grid,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (  # noqa: E402
    Q2_BASIS_COUNT,
    TRACE_COUNT,
    _family_coeff_table,
    _family_payload_bytes,
    _q2_basis,
    _render_config,
    _slice_payload_bytes,
    _tensor_bytes,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_camera_family_2d_native_interval_forward"
)


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _q_basis_table(q_pairs: list[tuple[float, float]], *, device: torch.device | str) -> torch.Tensor:
    return torch.stack(
        [_q2_basis(q_phase, q_height, device=device) for q_phase, q_height in q_pairs],
        dim=0,
    ).contiguous()


def _base_opacity_time_coeffs(*, device: torch.device | str) -> torch.Tensor:
    return torch.zeros((TRACE_COUNT, 3), dtype=torch.float32, device=device).contiguous()


def _base_spatial_precision_uv(*, device: torch.device | str, sigma_px: float) -> torch.Tensor:
    precision = torch.zeros((TRACE_COUNT, 3), dtype=torch.float32, device=device)
    inv_sigma2 = 1.0 / (float(sigma_px) * float(sigma_px))
    precision[:, 0] = inv_sigma2
    precision[:, 2] = inv_sigma2
    return precision.contiguous()


def _base_depth_affine_uv(*, device: torch.device | str) -> torch.Tensor:
    return torch.zeros((TRACE_COUNT, 6), dtype=torch.float32, device=device).contiguous()


def _family_forward_payload_bytes(
    family_coeffs: torch.Tensor,
    q_basis: torch.Tensor,
    opacity: torch.Tensor,
    opacity_time_coeffs: torch.Tensor,
    spatial_precision_uv: torch.Tensor,
    depth_affine_uv: torch.Tensor,
    color: torch.Tensor,
) -> int:
    return (
        _tensor_bytes(family_coeffs)
        + _tensor_bytes(q_basis)
        + _tensor_bytes(opacity)
        + _tensor_bytes(opacity_time_coeffs)
        + _tensor_bytes(spatial_precision_uv)
        + _tensor_bytes(depth_affine_uv)
        + _tensor_bytes(color)
    )


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    materialized_payload = int(report["materialized_trace_payload_bytes"])
    return {
        "q_axis_count": int(report["q_axis_count"]),
        "q_pair_count": int(report["q_pair_count"]),
        "trace_count": int(report["trace_count"]),
        "family_basis_count": int(report["family_basis_count"]),
        "frames_per_q": int(report["frames_per_q"]),
        "batched_frames": int(report["batched_frames"]),
        "family_coeff_payload_bytes": int(report["family_coeff_payload_bytes"]),
        "q_basis_payload_bytes": int(report["q_basis_payload_bytes"]),
        "family_forward_payload_bytes": int(report["family_forward_payload_bytes"]),
        "materialized_trace_payload_bytes": materialized_payload,
        "family_coeff_to_materialized_trace_payload_ratio": float(report["family_coeff_payload_bytes"])
        / float(materialized_payload),
        "family_forward_to_materialized_trace_payload_ratio": float(report["family_forward_payload_bytes"])
        / float(materialized_payload),
        "native_family_forward_max_abs_error": float(report["native_family_forward_max_abs_error"]),
        "native_family_forward_max_rel_error": float(report["native_family_forward_max_rel_error"]),
        "materialized_image_abs_sum": float(report["materialized_image_abs_sum"]),
        "native_family_image_abs_sum": float(report["native_family_image_abs_sum"]),
    }


def run_report(
    *,
    q_axis_count: int = 5,
    frames: int = 4,
    image_size: int = 8,
    tile_size: int = 8,
    sigma_px: float = 1.7,
) -> dict[str, Any]:
    interval_metal_available = bool(torch.backends.mps.is_available() and has_projective_trace_cell_interval_metal())
    family_interval_metal_available = bool(
        torch.backends.mps.is_available() and has_projective_trace_family_interval_metal()
    )
    if not interval_metal_available or not family_interval_metal_available:
        return {
            "status": "failed",
            "benchmark": "star_uvt_projective_camera_family_2d_native_interval_forward",
            "base_domain": "Q2 x Omega x T native family interval forward",
            "theory_contract": "The Metal interval compositor consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace rendering. This is native forward rendering/compositing/visibility over family traces, not the full backward renderer VJP.",
            "interval_metal_available": interval_metal_available,
            "family_interval_metal_available": family_interval_metal_available,
            "metal_ran": False,
            "errors": ["MPS interval Metal and native family interval forward ops are required."],
            "rows": [],
            "summary": {},
        }

    device = torch.device("mps")
    q_pairs = _q_grid(int(q_axis_count))
    family_coeffs = _family_coeff_table(device=device).contiguous()
    q_basis = _q_basis_table(q_pairs, device=device)
    config = _render_config(
        frames=int(frames) * len(q_pairs),
        image_size=int(image_size),
        tile_size=int(tile_size),
    )
    times = torch.linspace(0.0, 1.0, int(frames), dtype=torch.float32, device=device).repeat(len(q_pairs))
    times = times.contiguous()
    materialized_atlas = _batched_atlas_from_family(
        family_coeffs,
        q_pairs,
        frames_per_q=int(frames),
        device=device,
    )
    materialized_image = render_projective_trace_cell_interval_atlas_metal(
        materialized_atlas,
        times,
        config,
        sigma_px=float(sigma_px),
    )
    opacity = _trace_opacity(device=device).contiguous()
    opacity_time_coeffs = _base_opacity_time_coeffs(device=device)
    spatial_precision_uv = _base_spatial_precision_uv(device=device, sigma_px=float(sigma_px))
    depth_affine_uv = _base_depth_affine_uv(device=device)
    color = _trace_color(device=device).contiguous()
    native_family_image = render_projective_trace_family_interval_atlas_metal(
        materialized_atlas.cells,
        family_coeffs,
        q_basis,
        times,
        opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        depth_affine_uv,
        color,
        config,
        sigma_px=float(sigma_px),
    )
    torch.mps.synchronize()

    delta = native_family_image - materialized_image
    scale = torch.maximum(native_family_image.abs(), materialized_image.abs()).amax().clamp_min(1.0e-6)
    materialized_payload = _slice_payload_bytes(materialized_atlas)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_native_interval_forward",
        "base_domain": "Q2 x Omega x T native family interval forward",
        "theory_contract": "The Metal interval compositor consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace rendering. This is native forward rendering/compositing/visibility over family traces, not the full backward renderer VJP.",
        "interval_metal_available": interval_metal_available,
        "family_interval_metal_available": family_interval_metal_available,
        "metal_ran": True,
        "q_axis_count": int(q_axis_count),
        "q_pair_count": len(q_pairs),
        "frames_per_q": int(frames),
        "batched_frames": int(frames) * len(q_pairs),
        "image_size": int(image_size),
        "trace_count": TRACE_COUNT,
        "family_basis_count": Q2_BASIS_COUNT,
        "family_coeff_payload_bytes": _tensor_bytes(family_coeffs),
        "q_basis_payload_bytes": _tensor_bytes(q_basis),
        "family_static_payload_bytes": _family_payload_bytes(family_coeffs),
        "family_forward_payload_bytes": _family_forward_payload_bytes(
            family_coeffs,
            q_basis,
            opacity,
            opacity_time_coeffs,
            spatial_precision_uv,
            depth_affine_uv,
            color,
        ),
        "materialized_trace_payload_bytes": int(materialized_payload),
        "native_family_forward_max_abs_error": float(delta.abs().amax().detach().cpu().item()),
        "native_family_forward_max_rel_error": float((delta.abs().amax() / scale).detach().cpu().item()),
        "materialized_image_abs_sum": float(materialized_image.abs().sum().detach().cpu().item()),
        "native_family_image_abs_sum": float(native_family_image.abs().sum().detach().cpu().item()),
        "rows": [
            {
                "q_phase": float(q_phase),
                "q_height": float(q_height),
                "basis_abs_sum": float(q_basis[row_index].abs().sum().detach().cpu().item()),
                "native_image_abs_sum": float(
                    native_family_image[
                        row_index * int(frames) : (row_index + 1) * int(frames)
                    ]
                    .abs()
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                ),
            }
            for row_index, (q_phase, q_height) in enumerate(q_pairs)
        ],
    }
    report["summary"] = summarize(report)
    errors = verify_camera_family_2d_native_interval_forward_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _assert_summary_close(actual: Any, expected: Any, key: str, errors: list[str]) -> None:
    if isinstance(expected, float):
        if not _finite_float(actual) or abs(float(actual) - expected) > 1.0e-8:
            errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")


def verify_camera_family_2d_native_interval_forward_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_native_interval_forward":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T native family interval forward":
        errors.append(f"base_domain must name native family interval forward, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "pi_* Gamma^*" not in theory_contract
        or "rendering/compositing/visibility" not in theory_contract
        or "not the full backward" not in theory_contract
    ):
        errors.append("theory_contract must preserve the native-forward-but-not-full-backward contract")
    if report.get("metal_ran") is not True:
        errors.append("metal_ran must be true for this evidence artifact")
    if report.get("interval_metal_available") is not True or report.get("family_interval_metal_available") is not True:
        errors.append("interval and family interval Metal availability must both be true")

    q_axis_count = report.get("q_axis_count")
    q_pair_count = report.get("q_pair_count")
    if not isinstance(q_axis_count, int) or q_axis_count < 5:
        errors.append(f"q_axis_count must be an int >= 5, got {q_axis_count!r}")
        q_axis_count = 0
    if not isinstance(q_pair_count, int) or q_pair_count != int(q_axis_count) * int(q_axis_count):
        errors.append(f"q_pair_count must equal q_axis_count^2, got {q_pair_count!r}")
        q_pair_count = 0
    if report.get("trace_count") != TRACE_COUNT:
        errors.append(f"trace_count must be {TRACE_COUNT}, got {report.get('trace_count')!r}")
    if report.get("family_basis_count") != Q2_BASIS_COUNT:
        errors.append(f"family_basis_count must be {Q2_BASIS_COUNT}, got {report.get('family_basis_count')!r}")

    rows = report.get("rows")
    if not isinstance(rows, list):
        errors.append("rows must be a list")
        rows = []
    elif len(rows) != int(q_pair_count):
        errors.append(f"rows must contain one native forward row per q-pair, expected {q_pair_count}, got {len(rows)}")
    row_abs_sum = 0.0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {idx} must be an object")
            continue
        for key in ("q_phase", "q_height", "basis_abs_sum", "native_image_abs_sum"):
            if not _finite_float(row.get(key)):
                errors.append(f"row {idx} {key} must be finite")
        if _finite_float(row.get("native_image_abs_sum")):
            row_abs_sum += float(row["native_image_abs_sum"])

    required_float_keys = (
        "family_coeff_payload_bytes",
        "q_basis_payload_bytes",
        "family_forward_payload_bytes",
        "materialized_trace_payload_bytes",
        "native_family_forward_max_abs_error",
        "native_family_forward_max_rel_error",
        "materialized_image_abs_sum",
        "native_family_image_abs_sum",
    )
    for key in required_float_keys:
        if not _finite_float(report.get(key)):
            errors.append(f"{key} must be finite")
    if _finite_float(report.get("family_forward_payload_bytes")) and _finite_float(
        report.get("materialized_trace_payload_bytes")
    ):
        materialized_payload = float(report["materialized_trace_payload_bytes"])
        if materialized_payload <= 0.0:
            errors.append("materialized_trace_payload_bytes must be positive")
        else:
            ratio = float(report["family_forward_payload_bytes"]) / materialized_payload
            if ratio >= 0.50:
                errors.append(f"family forward/materialized trace payload ratio regressed: {ratio:.6g} >= 0.50")
    thresholds = {
        "native_family_forward_max_abs_error": 1.0e-6,
        "native_family_forward_max_rel_error": 1.0e-6,
    }
    for key, threshold in thresholds.items():
        if _finite_float(report.get(key)) and float(report[key]) > threshold:
            errors.append(f"{key}={float(report[key]):.6g} exceeds {threshold:.6g}")
    if _finite_float(report.get("native_family_image_abs_sum")) and float(report["native_family_image_abs_sum"]) <= 0.0:
        errors.append("native_family_image_abs_sum must be positive")
    if _finite_float(report.get("native_family_image_abs_sum")) and abs(
        row_abs_sum - float(report["native_family_image_abs_sum"])
    ) > 1.0e-4:
        errors.append("row native_image_abs_sum values must sum to native_family_image_abs_sum")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        expected = summarize(report)
        for key, value in expected.items():
            _assert_summary_close(summary.get(key), value, key, errors)
    return errors


def assert_camera_family_2d_native_interval_forward_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_native_interval_forward_report(report)
    if errors:
        raise AssertionError("\n".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = report.get("summary", {})
    lines = [
        "# STAR UVT Q2 Native Family Interval Forward",
        "",
        f"status: `{report.get('status')}`",
        f"q pairs: `{summary.get('q_pair_count')}`",
        f"family forward/materialized payload ratio: `{summary.get('family_forward_to_materialized_trace_payload_ratio')}`",
        f"max abs error: `{summary.get('native_family_forward_max_abs_error')}`",
        f"max rel error: `{summary.get('native_family_forward_max_rel_error')}`",
        "",
        "This proves native forward interval rendering/compositing/visibility over shared family coefficients.",
        "It does not prove the matching full native backward/VJP path.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_camera_family_2d_native_interval_forward_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report()
    write_report(report, args.out_dir)
    assert_camera_family_2d_native_interval_forward_report(report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
