from __future__ import annotations

import argparse
from dataclasses import replace
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
    direct_backward_projective_trace_cell_interval_atlas_metal,
    direct_backward_projective_trace_family_interval_atlas_metal,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_family_interval_backward_metal,
)

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_materialized_batch_report import (  # noqa: E402
    _batched_atlas_from_family,
    _trace_color,
    _trace_opacity,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_chain_rule_report import (  # noqa: E402
    _grad_image,
    _q_grid,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (  # noqa: E402
    Q2_BASIS_COUNT,
    TRACE_COUNT,
    _family_coeff_table,
    _q2_basis,
    _render_config,
    _slice_payload_bytes,
    _tensor_bytes,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_native_interval_forward_report import (  # noqa: E402
    _base_depth_affine_uv,
    _base_opacity_time_coeffs,
    _base_spatial_precision_uv,
    _family_forward_payload_bytes,
    _q_basis_table,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_camera_family_2d_native_interval_backward"
)


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _relative_error(delta: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    scale = torch.maximum(lhs.abs(), rhs.abs()).amax().clamp_min(1.0e-6)
    return float((delta.abs().amax() / scale).detach().cpu().item())


def _gradient_payload_bytes(
    grad_family_coeffs: torch.Tensor,
    grad_q_basis: torch.Tensor,
    grad_opacity: torch.Tensor,
    grad_opacity_time_coeffs: torch.Tensor,
    grad_spatial_precision_uv: torch.Tensor,
    grad_color: torch.Tensor,
) -> int:
    return (
        _tensor_bytes(grad_family_coeffs)
        + _tensor_bytes(grad_q_basis)
        + _tensor_bytes(grad_opacity)
        + _tensor_bytes(grad_opacity_time_coeffs)
        + _tensor_bytes(grad_spatial_precision_uv)
        + _tensor_bytes(grad_color)
    )


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    materialized_gradient_payload = int(report["materialized_gradient_payload_bytes"])
    return {
        "q_axis_count": int(report["q_axis_count"]),
        "q_pair_count": int(report["q_pair_count"]),
        "trace_count": int(report["trace_count"]),
        "family_basis_count": int(report["family_basis_count"]),
        "frames_per_q": int(report["frames_per_q"]),
        "batched_frames": int(report["batched_frames"]),
        "family_forward_payload_bytes": int(report["family_forward_payload_bytes"]),
        "native_family_gradient_payload_bytes": int(report["native_family_gradient_payload_bytes"]),
        "native_family_coeff_gradient_payload_bytes": int(report["native_family_coeff_gradient_payload_bytes"]),
        "native_q_basis_gradient_payload_bytes": int(report["native_q_basis_gradient_payload_bytes"]),
        "materialized_gradient_payload_bytes": materialized_gradient_payload,
        "native_family_gradient_to_materialized_gradient_payload_ratio": float(
            report["native_family_gradient_payload_bytes"]
        )
        / float(materialized_gradient_payload),
        "native_family_coeff_gradient_to_materialized_gradient_payload_ratio": float(
            report["native_family_coeff_gradient_payload_bytes"]
        )
        / float(materialized_gradient_payload),
        "native_family_interval_backward_max_family_grad_rel_error": float(
            report["native_family_interval_backward_max_family_grad_rel_error"]
        ),
        "native_family_interval_backward_max_q_basis_grad_rel_error": float(
            report["native_family_interval_backward_max_q_basis_grad_rel_error"]
        ),
        "native_family_interval_backward_max_opacity_grad_rel_error": float(
            report["native_family_interval_backward_max_opacity_grad_rel_error"]
        ),
        "native_family_interval_backward_max_color_grad_rel_error": float(
            report["native_family_interval_backward_max_color_grad_rel_error"]
        ),
        "native_family_interval_backward_max_opacity_time_grad_rel_error": float(
            report["native_family_interval_backward_max_opacity_time_grad_rel_error"]
        ),
        "native_family_interval_backward_max_spatial_precision_grad_rel_error": float(
            report["native_family_interval_backward_max_spatial_precision_grad_rel_error"]
        ),
        "native_family_grad_abs_sum": float(report["native_family_grad_abs_sum"]),
        "native_q_basis_grad_abs_sum": float(report["native_q_basis_grad_abs_sum"]),
    }


def run_report(
    *,
    q_axis_count: int = 5,
    frames: int = 4,
    image_size: int = 8,
    tile_size: int = 8,
    sigma_px: float = 1.7,
) -> dict[str, Any]:
    interval_backward_available = bool(
        torch.backends.mps.is_available() and has_projective_trace_cell_interval_backward_metal()
    )
    family_interval_backward_available = bool(
        torch.backends.mps.is_available() and has_projective_trace_family_interval_backward_metal()
    )
    if not interval_backward_available or not family_interval_backward_available:
        return {
            "status": "failed",
            "benchmark": "star_uvt_projective_camera_family_2d_native_interval_backward",
            "base_domain": "Q2 x Omega x T native family interval backward",
            "theory_contract": "The Metal interval VJP consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace rendering. This is native backward/VJP over family traces with compiled visibility held fixed.",
            "interval_backward_available": interval_backward_available,
            "family_interval_backward_available": family_interval_backward_available,
            "metal_ran": False,
            "errors": ["MPS interval backward and native family interval backward ops are required."],
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
    opacity = _trace_opacity(device=device).contiguous()
    opacity_time_coeffs = _base_opacity_time_coeffs(device=device)
    spatial_precision_uv = _base_spatial_precision_uv(device=device, sigma_px=float(sigma_px))
    depth_affine_uv = _base_depth_affine_uv(device=device)
    color = _trace_color(device=device).contiguous()
    materialized_atlas = replace(
        materialized_atlas,
        opacity_time_coeffs=opacity_time_coeffs.repeat((len(q_pairs), 1)).contiguous(),
        spatial_precision_uv=spatial_precision_uv.repeat((len(q_pairs), 1)).contiguous(),
        depth_affine_uv=depth_affine_uv.repeat((len(q_pairs), 1)).contiguous(),
    )
    per_q_config = _render_config(frames=int(frames), image_size=int(image_size), tile_size=int(tile_size))
    grad_image = torch.cat(
        [_grad_image(per_q_config, device=device, row_index=row_index) for row_index in range(len(q_pairs))],
        dim=0,
    ).contiguous()

    materialized_grads = direct_backward_projective_trace_cell_interval_atlas_metal(
        materialized_atlas,
        times,
        grad_image,
        config,
        sigma_px=float(sigma_px),
    )
    native_grads = direct_backward_projective_trace_family_interval_atlas_metal(
        materialized_atlas.cells,
        family_coeffs,
        q_basis,
        times,
        opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        depth_affine_uv,
        color,
        grad_image,
        config,
        sigma_px=float(sigma_px),
    )
    torch.mps.synchronize()

    q_count = len(q_pairs)
    trace_count = int(family_coeffs.shape[0])
    materialized_coeff_grads = materialized_grads.grad_coeffs.reshape(q_count, trace_count, 9)
    ref_family_grad = torch.einsum("qnk,qb->nkb", materialized_coeff_grads, q_basis)
    ref_q_basis_grad = torch.einsum("qnk,nkb->qb", materialized_coeff_grads, family_coeffs)
    ref_opacity_grad = materialized_grads.grad_opacity.reshape(q_count, trace_count).sum(dim=0)
    ref_color_grad = materialized_grads.grad_color.reshape(q_count, trace_count, 3).sum(dim=0)
    ref_opacity_time_grad = materialized_grads.grad_opacity_time_coeffs.reshape(q_count, trace_count, 3).sum(dim=0)
    ref_spatial_grad = materialized_grads.grad_spatial_precision_uv.reshape(q_count, trace_count, 3).sum(dim=0)

    family_delta = native_grads.grad_family_coeffs - ref_family_grad
    q_delta = native_grads.grad_q_basis - ref_q_basis_grad
    opacity_delta = native_grads.grad_opacity - ref_opacity_grad
    color_delta = native_grads.grad_color - ref_color_grad
    opacity_time_delta = native_grads.grad_opacity_time_coeffs - ref_opacity_time_grad
    spatial_delta = native_grads.grad_spatial_precision_uv - ref_spatial_grad

    materialized_gradient_payload = _gradient_payload_bytes(
        materialized_grads.grad_coeffs,
        torch.empty((0,), dtype=torch.float32, device=device),
        materialized_grads.grad_opacity,
        materialized_grads.grad_opacity_time_coeffs,
        materialized_grads.grad_spatial_precision_uv,
        materialized_grads.grad_color,
    )
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_native_interval_backward",
        "base_domain": "Q2 x Omega x T native family interval backward",
        "theory_contract": "The Metal interval VJP consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace rendering. This is native backward/VJP over family traces with compiled visibility held fixed.",
        "interval_backward_available": interval_backward_available,
        "family_interval_backward_available": family_interval_backward_available,
        "metal_ran": True,
        "q_axis_count": int(q_axis_count),
        "q_pair_count": q_count,
        "frames_per_q": int(frames),
        "batched_frames": int(frames) * q_count,
        "image_size": int(image_size),
        "trace_count": TRACE_COUNT,
        "family_basis_count": Q2_BASIS_COUNT,
        "family_forward_payload_bytes": _family_forward_payload_bytes(
            family_coeffs,
            q_basis,
            opacity,
            opacity_time_coeffs,
            spatial_precision_uv,
            depth_affine_uv,
            color,
        ),
        "native_family_gradient_payload_bytes": _gradient_payload_bytes(
            native_grads.grad_family_coeffs,
            native_grads.grad_q_basis,
            native_grads.grad_opacity,
            native_grads.grad_opacity_time_coeffs,
            native_grads.grad_spatial_precision_uv,
            native_grads.grad_color,
        ),
        "native_family_coeff_gradient_payload_bytes": _tensor_bytes(native_grads.grad_family_coeffs),
        "native_q_basis_gradient_payload_bytes": _tensor_bytes(native_grads.grad_q_basis),
        "materialized_gradient_payload_bytes": materialized_gradient_payload,
        "materialized_trace_payload_bytes": _slice_payload_bytes(materialized_atlas),
        "native_family_interval_backward_max_family_grad_abs_error": float(
            family_delta.abs().amax().detach().cpu().item()
        ),
        "native_family_interval_backward_max_family_grad_rel_error": _relative_error(
            family_delta,
            native_grads.grad_family_coeffs,
            ref_family_grad,
        ),
        "native_family_interval_backward_max_q_basis_grad_abs_error": float(q_delta.abs().amax().detach().cpu().item()),
        "native_family_interval_backward_max_q_basis_grad_rel_error": _relative_error(
            q_delta,
            native_grads.grad_q_basis,
            ref_q_basis_grad,
        ),
        "native_family_interval_backward_max_opacity_grad_abs_error": float(
            opacity_delta.abs().amax().detach().cpu().item()
        ),
        "native_family_interval_backward_max_opacity_grad_rel_error": _relative_error(
            opacity_delta,
            native_grads.grad_opacity,
            ref_opacity_grad,
        ),
        "native_family_interval_backward_max_color_grad_abs_error": float(color_delta.abs().amax().detach().cpu().item()),
        "native_family_interval_backward_max_color_grad_rel_error": _relative_error(
            color_delta,
            native_grads.grad_color,
            ref_color_grad,
        ),
        "native_family_interval_backward_max_opacity_time_grad_abs_error": float(
            opacity_time_delta.abs().amax().detach().cpu().item()
        ),
        "native_family_interval_backward_max_opacity_time_grad_rel_error": _relative_error(
            opacity_time_delta,
            native_grads.grad_opacity_time_coeffs,
            ref_opacity_time_grad,
        ),
        "native_family_interval_backward_max_spatial_precision_grad_abs_error": float(
            spatial_delta.abs().amax().detach().cpu().item()
        ),
        "native_family_interval_backward_max_spatial_precision_grad_rel_error": _relative_error(
            spatial_delta,
            native_grads.grad_spatial_precision_uv,
            ref_spatial_grad,
        ),
        "native_family_grad_abs_sum": float(native_grads.grad_family_coeffs.abs().sum().detach().cpu().item()),
        "native_q_basis_grad_abs_sum": float(native_grads.grad_q_basis.abs().sum().detach().cpu().item()),
        "native_opacity_grad_abs_sum": float(native_grads.grad_opacity.abs().sum().detach().cpu().item()),
        "native_color_grad_abs_sum": float(native_grads.grad_color.abs().sum().detach().cpu().item()),
        "rows": [
            {
                "q_phase": float(q_phase),
                "q_height": float(q_height),
                "basis_abs_sum": float(_q2_basis(q_phase, q_height, device=device).abs().sum().detach().cpu().item()),
                "native_q_basis_grad_abs_sum": float(
                    native_grads.grad_q_basis[row_index].abs().sum().detach().cpu().item()
                ),
            }
            for row_index, (q_phase, q_height) in enumerate(q_pairs)
        ],
    }
    report["summary"] = summarize(report)
    errors = verify_camera_family_2d_native_interval_backward_report(report)
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


def verify_camera_family_2d_native_interval_backward_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_native_interval_backward":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T native family interval backward":
        errors.append(f"base_domain must name native family interval backward, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "pi_* Gamma^*" not in theory_contract
        or "native backward/VJP" not in theory_contract
        or "compiled visibility held fixed" not in theory_contract
    ):
        errors.append("theory_contract must preserve the native-backward compiled-visibility contract")
    if report.get("metal_ran") is not True:
        errors.append("metal_ran must be true for this evidence artifact")
    if (
        report.get("interval_backward_available") is not True
        or report.get("family_interval_backward_available") is not True
    ):
        errors.append("interval and family interval backward Metal availability must both be true")

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
        errors.append(f"rows must contain one native backward row per q-pair, expected {q_pair_count}, got {len(rows)}")
    row_grad_sum = 0.0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {idx} must be an object")
            continue
        for key in ("q_phase", "q_height", "basis_abs_sum", "native_q_basis_grad_abs_sum"):
            if not _finite_float(row.get(key)):
                errors.append(f"row {idx} {key} must be finite")
        if _finite_float(row.get("native_q_basis_grad_abs_sum")):
            row_grad_sum += float(row["native_q_basis_grad_abs_sum"])

    required_float_keys = (
        "family_forward_payload_bytes",
        "native_family_gradient_payload_bytes",
        "native_family_coeff_gradient_payload_bytes",
        "native_q_basis_gradient_payload_bytes",
        "materialized_gradient_payload_bytes",
        "native_family_interval_backward_max_family_grad_abs_error",
        "native_family_interval_backward_max_family_grad_rel_error",
        "native_family_interval_backward_max_q_basis_grad_abs_error",
        "native_family_interval_backward_max_q_basis_grad_rel_error",
        "native_family_interval_backward_max_opacity_grad_abs_error",
        "native_family_interval_backward_max_opacity_grad_rel_error",
        "native_family_interval_backward_max_color_grad_abs_error",
        "native_family_interval_backward_max_color_grad_rel_error",
        "native_family_interval_backward_max_opacity_time_grad_abs_error",
        "native_family_interval_backward_max_opacity_time_grad_rel_error",
        "native_family_interval_backward_max_spatial_precision_grad_abs_error",
        "native_family_interval_backward_max_spatial_precision_grad_rel_error",
        "native_family_grad_abs_sum",
        "native_q_basis_grad_abs_sum",
        "native_opacity_grad_abs_sum",
        "native_color_grad_abs_sum",
    )
    for key in required_float_keys:
        if not _finite_float(report.get(key)):
            errors.append(f"{key} must be finite")
    if _finite_float(report.get("native_family_gradient_payload_bytes")) and _finite_float(
        report.get("materialized_gradient_payload_bytes")
    ):
        materialized_payload = float(report["materialized_gradient_payload_bytes"])
        if materialized_payload <= 0.0:
            errors.append("materialized_gradient_payload_bytes must be positive")
        else:
            ratio = float(report["native_family_gradient_payload_bytes"]) / materialized_payload
            if ratio >= 0.35:
                errors.append(f"native family/materialized gradient payload ratio regressed: {ratio:.6g} >= 0.35")
    if _finite_float(report.get("native_family_grad_abs_sum")) and float(report["native_family_grad_abs_sum"]) <= 0.0:
        errors.append("native_family_grad_abs_sum must be positive")
    if _finite_float(report.get("native_q_basis_grad_abs_sum")) and float(report["native_q_basis_grad_abs_sum"]) <= 0.0:
        errors.append("native_q_basis_grad_abs_sum must be positive")
    if _finite_float(report.get("native_q_basis_grad_abs_sum")) and abs(
        row_grad_sum - float(report["native_q_basis_grad_abs_sum"])
    ) > 1.0e-4:
        errors.append("row native_q_basis_grad_abs_sum values must sum to native_q_basis_grad_abs_sum")

    thresholds = {
        "native_family_interval_backward_max_family_grad_rel_error": 1.0e-5,
        "native_family_interval_backward_max_q_basis_grad_rel_error": 1.0e-5,
        "native_family_interval_backward_max_opacity_grad_rel_error": 1.0e-5,
        "native_family_interval_backward_max_color_grad_rel_error": 1.0e-5,
        "native_family_interval_backward_max_opacity_time_grad_rel_error": 1.0e-5,
        "native_family_interval_backward_max_spatial_precision_grad_rel_error": 1.0e-5,
    }
    for key, threshold in thresholds.items():
        if _finite_float(report.get(key)) and float(report[key]) > threshold:
            errors.append(f"{key}={float(report[key]):.6g} exceeds {threshold:.6g}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        expected = summarize(report)
        for key, value in expected.items():
            _assert_summary_close(summary.get(key), value, key, errors)
    return errors


def assert_camera_family_2d_native_interval_backward_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_native_interval_backward_report(report)
    if errors:
        raise AssertionError("\n".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = report.get("summary", {})
    lines = [
        "# STAR UVT Q2 Native Family Interval Backward",
        "",
        f"status: `{report.get('status')}`",
        f"q pairs: `{summary.get('q_pair_count')}`",
        f"native family/materialized gradient payload ratio: `{summary.get('native_family_gradient_to_materialized_gradient_payload_ratio')}`",
        f"family grad rel error: `{summary.get('native_family_interval_backward_max_family_grad_rel_error')}`",
        f"q-basis grad rel error: `{summary.get('native_family_interval_backward_max_q_basis_grad_rel_error')}`",
        "",
        "This proves native interval-renderer VJP over shared family coefficients and q-basis values,",
        "with visibility/order treated as compiled constants.",
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
        assert_camera_family_2d_native_interval_backward_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report()
    write_report(report, args.out_dir)
    assert_camera_family_2d_native_interval_backward_report(report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
