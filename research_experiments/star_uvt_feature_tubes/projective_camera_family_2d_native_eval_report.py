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
    direct_backward_projective_trace_family_metal,
    eval_projective_trace_family,
    eval_projective_trace_family_torch,
    has_projective_trace_family_backward_metal,
    has_projective_trace_family_metal,
)

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_chain_rule_report import (  # noqa: E402
    _q_grid,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (  # noqa: E402
    Q2_BASIS_COUNT,
    TRACE_COUNT,
    _family_coeff_table,
    _q2_basis,
    _tensor_bytes,
    lower_q2_family_coeffs,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_camera_family_2d_native_eval"
)


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _q_basis_table(q_pairs: list[tuple[float, float]], *, device: torch.device | str) -> torch.Tensor:
    return torch.stack(
        [_q2_basis(q_phase, q_height, device=device) for q_phase, q_height in q_pairs],
        dim=0,
    ).contiguous()


def _materialized_coeff_payload_bytes(
    family_coeffs: torch.Tensor,
    q_pairs: list[tuple[float, float]],
) -> int:
    coeffs = torch.stack(
        [
            lower_q2_family_coeffs(family_coeffs, q_phase=q_phase, q_height=q_height)
            for q_phase, q_height in q_pairs
        ],
        dim=0,
    )
    return _tensor_bytes(coeffs)


def _grad_output(shape: torch.Size, *, device: torch.device | str) -> torch.Tensor:
    grad = torch.linspace(-0.19, 0.33, steps=math.prod(shape), dtype=torch.float32, device=device).reshape(shape)
    grad = grad.contiguous()
    grad[..., 3] = 0.0
    return grad


def _relative_error(delta: torch.Tensor, reference_a: torch.Tensor, reference_b: torch.Tensor) -> float:
    scale = torch.maximum(reference_a.abs(), reference_b.abs()).amax().clamp_min(1.0e-6)
    return float((delta.abs().amax() / scale).detach().cpu().item())


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    q_pair_count = int(report["q_pair_count"])
    family_payload = int(report["family_coeff_payload_bytes"])
    q_basis_payload = int(report["q_basis_payload_bytes"])
    materialized_payload = int(report["materialized_coeff_payload_bytes"])
    return {
        "q_axis_count": int(report["q_axis_count"]),
        "q_pair_count": q_pair_count,
        "trace_count": int(report["trace_count"]),
        "family_basis_count": int(report["family_basis_count"]),
        "time_sample_count": int(report["time_sample_count"]),
        "family_coeff_payload_bytes": family_payload,
        "q_basis_payload_bytes": q_basis_payload,
        "family_plus_q_basis_payload_bytes": family_payload + q_basis_payload,
        "materialized_coeff_payload_bytes": materialized_payload,
        "family_coeff_to_materialized_coeff_payload_ratio": float(family_payload) / float(materialized_payload),
        "family_plus_q_basis_to_materialized_coeff_payload_ratio": float(family_payload + q_basis_payload)
        / float(materialized_payload),
        "native_eval_max_abs_error": float(report["native_eval_max_abs_error"]),
        "native_eval_max_rel_error": float(report["native_eval_max_rel_error"]),
        "native_grad_family_max_abs_error": float(report["native_grad_family_max_abs_error"]),
        "native_grad_family_max_rel_error": float(report["native_grad_family_max_rel_error"]),
        "native_grad_q_basis_max_abs_error": float(report["native_grad_q_basis_max_abs_error"]),
        "native_grad_q_basis_max_rel_error": float(report["native_grad_q_basis_max_rel_error"]),
        "metal_output_abs_sum": float(report["metal_output_abs_sum"]),
        "metal_grad_family_abs_sum": float(report["metal_grad_family_abs_sum"]),
        "metal_grad_q_basis_abs_sum": float(report["metal_grad_q_basis_abs_sum"]),
    }


def run_report(
    *,
    q_axis_count: int = 5,
    frames: int = 4,
    eps: float = 1.0e-6,
) -> dict[str, Any]:
    family_metal_available = bool(torch.backends.mps.is_available() and has_projective_trace_family_metal())
    family_backward_metal_available = bool(
        torch.backends.mps.is_available() and has_projective_trace_family_backward_metal()
    )
    if not family_metal_available or not family_backward_metal_available:
        return {
            "status": "failed",
            "benchmark": "star_uvt_projective_camera_family_2d_native_eval",
            "base_domain": "Q2 x Omega x T native family trace eval/VJP",
            "theory_contract": "The Metal shader consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace evaluation and VJP. This is native family trace evaluation, not the full compositing renderer.",
            "family_metal_available": family_metal_available,
            "family_backward_metal_available": family_backward_metal_available,
            "metal_ran": False,
            "errors": ["MPS projective trace family eval/backward ops are required for this saved evidence artifact."],
            "rows": [],
            "summary": {},
        }

    device = torch.device("mps")
    q_pairs = _q_grid(int(q_axis_count))
    family_coeffs = _family_coeff_table(device=device).contiguous()
    q_basis = _q_basis_table(q_pairs, device=device)
    times = torch.linspace(0.0, 1.0, int(frames), dtype=torch.float32, device=device).contiguous()
    metal_out = eval_projective_trace_family(family_coeffs, q_basis, times, eps=float(eps))
    torch_ref = eval_projective_trace_family_torch(family_coeffs, q_basis, times, eps=float(eps))
    grad_out = _grad_output(metal_out.shape, device=device)
    metal_grad_family, metal_grad_q_basis = direct_backward_projective_trace_family_metal(
        family_coeffs,
        q_basis,
        times,
        grad_out,
        eps=float(eps),
    )

    ref_family = family_coeffs.detach().clone().requires_grad_(True)
    ref_q_basis = q_basis.detach().clone().requires_grad_(True)
    ref_out_for_grad = eval_projective_trace_family_torch(ref_family, ref_q_basis, times, eps=float(eps))
    objective = torch.sum(ref_out_for_grad * grad_out)
    objective.backward()
    ref_grad_family = ref_family.grad.detach()
    ref_grad_q_basis = ref_q_basis.grad.detach()
    torch.mps.synchronize()

    eval_delta = metal_out - torch_ref
    grad_family_delta = metal_grad_family - ref_grad_family
    grad_q_delta = metal_grad_q_basis - ref_grad_q_basis
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_native_eval",
        "base_domain": "Q2 x Omega x T native family trace eval/VJP",
        "theory_contract": "The Metal shader consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace evaluation and VJP. This is native family trace evaluation and shared-family VJP, not the full visibility/compositing renderer.",
        "family_metal_available": family_metal_available,
        "family_backward_metal_available": family_backward_metal_available,
        "metal_ran": True,
        "q_axis_count": int(q_axis_count),
        "q_pair_count": len(q_pairs),
        "trace_count": TRACE_COUNT,
        "family_basis_count": Q2_BASIS_COUNT,
        "time_sample_count": int(frames),
        "family_coeff_payload_bytes": _tensor_bytes(family_coeffs),
        "q_basis_payload_bytes": _tensor_bytes(q_basis),
        "materialized_coeff_payload_bytes": _materialized_coeff_payload_bytes(family_coeffs, q_pairs),
        "native_eval_max_abs_error": float(eval_delta.abs().amax().detach().cpu().item()),
        "native_eval_max_rel_error": _relative_error(eval_delta, metal_out, torch_ref),
        "native_grad_family_max_abs_error": float(grad_family_delta.abs().amax().detach().cpu().item()),
        "native_grad_family_max_rel_error": _relative_error(grad_family_delta, metal_grad_family, ref_grad_family),
        "native_grad_q_basis_max_abs_error": float(grad_q_delta.abs().amax().detach().cpu().item()),
        "native_grad_q_basis_max_rel_error": _relative_error(grad_q_delta, metal_grad_q_basis, ref_grad_q_basis),
        "metal_output_abs_sum": float(metal_out.abs().sum().detach().cpu().item()),
        "metal_grad_family_abs_sum": float(metal_grad_family.abs().sum().detach().cpu().item()),
        "metal_grad_q_basis_abs_sum": float(metal_grad_q_basis.abs().sum().detach().cpu().item()),
        "rows": [
            {
                "q_phase": float(q_phase),
                "q_height": float(q_height),
                "basis_abs_sum": float(q_basis[row_index].abs().sum().detach().cpu().item()),
                "output_abs_sum": float(metal_out[row_index].abs().sum().detach().cpu().item()),
            }
            for row_index, (q_phase, q_height) in enumerate(q_pairs)
        ],
    }
    report["summary"] = summarize(report)
    errors = verify_camera_family_2d_native_eval_report(report)
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


def verify_camera_family_2d_native_eval_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_native_eval":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T native family trace eval/VJP":
        errors.append(f"base_domain must name native family trace eval/VJP, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "pi_* Gamma^*" not in theory_contract
        or "family coefficients" not in theory_contract
        or "not the full" not in theory_contract
    ):
        errors.append("theory_contract must preserve the pi_* Gamma^* native-family-but-not-full-renderer contract")
    if report.get("metal_ran") is not True:
        errors.append("metal_ran must be true for this evidence artifact")
    if report.get("family_metal_available") is not True or report.get("family_backward_metal_available") is not True:
        errors.append("family eval/backward Metal availability must both be true")

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
        errors.append(f"rows must contain one native-eval row per q-pair, expected {q_pair_count}, got {len(rows)}")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {idx} must be an object")
            continue
        for key in ("q_phase", "q_height", "basis_abs_sum", "output_abs_sum"):
            if not _finite_float(row.get(key)):
                errors.append(f"row {idx} {key} must be finite, got {row.get(key)!r}")
        if _finite_float(row.get("output_abs_sum")) and float(row["output_abs_sum"]) <= 1.0e-6:
            errors.append(f"row {idx} output_abs_sum must be nonzero")

    for key in (
        "family_coeff_payload_bytes",
        "q_basis_payload_bytes",
        "materialized_coeff_payload_bytes",
    ):
        if not isinstance(report.get(key), int) or int(report[key]) <= 0:
            errors.append(f"{key} must be a positive integer")
    for key in (
        "native_eval_max_abs_error",
        "native_eval_max_rel_error",
        "native_grad_family_max_abs_error",
        "native_grad_family_max_rel_error",
        "native_grad_q_basis_max_abs_error",
        "native_grad_q_basis_max_rel_error",
        "metal_output_abs_sum",
        "metal_grad_family_abs_sum",
        "metal_grad_q_basis_abs_sum",
    ):
        if not _finite_float(report.get(key)):
            errors.append(f"{key} must be finite, got {report.get(key)!r}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        try:
            expected = summarize(report)
            for key, value in expected.items():
                _assert_summary_close(summary.get(key), value, key, errors)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"summary could not be recomputed: {exc}")
        if (
            _finite_float(summary.get("family_coeff_to_materialized_coeff_payload_ratio"))
            and float(summary["family_coeff_to_materialized_coeff_payload_ratio"]) >= 0.30
        ):
            errors.append("family/materialized coefficient payload ratio must stay below 0.30")
        if (
            _finite_float(summary.get("family_plus_q_basis_to_materialized_coeff_payload_ratio"))
            and float(summary["family_plus_q_basis_to_materialized_coeff_payload_ratio"]) >= 0.65
        ):
            errors.append("family-plus-q-basis/materialized coefficient payload ratio must stay below 0.65")
        thresholds = {
            "native_eval_max_abs_error": 1.0e-6,
            "native_eval_max_rel_error": 1.0e-6,
            "native_grad_family_max_abs_error": 5.0e-5,
            "native_grad_family_max_rel_error": 2.0e-5,
            "native_grad_q_basis_max_abs_error": 5.0e-5,
            "native_grad_q_basis_max_rel_error": 2.0e-5,
        }
        for key, threshold in thresholds.items():
            if _finite_float(summary.get(key)) and float(summary[key]) > threshold:
                errors.append(f"{key} must stay below {threshold:g}")
        for key in ("metal_output_abs_sum", "metal_grad_family_abs_sum", "metal_grad_q_basis_abs_sum"):
            if _finite_float(summary.get(key)) and float(summary[key]) <= 1.0e-6:
                errors.append(f"summary {key} must be nonzero")
    return errors


def assert_camera_family_2d_native_eval_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_native_eval_report(report)
    if errors:
        raise AssertionError("camera-family 2D native eval report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Camera-Family 2D Native Eval",
        "",
        "This is native family trace evaluation/VJP. It is not the full visibility/compositing renderer.",
        "",
        "## Contract",
        "",
        report["theory_contract"],
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, default=None)
    parser.add_argument("--q-axis-count", type=int, default=5)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_camera_family_2d_native_eval_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(q_axis_count=int(args.q_axis_count), frames=int(args.frames))
    assert_camera_family_2d_native_eval_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
