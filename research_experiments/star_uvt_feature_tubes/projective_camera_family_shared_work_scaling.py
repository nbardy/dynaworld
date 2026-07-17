from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_experiments.star_uvt_feature_tubes.projective_camera_family_gauge_report import (
    _camera_family_frame,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_camera_family_shared_work_scaling"
)


PRIMITIVES = (
    torch.tensor([0.08, 0.03, -0.02], dtype=torch.float64),
    torch.tensor([-0.20, 0.10, 0.04], dtype=torch.float64),
    torch.tensor([0.18, -0.06, 0.03], dtype=torch.float64),
)
FRAME_COUNT = 16
Q_MIN = -0.30
Q_MAX = 0.32
FLOAT_BYTES = 4
CHANNELS = 3
FAMILY_BASIS_COUNT = 6
PATH_BASIS_COUNT = 3
PRIMITIVE_ATTRIBUTE_FLOATS = 4


def _project_center(q: torch.Tensor, tau: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    eye, right, up, forward = _camera_family_frame(q, tau)
    rel = point - eye
    x_cam = torch.dot(rel, right)
    y_cam = torch.dot(rel, up)
    z_cam = torch.dot(rel, forward)
    image_size = 64.0
    fx = 1.9 * image_size
    fy = 1.8 * image_size
    cx = 0.5 * image_size
    cy = 0.5 * image_size
    return torch.stack((fx * x_cam / z_cam + cx, fy * y_cam / z_cam + cy, z_cam))


def _family_basis(q: float, tau: float) -> torch.Tensor:
    return torch.tensor((1.0, q, tau, q * q, q * tau, tau * tau), dtype=torch.float64)


def _path_basis(tau: float) -> torch.Tensor:
    return torch.tensor((1.0, tau, tau * tau), dtype=torch.float64)


def _fit_error_for_family(q_samples: torch.Tensor, tau_samples: torch.Tensor, point: torch.Tensor) -> dict[str, float]:
    basis_rows = []
    values = []
    for q in q_samples:
        for tau in tau_samples:
            basis_rows.append(_family_basis(float(q), float(tau)))
            values.append(_project_center(q, tau, point))
    basis = torch.stack(basis_rows)
    targets = torch.stack(values)
    coeffs = torch.linalg.lstsq(basis, targets).solution
    error = (basis @ coeffs - targets).abs()
    return {
        "max_error": float(error.max().item()),
        "max_uv_error": float(error[:, :2].max().item()),
        "max_depth_error": float(error[:, 2].max().item()),
        "rms_error": float(torch.sqrt(error.square().mean()).item()),
    }


def _fit_error_for_replayed_paths(q_samples: torch.Tensor, tau_samples: torch.Tensor, point: torch.Tensor) -> dict[str, float]:
    max_error = 0.0
    max_uv_error = 0.0
    max_depth_error = 0.0
    squared_sum = 0.0
    value_count = 0
    basis = torch.stack([_path_basis(float(tau)) for tau in tau_samples])
    for q in q_samples:
        targets = torch.stack([_project_center(q, tau, point) for tau in tau_samples])
        coeffs = torch.linalg.lstsq(basis, targets).solution
        error = (basis @ coeffs - targets).abs()
        max_error = max(max_error, float(error.max().item()))
        max_uv_error = max(max_uv_error, float(error[:, :2].max().item()))
        max_depth_error = max(max_depth_error, float(error[:, 2].max().item()))
        squared_sum += float(error.square().sum().item())
        value_count += int(error.numel())
    return {
        "max_error": max_error,
        "max_uv_error": max_uv_error,
        "max_depth_error": max_depth_error,
        "rms_error": math.sqrt(squared_sum / float(value_count)),
    }


def _payload_bytes(*, chart_count: int, basis_count: int) -> int:
    coeff_floats = chart_count * basis_count * CHANNELS
    attr_floats = chart_count * PRIMITIVE_ATTRIBUTE_FLOATS
    return (coeff_floats + attr_floats) * FLOAT_BYTES


def _row(route: str, q_count: int, *, tau_samples: torch.Tensor) -> dict[str, Any]:
    q_samples = torch.linspace(Q_MIN, Q_MAX, int(q_count), dtype=torch.float64)
    if route == "family_chart":
        fit_rows = [_fit_error_for_family(q_samples, tau_samples, point) for point in PRIMITIVES]
        chart_count = len(PRIMITIVES)
        basis_count = FAMILY_BASIS_COUNT
    elif route == "per_q_replay":
        fit_rows = [_fit_error_for_replayed_paths(q_samples, tau_samples, point) for point in PRIMITIVES]
        chart_count = len(PRIMITIVES) * int(q_count)
        basis_count = PATH_BASIS_COUNT
    else:
        raise ValueError(f"unknown route {route!r}")
    return {
        "route": route,
        "q_count": int(q_count),
        "frame_count": int(FRAME_COUNT),
        "base_domain": "Q x Omega x T" if route == "family_chart" else "Omega x T per q",
        "primitive_count": len(PRIMITIVES),
        "chart_count": int(chart_count),
        "basis_count": int(basis_count),
        "coeff_float_count": int(chart_count * basis_count * CHANNELS),
        "payload_bytes": _payload_bytes(chart_count=chart_count, basis_count=basis_count),
        "dense_trace_samples": int(len(PRIMITIVES) * int(q_count) * int(FRAME_COUNT)),
        "max_fit_error_px": max(float(row["max_error"]) for row in fit_rows),
        "max_fit_uv_error_px": max(float(row["max_uv_error"]) for row in fit_rows),
        "max_fit_depth_error": max(float(row["max_depth_error"]) for row in fit_rows),
        "rms_fit_error": math.sqrt(sum(float(row["rms_error"]) ** 2 for row in fit_rows) / float(len(fit_rows))),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family = sorted((row for row in rows if row["route"] == "family_chart"), key=lambda row: int(row["q_count"]))
    replay = sorted((row for row in rows if row["route"] == "per_q_replay"), key=lambda row: int(row["q_count"]))
    final_family = family[-1]
    final_replay = replay[-1]
    return {
        "q_counts": [int(row["q_count"]) for row in family],
        "family_payload_growth": float(family[-1]["payload_bytes"]) / float(family[0]["payload_bytes"]),
        "per_q_replay_payload_growth": float(replay[-1]["payload_bytes"]) / float(replay[0]["payload_bytes"]),
        "final_payload_ratio": float(final_family["payload_bytes"]) / float(final_replay["payload_bytes"]),
        "family_chart_growth": float(family[-1]["chart_count"]) / float(family[0]["chart_count"]),
        "per_q_replay_chart_growth": float(replay[-1]["chart_count"]) / float(replay[0]["chart_count"]),
        "final_chart_ratio": float(final_family["chart_count"]) / float(final_replay["chart_count"]),
        "max_family_fit_uv_error_px": max(float(row["max_fit_uv_error_px"]) for row in family),
        "max_replay_fit_uv_error_px": max(float(row["max_fit_uv_error_px"]) for row in replay),
        "final_dense_sample_count": int(final_family["dense_trace_samples"]),
    }


def run_report(*, q_counts: tuple[int, ...] = (1, 2, 4, 8, 16)) -> dict[str, Any]:
    tau_samples = torch.linspace(0.05, 0.95, FRAME_COUNT, dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    for q_count in q_counts:
        rows.append(_row("family_chart", int(q_count), tau_samples=tau_samples))
        rows.append(_row("per_q_replay", int(q_count), tau_samples=tau_samples))
    report = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_shared_work_scaling",
        "theory_contract": (
            "A local Q x Omega x T camera-family chart can store projection metadata once over q,tau, "
            "while replaying one Omega x T atlas per q sample grows linearly with q samples."
        ),
        "q_min": Q_MIN,
        "q_max": Q_MAX,
        "frame_count": FRAME_COUNT,
        "primitive_count": len(PRIMITIVES),
        "rows": rows,
        "summary": summarize(rows),
    }
    errors = verify_camera_family_shared_work_scaling_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _assert_summary_close(summary: dict[str, Any], expected: dict[str, Any], key: str, errors: list[str]) -> None:
    actual = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    elif actual != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")


def verify_camera_family_shared_work_scaling_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_shared_work_scaling":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if not isinstance(report.get("theory_contract"), str) or "Q x Omega x T" not in report["theory_contract"]:
        errors.append("theory_contract must mention Q x Omega x T")
    q_min = _finite_float(report.get("q_min"), "q_min", errors)
    q_max = _finite_float(report.get("q_max"), "q_max", errors)
    if not q_min < 0.0 < q_max:
        errors.append("q range must span a local camera family around zero")
    frame_count = _finite_int(report.get("frame_count"), "frame_count", errors)
    primitive_count = _finite_int(report.get("primitive_count"), "primitive_count", errors)
    if frame_count <= 1:
        errors.append("frame_count must be greater than one")
    if primitive_count <= 0:
        errors.append("primitive_count must be positive")
    rows = report.get("rows")
    summary = report.get("summary")
    if not isinstance(rows, list) or len(rows) < 4 or any(not isinstance(row, dict) for row in rows):
        errors.append("rows must contain family_chart and per_q_replay rows")
        return errors
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors

    by_route: dict[str, list[dict[str, Any]]] = {"family_chart": [], "per_q_replay": []}
    for idx, row in enumerate(rows):
        route = row.get("route")
        if route not in by_route:
            errors.append(f"row {idx} has unknown route {route!r}")
            continue
        by_route[str(route)].append(row)
        q_count = _finite_int(row.get("q_count"), f"row {idx} q_count", errors)
        row_frame_count = _finite_int(row.get("frame_count"), f"row {idx} frame_count", errors)
        row_primitive_count = _finite_int(row.get("primitive_count"), f"row {idx} primitive_count", errors)
        chart_count = _finite_int(row.get("chart_count"), f"row {idx} chart_count", errors)
        basis_count = _finite_int(row.get("basis_count"), f"row {idx} basis_count", errors)
        coeff_float_count = _finite_int(row.get("coeff_float_count"), f"row {idx} coeff_float_count", errors)
        payload_bytes = _finite_int(row.get("payload_bytes"), f"row {idx} payload_bytes", errors)
        dense_samples = _finite_int(row.get("dense_trace_samples"), f"row {idx} dense_trace_samples", errors)
        if row_frame_count != frame_count:
            errors.append(f"row {idx} frame_count must match report frame_count")
        if row_primitive_count != primitive_count:
            errors.append(f"row {idx} primitive_count must match report primitive_count")
        expected_chart_count = primitive_count if route == "family_chart" else primitive_count * q_count
        expected_basis_count = FAMILY_BASIS_COUNT if route == "family_chart" else PATH_BASIS_COUNT
        expected_coeff_float_count = expected_chart_count * expected_basis_count * CHANNELS
        expected_payload_bytes = _payload_bytes(chart_count=expected_chart_count, basis_count=expected_basis_count)
        expected_dense_samples = primitive_count * q_count * frame_count
        if chart_count != expected_chart_count:
            errors.append(f"row {idx} chart_count mismatch: expected {expected_chart_count}, got {chart_count}")
        if basis_count != expected_basis_count:
            errors.append(f"row {idx} basis_count mismatch: expected {expected_basis_count}, got {basis_count}")
        if coeff_float_count != expected_coeff_float_count:
            errors.append(f"row {idx} coeff_float_count mismatch: expected {expected_coeff_float_count}, got {coeff_float_count}")
        if payload_bytes != expected_payload_bytes:
            errors.append(f"row {idx} payload_bytes mismatch: expected {expected_payload_bytes}, got {payload_bytes}")
        if dense_samples != expected_dense_samples:
            errors.append(f"row {idx} dense_trace_samples mismatch: expected {expected_dense_samples}, got {dense_samples}")
        for key in ("max_fit_error_px", "max_fit_uv_error_px", "max_fit_depth_error", "rms_fit_error"):
            value = _finite_float(row.get(key), f"row {idx} {key}", errors)
            if value < 0.0:
                errors.append(f"row {idx} {key} must be nonnegative")

    family_q = [int(row["q_count"]) for row in sorted(by_route["family_chart"], key=lambda row: int(row["q_count"]))]
    replay_q = [int(row["q_count"]) for row in sorted(by_route["per_q_replay"], key=lambda row: int(row["q_count"]))]
    if family_q != replay_q:
        errors.append(f"family and replay q_counts must match, got {family_q} vs {replay_q}")
    if family_q != sorted(family_q) or len(set(family_q)) != len(family_q) or any(q <= 0 for q in family_q):
        errors.append(f"q_counts must be strictly increasing positive ints, got {family_q}")

    try:
        expected_summary = summarize(rows)
        for key in expected_summary:
            _assert_summary_close(summary, expected_summary, key, errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")

    if float(summary.get("family_payload_growth") or math.inf) > 1.05:
        errors.append("family payload growth must stay near constant across q samples")
    if float(summary.get("per_q_replay_payload_growth") or 0.0) < 4.0:
        errors.append("per-q replay payload growth must expose replay cost")
    if float(summary.get("final_payload_ratio") or math.inf) >= 0.30:
        errors.append("final family/per-q payload ratio must stay below 0.30")
    if float(summary.get("family_chart_growth") or math.inf) > 1.05:
        errors.append("family chart growth must stay near constant across q samples")
    if float(summary.get("per_q_replay_chart_growth") or 0.0) < 4.0:
        errors.append("per-q replay chart growth must expose replay cost")
    if float(summary.get("final_chart_ratio") or math.inf) >= 0.15:
        errors.append("final family/per-q chart ratio must stay below 0.15")
    if float(summary.get("max_family_fit_uv_error_px") or math.inf) > 0.50:
        errors.append("family QxT fit residual must stay below 0.50 px")
    if float(summary.get("max_replay_fit_uv_error_px") or math.inf) > 0.30:
        errors.append("per-q replay fit residual must stay below 0.30 px")
    return errors


def assert_camera_family_shared_work_scaling_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_shared_work_scaling_report(report)
    if errors:
        raise AssertionError("camera-family shared-work scaling report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Camera-Family Shared-Work Scaling",
        "",
        "This report compares one local Q x Omega x T chart against replaying one Omega x T chart per q sample.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Rows",
        "",
        "| route | q count | chart count | payload bytes | max uv error px |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {route} | {q_count} | {chart_count} | {payload_bytes} | {max_fit_uv_error_px:.6g} |".format(
                **row
            )
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--q-counts", type=str, default="1,2,4,8,16")
    parser.add_argument("--verify-report", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_camera_family_shared_work_scaling_report(report)
        print(f"verified {args.verify_report}")
        return

    q_counts = tuple(int(part.strip()) for part in args.q_counts.split(",") if part.strip())
    report = run_report(q_counts=q_counts)
    assert_camera_family_shared_work_scaling_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
