from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_bundle_gauge_invariance"
)


def _look_at_w2c(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1.0e-12)
    up_hint = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    right = torch.cross(up_hint, forward, dim=0)
    right = right / right.norm().clamp_min(1.0e-12)
    up = torch.cross(forward, right, dim=0)
    up = up / up.norm().clamp_min(1.0e-12)
    rotation = torch.stack((right, up, forward), dim=0)
    w2c = torch.eye(4, dtype=torch.float64)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -(rotation @ eye)
    return w2c


def _orbit_w2c(tau: torch.Tensor) -> torch.Tensor:
    theta = math.radians(120.0) * (float(tau) - 0.5)
    eye = torch.tensor(
        [2.5 * math.sin(theta), 0.6 + 0.2 * math.cos(0.5 * theta), -2.5 * math.cos(theta)],
        dtype=torch.float64,
    )
    return _look_at_w2c(eye, torch.tensor([0.05, 0.02, 0.0], dtype=torch.float64))


def _camera_ray_world_point(u: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    image_size = 64.0
    fx = 1.9 * image_size
    fy = 1.8 * image_size
    cx = 0.5 * image_size
    cy = 0.5 * image_size
    x_cam = depth * (u - cx) / fx
    y_cam = depth * (v - cy) / fy
    z_cam = depth
    ones = torch.ones_like(depth)
    p_cam = torch.stack((x_cam, y_cam, z_cam, ones), dim=-1)
    c2w = torch.linalg.inv(_orbit_w2c(tau))
    return (p_cam @ c2w.T)[..., :3]


def _spacetime_density(u: torch.Tensor, v: torch.Tensor, tau: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    world = _camera_ray_world_point(u, v, tau, depth)
    mean = torch.tensor([0.08, 0.03, -0.02, 0.47], dtype=torch.float64)
    precision = torch.tensor(
        [
            [8.0, 1.4, -0.6, 0.3],
            [1.4, 11.0, 0.5, -0.4],
            [-0.6, 0.5, 6.5, 0.2],
            [0.3, -0.4, 0.2, 3.0],
        ],
        dtype=torch.float64,
    )
    centered = torch.cat((world, tau.expand_as(depth).unsqueeze(-1)), dim=-1) - mean
    exponent = -0.5 * torch.einsum("...i,ij,...j->...", centered, precision, centered)
    return torch.exp(exponent)


def _trapz(values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(values, coordinates)


def _integrate_depth_gauge(u: float, v: float, tau: float, *, samples: int, near: float, far: float) -> torch.Tensor:
    depth = torch.linspace(near, far, samples, dtype=torch.float64)
    return _trapz(_spacetime_density(torch.tensor(u), torch.tensor(v), torch.tensor(tau), depth), depth)


def _integrate_log_gauge(
    u: float,
    v: float,
    tau: float,
    *,
    samples: int,
    near: float,
    far: float,
    include_jacobian: bool,
) -> torch.Tensor:
    log_depth = torch.linspace(math.log(near), math.log(far), samples, dtype=torch.float64)
    depth = torch.exp(log_depth)
    values = _spacetime_density(torch.tensor(u), torch.tensor(v), torch.tensor(tau), depth)
    if include_jacobian:
        values = values * depth
    return _trapz(values, log_depth)


def _row_for_sensor_point(u: float, v: float, tau: float, *, samples: int, near: float, far: float) -> dict[str, float]:
    depth_integral = _integrate_depth_gauge(u, v, tau, samples=samples, near=near, far=far)
    log_integral = _integrate_log_gauge(u, v, tau, samples=samples, near=near, far=far, include_jacobian=True)
    bad_log_integral = _integrate_log_gauge(u, v, tau, samples=samples, near=near, far=far, include_jacobian=False)
    denom = depth_integral.abs().clamp_min(1.0e-12)
    return {
        "u": float(u),
        "v": float(v),
        "tau": float(tau),
        "depth_integral": float(depth_integral.item()),
        "log_gauge_integral": float(log_integral.item()),
        "bad_no_jacobian_integral": float(bad_log_integral.item()),
        "abs_error": float((log_integral - depth_integral).abs().item()),
        "rel_error": float(((log_integral - depth_integral).abs() / denom).item()),
        "bad_no_jacobian_rel_error": float(((bad_log_integral - depth_integral).abs() / denom).item()),
    }


def _order_report(*, near: float, far: float) -> dict[str, Any]:
    depths_front = torch.linspace(near + 0.2, 0.5 * (near + far) - 0.1, 11, dtype=torch.float64)
    depths_back = depths_front + 0.8
    log_preserves = bool(torch.all(torch.log(depths_front) < torch.log(depths_back)).item())
    neg_log_flips = bool(torch.all(-torch.log(depths_front) > -torch.log(depths_back)).item())
    return {
        "monotone_log_order_preserved": log_preserves,
        "orientation_reversing_neg_log_order_flipped": neg_log_flips,
        "log_min_derivative": float((1.0 / torch.linspace(near, far, 64, dtype=torch.float64)).min().item()),
        "neg_log_max_derivative": float((-1.0 / torch.linspace(near, far, 64, dtype=torch.float64)).max().item()),
    }


def summarize(rows: list[dict[str, float]], order: dict[str, Any]) -> dict[str, Any]:
    rel_errors = [float(row["rel_error"]) for row in rows]
    bad_errors = [float(row["bad_no_jacobian_rel_error"]) for row in rows]
    return {
        "sensor_point_count": len(rows),
        "max_abs_error": max(float(row["abs_error"]) for row in rows),
        "max_rel_error": max(rel_errors),
        "mean_rel_error": sum(rel_errors) / float(len(rel_errors)),
        "min_bad_no_jacobian_rel_error": min(bad_errors),
        "mean_bad_no_jacobian_rel_error": sum(bad_errors) / float(len(bad_errors)),
        "monotone_log_order_preserved": bool(order["monotone_log_order_preserved"]),
        "orientation_reversing_neg_log_order_flipped": bool(order["orientation_reversing_neg_log_order_flipped"]),
    }


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _assert_close(actual: float, expected: float, label: str, errors: list[str], *, atol: float = 1.0e-9) -> None:
    if abs(actual - expected) > atol:
        errors.append(f"{label} mismatch: expected {expected:.9g}, got {actual:.9g}")


def _assert_summary_close(
    summary: dict[str, Any],
    expected: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    actual_value = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual_value, int | float) or abs(float(actual_value) - expected_value) > atol:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual_value!r}")
    elif actual_value != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual_value!r}")


def run_report(*, samples: int = 4097, near: float = 0.4, far: float = 5.0) -> dict[str, Any]:
    sensor_points = [
        (28.0, 31.5, 0.05),
        (32.0, 32.0, 0.25),
        (36.5, 29.0, 0.50),
        (30.0, 35.0, 0.75),
        (39.0, 33.5, 0.95),
    ]
    rows = [_row_for_sensor_point(u, v, tau, samples=samples, near=near, far=far) for u, v, tau in sensor_points]
    order = _order_report(near=near, far=far)
    report = {
        "status": "ok",
        "benchmark": "star_uvt_projective_bundle_gauge_invariance",
        "samples": int(samples),
        "near": float(near),
        "far": float(far),
        "theory_contract": "UVT trace = pi_* Gamma^* world_primitive is invariant under monotone fiber gauge changes when the fiber-measure Jacobian is included.",
        "rows": rows,
        "order": order,
        "summary": summarize(rows, order),
    }
    errors = verify_bundle_gauge_invariance_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def verify_bundle_gauge_invariance_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_bundle_gauge_invariance":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    samples = report.get("samples")
    if not isinstance(samples, int) or samples < 3:
        errors.append(f"samples must be an integer >= 3, got {samples!r}")
    near = _finite_float(report.get("near"), "near", errors)
    far = _finite_float(report.get("far"), "far", errors)
    if not far > near > 0.0:
        errors.append(f"near/far must satisfy 0 < near < far, got near={near}, far={far}")
    if not isinstance(report.get("rows"), list) or len(report["rows"]) < 3:
        errors.append("rows must contain at least three sensor-time samples")
        return errors
    rows = report["rows"]
    if any(not isinstance(row, dict) for row in rows):
        errors.append("all rows must be objects")
        return errors
    order = report.get("order")
    if not isinstance(order, dict):
        errors.append("order must be an object")
        return errors
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    if int(summary.get("sensor_point_count") or 0) != len(rows):
        errors.append("summary sensor_point_count must match rows")

    for idx, row in enumerate(rows):
        for key in (
            "u",
            "v",
            "tau",
            "depth_integral",
            "log_gauge_integral",
            "bad_no_jacobian_integral",
            "abs_error",
            "rel_error",
            "bad_no_jacobian_rel_error",
        ):
            _finite_float(row.get(key), f"row {idx} {key}", errors)
        depth_integral = _finite_float(row.get("depth_integral"), f"row {idx} depth_integral", errors)
        log_integral = _finite_float(row.get("log_gauge_integral"), f"row {idx} log_gauge_integral", errors)
        bad_integral = _finite_float(row.get("bad_no_jacobian_integral"), f"row {idx} bad_no_jacobian_integral", errors)
        abs_error = _finite_float(row.get("abs_error"), f"row {idx} abs_error", errors)
        rel_error = _finite_float(row.get("rel_error"), f"row {idx} rel_error", errors)
        bad_rel_error = _finite_float(
            row.get("bad_no_jacobian_rel_error"),
            f"row {idx} bad_no_jacobian_rel_error",
            errors,
        )
        denom = max(abs(depth_integral), 1.0e-12)
        _assert_close(abs_error, abs(log_integral - depth_integral), f"row {idx} abs_error", errors)
        _assert_close(rel_error, abs_error / denom, f"row {idx} rel_error", errors)
        _assert_close(
            bad_rel_error,
            abs(bad_integral - depth_integral) / denom,
            f"row {idx} bad_no_jacobian_rel_error",
            errors,
        )

    try:
        expected_summary = summarize(rows, order)
        for key in expected_summary:
            _assert_summary_close(summary, expected_summary, key, errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")

    log_min_derivative = _finite_float(order.get("log_min_derivative"), "order log_min_derivative", errors)
    neg_log_max_derivative = _finite_float(order.get("neg_log_max_derivative"), "order neg_log_max_derivative", errors)
    if log_min_derivative <= 0.0:
        errors.append(f"log_min_derivative must be positive, got {log_min_derivative}")
    if neg_log_max_derivative >= 0.0:
        errors.append(f"neg_log_max_derivative must be negative, got {neg_log_max_derivative}")

    max_rel_error = float(summary.get("max_rel_error") or math.inf)
    if not math.isfinite(max_rel_error) or max_rel_error > 2.0e-6:
        errors.append(f"max_rel_error must be <= 2e-6, got {max_rel_error}")
    min_bad_error = float(summary.get("min_bad_no_jacobian_rel_error") or 0.0)
    if not math.isfinite(min_bad_error) or min_bad_error < 0.05:
        errors.append(f"min_bad_no_jacobian_rel_error must expose missing-Jacobian failure, got {min_bad_error}")
    if summary.get("monotone_log_order_preserved") is not True:
        errors.append("monotone log gauge must preserve depth order")
    if summary.get("orientation_reversing_neg_log_order_flipped") is not True:
        errors.append("orientation-reversing gauge must flip depth order and be rejected as a visibility gauge")
    if order.get("monotone_log_order_preserved") is not summary.get("monotone_log_order_preserved"):
        errors.append("order and summary disagree on monotone_log_order_preserved")
    if order.get("orientation_reversing_neg_log_order_flipped") is not summary.get(
        "orientation_reversing_neg_log_order_flipped"
    ):
        errors.append("order and summary disagree on orientation_reversing_neg_log_order_flipped")
    return errors


def assert_bundle_gauge_invariance_report(report: dict[str, Any]) -> None:
    errors = verify_bundle_gauge_invariance_report(report)
    if errors:
        raise AssertionError("\n".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Bundle Gauge Invariance",
        "",
        "This report numerically checks the fiber-bundle contract:",
        "",
        "```text",
        "UVT trace = pi_* Gamma^* world_primitive",
        "```",
        "",
        "The same revolving-camera spacetime Gaussian is integrated in ordinary depth and log-depth gauges.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Rows",
        "",
        "| u | v | tau | depth_integral | log_gauge_integral | rel_error | bad_no_jacobian_rel_error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {u:.3f} | {v:.3f} | {tau:.3f} | {depth_integral:.9g} | {log_gauge_integral:.9g} | {rel_error:.3g} | {bad_no_jacobian_rel_error:.3g} |".format(
                **row
            )
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", type=int, default=4097)
    parser.add_argument("--near", type=float, default=0.4)
    parser.add_argument("--far", type=float, default=5.0)
    parser.add_argument("--verify-report", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_bundle_gauge_invariance_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(samples=args.samples, near=args.near, far=args.far)
    assert_bundle_gauge_invariance_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
