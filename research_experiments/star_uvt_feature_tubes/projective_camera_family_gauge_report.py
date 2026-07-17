from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_camera_family_gauge"


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm().clamp_min(1.0e-12)


def _as_scalar(value: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _camera_family_frame(q: float | torch.Tensor, tau: float | torch.Tensor) -> tuple[torch.Tensor, ...]:
    q_tensor = _as_scalar(q)
    tau_tensor = _as_scalar(tau)
    theta = math.radians(120.0) * (tau_tensor - 0.5) + 0.35 * q_tensor
    radius = 2.5 + 0.35 * q_tensor
    eye = torch.stack(
        (
            radius * torch.sin(theta),
            0.6 + 0.2 * torch.cos(0.5 * theta) + 0.10 * q_tensor,
            -radius * torch.cos(theta),
        )
    )
    target = torch.stack((0.05 + 0.04 * q_tensor, torch.zeros_like(q_tensor) + 0.02, torch.zeros_like(q_tensor)))
    forward = _normalize(target - eye)
    up_hint = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    right = _normalize(torch.cross(up_hint, forward, dim=0))
    up = _normalize(torch.cross(forward, right, dim=0))
    return eye, right, up, forward


def _camera_ray_world_point(
    q: float | torch.Tensor,
    u: float | torch.Tensor,
    v: float | torch.Tensor,
    tau: float | torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    image_size = 64.0
    fx = 1.9 * image_size
    fy = 1.8 * image_size
    cx = 0.5 * image_size
    cy = 0.5 * image_size
    depth = torch.as_tensor(depth, dtype=torch.float64)
    x_cam = depth * (_as_scalar(u) - cx) / fx
    y_cam = depth * (_as_scalar(v) - cy) / fy
    z_cam = depth
    eye, right, up, forward = _camera_family_frame(q, tau)
    return eye + x_cam.unsqueeze(-1) * right + y_cam.unsqueeze(-1) * up + z_cam.unsqueeze(-1) * forward


def _initial_params(*, requires_grad: bool) -> dict[str, torch.Tensor]:
    params = {
        "mean": torch.tensor([0.08, 0.03, -0.02, 0.47], dtype=torch.float64),
        "log_precision": torch.log(torch.tensor([8.0, 11.0, 6.5, 3.0], dtype=torch.float64)),
        "log_amplitude": torch.tensor(0.2, dtype=torch.float64),
    }
    if requires_grad:
        return {key: value.clone().detach().requires_grad_(True) for key, value in params.items()}
    return {key: value.clone().detach() for key, value in params.items()}


def _spacetime_density(
    params: dict[str, torch.Tensor],
    *,
    q: float | torch.Tensor,
    u: float | torch.Tensor,
    v: float | torch.Tensor,
    tau: float | torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    world = _camera_ray_world_point(q, u, v, tau, depth)
    tau_tensor = _as_scalar(tau).expand_as(depth)
    centered = torch.cat((world, tau_tensor.unsqueeze(-1)), dim=-1) - params["mean"]
    precision = torch.exp(params["log_precision"])
    exponent = -0.5 * torch.sum(precision * centered.square(), dim=-1)
    return torch.exp(params["log_amplitude"]) * torch.exp(exponent)


def _trapz(values: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(values, coordinates)


def _integral(
    params: dict[str, torch.Tensor],
    *,
    q: float | torch.Tensor,
    u: float,
    v: float,
    tau: float,
    samples: int,
    near: float,
    far: float,
    gauge: str,
    include_jacobian: bool = True,
) -> torch.Tensor:
    if gauge == "depth":
        depth = torch.linspace(near, far, samples, dtype=torch.float64)
        values = _spacetime_density(params, q=q, u=u, v=v, tau=tau, depth=depth)
        return _trapz(values, depth)
    if gauge == "log_depth":
        log_depth = torch.linspace(math.log(near), math.log(far), samples, dtype=torch.float64)
        depth = torch.exp(log_depth)
        values = _spacetime_density(params, q=q, u=u, v=v, tau=tau, depth=depth)
        if include_jacobian:
            values = values * depth
        return _trapz(values, log_depth)
    raise ValueError(f"unknown gauge {gauge!r}")


def _value_row(
    *,
    q: float,
    u: float,
    v: float,
    tau: float,
    samples: int,
    near: float,
    far: float,
) -> dict[str, float]:
    params = _initial_params(requires_grad=False)
    depth = _integral(params, q=q, u=u, v=v, tau=tau, samples=samples, near=near, far=far, gauge="depth")
    log_depth = _integral(params, q=q, u=u, v=v, tau=tau, samples=samples, near=near, far=far, gauge="log_depth")
    bad = _integral(
        params,
        q=q,
        u=u,
        v=v,
        tau=tau,
        samples=samples,
        near=near,
        far=far,
        gauge="log_depth",
        include_jacobian=False,
    )
    denom = depth.abs().clamp_min(1.0e-12)
    return {
        "q": float(q),
        "u": float(u),
        "v": float(v),
        "tau": float(tau),
        "depth_integral": float(depth.item()),
        "log_gauge_integral": float(log_depth.item()),
        "bad_no_jacobian_integral": float(bad.item()),
        "abs_error": float((log_depth - depth).abs().item()),
        "rel_error": float(((log_depth - depth).abs() / denom).item()),
        "bad_no_jacobian_rel_error": float(((bad - depth).abs() / denom).item()),
    }


def _objective(
    params: dict[str, torch.Tensor],
    *,
    samples: int,
    near: float,
    far: float,
    gauge: str,
    include_jacobian: bool = True,
) -> torch.Tensor:
    family_points = [
        (-0.30, 28.0, 31.5, 0.05, 0.90),
        (-0.10, 32.0, 32.0, 0.25, -0.25),
        (0.00, 36.5, 29.0, 0.50, 0.70),
        (0.18, 30.0, 35.0, 0.75, 0.45),
        (0.32, 39.0, 33.5, 0.95, -0.15),
    ]
    total = torch.zeros((), dtype=torch.float64)
    for q, u, v, tau, weight in family_points:
        total = total + float(weight) * _integral(
            params,
            q=q,
            u=u,
            v=v,
            tau=tau,
            samples=samples,
            near=near,
            far=far,
            gauge=gauge,
            include_jacobian=include_jacobian,
        )
    return total


def _loss_and_grads(
    *,
    samples: int,
    near: float,
    far: float,
    gauge: str,
    include_jacobian: bool = True,
) -> tuple[float, dict[str, torch.Tensor]]:
    params = _initial_params(requires_grad=True)
    loss = _objective(
        params,
        samples=samples,
        near=near,
        far=far,
        gauge=gauge,
        include_jacobian=include_jacobian,
    )
    grads = torch.autograd.grad(loss, tuple(params.values()))
    return float(loss.detach().item()), {key: grad.detach() for key, grad in zip(params, grads, strict=True)}


def _gradient_row(name: str, reference: torch.Tensor, candidate: torch.Tensor, bad: torch.Tensor) -> dict[str, float | str]:
    diff = candidate - reference
    bad_diff = bad - reference
    denom = reference.norm().clamp_min(1.0e-12)
    return {
        "param": name,
        "depth_grad_norm": float(reference.norm().item()),
        "log_gauge_grad_norm": float(candidate.norm().item()),
        "bad_no_jacobian_grad_norm": float(bad.norm().item()),
        "max_abs_error": float(diff.abs().max().item()),
        "rel_error": float((diff.norm() / denom).item()),
        "bad_no_jacobian_rel_error": float((bad_diff.norm() / denom).item()),
    }


def _q_grad(*, q_value: float, samples: int, near: float, far: float, gauge: str, include_jacobian: bool = True) -> tuple[float, float]:
    params = _initial_params(requires_grad=False)
    q = torch.tensor(q_value, dtype=torch.float64, requires_grad=True)
    value = _integral(
        params,
        q=q,
        u=34.0,
        v=30.5,
        tau=0.62,
        samples=samples,
        near=near,
        far=far,
        gauge=gauge,
        include_jacobian=include_jacobian,
    )
    (grad,) = torch.autograd.grad(value, (q,))
    return float(value.detach().item()), float(grad.detach().item())


def _q_gradient_row(*, samples: int, near: float, far: float, epsilon: float = 1.0e-5) -> dict[str, float | str]:
    q_value = 0.11
    depth_value, depth_grad = _q_grad(q_value=q_value, samples=samples, near=near, far=far, gauge="depth")
    log_value, log_grad = _q_grad(q_value=q_value, samples=samples, near=near, far=far, gauge="log_depth")
    bad_value, bad_grad = _q_grad(
        q_value=q_value,
        samples=samples,
        near=near,
        far=far,
        gauge="log_depth",
        include_jacobian=False,
    )
    params = _initial_params(requires_grad=False)
    minus = _integral(
        params,
        q=q_value - epsilon,
        u=34.0,
        v=30.5,
        tau=0.62,
        samples=samples,
        near=near,
        far=far,
        gauge="depth",
    )
    plus = _integral(
        params,
        q=q_value + epsilon,
        u=34.0,
        v=30.5,
        tau=0.62,
        samples=samples,
        near=near,
        far=far,
        gauge="depth",
    )
    finite_difference = float(((plus - minus) / (2.0 * epsilon)).item())
    denom = max(abs(depth_grad), 1.0e-12)
    return {
        "param": "camera_family_q",
        "q": float(q_value),
        "depth_value": depth_value,
        "log_gauge_value": log_value,
        "bad_no_jacobian_value": bad_value,
        "depth_grad": depth_grad,
        "log_gauge_grad": log_grad,
        "bad_no_jacobian_grad": bad_grad,
        "finite_difference": finite_difference,
        "rel_error": abs(log_grad - depth_grad) / denom,
        "bad_no_jacobian_rel_error": abs(bad_grad - depth_grad) / denom,
        "finite_difference_rel_error": abs(finite_difference - depth_grad) / denom,
    }


def summarize(
    rows: list[dict[str, float]],
    gradient_rows: list[dict[str, float | str]],
    q_gradient: dict[str, float | str],
) -> dict[str, Any]:
    rel_errors = [float(row["rel_error"]) for row in rows]
    bad_errors = [float(row["bad_no_jacobian_rel_error"]) for row in rows]
    grad_errors = [float(row["rel_error"]) for row in gradient_rows]
    bad_grad_errors = [float(row["bad_no_jacobian_rel_error"]) for row in gradient_rows]
    q_values = [float(row["q"]) for row in rows]
    return {
        "family_point_count": len(rows),
        "q_min": min(q_values),
        "q_max": max(q_values),
        "max_value_rel_error": max(rel_errors),
        "mean_value_rel_error": sum(rel_errors) / float(len(rel_errors)),
        "min_bad_no_jacobian_value_rel_error": min(bad_errors),
        "max_primitive_gradient_rel_error": max(grad_errors),
        "mean_primitive_gradient_rel_error": sum(grad_errors) / float(len(grad_errors)),
        "min_bad_no_jacobian_gradient_rel_error": min(bad_grad_errors),
        "q_gradient_rel_error": float(q_gradient["rel_error"]),
        "q_finite_difference_rel_error": float(q_gradient["finite_difference_rel_error"]),
        "q_bad_no_jacobian_rel_error": float(q_gradient["bad_no_jacobian_rel_error"]),
    }


def run_report(*, samples: int = 4097, near: float = 0.4, far: float = 5.0) -> dict[str, Any]:
    family_points = [
        (-0.30, 28.0, 31.5, 0.05),
        (-0.10, 32.0, 32.0, 0.25),
        (0.00, 36.5, 29.0, 0.50),
        (0.18, 30.0, 35.0, 0.75),
        (0.32, 39.0, 33.5, 0.95),
    ]
    rows = [_value_row(q=q, u=u, v=v, tau=tau, samples=samples, near=near, far=far) for q, u, v, tau in family_points]
    _, depth_grads = _loss_and_grads(samples=samples, near=near, far=far, gauge="depth")
    _, log_grads = _loss_and_grads(samples=samples, near=near, far=far, gauge="log_depth")
    _, bad_grads = _loss_and_grads(samples=samples, near=near, far=far, gauge="log_depth", include_jacobian=False)
    gradient_rows = [
        _gradient_row(name, depth_grads[name], log_grads[name], bad_grads[name])
        for name in ("mean", "log_precision", "log_amplitude")
    ]
    q_gradient = _q_gradient_row(samples=samples, near=near, far=far)
    report = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_gauge",
        "samples": int(samples),
        "near": float(near),
        "far": float(far),
        "base_domain": "Q x Omega x T",
        "theory_contract": "A local camera-family trace over Q x Omega x T remains pi_* Gamma(q)^* world_primitive; monotone fiber gauge changes preserve values and derivatives when the Jacobian is included.",
        "rows": rows,
        "gradient_rows": gradient_rows,
        "q_gradient": q_gradient,
        "summary": summarize(rows, gradient_rows, q_gradient),
    }
    errors = verify_camera_family_gauge_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _assert_summary_close(summary: dict[str, Any], expected: dict[str, Any], key: str, errors: list[str]) -> None:
    actual = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    elif actual != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")


def verify_camera_family_gauge_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_gauge":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q x Omega x T":
        errors.append(f"base_domain must be Q x Omega x T, got {report.get('base_domain')!r}")
    if not isinstance(report.get("theory_contract"), str) or "Gamma(q)" not in report["theory_contract"]:
        errors.append("theory_contract must mention Gamma(q)")
    samples = report.get("samples")
    if not isinstance(samples, int) or samples < 3:
        errors.append(f"samples must be an integer >= 3, got {samples!r}")
    near = _finite_float(report.get("near"), "near", errors)
    far = _finite_float(report.get("far"), "far", errors)
    if not far > near > 0.0:
        errors.append(f"near/far must satisfy 0 < near < far, got near={near}, far={far}")
    rows = report.get("rows")
    gradient_rows = report.get("gradient_rows")
    q_gradient = report.get("q_gradient")
    summary = report.get("summary")
    if not isinstance(rows, list) or len(rows) < 5 or any(not isinstance(row, dict) for row in rows):
        errors.append("rows must contain at least five family samples")
        return errors
    if not isinstance(gradient_rows, list) or len(gradient_rows) != 3 or any(not isinstance(row, dict) for row in gradient_rows):
        errors.append("gradient_rows must contain mean, log_precision, and log_amplitude rows")
        return errors
    if not isinstance(q_gradient, dict):
        errors.append("q_gradient must be an object")
        return errors
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors

    for idx, row in enumerate(rows):
        for key in (
            "q",
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
        depth = _finite_float(row.get("depth_integral"), f"row {idx} depth_integral", errors)
        log_depth = _finite_float(row.get("log_gauge_integral"), f"row {idx} log_gauge_integral", errors)
        bad = _finite_float(row.get("bad_no_jacobian_integral"), f"row {idx} bad_no_jacobian_integral", errors)
        abs_error = _finite_float(row.get("abs_error"), f"row {idx} abs_error", errors)
        rel_error = _finite_float(row.get("rel_error"), f"row {idx} rel_error", errors)
        bad_rel_error = _finite_float(row.get("bad_no_jacobian_rel_error"), f"row {idx} bad_no_jacobian_rel_error", errors)
        denom = max(abs(depth), 1.0e-12)
        if abs(abs_error - abs(log_depth - depth)) > 1.0e-9:
            errors.append(f"row {idx} abs_error mismatch")
        if abs(rel_error - abs_error / denom) > 1.0e-9:
            errors.append(f"row {idx} rel_error mismatch")
        if abs(bad_rel_error - abs(bad - depth) / denom) > 1.0e-9:
            errors.append(f"row {idx} bad_no_jacobian_rel_error mismatch")

    expected_params = {"mean", "log_precision", "log_amplitude"}
    seen_params = {row.get("param") for row in gradient_rows}
    if seen_params != expected_params:
        errors.append(f"gradient_rows must cover {sorted(expected_params)}, got {sorted(str(param) for param in seen_params)}")
    for idx, row in enumerate(gradient_rows):
        for key in (
            "depth_grad_norm",
            "log_gauge_grad_norm",
            "bad_no_jacobian_grad_norm",
            "max_abs_error",
            "rel_error",
            "bad_no_jacobian_rel_error",
        ):
            value = _finite_float(row.get(key), f"gradient row {idx} {key}", errors)
            if key.endswith("_norm") and value <= 0.0:
                errors.append(f"gradient row {idx} {key} must be positive")

    for key in (
        "q",
        "depth_value",
        "log_gauge_value",
        "bad_no_jacobian_value",
        "depth_grad",
        "log_gauge_grad",
        "bad_no_jacobian_grad",
        "finite_difference",
        "rel_error",
        "bad_no_jacobian_rel_error",
        "finite_difference_rel_error",
    ):
        _finite_float(q_gradient.get(key), f"q_gradient {key}", errors)
    q_depth_grad = _finite_float(q_gradient.get("depth_grad"), "q_gradient depth_grad", errors)
    denom = max(abs(q_depth_grad), 1.0e-12)
    if abs(
        _finite_float(q_gradient.get("rel_error"), "q_gradient rel_error", errors)
        - abs(_finite_float(q_gradient.get("log_gauge_grad"), "q_gradient log_gauge_grad", errors) - q_depth_grad) / denom
    ) > 1.0e-9:
        errors.append("q_gradient rel_error mismatch")
    if abs(
        _finite_float(q_gradient.get("finite_difference_rel_error"), "q_gradient finite_difference_rel_error", errors)
        - abs(_finite_float(q_gradient.get("finite_difference"), "q_gradient finite_difference", errors) - q_depth_grad) / denom
    ) > 1.0e-9:
        errors.append("q_gradient finite_difference_rel_error mismatch")

    try:
        expected_summary = summarize(rows, gradient_rows, q_gradient)
        for key in expected_summary:
            _assert_summary_close(summary, expected_summary, key, errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")

    if float(summary.get("q_min") or 0.0) >= 0.0 or float(summary.get("q_max") or 0.0) <= 0.0:
        errors.append("camera-family samples must span negative and positive q")
    if float(summary.get("max_value_rel_error") or math.inf) > 2.0e-6:
        errors.append("camera-family value gauge error must stay below 2e-6")
    if float(summary.get("max_primitive_gradient_rel_error") or math.inf) > 2.0e-6:
        errors.append("camera-family primitive gradient gauge error must stay below 2e-6")
    if float(summary.get("q_gradient_rel_error") or math.inf) > 2.0e-6:
        errors.append("camera-family q gradient gauge error must stay below 2e-6")
    if float(summary.get("q_finite_difference_rel_error") or math.inf) > 2.0e-6:
        errors.append("camera-family q finite-difference error must stay below 2e-6")
    if float(summary.get("min_bad_no_jacobian_value_rel_error") or 0.0) < 0.05:
        errors.append("camera-family value missing-Jacobian control must stay visibly wrong")
    if float(summary.get("min_bad_no_jacobian_gradient_rel_error") or 0.0) < 0.05:
        errors.append("camera-family primitive gradient missing-Jacobian control must stay visibly wrong")
    if float(summary.get("q_bad_no_jacobian_rel_error") or 0.0) < 0.05:
        errors.append("camera-family q-gradient missing-Jacobian control must stay visibly wrong")
    return errors


def assert_camera_family_gauge_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_gauge_report(report)
    if errors:
        raise AssertionError("camera-family gauge report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Camera-Family Gauge",
        "",
        "This report extends the screen-fiber gauge check from one camera path to a one-parameter local camera family.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Value Rows",
        "",
        "| q | u | v | tau | rel error | bad no-Jacobian rel error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {q:.3f} | {u:.3f} | {v:.3f} | {tau:.3f} | {rel_error:.3g} | {bad_no_jacobian_rel_error:.3g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Camera-Family Gradient",
            "",
            "```json",
            json.dumps(report["q_gradient"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


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
        assert_camera_family_gauge_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(samples=args.samples, near=args.near, far=args.far)
    assert_camera_family_gauge_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
