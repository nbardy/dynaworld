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

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_shared_work_scaling import (  # noqa: E402
    Q_HEIGHT_MAX,
    Q_HEIGHT_MIN,
    Q_PHASE_MAX,
    Q_PHASE_MIN,
    _camera_family_2d_frame,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_camera_family_2d_gauge"


def _as_scalar(value: float | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float64)
    return torch.tensor(float(value), dtype=torch.float64)


def _camera_ray_world_point(
    q_phase: float | torch.Tensor,
    q_height: float | torch.Tensor,
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
    depth = depth.to(dtype=torch.float64)
    x_cam = depth * (_as_scalar(u) - cx) / fx
    y_cam = depth * (_as_scalar(v) - cy) / fy
    eye, right, up, forward = _camera_family_2d_frame(_as_scalar(q_phase), _as_scalar(q_height), _as_scalar(tau))
    return eye + x_cam.unsqueeze(-1) * right + y_cam.unsqueeze(-1) * up + depth.unsqueeze(-1) * forward


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
    q_phase: float | torch.Tensor,
    q_height: float | torch.Tensor,
    u: float | torch.Tensor,
    v: float | torch.Tensor,
    tau: float | torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    world = _camera_ray_world_point(q_phase, q_height, u, v, tau, depth)
    tau_tensor = _as_scalar(tau).expand_as(depth)
    centered = torch.cat((world, tau_tensor.unsqueeze(-1)), dim=-1) - params["mean"]
    precision = torch.exp(params["log_precision"])
    exponent = -0.5 * torch.sum(precision * centered.square(), dim=-1)
    return torch.exp(params["log_amplitude"]) * torch.exp(exponent)


def _integral(
    params: dict[str, torch.Tensor],
    *,
    q_phase: float | torch.Tensor,
    q_height: float | torch.Tensor,
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
        values = _spacetime_density(params, q_phase=q_phase, q_height=q_height, u=u, v=v, tau=tau, depth=depth)
        return torch.trapezoid(values, depth)
    if gauge == "log_depth":
        log_depth = torch.linspace(math.log(near), math.log(far), samples, dtype=torch.float64)
        depth = torch.exp(log_depth)
        values = _spacetime_density(params, q_phase=q_phase, q_height=q_height, u=u, v=v, tau=tau, depth=depth)
        if include_jacobian:
            values = values * depth
        return torch.trapezoid(values, log_depth)
    raise ValueError(f"unknown gauge {gauge!r}")


def _family_points() -> list[tuple[float, float, float, float, float, float]]:
    return [
        (-0.30, -0.20, 28.0, 31.5, 0.05, 0.90),
        (-0.12, 0.10, 32.0, 32.0, 0.25, -0.25),
        (0.00, 0.00, 36.5, 29.0, 0.50, 0.70),
        (0.16, -0.06, 30.0, 35.0, 0.75, 0.45),
        (0.30, 0.22, 39.0, 33.5, 0.95, -0.15),
    ]


def _value_row(
    *,
    q_phase: float,
    q_height: float,
    u: float,
    v: float,
    tau: float,
    samples: int,
    near: float,
    far: float,
) -> dict[str, float]:
    params = _initial_params(requires_grad=False)
    depth = _integral(
        params,
        q_phase=q_phase,
        q_height=q_height,
        u=u,
        v=v,
        tau=tau,
        samples=samples,
        near=near,
        far=far,
        gauge="depth",
    )
    log_depth = _integral(
        params,
        q_phase=q_phase,
        q_height=q_height,
        u=u,
        v=v,
        tau=tau,
        samples=samples,
        near=near,
        far=far,
        gauge="log_depth",
    )
    bad = _integral(
        params,
        q_phase=q_phase,
        q_height=q_height,
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
        "q_phase": float(q_phase),
        "q_height": float(q_height),
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
    total = torch.zeros((), dtype=torch.float64)
    for q_phase, q_height, u, v, tau, weight in _family_points():
        total = total + float(weight) * _integral(
            params,
            q_phase=q_phase,
            q_height=q_height,
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


def _camera_param_grad(
    *,
    axis: str,
    q_phase_value: float,
    q_height_value: float,
    samples: int,
    near: float,
    far: float,
    gauge: str,
    include_jacobian: bool = True,
) -> tuple[float, float]:
    params = _initial_params(requires_grad=False)
    q_phase: float | torch.Tensor = q_phase_value
    q_height: float | torch.Tensor = q_height_value
    if axis == "q_phase":
        q_phase = torch.tensor(q_phase_value, dtype=torch.float64, requires_grad=True)
        grad_inputs = (q_phase,)
    elif axis == "q_height":
        q_height = torch.tensor(q_height_value, dtype=torch.float64, requires_grad=True)
        grad_inputs = (q_height,)
    else:
        raise ValueError(f"unknown camera-family axis {axis!r}")
    value = _integral(
        params,
        q_phase=q_phase,
        q_height=q_height,
        u=34.0,
        v=30.5,
        tau=0.62,
        samples=samples,
        near=near,
        far=far,
        gauge=gauge,
        include_jacobian=include_jacobian,
    )
    (grad,) = torch.autograd.grad(value, grad_inputs)
    return float(value.detach().item()), float(grad.detach().item())


def _camera_gradient_row(
    *,
    axis: str,
    samples: int,
    near: float,
    far: float,
    epsilon: float = 1.0e-5,
) -> dict[str, float | str]:
    q_phase_value = 0.11
    q_height_value = -0.07
    depth_value, depth_grad = _camera_param_grad(
        axis=axis,
        q_phase_value=q_phase_value,
        q_height_value=q_height_value,
        samples=samples,
        near=near,
        far=far,
        gauge="depth",
    )
    log_value, log_grad = _camera_param_grad(
        axis=axis,
        q_phase_value=q_phase_value,
        q_height_value=q_height_value,
        samples=samples,
        near=near,
        far=far,
        gauge="log_depth",
    )
    bad_value, bad_grad = _camera_param_grad(
        axis=axis,
        q_phase_value=q_phase_value,
        q_height_value=q_height_value,
        samples=samples,
        near=near,
        far=far,
        gauge="log_depth",
        include_jacobian=False,
    )
    params = _initial_params(requires_grad=False)
    q_phase_minus = q_phase_value - epsilon if axis == "q_phase" else q_phase_value
    q_phase_plus = q_phase_value + epsilon if axis == "q_phase" else q_phase_value
    q_height_minus = q_height_value - epsilon if axis == "q_height" else q_height_value
    q_height_plus = q_height_value + epsilon if axis == "q_height" else q_height_value
    minus = _integral(
        params,
        q_phase=q_phase_minus,
        q_height=q_height_minus,
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
        q_phase=q_phase_plus,
        q_height=q_height_plus,
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
        "param": axis,
        "q_phase": float(q_phase_value),
        "q_height": float(q_height_value),
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
    camera_gradients: list[dict[str, float | str]],
) -> dict[str, Any]:
    rel_errors = [float(row["rel_error"]) for row in rows]
    bad_errors = [float(row["bad_no_jacobian_rel_error"]) for row in rows]
    grad_errors = [float(row["rel_error"]) for row in gradient_rows]
    bad_grad_errors = [float(row["bad_no_jacobian_rel_error"]) for row in gradient_rows]
    camera_by_axis = {str(row["param"]): row for row in camera_gradients}
    return {
        "family_point_count": len(rows),
        "q_phase_min": min(float(row["q_phase"]) for row in rows),
        "q_phase_max": max(float(row["q_phase"]) for row in rows),
        "q_height_min": min(float(row["q_height"]) for row in rows),
        "q_height_max": max(float(row["q_height"]) for row in rows),
        "max_value_rel_error": max(rel_errors),
        "mean_value_rel_error": sum(rel_errors) / float(len(rel_errors)),
        "min_bad_no_jacobian_value_rel_error": min(bad_errors),
        "max_primitive_gradient_rel_error": max(grad_errors),
        "mean_primitive_gradient_rel_error": sum(grad_errors) / float(len(grad_errors)),
        "min_bad_no_jacobian_gradient_rel_error": min(bad_grad_errors),
        "q_phase_gradient_rel_error": float(camera_by_axis["q_phase"]["rel_error"]),
        "q_phase_finite_difference_rel_error": float(camera_by_axis["q_phase"]["finite_difference_rel_error"]),
        "q_phase_bad_no_jacobian_rel_error": float(camera_by_axis["q_phase"]["bad_no_jacobian_rel_error"]),
        "q_height_gradient_rel_error": float(camera_by_axis["q_height"]["rel_error"]),
        "q_height_finite_difference_rel_error": float(camera_by_axis["q_height"]["finite_difference_rel_error"]),
        "q_height_bad_no_jacobian_rel_error": float(camera_by_axis["q_height"]["bad_no_jacobian_rel_error"]),
    }


def run_report(*, samples: int = 4097, near: float = 0.4, far: float = 5.0) -> dict[str, Any]:
    rows = [
        _value_row(q_phase=q_phase, q_height=q_height, u=u, v=v, tau=tau, samples=samples, near=near, far=far)
        for q_phase, q_height, u, v, tau, _ in _family_points()
    ]
    _, depth_grads = _loss_and_grads(samples=samples, near=near, far=far, gauge="depth")
    _, log_grads = _loss_and_grads(samples=samples, near=near, far=far, gauge="log_depth")
    _, bad_grads = _loss_and_grads(samples=samples, near=near, far=far, gauge="log_depth", include_jacobian=False)
    gradient_rows = [
        _gradient_row(name, depth_grads[name], log_grads[name], bad_grads[name])
        for name in ("mean", "log_precision", "log_amplitude")
    ]
    camera_gradients = [
        _camera_gradient_row(axis="q_phase", samples=samples, near=near, far=far),
        _camera_gradient_row(axis="q_height", samples=samples, near=near, far=far),
    ]
    report = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_gauge",
        "samples": int(samples),
        "near": float(near),
        "far": float(far),
        "base_domain": "Q2 x Omega x T",
        "q_phase_min": Q_PHASE_MIN,
        "q_phase_max": Q_PHASE_MAX,
        "q_height_min": Q_HEIGHT_MIN,
        "q_height_max": Q_HEIGHT_MAX,
        "theory_contract": "A local two-parameter camera-family trace over Q2 x Omega x T remains pi_* Gamma(q_phase,q_height)^* world_primitive; monotone fiber gauge changes preserve values and derivatives when the Jacobian is included.",
        "rows": rows,
        "gradient_rows": gradient_rows,
        "camera_gradients": camera_gradients,
        "summary": summarize(rows, gradient_rows, camera_gradients),
    }
    errors = verify_camera_family_2d_gauge_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _assert_close(actual: float, expected: float, label: str, errors: list[str], *, atol: float = 1.0e-9) -> None:
    if abs(float(actual) - float(expected)) > atol:
        errors.append(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _assert_summary_close(summary: dict[str, Any], expected: dict[str, Any], key: str, errors: list[str]) -> None:
    actual = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    elif actual != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")


def verify_camera_family_2d_gauge_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_gauge":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T":
        errors.append(f"base_domain must be Q2 x Omega x T, got {report.get('base_domain')!r}")
    if not isinstance(report.get("theory_contract"), str) or "Gamma(q_phase,q_height)" not in report["theory_contract"]:
        errors.append("theory_contract must mention Gamma(q_phase,q_height)")
    samples = report.get("samples")
    if not isinstance(samples, int) or samples < 3:
        errors.append(f"samples must be an integer >= 3, got {samples!r}")
    near = _finite_float(report.get("near"), "near", errors)
    far = _finite_float(report.get("far"), "far", errors)
    if not far > near > 0.0:
        errors.append(f"near/far must satisfy 0 < near < far, got near={near}, far={far}")
    for key in ("q_phase_min", "q_phase_max", "q_height_min", "q_height_max"):
        _finite_float(report.get(key), key, errors)
    rows = report.get("rows")
    gradient_rows = report.get("gradient_rows")
    camera_gradients = report.get("camera_gradients")
    summary = report.get("summary")
    if not isinstance(rows, list) or len(rows) < 5 or any(not isinstance(row, dict) for row in rows):
        errors.append("rows must contain at least five Q2 family samples")
        return errors
    if not isinstance(gradient_rows, list) or len(gradient_rows) != 3 or any(not isinstance(row, dict) for row in gradient_rows):
        errors.append("gradient_rows must contain mean, log_precision, and log_amplitude rows")
        return errors
    if not isinstance(camera_gradients, list) or len(camera_gradients) != 2 or any(not isinstance(row, dict) for row in camera_gradients):
        errors.append("camera_gradients must contain q_phase and q_height rows")
        return errors
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors

    for idx, row in enumerate(rows):
        for key in (
            "q_phase",
            "q_height",
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
        _assert_close(abs_error, abs(log_depth - depth), f"row {idx} abs_error", errors)
        _assert_close(rel_error, abs_error / denom, f"row {idx} rel_error", errors)
        _assert_close(bad_rel_error, abs(bad - depth) / denom, f"row {idx} bad_no_jacobian_rel_error", errors)

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

    seen_camera_params = {row.get("param") for row in camera_gradients}
    if seen_camera_params != {"q_phase", "q_height"}:
        errors.append(f"camera_gradients must cover q_phase and q_height, got {seen_camera_params}")
    for idx, row in enumerate(camera_gradients):
        for key in (
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
            _finite_float(row.get(key), f"camera gradient row {idx} {key}", errors)
        depth_grad = _finite_float(row.get("depth_grad"), f"camera gradient row {idx} depth_grad", errors)
        log_grad = _finite_float(row.get("log_gauge_grad"), f"camera gradient row {idx} log_gauge_grad", errors)
        bad_grad = _finite_float(row.get("bad_no_jacobian_grad"), f"camera gradient row {idx} bad_no_jacobian_grad", errors)
        finite_difference = _finite_float(row.get("finite_difference"), f"camera gradient row {idx} finite_difference", errors)
        denom = max(abs(depth_grad), 1.0e-12)
        _assert_close(
            _finite_float(row.get("rel_error"), f"camera gradient row {idx} rel_error", errors),
            abs(log_grad - depth_grad) / denom,
            f"camera gradient row {idx} rel_error",
            errors,
        )
        _assert_close(
            _finite_float(row.get("bad_no_jacobian_rel_error"), f"camera gradient row {idx} bad_no_jacobian_rel_error", errors),
            abs(bad_grad - depth_grad) / denom,
            f"camera gradient row {idx} bad_no_jacobian_rel_error",
            errors,
        )
        _assert_close(
            _finite_float(row.get("finite_difference_rel_error"), f"camera gradient row {idx} finite_difference_rel_error", errors),
            abs(finite_difference - depth_grad) / denom,
            f"camera gradient row {idx} finite_difference_rel_error",
            errors,
        )

    try:
        expected_summary = summarize(rows, gradient_rows, camera_gradients)
        for key in expected_summary:
            _assert_summary_close(summary, expected_summary, key, errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")

    if float(summary.get("max_value_rel_error") or math.inf) > 2.0e-6:
        errors.append("2D camera-family value gauge error must stay below 2e-6")
    if float(summary.get("max_primitive_gradient_rel_error") or math.inf) > 2.0e-6:
        errors.append("2D camera-family primitive gradient gauge error must stay below 2e-6")
    for axis in ("q_phase", "q_height"):
        if float(summary.get(f"{axis}_gradient_rel_error") or math.inf) > 2.0e-6:
            errors.append(f"2D camera-family {axis} gradient gauge error must stay below 2e-6")
        if float(summary.get(f"{axis}_finite_difference_rel_error") or math.inf) > 2.0e-6:
            errors.append(f"2D camera-family {axis} finite-difference check must stay below 2e-6")
        if float(summary.get(f"{axis}_bad_no_jacobian_rel_error") or 0.0) < 0.05:
            errors.append(f"2D camera-family {axis} missing-Jacobian control must stay visibly wrong")
    if float(summary.get("min_bad_no_jacobian_value_rel_error") or 0.0) < 0.05:
        errors.append("2D camera-family value missing-Jacobian control must stay visibly wrong")
    if float(summary.get("min_bad_no_jacobian_gradient_rel_error") or 0.0) < 0.05:
        errors.append("2D camera-family primitive-gradient missing-Jacobian control must stay visibly wrong")
    return errors


def assert_camera_family_2d_gauge_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_gauge_report(report)
    if errors:
        raise AssertionError("2D camera-family gauge report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective 2D Camera-Family Gauge Report",
        "",
        "This report extends the screen-fiber gauge check from one camera-family coordinate to two.",
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
    parser.add_argument("--samples", type=int, default=4097)
    parser.add_argument("--near", type=float, default=0.4)
    parser.add_argument("--far", type=float, default=5.0)
    parser.add_argument("--verify-report", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_camera_family_2d_gauge_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(samples=int(args.samples), near=float(args.near), far=float(args.far))
    assert_camera_family_2d_gauge_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
