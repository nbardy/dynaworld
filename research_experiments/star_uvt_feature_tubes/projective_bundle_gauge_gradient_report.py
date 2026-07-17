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

from research_experiments.star_uvt_feature_tubes.projective_bundle_gauge_invariance_report import (
    _camera_ray_world_point,
    _assert_close,
    _assert_summary_close,
    _finite_float,
    _trapz,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_bundle_gauge_gradient"
)


def _initial_params(*, requires_grad: bool) -> dict[str, torch.Tensor]:
    params = {
        "mean": torch.tensor([0.08, 0.03, -0.02, 0.47], dtype=torch.float64),
        "log_precision": torch.log(torch.tensor([8.0, 11.0, 6.5, 3.0], dtype=torch.float64)),
        "log_amplitude": torch.tensor(0.2, dtype=torch.float64),
    }
    if requires_grad:
        return {key: value.clone().detach().requires_grad_(True) for key, value in params.items()}
    return {key: value.clone().detach() for key, value in params.items()}


def _spacetime_density_with_params(
    u: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    depth: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    world = _camera_ray_world_point(u, v, tau, depth)
    centered = torch.cat((world, tau.expand_as(depth).unsqueeze(-1)), dim=-1) - params["mean"]
    precision_diag = torch.exp(params["log_precision"])
    exponent = -0.5 * torch.sum(precision_diag * centered.square(), dim=-1)
    return torch.exp(params["log_amplitude"]) * torch.exp(exponent)


def _integral_for_sensor_point(
    params: dict[str, torch.Tensor],
    *,
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
        values = _spacetime_density_with_params(torch.tensor(u), torch.tensor(v), torch.tensor(tau), depth, params)
        return _trapz(values, depth)
    if gauge == "log_depth":
        log_depth = torch.linspace(math.log(near), math.log(far), samples, dtype=torch.float64)
        depth = torch.exp(log_depth)
        values = _spacetime_density_with_params(torch.tensor(u), torch.tensor(v), torch.tensor(tau), depth, params)
        if include_jacobian:
            values = values * depth
        return _trapz(values, log_depth)
    raise ValueError(f"unknown gauge {gauge!r}")


def _objective(
    params: dict[str, torch.Tensor],
    *,
    samples: int,
    near: float,
    far: float,
    gauge: str,
    include_jacobian: bool = True,
) -> torch.Tensor:
    sensor_points = [
        (28.0, 31.5, 0.05, 1.0),
        (32.0, 32.0, 0.25, -0.25),
        (36.5, 29.0, 0.50, 0.7),
        (30.0, 35.0, 0.75, 0.45),
        (39.0, 33.5, 0.95, -0.15),
    ]
    total = torch.zeros((), dtype=torch.float64)
    for u, v, tau, weight in sensor_points:
        total = total + float(weight) * _integral_for_sensor_point(
            params,
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


def _finite_difference_mean_x(*, samples: int, near: float, far: float, epsilon: float) -> dict[str, float]:
    params = _initial_params(requires_grad=True)
    loss = _objective(params, samples=samples, near=near, far=far, gauge="depth")
    (mean_grad,) = torch.autograd.grad(loss, (params["mean"],))

    minus_params = _initial_params(requires_grad=False)
    plus_params = _initial_params(requires_grad=False)
    minus_params["mean"][0] -= epsilon
    plus_params["mean"][0] += epsilon
    minus_loss = _objective(minus_params, samples=samples, near=near, far=far, gauge="depth")
    plus_loss = _objective(plus_params, samples=samples, near=near, far=far, gauge="depth")
    finite_diff = (plus_loss - minus_loss) / (2.0 * epsilon)
    autograd_value = mean_grad[0]
    abs_error = (finite_diff - autograd_value).abs()
    rel_error = abs_error / autograd_value.abs().clamp_min(1.0e-12)
    return {
        "param": "mean[0]",
        "epsilon": float(epsilon),
        "autograd": float(autograd_value.detach().item()),
        "finite_difference": float(finite_diff.detach().item()),
        "abs_error": float(abs_error.detach().item()),
        "rel_error": float(rel_error.detach().item()),
    }


def summarize(
    value_depth: float,
    value_log: float,
    value_bad: float,
    rows: list[dict[str, float | str]],
    finite_difference: dict[str, float],
) -> dict[str, Any]:
    value_denom = max(abs(value_depth), 1.0e-12)
    good_grad_errors = [float(row["rel_error"]) for row in rows]
    bad_grad_errors = [float(row["bad_no_jacobian_rel_error"]) for row in rows]
    return {
        "value_depth": float(value_depth),
        "value_log_gauge": float(value_log),
        "value_bad_no_jacobian": float(value_bad),
        "value_rel_error": abs(value_log - value_depth) / value_denom,
        "bad_value_rel_error": abs(value_bad - value_depth) / value_denom,
        "max_gradient_rel_error": max(good_grad_errors),
        "mean_gradient_rel_error": sum(good_grad_errors) / float(len(good_grad_errors)),
        "min_bad_no_jacobian_gradient_rel_error": min(bad_grad_errors),
        "mean_bad_no_jacobian_gradient_rel_error": sum(bad_grad_errors) / float(len(bad_grad_errors)),
        "finite_difference_mean_x_rel_error": float(finite_difference["rel_error"]),
    }


def run_report(*, samples: int = 4097, near: float = 0.4, far: float = 5.0) -> dict[str, Any]:
    value_depth, depth_grads = _loss_and_grads(samples=samples, near=near, far=far, gauge="depth")
    value_log, log_grads = _loss_and_grads(samples=samples, near=near, far=far, gauge="log_depth")
    value_bad, bad_grads = _loss_and_grads(
        samples=samples,
        near=near,
        far=far,
        gauge="log_depth",
        include_jacobian=False,
    )
    rows = [
        _gradient_row(name, depth_grads[name], log_grads[name], bad_grads[name])
        for name in ("mean", "log_precision", "log_amplitude")
    ]
    finite_difference = _finite_difference_mean_x(samples=samples, near=near, far=far, epsilon=1.0e-5)
    report = {
        "status": "ok",
        "benchmark": "star_uvt_projective_bundle_gauge_gradient",
        "samples": int(samples),
        "near": float(near),
        "far": float(far),
        "theory_contract": "Primitive gradients of pi_* Gamma^* rho are invariant under monotone fiber gauge changes when the fiber-measure Jacobian is included.",
        "rows": rows,
        "finite_difference": finite_difference,
        "summary": summarize(value_depth, value_log, value_bad, rows, finite_difference),
    }
    errors = verify_bundle_gauge_gradient_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def verify_bundle_gauge_gradient_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_bundle_gauge_gradient":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    samples = report.get("samples")
    if not isinstance(samples, int) or samples < 3:
        errors.append(f"samples must be an integer >= 3, got {samples!r}")
    near = _finite_float(report.get("near"), "near", errors)
    far = _finite_float(report.get("far"), "far", errors)
    if not far > near > 0.0:
        errors.append(f"near/far must satisfy 0 < near < far, got near={near}, far={far}")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        errors.append("rows must contain mean, log_precision, and log_amplitude gradient summaries")
        return errors
    if any(not isinstance(row, dict) for row in rows):
        errors.append("all rows must be objects")
        return errors
    finite_difference = report.get("finite_difference")
    if not isinstance(finite_difference, dict):
        errors.append("finite_difference must be an object")
        return errors
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors

    for key in ("value_depth", "value_log_gauge", "value_bad_no_jacobian"):
        _finite_float(summary.get(key), f"summary {key}", errors)

    for key in ("epsilon", "autograd", "finite_difference", "abs_error", "rel_error"):
        _finite_float(finite_difference.get(key), f"finite_difference {key}", errors)
    epsilon = _finite_float(finite_difference.get("epsilon"), "finite_difference epsilon", errors)
    if epsilon <= 0.0:
        errors.append(f"finite_difference epsilon must be positive, got {epsilon}")
    fd_autograd = _finite_float(finite_difference.get("autograd"), "finite_difference autograd", errors)
    fd_value = _finite_float(finite_difference.get("finite_difference"), "finite_difference finite_difference", errors)
    fd_abs_error = _finite_float(finite_difference.get("abs_error"), "finite_difference abs_error", errors)
    fd_rel_error = _finite_float(finite_difference.get("rel_error"), "finite_difference rel_error", errors)
    _assert_close(fd_abs_error, abs(fd_value - fd_autograd), "finite_difference abs_error", errors)
    _assert_close(
        fd_rel_error,
        fd_abs_error / max(abs(fd_autograd), 1.0e-12),
        "finite_difference rel_error",
        errors,
    )

    expected_params = {"mean", "log_precision", "log_amplitude"}
    seen_params = {row.get("param") for row in rows}
    if seen_params != expected_params:
        errors.append(f"rows must cover {sorted(expected_params)}, got {sorted(str(param) for param in seen_params)}")
    for idx, row in enumerate(rows):
        for key in (
            "depth_grad_norm",
            "log_gauge_grad_norm",
            "bad_no_jacobian_grad_norm",
            "max_abs_error",
            "rel_error",
            "bad_no_jacobian_rel_error",
        ):
            _finite_float(row.get(key), f"row {idx} {key}", errors)
        for key in ("depth_grad_norm", "log_gauge_grad_norm", "bad_no_jacobian_grad_norm"):
            if _finite_float(row.get(key), f"row {idx} {key}", errors) <= 0.0:
                errors.append(f"row {idx} {key} must be positive")

    try:
        expected_summary = summarize(
            _finite_float(summary.get("value_depth"), "summary value_depth", errors),
            _finite_float(summary.get("value_log_gauge"), "summary value_log_gauge", errors),
            _finite_float(summary.get("value_bad_no_jacobian"), "summary value_bad_no_jacobian", errors),
            rows,
            finite_difference,
        )
        for key in expected_summary:
            _assert_summary_close(summary, expected_summary, key, errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")

    max_grad_error = float(summary.get("max_gradient_rel_error") or math.inf)
    if not math.isfinite(max_grad_error) or max_grad_error > 2.0e-6:
        errors.append(f"max_gradient_rel_error must be <= 2e-6, got {max_grad_error}")
    value_error = float(summary.get("value_rel_error") or math.inf)
    if not math.isfinite(value_error) or value_error > 2.0e-6:
        errors.append(f"value_rel_error must be <= 2e-6, got {value_error}")
    min_bad_grad = float(summary.get("min_bad_no_jacobian_gradient_rel_error") or 0.0)
    if not math.isfinite(min_bad_grad) or min_bad_grad < 0.05:
        errors.append(f"min_bad_no_jacobian_gradient_rel_error must expose missing-Jacobian failure, got {min_bad_grad}")
    bad_value = float(summary.get("bad_value_rel_error") or 0.0)
    if not math.isfinite(bad_value) or bad_value < 0.05:
        errors.append(f"bad_value_rel_error must expose missing-Jacobian failure, got {bad_value}")
    finite_diff_error = float(summary.get("finite_difference_mean_x_rel_error") or math.inf)
    if not math.isfinite(finite_diff_error) or finite_diff_error > 1.0e-6:
        errors.append(f"finite_difference_mean_x_rel_error must be <= 1e-6, got {finite_diff_error}")
    return errors


def assert_bundle_gauge_gradient_report(report: dict[str, Any]) -> None:
    errors = verify_bundle_gauge_gradient_report(report)
    if errors:
        raise AssertionError("\n".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Bundle Gauge Gradient",
        "",
        "This report checks that primitive gradients of the fiber pushforward survive a depth-gauge change.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Gradient Rows",
        "",
        "| param | depth_grad_norm | log_gauge_grad_norm | rel_error | bad_no_jacobian_rel_error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {param} | {depth_grad_norm:.9g} | {log_gauge_grad_norm:.9g} | {rel_error:.3g} | {bad_no_jacobian_rel_error:.3g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Finite Difference",
            "",
            "```json",
            json.dumps(report["finite_difference"], indent=2, sort_keys=True),
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
        assert_bundle_gauge_gradient_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(samples=args.samples, near=args.near, far=args.far)
    assert_bundle_gauge_gradient_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
