"""Run the bounded M0--M5 fixed-segment foundation gate.

CPU float64 validation is the default.  The optional Metal path is deliberately
limited to the same small segment records and requires an explicit safety flag;
it is mechanical parity evidence, never a training or paper benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .finite_element_material_metal import FiniteElementMaterialMetal, SOURCE_PATH
from .finite_element_material_transfer import (
    MaterialMode,
    branch_status_counts,
    evaluate_material_segment,
    material_segment_vjp,
)


DTYPE = torch.float64
CPU_INTEGRAL_TOLERANCE = 1.0e-10
CPU_VJP_TOLERANCE = 1.0e-5
CPU_FINITE_DIFFERENCE_VJP_TOLERANCE = 1.0e-7
METAL_FORWARD_TOLERANCE = 1.0e-5
METAL_VJP_TOLERANCE = 1.0e-4


def _cases() -> list[tuple[MaterialMode, str, list[float]]]:
    return [
        (MaterialMode.M0_P0_CONSTANT, "direct", [0.7, 0.0, 0.0]),
        (MaterialMode.M0_P0_CONSTANT, "tiny_tau", [2.0e-5, 0.0, 0.0]),
        (MaterialMode.M1_P0_AFFINE_RGB, "direct", [0.7, 0.0, 0.0]),
        (MaterialMode.M2_POSITIVE_BERNSTEIN_P1, "direct", [0.25, 1.1, 0.0]),
        (MaterialMode.M3_POSITIVE_BERNSTEIN_P2, "direct", [0.2, 1.4, 0.5]),
        (MaterialMode.M4_LOG_P1, "series", [1.0e-7, -0.2, 0.0]),
        (MaterialMode.M4_LOG_P1, "ordinary", [0.9, -0.2, 0.0]),
        (MaterialMode.M4_LOG_P1, "scaled_endpoints", [-100.0, 100.0, 0.0]),
        (MaterialMode.M5_CONVEX_LOG_P2, "series", [1.0e-4, 0.3, -0.1]),
        (MaterialMode.M5_CONVEX_LOG_P2, "erf", [0.8, -0.35, 0.1]),
        (MaterialMode.M5_CONVEX_LOG_P2, "tail", [3.0, 28.0, 1.0]),
        (MaterialMode.M5_CONVEX_LOG_P2, "sharp_interior", [1000.0, -1000.0, 250.0]),
    ]


def _inputs(controls: list[float]) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(controls, dtype=DTYPE),
        torch.tensor(1.3, dtype=DTYPE),
        torch.tensor([0.2, 0.7, 0.4], dtype=DTYPE),
        torch.tensor([0.8, 0.1, 0.6], dtype=DTYPE),
    )


def _density(
    mode: MaterialMode,
    controls: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        return torch.ones_like(x) * controls[0]
    if mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
        return controls[0] * (1.0 - x) + controls[1] * x
    if mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        return (
            controls[0] * (1.0 - x).square()
            + 2.0 * controls[1] * x * (1.0 - x)
            + controls[2] * x.square()
        )
    if mode == MaterialMode.M4_LOG_P1:
        return torch.exp(-(controls[0] * x + controls[1]))
    return torch.exp(
        -(controls[0] * x.square() + controls[1] * x + controls[2])
    )


def _quadrature_reference(
    mode: MaterialMode,
    controls: torch.Tensor,
    length: torch.Tensor,
    color_front: torch.Tensor,
    color_back: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = _density(mode, controls, nodes)
    tau = length * torch.dot(weights, sigma)
    beta = torch.exp(-tau)
    if mode == MaterialMode.M1_P0_AFFINE_RGB:
        color = (
            color_front[None, :] * (1.0 - nodes[:, None])
            + color_back[None, :] * nodes[:, None]
        )
        m = torch.sum(
            weights[:, None]
            * length
            * sigma[:, None]
            * torch.exp(-length * controls[0] * nodes)[:, None]
            * color,
            dim=0,
        )
    else:
        m = (1.0 - beta) * color_front
    return tau, beta, m, sigma


def _normalized_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.maximum(torch.ones_like(expected), expected.abs())
    return float(((actual - expected).abs() / denominator).max().detach())


def _objective(
    mode: MaterialMode,
    values: list[torch.Tensor],
    grad_tau: torch.Tensor,
    grad_beta: torch.Tensor,
    grad_m: torch.Tensor,
) -> torch.Tensor:
    forward = evaluate_material_segment(mode, *values)
    return (
        grad_tau * forward.tau
        + grad_beta * forward.element.beta
        + torch.dot(grad_m, forward.element.m)
    )


def _central_difference_vjp(
    mode: MaterialMode,
    values: tuple[torch.Tensor, ...],
    grad_tau: torch.Tensor,
    grad_beta: torch.Tensor,
    grad_m: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    gradients = []
    for argument_index, argument in enumerate(values):
        gradient = torch.empty_like(argument)
        for scalar_index in range(argument.numel()):
            plus = [value.detach().clone() for value in values]
            minus = [value.detach().clone() for value in values]
            center = float(argument.detach().reshape(-1)[scalar_index])
            step = 1.0e-6 * max(1.0, abs(center))
            plus[argument_index].reshape(-1)[scalar_index] += step
            minus[argument_index].reshape(-1)[scalar_index] -= step
            gradient.reshape(-1)[scalar_index] = (
                _objective(mode, plus, grad_tau, grad_beta, grad_m)
                - _objective(mode, minus, grad_tau, grad_beta, grad_m)
            ) / (2.0 * step)
        gradients.append(gradient)
    return tuple(gradients)


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def build_cpu_report(*, quadrature_order: int = 128) -> tuple[dict[str, Any], dict[str, Any]]:
    np_nodes, np_weights = np.polynomial.legendre.leggauss(quadrature_order)
    nodes = torch.tensor(0.5 * (np_nodes + 1.0), dtype=DTYPE)
    weights = torch.tensor(0.5 * np_weights, dtype=DTYPE)
    grad_tau = torch.tensor(0.31, dtype=DTYPE)
    grad_beta = torch.tensor(-0.23, dtype=DTYPE)
    grad_m = torch.tensor([0.2, -0.4, 0.5], dtype=DTYPE)

    records: list[dict[str, Any]] = []
    statuses: list[int] = []
    metal_inputs: dict[str, list[torch.Tensor | int]] = {
        "controls": [],
        "lengths": [],
        "color_front": [],
        "color_back": [],
        "modes": [],
        "grad_tau": [],
        "grad_beta": [],
        "grad_m": [],
    }
    maximum_integral_error = 0.0
    maximum_vjp_error = 0.0
    maximum_finite_difference_vjp_error = 0.0
    maximum_bound_violation = 0.0
    started = time.perf_counter()
    for mode, branch_label, raw_controls in _cases():
        controls, length, color_front, color_back = [
            value.clone().requires_grad_(True) for value in _inputs(raw_controls)
        ]
        forward = evaluate_material_segment(
            mode, controls, length, color_front, color_back
        )
        reference_tau, reference_beta, reference_m, sampled_density = (
            _quadrature_reference(
                mode,
                controls,
                length,
                color_front,
                color_back,
                nodes,
                weights,
            )
        )
        integral_error = max(
            float((forward.tau - reference_tau).abs().detach()),
            float((forward.element.beta - reference_beta).abs().detach()),
            float((forward.element.m - reference_m).abs().max().detach()),
        )
        maximum_integral_error = max(maximum_integral_error, integral_error)
        lower_violation = float(
            torch.clamp(
                forward.density_bounds[0] - sampled_density.min(), min=0.0
            ).detach()
        )
        upper_violation = float(
            torch.clamp(
                sampled_density.max() - forward.density_bounds[1], min=0.0
            ).detach()
        )
        bound_violation = max(lower_violation, upper_violation)
        maximum_bound_violation = max(maximum_bound_violation, bound_violation)

        loss = (
            grad_tau * forward.tau
            + grad_beta * forward.element.beta
            + torch.dot(grad_m, forward.element.m)
        )
        autograd_raw = torch.autograd.grad(
            loss,
            (controls, color_front, color_back, length),
            allow_unused=True,
        )
        autograd_values = tuple(
            torch.zeros_like(value) if gradient is None else gradient
            for gradient, value in zip(
                autograd_raw,
                (controls, color_front, color_back, length),
                strict=True,
            )
        )
        explicit = material_segment_vjp(
            mode,
            controls,
            length,
            color_front,
            color_back,
            grad_tau=grad_tau,
            grad_beta=grad_beta,
            grad_m=grad_m,
        )
        explicit_values = (
            explicit.density_controls,
            explicit.color_front,
            explicit.color_back,
            explicit.length,
        )
        vjp_error = max(
            _normalized_error(actual, expected)
            for actual, expected in zip(
                explicit_values, autograd_values, strict=True
            )
        )
        maximum_vjp_error = max(maximum_vjp_error, vjp_error)
        finite_difference = _central_difference_vjp(
            mode,
            (controls, length, color_front, color_back),
            grad_tau,
            grad_beta,
            grad_m,
        )
        finite_difference_values = (
            finite_difference[0],
            finite_difference[2],
            finite_difference[3],
            finite_difference[1],
        )
        finite_difference_vjp_error = max(
            _normalized_error(actual, expected)
            for actual, expected in zip(
                explicit_values,
                finite_difference_values,
                strict=True,
            )
        )
        maximum_finite_difference_vjp_error = max(
            maximum_finite_difference_vjp_error,
            finite_difference_vjp_error,
        )
        statuses.append(int(forward.branch_status))
        records.append(
            {
                "mode": mode.name,
                "branch_fixture": branch_label,
                "controls": raw_controls,
                "tau": float(forward.tau.detach()),
                "density_bounds": [
                    float(value) for value in forward.density_bounds.detach()
                ],
                "branch_status": int(forward.branch_status),
                "quadrature_max_abs_error": integral_error,
                "vjp_normalized_error": vjp_error,
                "finite_difference_vjp_normalized_error": finite_difference_vjp_error,
                "density_bound_violation": bound_violation,
            }
        )
        for key, value in (
            ("controls", controls.detach().float()),
            ("lengths", length.detach().float()),
            ("color_front", color_front.detach().float()),
            ("color_back", color_back.detach().float()),
            ("modes", int(mode)),
            ("grad_tau", grad_tau.float()),
            ("grad_beta", grad_beta.float()),
            ("grad_m", grad_m.float()),
        ):
            metal_inputs[key].append(value)

    elapsed = time.perf_counter() - started
    branch_counts = branch_status_counts(statuses)
    report = {
        "device": "cpu",
        "dtype": "float64",
        "quadrature": {
            "rule": "independent_numpy_gauss_legendre",
            "order": quadrature_order,
        },
        "segment_count": len(records),
        "evaluation_seconds": elapsed,
        "max_integral_abs_error": maximum_integral_error,
        "max_vjp_normalized_error": maximum_vjp_error,
        "max_finite_difference_vjp_normalized_error": (
            maximum_finite_difference_vjp_error
        ),
        "max_density_bound_violation": maximum_bound_violation,
        "branch_counts": branch_counts,
        "records": records,
        "gate": {
            "integral_tolerance": CPU_INTEGRAL_TOLERANCE,
            "vjp_tolerance": CPU_VJP_TOLERANCE,
            "finite_difference_vjp_tolerance": (
                CPU_FINITE_DIFFERENCE_VJP_TOLERANCE
            ),
            "passed": (
                maximum_integral_error <= CPU_INTEGRAL_TOLERANCE
                and maximum_vjp_error <= CPU_VJP_TOLERANCE
                and maximum_finite_difference_vjp_error
                <= CPU_FINITE_DIFFERENCE_VJP_TOLERANCE
                and maximum_bound_violation == 0.0
                and branch_counts["small_tau_series"] > 0
            ),
        },
    }
    return report, metal_inputs


def build_metal_report(
    metal_inputs: dict[str, Any],
    cpu_report: dict[str, Any],
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    wrapper = FiniteElementMaterialMetal()
    controls = torch.stack(metal_inputs["controls"]).to("mps")
    lengths = torch.stack(metal_inputs["lengths"]).to("mps")
    color_front = torch.stack(metal_inputs["color_front"]).to("mps")
    color_back = torch.stack(metal_inputs["color_back"]).to("mps")
    modes = torch.tensor(metal_inputs["modes"], dtype=torch.int32, device="mps")
    grad_tau = torch.stack(metal_inputs["grad_tau"]).to("mps")
    grad_beta = torch.stack(metal_inputs["grad_beta"]).to("mps")
    grad_m = torch.stack(metal_inputs["grad_m"]).to("mps")

    torch.mps.synchronize()
    started = time.perf_counter()
    forward = wrapper.forward(controls, lengths, color_front, color_back, modes)
    vjp = wrapper.vjp(
        controls,
        lengths,
        color_front,
        color_back,
        modes,
        grad_tau,
        grad_beta,
        grad_m,
    )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started

    cpu_forward = {
        "tau": torch.tensor([record["tau"] for record in cpu_report["records"]]),
        "density_bounds": torch.tensor(
            [record["density_bounds"] for record in cpu_report["records"]]
        ),
    }
    expected_full = []
    for mode, _, controls_values in _cases():
        values = _inputs(controls_values)
        expected_full.append(evaluate_material_segment(mode, *values))
    cpu_forward["beta"] = torch.stack(
        [value.element.beta.float() for value in expected_full]
    )
    cpu_forward["m"] = torch.stack(
        [value.element.m.float() for value in expected_full]
    )
    forward_error = max(
        _normalized_error(forward[key].cpu(), expected)
        for key, expected in cpu_forward.items()
    )

    expected_vjp = []
    for index, ((mode, _, controls_values), values) in enumerate(
        zip(_cases(), (_inputs(case[2]) for case in _cases()), strict=True)
    ):
        expected_vjp.append(
            material_segment_vjp(
                mode,
                *values,
                grad_tau=metal_inputs["grad_tau"][index].double(),
                grad_beta=metal_inputs["grad_beta"][index].double(),
                grad_m=metal_inputs["grad_m"][index].double(),
            )
        )
    expected_vjp_tensors = {
        "density_controls": torch.stack(
            [value.density_controls.float() for value in expected_vjp]
        ),
        "color_front": torch.stack(
            [value.color_front.float() for value in expected_vjp]
        ),
        "color_back": torch.stack(
            [value.color_back.float() for value in expected_vjp]
        ),
        "length": torch.stack([value.length.float() for value in expected_vjp]),
    }
    vjp_error = max(
        _normalized_error(vjp[key].cpu(), expected)
        for key, expected in expected_vjp_tensors.items()
    )
    branch_counts = wrapper.count_branches(forward["status"])
    return {
        "device": "mps",
        "dtype": "float32",
        "segment_count": int(lengths.numel()),
        "compile_forward_vjp_seconds": elapsed,
        "max_forward_normalized_error": forward_error,
        "max_vjp_normalized_error": vjp_error,
        "branch_counts": branch_counts,
        "status_matches_vjp": bool(
            torch.equal(forward["status"].cpu(), vjp["status"].cpu())
        ),
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "gate": {
            "forward_tolerance": METAL_FORWARD_TOLERANCE,
            "vjp_tolerance": METAL_VJP_TOLERANCE,
            "passed": (
                forward_error <= METAL_FORWARD_TOLERANCE
                and vjp_error <= METAL_VJP_TOLERANCE
                and bool(torch.equal(forward["status"].cpu(), vjp["status"].cpu()))
                and branch_counts["small_tau_series"] > 0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument("--run-metal", action="store_true")
    parser.add_argument(
        "--allow-local-mps-mechanical-only",
        action="store_true",
        help="Required with --run-metal; authorizes only the tiny fixed-segment parity fixture.",
    )
    args = parser.parse_args()
    if args.quadrature_order < 32:
        parser.error("--quadrature-order must be at least 32")
    if args.run_metal and not args.allow_local_mps_mechanical_only:
        parser.error("--run-metal requires --allow-local-mps-mechanical-only")

    root = Path(__file__).resolve().parents[2]
    cpu_report, metal_inputs = build_cpu_report(
        quadrature_order=args.quadrature_order
    )
    payload: dict[str, Any] = {
        "schema_version": "worldfoam.material_foundation_gate.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(root),
        "seed": 0,
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "material_modes": [mode.name for mode in MaterialMode],
        "coefficient_convention": (
            "normalized xi in [0,1]; direct modes use Bernstein controls; "
            "log modes use negative-log power coefficients"
        ),
        "gauge_length_convention": "L is physical ray length after gauge Jacobian",
        "tape_dimensions": {"segments": len(_cases()), "density_slots": 3, "rgb": 3},
        "synchronized_timing": bool(args.run_metal),
        "cpu": cpu_report,
        "metal": None,
        "claim_scope": [
            "local fixed-segment material correctness only",
            "not trained image quality",
            "not renderer throughput",
            "not native-4D parameter or event scaling",
        ],
    }
    if args.run_metal:
        payload["metal"] = build_metal_report(metal_inputs, cpu_report)
    payload["passed"] = bool(cpu_report["gate"]["passed"]) and (
        payload["metal"] is None or bool(payload["metal"]["gate"]["passed"])
    )

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(args.output)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
