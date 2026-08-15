"""Run the tiny retained-depth SPD(4) CPU/Metal forward and VJP gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch

from .retained_fiber_metal import RetainedFiberMetal
from .retained_fiber_transfer import render_retained_fiber_reference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "foundation_gates" / "spd4_retained_fiber_cpu_metal.json"
)


def _scene() -> list[torch.Tensor]:
    return [
        torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32),
        torch.tensor(
            [
                [0.8, 0.0, 0.0, 0.8, 0.0, 1.0],
                [0.8, 0.0, 0.0, 0.8, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([-0.18, 0.16], dtype=torch.float32),
        torch.tensor(
            [[0.03, -0.02, 0.04], [-0.01, 0.025, -0.03]],
            dtype=torch.float32,
        ),
        torch.tensor([0.20, 0.16], dtype=torch.float32),
        torch.tensor([0.72, 0.61], dtype=torch.float32),
        torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.25, 0.95]], dtype=torch.float32),
        torch.tensor([-0.25], dtype=torch.float32),
    ]


def _normalized_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.maximum(torch.ones_like(expected), expected.abs())
    return float(((actual - expected).abs() / denominator).max())


def run_gate() -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("retained-fiber Metal gate requires MPS")
    cpu_values = _scene()
    differentiable = [
        value.clone().requires_grad_(index < 7)
        for index, value in enumerate(cpu_values)
    ]
    reference = render_retained_fiber_reference(
        *differentiable,
        height=2,
        width=2,
        depth_samples=32,
    )
    grad_output = torch.tensor(
        [
            [
                [[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]],
                [[-0.3, 0.2, 0.1], [0.25, -0.15, 0.35]],
            ]
        ],
        dtype=torch.float32,
    )
    reference_gradients = torch.autograd.grad(
        torch.sum(reference * grad_output),
        differentiable[:7],
    )

    dense = render_retained_fiber_reference(
        *cpu_values,
        height=2,
        width=2,
        depth_samples=1024,
    )
    quadrature_error = float((reference.detach() - dense).abs().max())

    torch.mps.empty_cache()
    mps_values = [value.to("mps").contiguous() for value in cpu_values]
    metal = RetainedFiberMetal()
    actual = metal.forward(
        *mps_values,
        height=2,
        width=2,
        depth_samples=32,
    )
    gradients = metal.vjp(
        grad_output.to("mps").contiguous(),
        *mps_values,
        height=2,
        width=2,
        depth_samples=32,
    )
    torch.mps.synchronize()
    forward_error = float((actual.cpu() - reference.detach()).abs().max())
    names = (
        "ma",
        "q_uvt",
        "depth0",
        "depth_beta",
        "depth_variance",
        "optical_thickness",
        "color",
    )
    vjp_errors = {
        name: _normalized_error(gradients[name].cpu(), expected)
        for name, expected in zip(names, reference_gradients, strict=True)
    }
    current_bytes = int(torch.mps.current_allocated_memory())
    driver_bytes = int(torch.mps.driver_allocated_memory())
    checks = {
        "forward_metal_matches_cpu": forward_error <= 3.0e-5,
        "all_vjps_match_cpu": max(vjp_errors.values()) <= 2.0e-4,
        "depth_variance_vjp_is_nonzero": (
            float(gradients["depth_variance"].abs().sum().cpu()) > 0.0
        ),
        "fixed_quadrature_converges": quadrature_error <= 3.0e-4,
        "driver_memory_below_100mb": driver_bytes < 100_000_000,
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "tiny_retained_fiber_cpu_metal_forward_vjp",
        "status": "pass" if all(checks.values()) else "fail",
        "fixture": {
            "atoms": 2,
            "frames": 1,
            "height": 2,
            "width": 2,
            "depth_samples": 32,
            "sigma_extent": 6.0,
        },
        "metrics": {
            "forward_max_abs_error": forward_error,
            "vjp_normalized_errors": vjp_errors,
            "cpu_32_vs_1024_sample_max_abs_error": quadrature_error,
            "mps_current_allocated_bytes": current_bytes,
            "mps_driver_allocated_bytes": driver_bytes,
        },
        "checks": checks,
        "claim_limits": {
            "production_static_camera_training_wiring": True,
            "bounded_full_image_training_smoke": True,
            "variance_aware_tile_order_certificate": True,
            "masked_retained_fiber_fallback": True,
            "adaptive_quadrature": False,
            "quadrature_policy": "fixed_midpoint_at_most_64_samples",
            "bound_derivatives": "detached_compiler_decision",
            "projective_atlas_fallback_wiring": False,
            "large_scene_fallback_selectivity": False,
            "purpose": (
                "native shader correctness/differentiability plus the "
                "production static-camera certificate/fallback boundary"
            ),
        },
    }
    torch.mps.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_gate()
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote retained-fiber gate to {output}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
