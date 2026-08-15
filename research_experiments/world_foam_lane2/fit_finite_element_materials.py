"""Fit M0/M1/M3/M5 on one shared partial-segment transfer fixture.

The foundation parity gate establishes that each material evaluator computes
what it claims.  This gate asks the next question: does a richer material law
carry observable value when the same spacetime cell is intersected over
different subintervals?

The target is an independently integrated positive Bernstein-P2 extinction
field with constant color.  Each record observes the transfer element
``(beta, m)`` over a different subinterval of the same normalized cell.  M0
and M1 share one constant extinction, M1 may additionally learn global affine
color, and M3/M5 each receive three density controls.  This is a synthetic
material-capacity/training gate, not an image-quality benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .finite_element_material_transfer import (
    MaterialMode,
    evaluate_material_segment,
)


DTYPE = torch.float64
TARGET_CONTROLS = torch.tensor([0.12, 2.0, 0.25], dtype=DTYPE)
TARGET_COLOR = torch.tensor([0.72, 0.18, 0.46], dtype=DTYPE)
DEFAULT_OUTPUT = Path(
    "artifacts/foundation_gates/worldfoam_material_value_fit_cpu.json"
)
_GL_NODES_RAW, _GL_WEIGHTS_RAW = np.polynomial.legendre.leggauss(48)
_GL_NODES = torch.tensor(_GL_NODES_RAW, dtype=DTYPE)
_GL_WEIGHTS = torch.tensor(_GL_WEIGHTS_RAW, dtype=DTYPE)


@dataclass(frozen=True)
class IntervalTarget:
    start: float
    stop: float
    beta: torch.Tensor
    m: torch.Tensor


def interval_fixture() -> tuple[tuple[float, float], ...]:
    """Return a fixed tape exposing endpoints, interiors, prefixes, and suffixes."""

    return (
        (0.00, 0.10),
        (0.10, 0.20),
        (0.20, 0.30),
        (0.30, 0.40),
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.00),
        (0.00, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.00),
        (0.00, 0.50),
        (0.25, 0.75),
        (0.50, 1.00),
        (0.00, 0.75),
        (0.25, 1.00),
        (0.00, 1.00),
    )


def _bernstein_power_coefficients(controls: torch.Tensor) -> tuple[torch.Tensor, ...]:
    c0, c1, c2 = controls.unbind()
    return c0, -2.0 * c0 + 2.0 * c1, c0 - 2.0 * c1 + c2


def _positive_p2_integral(
    controls: torch.Tensor,
    start: float,
    stop: float,
) -> torch.Tensor:
    p0, p1, p2 = _bernstein_power_coefficients(controls)
    a = torch.as_tensor(start, dtype=controls.dtype, device=controls.device)
    b = torch.as_tensor(stop, dtype=controls.dtype, device=controls.device)
    return (
        p0 * (b - a)
        + 0.5 * p1 * (b.square() - a.square())
        + (p2 / 3.0) * (b.pow(3) - a.pow(3))
    )


def make_targets() -> tuple[IntervalTarget, ...]:
    records: list[IntervalTarget] = []
    for start, stop in interval_fixture():
        tau = _positive_p2_integral(TARGET_CONTROLS, start, stop)
        beta = torch.exp(-tau)
        records.append(
            IntervalTarget(
                start=start,
                stop=stop,
                beta=beta,
                m=(1.0 - beta) * TARGET_COLOR,
            )
        )
    return tuple(records)


def _positive(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw) + 1.0e-8


def _unit_color(raw: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(raw)


def _bernstein_value(controls: torch.Tensor, x: float) -> torch.Tensor:
    u = torch.as_tensor(x, dtype=controls.dtype, device=controls.device)
    return (
        controls[0] * (1.0 - u).square()
        + 2.0 * controls[1] * u * (1.0 - u)
        + controls[2] * u.square()
    )


def _restrict_positive_p2(
    controls: torch.Tensor,
    start: float,
    stop: float,
) -> torch.Tensor:
    midpoint = 0.5 * (float(start) + float(stop))
    local_front = _bernstein_value(controls, start)
    local_back = _bernstein_value(controls, stop)
    local_mid = _bernstein_value(controls, midpoint)
    local_control = 2.0 * local_mid - 0.5 * (local_front + local_back)
    return torch.stack((local_front, local_control, local_back))


def _restrict_log_p2(
    controls: torch.Tensor,
    start: float,
    stop: float,
) -> torch.Tensor:
    a, b, c = controls.unbind()
    x0 = torch.as_tensor(start, dtype=controls.dtype, device=controls.device)
    delta = torch.as_tensor(stop - start, dtype=controls.dtype, device=controls.device)
    return torch.stack(
        (
            a * delta.square(),
            delta * (2.0 * a * x0 + b),
            a * x0.square() + b * x0 + c,
        )
    )


def _mode_parameters(
    mode: MaterialMode,
    *,
    generator: torch.Generator,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        raw_density = torch.nn.Parameter(
            torch.randn((1,), generator=generator, dtype=DTYPE) * 0.15
        )
    else:
        raw_density = torch.nn.Parameter(
            torch.randn((3,), generator=generator, dtype=DTYPE) * 0.15
        )
    color_count = 6 if mode == MaterialMode.M1_P0_AFFINE_RGB else 3
    target_logit = torch.logit(TARGET_COLOR.clamp(1.0e-4, 1.0 - 1.0e-4))
    raw_color = torch.nn.Parameter(
        target_logit.repeat(2 if color_count == 6 else 1)
        + torch.randn((color_count,), generator=generator, dtype=DTYPE) * 0.1
    )
    return raw_density, raw_color


def _decoded_density(mode: MaterialMode, raw_density: torch.Tensor) -> torch.Tensor:
    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        return _positive(raw_density)
    if mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        return _positive(raw_density)
    if mode == MaterialMode.M5_CONVEX_LOG_P2:
        return torch.stack((_positive(raw_density[:1])[0], raw_density[1], raw_density[2]))
    raise ValueError(f"unsupported fit mode {mode.name}")


def _predictions(
    mode: MaterialMode,
    raw_density: torch.Tensor,
    raw_color: torch.Tensor,
    targets: tuple[IntervalTarget, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized differentiable fixed-tape evaluation.

    M0/M1/M3 use their exact elementary integrals. M5 uses fixed GL48
    quadrature during optimization; the final fitted record is independently
    re-evaluated through the production material evaluator below.
    """

    density = _decoded_density(mode, raw_density)
    starts = torch.tensor([record.start for record in targets], dtype=DTYPE)
    stops = torch.tensor([record.stop for record in targets], dtype=DTYPE)
    lengths = stops - starts
    if mode == MaterialMode.M1_P0_AFFINE_RGB:
        color_front_global = _unit_color(raw_color[:3])
        color_back_global = _unit_color(raw_color[3:])
    else:
        color_front_global = _unit_color(raw_color)
        color_back_global = color_front_global

    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        tau = density[0] * lengths
    elif mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        tau = _positive_p2_integral(density, starts, stops)
    else:
        nodes = (
            0.5 * lengths[:, None] * _GL_NODES[None, :]
            + 0.5 * (starts + stops)[:, None]
        )
        q = (
            density[0] * nodes.square()
            + density[1] * nodes
            + density[2]
        )
        tau = 0.5 * lengths * torch.sum(
            _GL_WEIGHTS[None, :] * torch.exp(-q),
            dim=1,
        )
    beta = torch.exp(-tau)
    if mode != MaterialMode.M1_P0_AFFINE_RGB:
        return beta, (1.0 - beta)[:, None] * color_front_global[None, :]

    color_front = (
        (1.0 - starts)[:, None] * color_front_global[None, :]
        + starts[:, None] * color_back_global[None, :]
    )
    color_back = (
        (1.0 - stops)[:, None] * color_front_global[None, :]
        + stops[:, None] * color_back_global[None, :]
    )
    total_weight = -torch.expm1(-tau)
    ordinary_first_moment = (
        1.0 - (1.0 + tau) * beta
    ) / tau.clamp_min(1.0e-12)
    series_first_moment = (
        0.5 * tau
        - tau.square() / 3.0
        + tau.pow(3) / 8.0
        - tau.pow(4) / 30.0
    )
    first_moment = torch.where(
        tau.abs() < 1.0e-4,
        series_first_moment,
        ordinary_first_moment,
    )
    m = (
        (total_weight - first_moment)[:, None] * color_front
        + first_moment[:, None] * color_back
    )
    return beta, m


def _reference_predictions(
    mode: MaterialMode,
    raw_density: torch.Tensor,
    raw_color: torch.Tensor,
    targets: tuple[IntervalTarget, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-evaluate a final fit through the shared production CPU evaluator."""

    density = _decoded_density(mode, raw_density)
    if mode == MaterialMode.M1_P0_AFFINE_RGB:
        color_front_global = _unit_color(raw_color[:3])
        color_back_global = _unit_color(raw_color[3:])
    else:
        color_front_global = _unit_color(raw_color)
        color_back_global = color_front_global
    beta_rows: list[torch.Tensor] = []
    m_rows: list[torch.Tensor] = []
    for record in targets:
        start = float(record.start)
        stop = float(record.stop)
        length = torch.as_tensor(stop - start, dtype=DTYPE)
        if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
            controls = torch.stack(
                (density[0], torch.zeros_like(density[0]), torch.zeros_like(density[0]))
            )
        elif mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
            controls = _restrict_positive_p2(density, start, stop)
        else:
            controls = _restrict_log_p2(density, start, stop)
        if mode == MaterialMode.M1_P0_AFFINE_RGB:
            color_front = (
                (1.0 - start) * color_front_global + start * color_back_global
            )
            color_back = (
                (1.0 - stop) * color_front_global + stop * color_back_global
            )
        else:
            color_front = color_front_global
            color_back = color_back_global
        transfer = evaluate_material_segment(
            mode,
            controls,
            length,
            color_front,
            color_back,
        )
        beta_rows.append(transfer.element.beta)
        m_rows.append(transfer.element.m)
    return torch.stack(beta_rows), torch.stack(m_rows)


def _loss(
    mode: MaterialMode,
    raw_density: torch.Tensor,
    raw_color: torch.Tensor,
    targets: tuple[IntervalTarget, ...],
) -> torch.Tensor:
    beta, m = _predictions(mode, raw_density, raw_color, targets)
    target_beta = torch.stack([record.beta for record in targets])
    target_m = torch.stack([record.m for record in targets])
    return (beta - target_beta).square().mean() + (m - target_m).square().mean()


def fit_mode(
    mode: MaterialMode,
    *,
    steps: int = 700,
    seed: int = 17,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if mode not in (
        MaterialMode.M0_P0_CONSTANT,
        MaterialMode.M1_P0_AFFINE_RGB,
        MaterialMode.M3_POSITIVE_BERNSTEIN_P2,
        MaterialMode.M5_CONVEX_LOG_P2,
    ):
        raise ValueError(f"unsupported fit mode {mode.name}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(mode) * 101)
    raw_density, raw_color = _mode_parameters(mode, generator=generator)
    targets = make_targets()
    optimizer = torch.optim.Adam((raw_density, raw_color), lr=0.035)
    loss_trace: list[dict[str, float | int]] = []
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(mode, raw_density, raw_color, targets)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(steps):
            loss_trace.append({"step": step, "loss": float(loss.detach())})

    # A deterministic quasi-Newton polish makes the exact M3 capacity visible
    # without hiding optimizer state in hand-set target coefficients.
    polish = torch.optim.LBFGS(
        (raw_density, raw_color),
        lr=0.8,
        max_iter=80,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        polish.zero_grad(set_to_none=True)
        value = _loss(mode, raw_density, raw_color, targets)
        value.backward()
        return value

    polish.step(closure)
    final_loss = _loss(mode, raw_density, raw_color, targets)
    beta, m = _predictions(mode, raw_density, raw_color, targets)
    reference_beta, reference_m = _reference_predictions(
        mode, raw_density, raw_color, targets
    )
    target_beta = torch.stack([record.beta for record in targets])
    target_m = torch.stack([record.m for record in targets])
    decoded_density = _decoded_density(mode, raw_density).detach()
    decoded_color = _unit_color(raw_color).detach()
    return {
        "mode": mode.name,
        "steps": int(steps),
        "trainable_scalars": int(raw_density.numel() + raw_color.numel()),
        "final_loss": float(final_loss.detach()),
        "max_abs_beta_error": float((beta - target_beta).abs().max().detach()),
        "max_abs_m_error": float((m - target_m).abs().max().detach()),
        "production_evaluator_max_abs_error": max(
            float((beta - reference_beta).abs().max().detach()),
            float((m - reference_m).abs().max().detach()),
        ),
        "density_parameters": [float(value) for value in decoded_density],
        "color_parameters": [float(value) for value in decoded_color],
        "loss_trace": loss_trace,
        "finite": bool(
            torch.isfinite(final_loss)
            and torch.isfinite(decoded_density).all()
            and torch.isfinite(decoded_color).all()
        ),
    }


def run_gate(*, steps: int = 700, seed: int = 17) -> dict[str, Any]:
    modes = (
        MaterialMode.M0_P0_CONSTANT,
        MaterialMode.M1_P0_AFFINE_RGB,
        MaterialMode.M3_POSITIVE_BERNSTEIN_P2,
        MaterialMode.M5_CONVEX_LOG_P2,
    )
    fits = {mode.name: fit_mode(mode, steps=steps, seed=seed) for mode in modes}
    m0 = fits[MaterialMode.M0_P0_CONSTANT.name]["final_loss"]
    m1 = fits[MaterialMode.M1_P0_AFFINE_RGB.name]["final_loss"]
    m3 = fits[MaterialMode.M3_POSITIVE_BERNSTEIN_P2.name]["final_loss"]
    m5 = fits[MaterialMode.M5_CONVEX_LOG_P2.name]["final_loss"]
    checks = {
        "all_fits_finite": all(bool(record["finite"]) for record in fits.values()),
        "optimized_and_production_evaluators_agree": all(
            float(record["production_evaluator_max_abs_error"]) <= 2.0e-10
            for record in fits.values()
        ),
        "m3_reaches_near_exact_fit": float(m3) <= 1.0e-12,
        "m3_beats_m0_by_1000x": float(m3) * 1000.0 < float(m0),
        "m3_beats_m1_by_1000x": float(m3) * 1000.0 < float(m1),
        "m3_challenges_m5": float(m3) <= float(m5),
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic_cpu_fixed_tape_material_capacity",
        "status": "pass" if all(checks.values()) else "fail",
        "fixture": {
            "interval_count": len(interval_fixture()),
            "target_density": "positive_bernstein_p2",
            "target_controls": [float(value) for value in TARGET_CONTROLS],
            "target_color": [float(value) for value in TARGET_COLOR],
            "observed_transfer": ["beta", "m"],
            "seed": int(seed),
            "adam_steps": int(steps),
        },
        "checks": checks,
        "fits": fits,
        "comparisons": {
            "m0_over_m3_loss": float(m0) / max(float(m3), 1.0e-300),
            "m1_over_m3_loss": float(m1) / max(float(m3), 1.0e-300),
            "m5_over_m3_loss": float(m5) / max(float(m3), 1.0e-300),
        },
        "claim_limits": {
            "public_scene_quality": False,
            "metal_throughput": False,
            "cell_geometry_training": False,
            "purpose": "material-law identifiability and optimizer plumbing on one fixed synthetic tape",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_gate(steps=args.steps, seed=args.seed)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote material-value fit gate to {output}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
