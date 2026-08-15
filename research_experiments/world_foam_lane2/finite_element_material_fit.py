"""Fixed-field material-value fitting for the WorldFoam M0--M5 matrix.

The local material microkernel evaluates one complete normalized segment.  A
single complete segment with constant color cannot identify density shape:
every density law that matches total optical depth produces the same
``(beta, m)``.  This fixture therefore shares one material field across many
partial chords ``[xi_start, xi_stop]`` of the same normalized cell.  Each chord
is lowered exactly to the existing segment evaluator.

This is a tiny CPU optimization/capacity gate.  It is not image training,
camera-program compilation, or renderer-throughput evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from research_experiments.world_foam_lane2.finite_element_material_transfer import (
    MaterialMode,
    MaterialTransfer,
    evaluate_material_segment,
)


DEFAULT_INTERVALS = (
    (0.00, 0.12),
    (0.04, 0.24),
    (0.12, 0.38),
    (0.20, 0.52),
    (0.31, 0.64),
    (0.43, 0.78),
    (0.58, 0.91),
    (0.76, 1.00),
    (0.00, 0.45),
    (0.18, 0.72),
    (0.37, 1.00),
    (0.00, 1.00),
)

HELDOUT_INTERVALS = (
    (0.01, 0.18),
    (0.09, 0.33),
    (0.24, 0.57),
    (0.35, 0.69),
    (0.49, 0.84),
    (0.67, 0.97),
    (0.00, 0.68),
    (0.22, 1.00),
)

MATERIAL_SCALARS = {
    MaterialMode.M0_P0_CONSTANT: 4,
    MaterialMode.M1_P0_AFFINE_RGB: 7,
    MaterialMode.M2_POSITIVE_BERNSTEIN_P1: 5,
    MaterialMode.M3_POSITIVE_BERNSTEIN_P2: 6,
    MaterialMode.M4_LOG_P1: 5,
    MaterialMode.M5_CONVEX_LOG_P2: 6,
}


@dataclass(frozen=True)
class RestrictedMaterial:
    density_controls: Tensor
    length: Tensor
    color_front: Tensor
    color_back: Tensor


@dataclass(frozen=True)
class TargetField:
    name: str
    mode: MaterialMode
    density_controls: tuple[float, float, float]
    length: float
    color_front: tuple[float, float, float]
    color_back: tuple[float, float, float]


TARGET_FIELDS = (
    TargetField(
        name="positive_p2_hump",
        mode=MaterialMode.M3_POSITIVE_BERNSTEIN_P2,
        density_controls=(0.08, 3.20, 0.18),
        length=1.7,
        color_front=(0.82, 0.27, 0.11),
        color_back=(0.82, 0.27, 0.11),
    ),
    TargetField(
        name="convex_log_p2_hump",
        mode=MaterialMode.M5_CONVEX_LOG_P2,
        density_controls=(12.0, -12.0, 3.0),
        length=1.7,
        color_front=(0.18, 0.73, 0.36),
        color_back=(0.18, 0.73, 0.36),
    ),
)


def _require_interval(xi_start: float, xi_stop: float) -> tuple[float, float]:
    start = float(xi_start)
    stop = float(xi_stop)
    if not (math.isfinite(start) and math.isfinite(stop)):
        raise ValueError("interval endpoints must be finite")
    if not 0.0 <= start < stop <= 1.0:
        raise ValueError("interval must satisfy 0 <= xi_start < xi_stop <= 1")
    return start, stop


def _bernstein_p2_value(controls: Tensor, xi: float) -> Tensor:
    one_minus = 1.0 - xi
    return (
        one_minus * one_minus * controls[0]
        + 2.0 * xi * one_minus * controls[1]
        + xi * xi * controls[2]
    )


def _bernstein_p2_derivative(controls: Tensor, xi: float) -> Tensor:
    return 2.0 * (
        (1.0 - xi) * (controls[1] - controls[0])
        + xi * (controls[2] - controls[1])
    )


def restrict_material_interval(
    mode: MaterialMode | int,
    density_controls: Tensor,
    length: Tensor,
    color_front: Tensor,
    color_back: Tensor,
    xi_start: float,
    xi_stop: float,
) -> RestrictedMaterial:
    """Restrict one global normalized material field to a partial chord.

    Direct P1/P2 modes use exact Bernstein subdivision.  Log modes substitute
    ``xi = xi_start + local_xi * (xi_stop - xi_start)`` into their negative-log
    polynomial.  M1 restricts its global affine color in the same way.
    """

    mode = MaterialMode(mode)
    start, stop = _require_interval(xi_start, xi_stop)
    delta = stop - start
    if density_controls.shape != (3,):
        raise ValueError("density_controls must have shape (3,)")
    if length.numel() != 1:
        raise ValueError("length must be scalar")
    if color_front.shape != (3,) or color_back.shape != (3,):
        raise ValueError("colors must have shape (3,)")

    zeros = torch.zeros_like(density_controls)
    if mode in (MaterialMode.M0_P0_CONSTANT, MaterialMode.M1_P0_AFFINE_RGB):
        local_controls = torch.stack(
            (density_controls[0], zeros[1], zeros[2])
        )
    elif mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
        c0, c1 = density_controls[:2]
        local_controls = torch.stack(
            (
                (1.0 - start) * c0 + start * c1,
                (1.0 - stop) * c0 + stop * c1,
                zeros[2],
            )
        )
    elif mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
        value_start = _bernstein_p2_value(density_controls, start)
        value_stop = _bernstein_p2_value(density_controls, stop)
        middle = value_start + (
            0.5
            * delta
            * _bernstein_p2_derivative(density_controls, start)
        )
        local_controls = torch.stack((value_start, middle, value_stop))
    elif mode == MaterialMode.M4_LOG_P1:
        b, c = density_controls[:2]
        local_controls = torch.stack(
            (delta * b, start * b + c, zeros[2])
        )
    elif mode == MaterialMode.M5_CONVEX_LOG_P2:
        a, b, c = density_controls
        local_controls = torch.stack(
            (
                delta * delta * a,
                delta * (2.0 * a * start + b),
                a * start * start + b * start + c,
            )
        )
    else:  # pragma: no cover - IntEnum exhaustiveness guard.
        raise ValueError(f"unsupported material mode {mode}")

    if mode == MaterialMode.M1_P0_AFFINE_RGB:
        color_delta = color_back - color_front
        local_color_front = color_front + start * color_delta
        local_color_back = color_front + stop * color_delta
    else:
        local_color_front = color_front
        local_color_back = color_front

    return RestrictedMaterial(
        density_controls=local_controls,
        length=length * delta,
        color_front=local_color_front,
        color_back=local_color_back,
    )


def evaluate_material_field(
    mode: MaterialMode | int,
    density_controls: Tensor,
    length: Tensor,
    color_front: Tensor,
    color_back: Tensor,
    intervals: Iterable[tuple[float, float]] = DEFAULT_INTERVALS,
) -> list[MaterialTransfer]:
    result: list[MaterialTransfer] = []
    for start, stop in intervals:
        local = restrict_material_interval(
            mode,
            density_controls,
            length,
            color_front,
            color_back,
            start,
            stop,
        )
        result.append(
            evaluate_material_segment(
                mode,
                local.density_controls,
                local.length,
                local.color_front,
                local.color_back,
            )
        )
    return result


def _inv_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _logit(value: Tensor) -> Tensor:
    clamped = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(clamped) - torch.log1p(-clamped)


class FittedMaterial(nn.Module):
    """Small constrained chart for one global M0--M5 material field."""

    def __init__(self, mode: MaterialMode, seed: int) -> None:
        super().__init__()
        self.mode = MaterialMode(mode)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = 0.04 * torch.randn(3, generator=generator, dtype=torch.float64)

        if self.mode in (
            MaterialMode.M0_P0_CONSTANT,
            MaterialMode.M1_P0_AFFINE_RGB,
        ):
            initial = torch.tensor((0.8,), dtype=torch.float64)
            self.raw_positive = nn.Parameter(_inv_softplus(initial) + noise[:1])
            self.free_density = None
        elif self.mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
            initial = torch.tensor((0.7, 0.9), dtype=torch.float64)
            self.raw_positive = nn.Parameter(_inv_softplus(initial) + noise[:2])
            self.free_density = None
        elif self.mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
            initial = torch.tensor((0.6, 1.0, 0.7), dtype=torch.float64)
            self.raw_positive = nn.Parameter(_inv_softplus(initial) + noise)
            self.free_density = None
        elif self.mode == MaterialMode.M4_LOG_P1:
            self.raw_positive = None
            self.free_density = nn.Parameter(
                torch.tensor((0.0, 0.2), dtype=torch.float64) + noise[:2]
            )
        elif self.mode == MaterialMode.M5_CONVEX_LOG_P2:
            self.raw_positive = nn.Parameter(
                _inv_softplus(torch.tensor((0.4,), dtype=torch.float64))
                + noise[:1]
            )
            self.free_density = nn.Parameter(
                torch.tensor((0.0, 0.2), dtype=torch.float64) + noise[1:]
            )
        else:  # pragma: no cover - IntEnum exhaustiveness guard.
            raise ValueError(f"unsupported material mode {self.mode}")

        base_color = torch.tensor((0.45, 0.45, 0.45), dtype=torch.float64)
        self.raw_color_front = nn.Parameter(_logit(base_color) + noise)
        self.raw_color_back = (
            nn.Parameter(_logit(base_color) - noise)
            if self.mode == MaterialMode.M1_P0_AFFINE_RGB
            else None
        )

    def density_controls(self) -> Tensor:
        zeros = torch.zeros(3, dtype=torch.float64)
        if self.mode in (
            MaterialMode.M0_P0_CONSTANT,
            MaterialMode.M1_P0_AFFINE_RGB,
        ):
            return torch.cat((F.softplus(self.raw_positive), zeros[1:]))
        if self.mode == MaterialMode.M2_POSITIVE_BERNSTEIN_P1:
            return torch.cat((F.softplus(self.raw_positive), zeros[2:]))
        if self.mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
            return F.softplus(self.raw_positive)
        if self.mode == MaterialMode.M4_LOG_P1:
            return torch.cat((self.free_density, zeros[2:]))
        if self.mode == MaterialMode.M5_CONVEX_LOG_P2:
            return torch.cat((F.softplus(self.raw_positive), self.free_density))
        raise ValueError(f"unsupported material mode {self.mode}")

    def colors(self) -> tuple[Tensor, Tensor]:
        front = torch.sigmoid(self.raw_color_front)
        back = (
            torch.sigmoid(self.raw_color_back)
            if self.raw_color_back is not None
            else front
        )
        return front, back


def _stack_outputs(transfers: list[MaterialTransfer]) -> tuple[Tensor, Tensor]:
    beta = torch.stack([transfer.element.beta for transfer in transfers])
    rgb = torch.stack([transfer.element.m for transfer in transfers])
    return beta, rgb


def independent_target_outputs(
    target: TargetField,
    intervals: Iterable[tuple[float, float]],
) -> tuple[Tensor, Tensor]:
    """Integrate target fields without calling the fitted production evaluator."""

    beta_values: list[Tensor] = []
    rgb_values: list[Tensor] = []
    color = torch.tensor(target.color_front, dtype=torch.float64)
    if target.color_front != target.color_back:
        raise ValueError("independent target oracle currently requires constant color")
    controls = torch.tensor(target.density_controls, dtype=torch.float64)
    for start, stop in intervals:
        start, stop = _require_interval(start, stop)
        if target.mode == MaterialMode.M3_POSITIVE_BERNSTEIN_P2:
            c0, c1, c2 = controls
            p0 = c0
            p1 = 2.0 * (c1 - c0)
            p2 = c0 - 2.0 * c1 + c2
            density_integral = (
                p0 * (stop - start)
                + 0.5 * p1 * (stop * stop - start * start)
                + (p2 / 3.0) * (stop**3 - start**3)
            )
        elif target.mode == MaterialMode.M5_CONVEX_LOG_P2:
            # Composite Simpson is deliberately independent of the analytic
            # erf/series/tail branches in the production M5 evaluator.
            xi = torch.linspace(start, stop, 1025, dtype=torch.float64)
            q = controls[0] * xi.square() + controls[1] * xi + controls[2]
            density = torch.exp(-q)
            step = (stop - start) / 1024.0
            density_integral = (step / 3.0) * (
                density[0]
                + density[-1]
                + 4.0 * density[1:-1:2].sum()
                + 2.0 * density[2:-1:2].sum()
            )
        else:
            raise ValueError(
                f"unsupported independent target mode {target.mode.name}"
            )
        tau = float(target.length) * density_integral
        beta = torch.exp(-tau)
        beta_values.append(beta)
        rgb_values.append((1.0 - beta) * color)
    return torch.stack(beta_values), torch.stack(rgb_values)


def fit_material_mode(
    target: TargetField,
    mode: MaterialMode | int,
    *,
    seed: int,
    steps: int = 800,
    learning_rate: float = 0.04,
    refinement_steps: int = 50,
    train_intervals: Iterable[tuple[float, float]] = DEFAULT_INTERVALS,
    heldout_intervals: Iterable[tuple[float, float]] = HELDOUT_INTERVALS,
) -> dict[str, object]:
    mode = MaterialMode(mode)
    train_intervals = tuple(train_intervals)
    heldout_intervals = tuple(heldout_intervals)
    target_length = torch.tensor(target.length, dtype=torch.float64)
    with torch.no_grad():
        target_train_beta, target_train_rgb = independent_target_outputs(
            target,
            train_intervals,
        )
        target_heldout_beta, target_heldout_rgb = independent_target_outputs(
            target,
            heldout_intervals,
        )

    torch.manual_seed(int(seed))
    random.seed(int(seed))
    model = FittedMaterial(mode, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    started_at = time.perf_counter()
    loss_trace: list[dict[str, float | int | str]] = []

    def evaluate_loss() -> tuple[Tensor, Tensor, Tensor]:
        controls = model.density_controls()
        color_front, color_back = model.colors()
        predicted_beta, predicted_rgb = _stack_outputs(
            evaluate_material_field(
                mode,
                controls,
                target_length,
                color_front,
                color_back,
                train_intervals,
            )
        )
        beta_mse = (predicted_beta - target_train_beta).square().mean()
        rgb_mse = (predicted_rgb - target_train_rgb).square().mean()
        return beta_mse + rgb_mse, beta_mse, rgb_mse

    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, beta_mse, rgb_mse = evaluate_loss()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            loss_trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "beta_mse": float(beta_mse.detach()),
                    "rgb_mse": float(rgb_mse.detach()),
                }
            )

    # The gate measures material capacity, not sensitivity to Adam's scale.
    # A common deterministic quasi-Newton polish keeps exact-family targets
    # from looking worse merely because log-polynomial coefficients span a
    # larger numerical range than positive Bernstein controls.
    if refinement_steps > 0:
        refinement = torch.optim.LBFGS(
            model.parameters(),
            max_iter=int(refinement_steps),
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            line_search_fn="strong_wolfe",
        )

        def refinement_closure() -> Tensor:
            refinement.zero_grad(set_to_none=True)
            loss, _, _ = evaluate_loss()
            loss.backward()
            return loss

        refinement.step(refinement_closure)
        loss, beta_mse, rgb_mse = evaluate_loss()
        loss_trace.append(
            {
                "phase": "lbfgs_refinement",
                "step": int(steps) + int(refinement_steps),
                "loss": float(loss.detach()),
                "beta_mse": float(beta_mse.detach()),
                "rgb_mse": float(rgb_mse.detach()),
            }
        )

    controls = model.density_controls().detach()
    color_front, color_back = (value.detach() for value in model.colors())
    predicted_train_beta, predicted_train_rgb = _stack_outputs(
        evaluate_material_field(
            mode,
            controls,
            target_length,
            color_front,
            color_back,
            train_intervals,
        )
    )
    predicted_heldout_beta, predicted_heldout_rgb = _stack_outputs(
        evaluate_material_field(
            mode,
            controls,
            target_length,
            color_front,
            color_back,
            heldout_intervals,
        )
    )
    train_beta_error = predicted_train_beta - target_train_beta
    train_rgb_error = predicted_train_rgb - target_train_rgb
    heldout_beta_error = predicted_heldout_beta - target_heldout_beta
    heldout_rgb_error = predicted_heldout_rgb - target_heldout_rgb
    train_beta_mse = float(train_beta_error.square().mean())
    train_rgb_mse = float(train_rgb_error.square().mean())
    heldout_beta_mse = float(heldout_beta_error.square().mean())
    heldout_rgb_mse = float(heldout_rgb_error.square().mean())
    return {
        "target": target.name,
        "target_mode": target.mode.name,
        "mode": mode.name,
        "seed": int(seed),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "refinement_steps": int(refinement_steps),
        "optimizer": "adam_then_lbfgs_strong_wolfe",
        "elapsed_s": time.perf_counter() - started_at,
        "trainable_scalars": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "serialized_material_scalars": MATERIAL_SCALARS[mode],
        "serialized_material_bytes_float32": 4 * MATERIAL_SCALARS[mode],
        "loss": heldout_beta_mse + heldout_rgb_mse,
        "beta_mse": heldout_beta_mse,
        "rgb_mse": heldout_rgb_mse,
        "max_beta_abs_error": float(heldout_beta_error.abs().max()),
        "max_rgb_abs_error": float(heldout_rgb_error.abs().max()),
        "train_loss": train_beta_mse + train_rgb_mse,
        "train_beta_mse": train_beta_mse,
        "train_rgb_mse": train_rgb_mse,
        "train_max_beta_abs_error": float(train_beta_error.abs().max()),
        "train_max_rgb_abs_error": float(train_rgb_error.abs().max()),
        "heldout_loss": heldout_beta_mse + heldout_rgb_mse,
        "heldout_beta_mse": heldout_beta_mse,
        "heldout_rgb_mse": heldout_rgb_mse,
        "heldout_max_beta_abs_error": float(heldout_beta_error.abs().max()),
        "heldout_max_rgb_abs_error": float(heldout_rgb_error.abs().max()),
        "density_controls": controls.tolist(),
        "color_front": color_front.tolist(),
        "color_back": color_back.tolist(),
        "loss_trace": loss_trace,
    }


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _git_metadata(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("finite_element_material_transfer.py").resolve(),
    )
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def run_material_value_gate(
    *,
    seeds: Iterable[int] = (17, 29, 43),
    steps: int = 800,
    learning_rate: float = 0.04,
    refinement_steps: int = 50,
) -> dict[str, object]:
    seeds = tuple(int(seed) for seed in seeds)
    rows = [
        fit_material_mode(
            target,
            mode,
            seed=seed,
            steps=steps,
            learning_rate=learning_rate,
            refinement_steps=refinement_steps,
        )
        for target in TARGET_FIELDS
        for mode in MaterialMode
        for seed in seeds
    ]
    medians: dict[str, dict[str, dict[str, float]]] = {}
    for target in TARGET_FIELDS:
        target_rows = [row for row in rows if row["target"] == target.name]
        medians[target.name] = {}
        for mode in MaterialMode:
            mode_rows = [row for row in target_rows if row["mode"] == mode.name]
            medians[target.name][mode.name] = {
                key: _median(float(row[key]) for row in mode_rows)
                for key in (
                    "loss",
                    "beta_mse",
                    "rgb_mse",
                    "max_beta_abs_error",
                    "max_rgb_abs_error",
                    "train_loss",
                    "heldout_loss",
                    "elapsed_s",
                )
            }

    direct = medians["positive_p2_hump"]
    log = medians["convex_log_p2_hump"]
    direct_m3 = direct[MaterialMode.M3_POSITIVE_BERNSTEIN_P2.name]["loss"]
    log_m5 = log[MaterialMode.M5_CONVEX_LOG_P2.name]["loss"]
    checks = {
        "positive_p2_m3_beats_m0_100x": direct_m3
        <= 0.01 * direct[MaterialMode.M0_P0_CONSTANT.name]["loss"],
        "positive_p2_m3_beats_m1_100x": direct_m3
        <= 0.01 * direct[MaterialMode.M1_P0_AFFINE_RGB.name]["loss"],
        "positive_p2_m3_challenges_m5": direct_m3
        <= direct[MaterialMode.M5_CONVEX_LOG_P2.name]["loss"],
        "positive_p2_m3_beats_m5_100x": direct_m3
        <= 0.01 * direct[MaterialMode.M5_CONVEX_LOG_P2.name]["loss"],
        "log_p2_m5_beats_m0_100x": log_m5
        <= 0.01 * log[MaterialMode.M0_P0_CONSTANT.name]["loss"],
        "log_p2_m5_beats_m1_100x": log_m5
        <= 0.01 * log[MaterialMode.M1_P0_AFFINE_RGB.name]["loss"],
        "log_p2_m5_beats_m3_100x": log_m5
        <= 0.01 * log[MaterialMode.M3_POSITIVE_BERNSTEIN_P2.name]["loss"],
        "m3_m5_matched_serialized_bytes": MATERIAL_SCALARS[
            MaterialMode.M3_POSITIVE_BERNSTEIN_P2
        ]
        == MATERIAL_SCALARS[MaterialMode.M5_CONVEX_LOG_P2],
        "all_rows_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "loss",
                "beta_mse",
                "rgb_mse",
                "max_beta_abs_error",
                "max_rgb_abs_error",
                "train_loss",
                "heldout_loss",
            )
        ),
    }
    root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 2,
        "claim_scope": [
            "shared partial-chord synthetic material train/heldout capacity only",
            "not image training or heldout view quality",
            "not renderer or camera-program compiler throughput",
            "not native-4D cell parameter/event scaling",
        ],
        "git": _git_metadata(root),
        "source_sha256": _source_hashes(),
        "dtype": "float64",
        "device": "cpu",
        "seeds": list(seeds),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "refinement_steps": int(refinement_steps),
        "optimizer": "adam_then_lbfgs_strong_wolfe",
        "train_intervals": [list(interval) for interval in DEFAULT_INTERVALS],
        "heldout_intervals": [list(interval) for interval in HELDOUT_INTERVALS],
        "material_scalars": {
            mode.name: MATERIAL_SCALARS[mode] for mode in MaterialMode
        },
        "promotion": {
            "winner": None,
            "decision": (
                "no universal M3/M5 winner: each exact family wins its own "
                "held-out target"
            ),
            "eligible_for_native_4d_integration": False,
            "next_gate": (
                "adaptive per-cell basis selection or real-data held-out "
                "material comparison"
            ),
        },
        "targets": [
            {
                "name": target.name,
                "mode": target.mode.name,
                "density_controls": list(target.density_controls),
                "length": target.length,
                "color_front": list(target.color_front),
                "color_back": list(target.color_back),
            }
            for target in TARGET_FIELDS
        ],
        "medians": medians,
        "checks": checks,
        "passed": all(checks.values()),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "artifacts/foundation_gates/"
            "worldfoam_material_value_fit_cpu_20260727.json"
        ),
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--refinement-steps", type=int, default=50)
    parser.add_argument("--seeds", default="17,29,43")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    payload = run_material_value_gate(
        seeds=seeds,
        steps=args.steps,
        learning_rate=args.learning_rate,
        refinement_steps=args.refinement_steps,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_json": str(args.out_json), **payload["checks"]}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
