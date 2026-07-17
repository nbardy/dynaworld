from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True)
class TraceSpec:
    center_u: float
    center_v: float
    opacity: float
    precision_uu: float
    precision_uv: float
    precision_vv: float
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class AnisotropicCaseResult:
    name: str
    expectation: str
    passed: bool
    certified_reused: bool
    tail_alpha_epsilon: float
    omitted_alpha_bound: float
    max_abs_error: float
    mean_abs_error: float
    forced_bad_max_abs_error: float | None = None
    omitted_tiles: tuple[tuple[int, int], ...] = ()
    notes: str = ""


def _rotated_precision(*, sigma_major: float, sigma_minor: float, angle_degrees: float) -> tuple[float, float, float]:
    theta = math.radians(float(angle_degrees))
    c = math.cos(theta)
    s = math.sin(theta)
    q_major = 1.0 / float(sigma_major * sigma_major)
    q_minor = 1.0 / float(sigma_minor * sigma_minor)
    q_uu = c * c * q_major + s * s * q_minor
    q_uv = c * s * (q_major - q_minor)
    q_vv = s * s * q_major + c * c * q_minor
    return (q_uu, q_uv, q_vv)


def _quadratic(trace: TraceSpec, u: float, v: float) -> float:
    du = float(u) - float(trace.center_u)
    dv = float(v) - float(trace.center_v)
    return (
        float(trace.precision_uu) * du * du
        + 2.0 * float(trace.precision_uv) * du * dv
        + float(trace.precision_vv) * dv * dv
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _min_quadratic_on_rect(
    trace: TraceSpec,
    *,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
) -> float:
    """Exact 2D SPD quadratic minimum on an axis-aligned rectangle.

    The minimum of a convex quadratic over a rectangle lies in the interior, on
    an edge stationary point, or at a corner. We enumerate those candidates.
    """

    p00 = float(trace.precision_uu)
    p01 = float(trace.precision_uv)
    p11 = float(trace.precision_vv)
    if p00 <= 0.0 or p11 <= 0.0 or p00 * p11 - p01 * p01 <= 0.0:
        raise ValueError("trace precision must be positive definite")

    candidates: list[tuple[float, float]] = []
    if float(u0) <= trace.center_u <= float(u1) and float(v0) <= trace.center_v <= float(v1):
        candidates.append((float(trace.center_u), float(trace.center_v)))

    for u in (float(u0), float(u1)):
        du = u - float(trace.center_u)
        v = float(trace.center_v) - (p01 / p11) * du
        candidates.append((u, _clamp(v, float(v0), float(v1))))

    for v in (float(v0), float(v1)):
        dv = v - float(trace.center_v)
        u = float(trace.center_u) - (p01 / p00) * dv
        candidates.append((_clamp(u, float(u0), float(u1)), v))

    for u in (float(u0), float(u1)):
        for v in (float(v0), float(v1)):
            candidates.append((u, v))

    return min(_quadratic(trace, u, v) for u, v in candidates)


def _render_density(
    traces: Sequence[TraceSpec],
    tiles: set[tuple[int, int]],
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
) -> torch.Tensor:
    image = torch.zeros((int(image_height), int(image_width), 3), dtype=torch.float32)
    xs = torch.arange(int(image_width), dtype=torch.float32) + 0.5
    ys = torch.arange(int(image_height), dtype=torch.float32) + 0.5
    grid_v, grid_u = torch.meshgrid(ys, xs, indexing="ij")
    tile_u = torch.div(torch.arange(int(image_width)), int(tile_size), rounding_mode="floor")
    tile_v = torch.div(torch.arange(int(image_height)), int(tile_size), rounding_mode="floor")
    tile_v_grid, tile_u_grid = torch.meshgrid(tile_v, tile_u, indexing="ij")
    tile_mask = torch.zeros((int(image_height), int(image_width)), dtype=torch.bool)
    for u_tile, v_tile in tiles:
        tile_mask |= (tile_u_grid == int(u_tile)) & (tile_v_grid == int(v_tile))

    for trace in traces:
        du = grid_u - float(trace.center_u)
        dv = grid_v - float(trace.center_v)
        q = (
            float(trace.precision_uu) * du.square()
            + 2.0 * float(trace.precision_uv) * du * dv
            + float(trace.precision_vv) * dv.square()
        )
        alpha = float(trace.opacity) * torch.exp(-0.5 * q)
        alpha = torch.where(tile_mask, alpha, torch.zeros_like(alpha))
        color = torch.tensor(trace.color, dtype=torch.float32).reshape(1, 1, 3)
        image = image + alpha.unsqueeze(-1) * color
    return image


def _omitted_alpha_bound(
    traces: Sequence[TraceSpec],
    omitted_tiles: set[tuple[int, int]],
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
) -> float:
    max_tile_bound = 0.0
    for tile_u, tile_v in sorted(omitted_tiles):
        u0 = float(tile_u * int(tile_size))
        u1 = float(min(int(image_width), (int(tile_u) + 1) * int(tile_size)))
        v0 = float(tile_v * int(tile_size))
        v1 = float(min(int(image_height), (int(tile_v) + 1) * int(tile_size)))
        tile_bound = 0.0
        for trace in traces:
            q_min = _min_quadratic_on_rect(trace, u0=u0, u1=u1, v0=v0, v1=v1)
            color_gain = max(abs(float(channel)) for channel in trace.color)
            tile_bound += float(trace.opacity) * color_gain * math.exp(-0.5 * q_min)
        max_tile_bound = max(max_tile_bound, tile_bound)
    return float(max_tile_bound)


def _run_case(
    *,
    name: str,
    traces: Sequence[TraceSpec],
    stale_tiles: set[tuple[int, int]],
    strict_tiles: set[tuple[int, int]],
    tail_alpha_epsilon: float,
    expectation: str = "certified_anisotropic_tail_reuse",
    require_reuse: bool = True,
    image_width: int = 16,
    image_height: int = 8,
    tile_size: int = 8,
    notes: str = "",
) -> AnisotropicCaseResult:
    omitted_tiles = set(strict_tiles) - set(stale_tiles)
    strict_image = _render_density(
        traces,
        set(strict_tiles),
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )
    stale_image = _render_density(
        traces,
        set(stale_tiles),
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )
    diff = (strict_image - stale_image).abs()
    omitted_bound = _omitted_alpha_bound(
        traces,
        omitted_tiles,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )
    certified_reused = omitted_bound <= float(tail_alpha_epsilon)
    max_error = float(diff.max().item())
    if require_reuse:
        passed = certified_reused and max_error <= omitted_bound * 1.01 + 1.0e-7
        forced_bad: float | None = None
    else:
        passed = (not certified_reused) and max_error > 0.25 and omitted_bound > float(tail_alpha_epsilon)
        forced_bad = max_error
    return AnisotropicCaseResult(
        name=name,
        expectation=expectation,
        passed=bool(passed),
        certified_reused=bool(certified_reused),
        tail_alpha_epsilon=float(tail_alpha_epsilon),
        omitted_alpha_bound=float(omitted_bound),
        max_abs_error=max_error,
        mean_abs_error=float(diff.mean().item()),
        forced_bad_max_abs_error=forced_bad,
        omitted_tiles=tuple(sorted(omitted_tiles)),
        notes=notes,
    )


def run_verifier(*, tail_alpha_epsilon: float) -> dict[str, object]:
    rotated_uu, rotated_uv, rotated_vv = _rotated_precision(
        sigma_major=1.15,
        sigma_minor=0.55,
        angle_degrees=35.0,
    )
    sum_uu, sum_uv, sum_vv = _rotated_precision(
        sigma_major=1.05,
        sigma_minor=0.70,
        angle_degrees=-25.0,
    )
    cases = [
        _run_case(
            name="diagonal_sigma_u1_v2_tail",
            traces=[
                TraceSpec(
                    center_u=4.05,
                    center_v=4.0,
                    opacity=0.5,
                    precision_uu=1.0,
                    precision_uv=0.0,
                    precision_vv=0.25,
                )
            ],
            stale_tiles={(0, 0)},
            strict_tiles={(0, 0), (1, 0)},
            tail_alpha_epsilon=tail_alpha_epsilon,
            notes="axis-aligned anisotropic footprint omitted only in the narrow u direction",
        ),
        _run_case(
            name="rotated_precision_tail",
            traces=[
                TraceSpec(
                    center_u=4.05,
                    center_v=4.0,
                    opacity=0.5,
                    precision_uu=rotated_uu,
                    precision_uv=rotated_uv,
                    precision_vv=rotated_vv,
                )
            ],
            stale_tiles={(0, 0)},
            strict_tiles={(0, 0), (1, 0)},
            tail_alpha_epsilon=tail_alpha_epsilon,
            notes="rotated ellipse uses exact rectangle Mahalanobis minimization",
        ),
        _run_case(
            name="two_trace_same_omitted_tile_sum",
            traces=[
                TraceSpec(
                    center_u=4.05,
                    center_v=3.2,
                    opacity=0.35,
                    precision_uu=sum_uu,
                    precision_uv=sum_uv,
                    precision_vv=sum_vv,
                ),
                TraceSpec(
                    center_u=4.10,
                    center_v=5.0,
                    opacity=0.40,
                    precision_uu=1.10,
                    precision_uv=0.0,
                    precision_vv=0.50,
                    color=(0.8, 0.9, 1.0),
                ),
            ],
            stale_tiles={(0, 0)},
            strict_tiles={(0, 0), (1, 0)},
            tail_alpha_epsilon=tail_alpha_epsilon,
            notes="bounds add omitted alpha from multiple traces landing in the same missing tile",
        ),
        _run_case(
            name="anisotropic_core_loss_rejected",
            traces=[
                TraceSpec(
                    center_u=8.05,
                    center_v=4.0,
                    opacity=0.5,
                    precision_uu=1.0,
                    precision_uv=0.0,
                    precision_vv=0.25,
                )
            ],
            stale_tiles={(0, 0)},
            strict_tiles={(0, 0), (1, 0)},
            tail_alpha_epsilon=tail_alpha_epsilon,
            expectation="anisotropic_certificate_rejects_core_loss",
            require_reuse=False,
            notes="same pixel-scale crossing, but omitted tile contains the Gaussian core",
        ),
    ]
    return {
        "tail_alpha_epsilon": float(tail_alpha_epsilon),
        "all_passed": all(case.passed for case in cases),
        "cases": [asdict(case) for case in cases],
    }


def verify_anisotropic_tail_bound_report(payload: dict[str, object]) -> list[str]:
    """Return contract failures for a saved anisotropic tail-bound report."""

    errors: list[str] = []
    if not bool(payload.get("all_passed")):
        errors.append("all_passed must be true")
    tail_alpha_epsilon = payload.get("tail_alpha_epsilon")
    if not isinstance(tail_alpha_epsilon, int | float) or not math.isfinite(float(tail_alpha_epsilon)):
        errors.append(f"tail_alpha_epsilon must be finite, got {tail_alpha_epsilon!r}")
        tail_alpha = 0.0
    else:
        tail_alpha = float(tail_alpha_epsilon)
        if tail_alpha <= 0.0:
            errors.append(f"tail_alpha_epsilon must be positive, got {tail_alpha}")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        errors.append("cases must be a non-empty list")
        return errors
    cases = [case for case in raw_cases if isinstance(case, dict)]
    if len(cases) != len(raw_cases):
        errors.append("all cases must be objects")

    by_name = {str(case.get("name")): case for case in cases}
    required_names = {
        "diagonal_sigma_u1_v2_tail",
        "rotated_precision_tail",
        "two_trace_same_omitted_tile_sum",
        "anisotropic_core_loss_rejected",
    }
    missing = sorted(required_names - set(by_name))
    if missing:
        errors.append(f"missing required cases: {missing}")

    positive_cases = [case for case in cases if case.get("expectation") == "certified_anisotropic_tail_reuse"]
    if len(positive_cases) < 3:
        errors.append("need at least three certified anisotropic tail-reuse cases")
    for case in positive_cases:
        name = str(case.get("name"))
        omitted_bound = float(case.get("omitted_alpha_bound") or 0.0)
        max_error = float(case.get("max_abs_error") or 0.0)
        mean_error = float(case.get("mean_abs_error") or 0.0)
        omitted_tiles = case.get("omitted_tiles")
        if not bool(case.get("passed")):
            errors.append(f"{name} did not pass")
        if not bool(case.get("certified_reused")):
            errors.append(f"{name} must certify reuse")
        if not omitted_tiles:
            errors.append(f"{name} must identify omitted tiles")
        if not 0.0 < omitted_bound <= tail_alpha:
            errors.append(f"{name} omitted alpha bound must be in (0, epsilon], got {omitted_bound}")
        if max_error > omitted_bound * 1.01 + 1.0e-7:
            errors.append(f"{name} max image error {max_error} exceeds omitted bound {omitted_bound}")
        if mean_error > max_error + 1.0e-12:
            errors.append(f"{name} mean image error {mean_error} exceeds max image error {max_error}")

    sum_case = by_name.get("two_trace_same_omitted_tile_sum")
    if sum_case is not None:
        diagonal_bound = float(by_name.get("diagonal_sigma_u1_v2_tail", {}).get("omitted_alpha_bound") or 0.0)
        rotated_bound = float(by_name.get("rotated_precision_tail", {}).get("omitted_alpha_bound") or 0.0)
        sum_bound = float(sum_case.get("omitted_alpha_bound") or 0.0)
        if sum_bound <= max(diagonal_bound, rotated_bound):
            errors.append("two_trace_same_omitted_tile_sum must exceed each single-tail bound")

    core_case = by_name.get("anisotropic_core_loss_rejected")
    if core_case is not None:
        omitted_bound = float(core_case.get("omitted_alpha_bound") or 0.0)
        forced_bad = core_case.get("forced_bad_max_abs_error")
        if core_case.get("expectation") != "anisotropic_certificate_rejects_core_loss":
            errors.append("anisotropic_core_loss_rejected has wrong expectation")
        if not bool(core_case.get("passed")):
            errors.append("anisotropic_core_loss_rejected did not pass")
        if bool(core_case.get("certified_reused")):
            errors.append("anisotropic_core_loss_rejected must reject stale reuse")
        if omitted_bound <= tail_alpha:
            errors.append(f"anisotropic_core_loss_rejected bound must exceed epsilon, got {omitted_bound}")
        if forced_bad is None or float(forced_bad) <= 0.25:
            errors.append(f"anisotropic_core_loss_rejected forced-bad error must be large, got {forced_bad!r}")

    return errors


def assert_anisotropic_tail_bound_report(payload: dict[str, object]) -> None:
    errors = verify_anisotropic_tail_bound_report(payload)
    if errors:
        raise AssertionError("anisotropic tail-bound report failed:\n- " + "\n- ".join(errors))


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    cases = payload["cases"]
    assert isinstance(cases, list)
    lines = [
        "# Projective Anisotropic Tail-Bound Verifier",
        "",
        "This verifier is a CPU/theory gate for richer projective footprint",
        "certificates. It bounds omitted support by minimizing the anisotropic",
        "Mahalanobis quadratic over each omitted tile rectangle.",
        "",
        f"- tail_alpha_epsilon: `{payload['tail_alpha_epsilon']}`",
        f"- all_passed: `{payload['all_passed']}`",
        "",
        "| case | expectation | pass | certified_reused | omitted_bound | max_abs_error | mean_abs_error | forced_bad_max_abs_error | omitted_tiles |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for raw_case in cases:
        assert isinstance(raw_case, dict)
        forced_bad = raw_case.get("forced_bad_max_abs_error")
        lines.append(
            "| {name} | {expectation} | {passed} | {certified_reused} | "
            "{bound:.8g} | {max_err:.8g} | {mean_err:.8g} | {forced_bad} | {tiles} |".format(
                name=raw_case["name"],
                expectation=raw_case["expectation"],
                passed=raw_case["passed"],
                certified_reused=raw_case["certified_reused"],
                bound=float(raw_case["omitted_alpha_bound"]),
                max_err=float(raw_case["max_abs_error"]),
                mean_err=float(raw_case["mean_abs_error"]),
                forced_bad="" if forced_bad is None else f"{float(forced_bad):.8g}",
                tiles=raw_case["omitted_tiles"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this verifier is still a CPU/theory gate for the support",
            "certificate. It is paired with focused Metal interval parity tests that",
            "exercise the same per-trace precision in forward/backward rendering. For a local",
            "projective/gauged footprint with SPD precision `P`, omitted support is",
            "bounded by `opacity * exp(-0.5 * min_rect (x-mu)^T P (x-mu))`, summed",
            "per omitted tile for overlapping traces.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail-alpha-epsilon", type=float, default=1.0e-3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound"),
    )
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()
    if args.verify_report is not None:
        payload = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_anisotropic_tail_bound_report(payload)
        print(f"verified {args.verify_report}")
        return
    if args.tail_alpha_epsilon <= 0.0:
        raise ValueError("--tail-alpha-epsilon must be positive")
    payload = run_verifier(tail_alpha_epsilon=float(args.tail_alpha_epsilon))
    assert_anisotropic_tail_bound_report(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(payload, args.out_dir / "summary.md")
    print(args.out_dir / "summary.md")
    if not bool(payload["all_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
