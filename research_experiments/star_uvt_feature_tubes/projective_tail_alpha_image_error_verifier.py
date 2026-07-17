from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from star_uvt_runtime import ensure_star_uvt_on_path


ensure_star_uvt_on_path()

from research_project.trainer_harness.tile_metal_autograd import (  # noqa: E402
    refresh_projective_cell_interval_atlas_if_stale,
)
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    projective_trace_windows_to_cell_trace_atlas,
    render_projective_trace_cell_atlas_reference,
    split_projective_trace_windows,
)


@dataclass(frozen=True)
class ImageErrorCaseResult:
    name: str
    expectation: str
    passed: bool
    strict_rebinned: bool
    certified_rebinned: bool
    certified_reused: bool
    tail_alpha_epsilon: float
    support_tail_alpha_bound: float
    max_support_overshoot_px: float
    max_abs_error: float
    mean_abs_error: float
    forced_bad_max_abs_error: float | None = None
    notes: str = ""


def _single_trace_atlas(
    *,
    center_u: float,
    center_v: float,
    depth: float,
    frames: int,
    opacity: float,
    tile_u: int = 0,
    tile_v: int = 0,
    color: Sequence[float] = (1.0, 1.0, 1.0),
) -> ProjectiveTraceCellTraceAtlas:
    return ProjectiveTraceCellTraceAtlas(
        coeffs=torch.tensor(
            [[center_u, 0.0, 0.0, center_v, 0.0, 0.0, depth, 0.0, 0.0]],
            dtype=torch.float32,
        ).contiguous(),
        opacity=torch.tensor([opacity], dtype=torch.float32),
        color=torch.tensor([list(color)], dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=int(tile_u),
                tile_v=int(tile_v),
                start=0,
                stop=int(frames),
                primitive_ids=(0,),
                ordered_primitive_ids=(0,),
                depth_intervals=((float(depth), float(depth)),),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=(0,),
        source_primitive_ids=(0,),
        active_start=(0,),
        active_stop=(int(frames),),
    )


def _render(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    sigma_px: float,
) -> torch.Tensor:
    return render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=int(image_width),
        image_height=int(image_height),
        tile_size=int(tile_size),
        sigma_px=float(sigma_px),
    )


def _positive_tail_case(
    *,
    name: str,
    center_u: float,
    center_v: float,
    uv_padding: float,
    sigma_px: float,
    opacity: float,
    tail_alpha_epsilon: float,
    image_width: int = 16,
    image_height: int = 8,
    tile_size: int = 8,
    frames: int = 2,
) -> ImageErrorCaseResult:
    times = torch.arange(int(frames), dtype=torch.float32).contiguous()
    stale_atlas = _single_trace_atlas(
        center_u=float(center_u),
        center_v=float(center_v),
        depth=1.0,
        frames=int(frames),
        opacity=float(opacity),
    )
    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        stale_atlas,
        times,
        image_width=int(image_width),
        image_height=int(image_height),
        tile_size=int(tile_size),
        uv_padding=float(uv_padding),
        sigma_px=float(sigma_px),
        check_visibility=False,
    )
    certified_refresh = refresh_projective_cell_interval_atlas_if_stale(
        stale_atlas,
        times,
        image_width=int(image_width),
        image_height=int(image_height),
        tile_size=int(tile_size),
        uv_padding=float(uv_padding),
        sigma_px=float(sigma_px),
        support_stale_tail_alpha_epsilon=float(tail_alpha_epsilon),
        check_visibility=False,
    )
    strict_image = _render(
        strict_refresh.atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        sigma_px=sigma_px,
    )
    certified_image = _render(
        certified_refresh.atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        sigma_px=sigma_px,
    )
    diff = (strict_image - certified_image).abs()
    tail_bound = float(certified_refresh.support_tail_alpha_bound_before)
    max_error = float(diff.max().item())
    passed = (
        bool(strict_refresh.rebinned)
        and not bool(certified_refresh.rebinned)
        and tail_bound <= float(tail_alpha_epsilon)
        and max_error <= max(tail_bound * 1.05 + 1.0e-7, 1.0e-7)
    )
    return ImageErrorCaseResult(
        name=name,
        expectation="certified_tail_reuse",
        passed=passed,
        strict_rebinned=bool(strict_refresh.rebinned),
        certified_rebinned=bool(certified_refresh.rebinned),
        certified_reused=not bool(certified_refresh.rebinned),
        tail_alpha_epsilon=float(tail_alpha_epsilon),
        support_tail_alpha_bound=tail_bound,
        max_support_overshoot_px=float(certified_refresh.support_margin_before.max_boundary_overshoot_px),
        max_abs_error=max_error,
        mean_abs_error=float(diff.mean().item()),
        notes="strict rebin is the reference; certified reuse must stay under the omitted-alpha bound",
    )


def _orbit_tail_case(*, tail_alpha_epsilon: float) -> ImageErrorCaseResult:
    theta = torch.linspace(-math.radians(3.0), math.radians(3.0), 4, dtype=torch.float32)
    times = torch.tan(0.5 * theta).contiguous()
    point_x = 0.05
    base_depth = 2.6
    vertical = 0.02
    center_u = 11.92
    center_v = 8.0
    raw_u = torch.tensor([point_x, 2.0, -point_x], dtype=torch.float32)
    raw_v = torch.tensor([vertical, 0.0, vertical], dtype=torch.float32)
    depth = torch.tensor([base_depth + 0.25, 2.0 * point_x, base_depth - 0.25], dtype=torch.float32)
    pixel_u = center_u * depth + raw_u
    pixel_v = center_v * depth + raw_v
    coeffs = torch.tensor([[*pixel_u.tolist(), *pixel_v.tolist(), *depth.tolist()]], dtype=torch.float32).contiguous()
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=0.01,
        min_denominator_abs=1.0e-3,
        min_samples=2,
    )
    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=torch.tensor([0.5], dtype=torch.float32),
        color=torch.ones((1, 3), dtype=torch.float32),
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
    )
    moved_coeffs = atlas.coeffs.clone()
    moved_coeffs[:, 0] += 0.10
    moved_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=moved_coeffs,
        opacity=atlas.opacity,
        color=atlas.color,
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )

    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        moved_atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
        sigma_px=1.0,
        check_visibility=False,
    )
    certified_refresh = refresh_projective_cell_interval_atlas_if_stale(
        moved_atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        uv_padding=4.0,
        sigma_px=1.0,
        support_stale_tail_alpha_epsilon=float(tail_alpha_epsilon),
        check_visibility=False,
    )
    strict_image = _render(
        strict_refresh.atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        sigma_px=1.0,
    )
    certified_image = _render(
        certified_refresh.atlas,
        times,
        image_width=32,
        image_height=16,
        tile_size=16,
        sigma_px=1.0,
    )
    diff = (strict_image - certified_image).abs()
    tail_bound = float(certified_refresh.support_tail_alpha_bound_before)
    max_error = float(diff.max().item())
    passed = (
        bool(strict_refresh.rebinned)
        and not bool(certified_refresh.rebinned)
        and tail_bound <= float(tail_alpha_epsilon)
        and max_error <= max(tail_bound * 1.05 + 1.0e-7, 1.0e-7)
    )
    return ImageErrorCaseResult(
        name="orbit_rational_tail_reuse",
        expectation="certified_tail_reuse",
        passed=passed,
        strict_rebinned=bool(strict_refresh.rebinned),
        certified_rebinned=bool(certified_refresh.rebinned),
        certified_reused=not bool(certified_refresh.rebinned),
        tail_alpha_epsilon=float(tail_alpha_epsilon),
        support_tail_alpha_bound=tail_bound,
        max_support_overshoot_px=float(certified_refresh.support_margin_before.max_boundary_overshoot_px),
        max_abs_error=max_error,
        mean_abs_error=float(diff.mean().item()),
        notes="same certificate on a tiny rational revolving-camera chart",
    )


def _core_loss_rejection_case(*, tail_alpha_epsilon: float) -> ImageErrorCaseResult:
    times = torch.arange(2, dtype=torch.float32).contiguous()
    atlas = _single_trace_atlas(center_u=8.05, center_v=4.0, depth=1.0, frames=2, opacity=0.5)
    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=0.0,
        sigma_px=1.0,
        check_visibility=False,
    )
    certified_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=0.0,
        sigma_px=1.0,
        support_stale_tail_alpha_epsilon=float(tail_alpha_epsilon),
        check_visibility=False,
    )
    forced_bad_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=0.0,
        support_stale_overshoot_epsilon=0.10,
        check_visibility=False,
    )
    strict_image = _render(
        strict_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    certified_image = _render(
        certified_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    forced_bad_image = _render(
        forced_bad_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    diff = (strict_image - certified_image).abs()
    forced_bad_diff = (strict_image - forced_bad_image).abs()
    tail_bound = float(certified_refresh.support_tail_alpha_bound_before)
    forced_bad = float(forced_bad_diff.max().item())
    passed = (
        bool(strict_refresh.rebinned)
        and bool(certified_refresh.rebinned)
        and tail_bound > float(tail_alpha_epsilon)
        and bool(forced_bad_refresh.before.stale)
        and not bool(forced_bad_refresh.rebinned)
        and forced_bad > 0.35
    )
    return ImageErrorCaseResult(
        name="core_loss_rejected",
        expectation="tail_certificate_rejects_core_loss",
        passed=passed,
        strict_rebinned=bool(strict_refresh.rebinned),
        certified_rebinned=bool(certified_refresh.rebinned),
        certified_reused=not bool(certified_refresh.rebinned),
        tail_alpha_epsilon=float(tail_alpha_epsilon),
        support_tail_alpha_bound=tail_bound,
        max_support_overshoot_px=float(certified_refresh.support_margin_before.max_boundary_overshoot_px),
        max_abs_error=float(diff.max().item()),
        mean_abs_error=float(diff.mean().item()),
        forced_bad_max_abs_error=forced_bad,
        notes="pixel overshoot would reuse the stale atlas, but the alpha certificate rejects the missing core",
    )


def _overlapping_tail_aggregation_case(*, tail_alpha_epsilon: float) -> ImageErrorCaseResult:
    frames = 2
    trace_count = 64
    times = torch.arange(frames, dtype=torch.float32).contiguous()
    coeffs = torch.tensor(
        [[4.05, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0 + 0.001 * index, 0.0, 0.0] for index in range(trace_count)],
        dtype=torch.float32,
    ).contiguous()
    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=torch.full((trace_count,), 0.5, dtype=torch.float32),
        color=torch.ones((trace_count, 3), dtype=torch.float32),
        cells=[
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=0,
                stop=frames,
                primitive_ids=tuple(range(trace_count)),
                ordered_primitive_ids=tuple(range(trace_count)),
                depth_intervals=tuple((1.0 + 0.001 * index, 1.0 + 0.001 * index) for index in range(trace_count)),
                fallback=False,
                fallback_reasons=(),
            )
        ],
        source_window_indices=tuple(range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(frames for _ in range(trace_count)),
    )
    strict_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        sigma_px=1.0,
        check_visibility=False,
    )
    certified_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        sigma_px=1.0,
        support_stale_tail_alpha_epsilon=float(tail_alpha_epsilon),
        check_visibility=False,
    )
    aggregate_tail_bound = float(certified_refresh.support_tail_alpha_bound_before)
    forced_bad_refresh = refresh_projective_cell_interval_atlas_if_stale(
        atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        uv_padding=4.0,
        sigma_px=1.0,
        support_stale_tail_alpha_epsilon=max(aggregate_tail_bound * 1.1, float(tail_alpha_epsilon) * 2.0),
        check_visibility=False,
    )
    strict_image = _render(
        strict_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    certified_image = _render(
        certified_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    forced_bad_image = _render(
        forced_bad_refresh.atlas,
        times,
        image_width=16,
        image_height=8,
        tile_size=8,
        sigma_px=1.0,
    )
    diff = (strict_image - certified_image).abs()
    forced_bad_diff = (strict_image - forced_bad_image).abs()
    forced_bad = float(forced_bad_diff.max().item())
    passed = (
        bool(strict_refresh.rebinned)
        and bool(certified_refresh.rebinned)
        and aggregate_tail_bound > float(tail_alpha_epsilon)
        and not bool(forced_bad_refresh.rebinned)
        and forced_bad > float(tail_alpha_epsilon)
    )
    return ImageErrorCaseResult(
        name="overlapping_tail_aggregate_rejected",
        expectation="tail_certificate_sums_overlapping_omitted_tails",
        passed=passed,
        strict_rebinned=bool(strict_refresh.rebinned),
        certified_rebinned=bool(certified_refresh.rebinned),
        certified_reused=not bool(certified_refresh.rebinned),
        tail_alpha_epsilon=float(tail_alpha_epsilon),
        support_tail_alpha_bound=aggregate_tail_bound,
        max_support_overshoot_px=float(certified_refresh.support_margin_before.max_boundary_overshoot_px),
        max_abs_error=float(diff.max().item()),
        mean_abs_error=float(diff.mean().item()),
        forced_bad_max_abs_error=forced_bad,
        notes="many individually tiny tails on the same tile must be bounded by their aggregate, not by max per trace",
    )


def run_verifier(*, tail_alpha_epsilon: float) -> dict[str, object]:
    cases = [
        _positive_tail_case(
            name="axis_r4_sigma1_opacity05",
            center_u=4.05,
            center_v=4.0,
            uv_padding=4.0,
            sigma_px=1.0,
            opacity=0.5,
            tail_alpha_epsilon=tail_alpha_epsilon,
        ),
        _positive_tail_case(
            name="axis_r5_sigma125_opacity08",
            center_u=3.05,
            center_v=4.0,
            uv_padding=5.0,
            sigma_px=1.25,
            opacity=0.8,
            tail_alpha_epsilon=tail_alpha_epsilon,
        ),
        _positive_tail_case(
            name="axis_r6_sigma15_opacity09",
            center_u=2.05,
            center_v=4.0,
            uv_padding=6.0,
            sigma_px=1.5,
            opacity=0.9,
            tail_alpha_epsilon=tail_alpha_epsilon,
        ),
        _orbit_tail_case(tail_alpha_epsilon=tail_alpha_epsilon),
        _core_loss_rejection_case(tail_alpha_epsilon=tail_alpha_epsilon),
        _overlapping_tail_aggregation_case(tail_alpha_epsilon=tail_alpha_epsilon),
    ]
    return {
        "tail_alpha_epsilon": float(tail_alpha_epsilon),
        "all_passed": all(case.passed for case in cases),
        "cases": [asdict(case) for case in cases],
    }


def verify_tail_alpha_image_error_report(payload: dict[str, object]) -> list[str]:
    """Return contract failures for a saved tail-alpha image-error report."""

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
        "axis_r4_sigma1_opacity05",
        "axis_r5_sigma125_opacity08",
        "axis_r6_sigma15_opacity09",
        "orbit_rational_tail_reuse",
        "core_loss_rejected",
        "overlapping_tail_aggregate_rejected",
    }
    missing = sorted(required_names - set(by_name))
    if missing:
        errors.append(f"missing required cases: {missing}")

    positive_cases = [case for case in cases if case.get("expectation") == "certified_tail_reuse"]
    if len(positive_cases) < 4:
        errors.append("need at least four certified_tail_reuse positive cases")
    for case in positive_cases:
        name = str(case.get("name"))
        tail_bound = float(case.get("support_tail_alpha_bound") or 0.0)
        max_error = float(case.get("max_abs_error") or 0.0)
        mean_error = float(case.get("mean_abs_error") or 0.0)
        if not bool(case.get("passed")):
            errors.append(f"{name} did not pass")
        if not bool(case.get("strict_rebinned")):
            errors.append(f"{name} must strict-rebin before certification")
        if bool(case.get("certified_rebinned")) or not bool(case.get("certified_reused")):
            errors.append(f"{name} must reuse certified stale support")
        if not 0.0 < tail_bound <= tail_alpha:
            errors.append(f"{name} tail bound must be in (0, epsilon], got {tail_bound}")
        if max_error > tail_bound * 1.05 + 1.0e-7:
            errors.append(f"{name} max image error {max_error} exceeds tail bound {tail_bound}")
        if mean_error > max_error + 1.0e-12:
            errors.append(f"{name} mean image error {mean_error} exceeds max image error {max_error}")

    for name, expectation in (
        ("core_loss_rejected", "tail_certificate_rejects_core_loss"),
        ("overlapping_tail_aggregate_rejected", "tail_certificate_sums_overlapping_omitted_tails"),
    ):
        case = by_name.get(name)
        if case is None:
            continue
        tail_bound = float(case.get("support_tail_alpha_bound") or 0.0)
        forced_bad = case.get("forced_bad_max_abs_error")
        if case.get("expectation") != expectation:
            errors.append(f"{name} expectation must be {expectation!r}")
        if not bool(case.get("passed")):
            errors.append(f"{name} did not pass")
        if not bool(case.get("certified_rebinned")) or bool(case.get("certified_reused")):
            errors.append(f"{name} must reject stale reuse and rebin")
        if tail_bound <= tail_alpha:
            errors.append(f"{name} tail bound must exceed epsilon, got {tail_bound} <= {tail_alpha}")
        if forced_bad is None or float(forced_bad) <= tail_alpha:
            errors.append(f"{name} forced-bad image error must exceed epsilon, got {forced_bad!r}")

    return errors


def assert_tail_alpha_image_error_report(payload: dict[str, object]) -> None:
    errors = verify_tail_alpha_image_error_report(payload)
    if errors:
        raise AssertionError("tail-alpha image-error report failed:\n- " + "\n- ".join(errors))


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    cases = payload["cases"]
    assert isinstance(cases, list)
    lines = [
        "# Projective Tail-Alpha Image Error Verifier",
        "",
        "This verifier compares strict support rebinning against certified stale",
        "support reuse. Positive cases must keep image error below the omitted",
        "tail-alpha bound; the negative case proves missing core support still",
        "rebins and that a pure pixel-overshoot pardon would be unsafe.",
        "",
        f"- tail_alpha_epsilon: `{payload['tail_alpha_epsilon']}`",
        f"- all_passed: `{payload['all_passed']}`",
        "",
        "| case | expectation | pass | strict_rebinned | certified_reused | tail_bound | overshoot_px | max_abs_error | mean_abs_error | forced_bad_max_abs_error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for raw_case in cases:
        assert isinstance(raw_case, dict)
        forced_bad = raw_case.get("forced_bad_max_abs_error")
        lines.append(
            "| {name} | {expectation} | {passed} | {strict_rebinned} | {certified_reused} | "
            "{tail:.8g} | {overshoot:.6g} | {max_err:.8g} | {mean_err:.8g} | {forced_bad} |".format(
                name=raw_case["name"],
                expectation=raw_case["expectation"],
                passed=raw_case["passed"],
                strict_rebinned=raw_case["strict_rebinned"],
                certified_reused=raw_case["certified_reused"],
                tail=float(raw_case["support_tail_alpha_bound"]),
                overshoot=float(raw_case["max_support_overshoot_px"]),
                max_err=float(raw_case["max_abs_error"]),
                mean_err=float(raw_case["mean_abs_error"]),
                forced_bad="" if forced_bad is None else f"{float(forced_bad):.8g}",
            )
        )
    lines.append("")
    lines.append("Interpretation: this is still a local isotropic/projective check, not a")
    lines.append("global renderer theorem. It strengthens the cache-policy artifact by")
    lines.append("binding skipped support to observed image residuals and a core-loss")
    lines.append("negative control.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail-alpha-epsilon", type=float, default=1.0e-3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error"),
    )
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        payload = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_tail_alpha_image_error_report(payload)
        print(f"verified {args.verify_report}")
        return

    if args.tail_alpha_epsilon <= 0.0:
        raise ValueError("--tail-alpha-epsilon must be positive")
    payload = run_verifier(tail_alpha_epsilon=float(args.tail_alpha_epsilon))
    assert_tail_alpha_image_error_report(payload)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "summary.json"
    md_path = args.out_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload, md_path)
    print(md_path)
    if not bool(payload["all_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
