from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    UVTRenderConfig,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
    render_projective_trace_cell_interval_atlas_metal,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_camera_family_2d_metal_lowering"
)

TRACE_COUNT = 2
COEFF_COUNT = 9
Q2_BASIS_COUNT = 6
FLOAT_BYTES = 4


def _q2_basis(q_phase: float, q_height: float, *, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(
        (
            1.0,
            float(q_phase),
            float(q_height),
            float(q_phase) * float(q_height),
            float(q_phase) * float(q_phase),
            float(q_height) * float(q_height),
        ),
        dtype=torch.float32,
        device=device,
    )


def _family_coeff_table(*, device: torch.device | str = "cpu") -> torch.Tensor:
    """Q2 coefficients for ordinary [u0,u1,u2,v0,v1,v2,z0,z1,z2] time traces."""

    base = torch.tensor(
        [
            [3.50, 0.25, 0.00, 3.50, 0.08, 0.00, 1.00, 0.10, 0.00],
            [4.60, -0.18, 0.00, 3.20, 0.12, 0.00, 1.80, -0.06, 0.00],
        ],
        dtype=torch.float32,
        device=device,
    )
    q_phase = torch.tensor(
        [
            [0.25, 0.03, 0.00, -0.12, 0.02, 0.00, 0.05, 0.01, 0.00],
            [-0.16, -0.02, 0.00, 0.18, 0.01, 0.00, -0.04, 0.01, 0.00],
        ],
        dtype=torch.float32,
        device=device,
    )
    q_height = torch.tensor(
        [
            [-0.10, -0.01, 0.00, 0.22, 0.02, 0.00, 0.03, 0.00, 0.00],
            [0.14, 0.02, 0.00, -0.16, -0.01, 0.00, 0.02, -0.01, 0.00],
        ],
        dtype=torch.float32,
        device=device,
    )
    cross = torch.tensor(
        [
            [0.05, 0.00, 0.00, -0.04, 0.00, 0.00, 0.01, 0.00, 0.00],
            [-0.03, 0.00, 0.00, 0.05, 0.00, 0.00, -0.01, 0.00, 0.00],
        ],
        dtype=torch.float32,
        device=device,
    )
    phase2 = torch.tensor(
        [
            [-0.04, 0.00, 0.00, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.03, 0.00, 0.00, -0.02, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
        device=device,
    )
    height2 = torch.tensor(
        [
            [0.02, 0.00, 0.00, -0.03, 0.00, 0.00, 0.00, 0.00, 0.00],
            [-0.02, 0.00, 0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
        device=device,
    )
    return torch.stack((base, q_phase, q_height, cross, phase2, height2), dim=-1).contiguous()


def _trace_color(*, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([[1.0, 0.1, 0.05], [0.05, 0.25, 1.0]], dtype=torch.float32, device=device)


def _trace_opacity(*, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([0.65, 0.45], dtype=torch.float32, device=device)


def lower_q2_family_coeffs(
    family_coeffs: torch.Tensor,
    *,
    q_phase: float,
    q_height: float,
) -> torch.Tensor:
    basis = _q2_basis(q_phase, q_height, device=family_coeffs.device)
    return torch.einsum("nkb,b->nk", family_coeffs, basis).to(dtype=torch.float32).contiguous()


def _depth_interval(coeff: torch.Tensor, times: torch.Tensor) -> tuple[float, float]:
    depth = coeff[6] + coeff[7] * times + coeff[8] * times.square()
    return (float(depth.min().item()), float(depth.max().item()))


def _atlas_from_coeffs(
    coeffs: torch.Tensor,
    *,
    frames: int,
    device: torch.device | str,
) -> ProjectiveTraceCellTraceAtlas:
    coeffs = coeffs.to(device=device, dtype=torch.float32).contiguous()
    opacity = _trace_opacity(device=device).contiguous()
    color = _trace_color(device=device).contiguous()
    times_cpu = torch.linspace(0.0, 1.0, int(frames), dtype=torch.float32)
    mid_t = torch.tensor(0.5, dtype=torch.float32, device=coeffs.device)
    mid_depth = coeffs[:, 6] + coeffs[:, 7] * mid_t + coeffs[:, 8] * mid_t * mid_t
    order = tuple(int(idx) for idx in torch.argsort(mid_depth).detach().cpu().tolist())
    coeffs_cpu = coeffs.detach().cpu()
    depth_intervals = tuple(_depth_interval(coeffs_cpu[idx], times_cpu) for idx in range(int(coeffs.shape[0])))
    cell = ProjectiveTraceTileTimeCell(
        tile_u=0,
        tile_v=0,
        start=0,
        stop=int(frames),
        primitive_ids=tuple(range(int(coeffs.shape[0]))),
        ordered_primitive_ids=order,
        depth_intervals=depth_intervals,
        fallback=False,
        fallback_reasons=(),
    )
    trace_count = int(coeffs.shape[0])
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=opacity,
        color=color,
        cells=[cell],
        source_window_indices=tuple(0 for _ in range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(int(frames) for _ in range(trace_count)),
    )


def _render_config(*, frames: int, image_size: int, tile_size: int) -> UVTRenderConfig:
    return UVTRenderConfig(
        height=int(image_size),
        width=int(image_size),
        frames=int(frames),
        tile_x=int(tile_size),
        tile_y=int(tile_size),
        tile_t=1,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )


def _apply_metal_tile_env(config: UVTRenderConfig) -> None:
    os.environ["STAR_UVT_TILE_X"] = str(config.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(config.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(config.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(config.tile_capacity)


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _slice_payload_bytes(atlas: ProjectiveTraceCellTraceAtlas) -> int:
    return (
        _tensor_bytes(atlas.coeffs)
        + _tensor_bytes(atlas.opacity)
        + _tensor_bytes(atlas.color)
        + _tensor_bytes(atlas.opacity_time_coeffs)
        + _tensor_bytes(atlas.spatial_precision_uv)
        + _tensor_bytes(atlas.depth_affine_uv)
    )


def _family_payload_bytes(family_coeffs: torch.Tensor) -> int:
    return _tensor_bytes(family_coeffs) + _tensor_bytes(_trace_opacity()) + _tensor_bytes(_trace_color())


def _metal_row(
    *,
    q_phase: float,
    q_height: float,
    family_coeffs: torch.Tensor,
    times: torch.Tensor,
    config: UVTRenderConfig,
    sigma_px: float,
) -> dict[str, Any]:
    coeffs = lower_q2_family_coeffs(family_coeffs, q_phase=q_phase, q_height=q_height)
    atlas = _atlas_from_coeffs(coeffs, frames=int(config.frames), device=times.device)
    image = render_projective_trace_cell_interval_atlas_metal(atlas, times, config, sigma_px=float(sigma_px))
    sample_count = int(image.numel())
    grad_image = torch.linspace(-0.20, 0.35, steps=sample_count, dtype=torch.float32, device=times.device)
    grad_image = grad_image.reshape_as(image).contiguous()
    grads = direct_backward_projective_trace_cell_interval_atlas_metal(
        atlas,
        times,
        grad_image,
        config,
        sigma_px=float(sigma_px),
    )
    torch.mps.synchronize()
    coeffs_cpu = coeffs.detach().cpu()
    return {
        "q_phase": float(q_phase),
        "q_height": float(q_height),
        "coeff_checksum": float(coeffs_cpu.sum().item()),
        "slice_payload_bytes": _slice_payload_bytes(atlas),
        "ordered_primitive_ids": list(atlas.cells[0].ordered_primitive_ids),
        "image_sum": float(image.sum().detach().cpu().item()),
        "image_max": float(image.max().detach().cpu().item()),
        "grad_coeff_abs_sum": float(grads.grad_coeffs.abs().sum().detach().cpu().item()),
        "grad_opacity_abs_sum": float(grads.grad_opacity.abs().sum().detach().cpu().item()),
        "grad_color_abs_sum": float(grads.grad_color.abs().sum().detach().cpu().item()),
    }


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    family_payload = int(report["family_payload_bytes"])
    slice_payload = int(report["slice_payload_bytes"])
    q_pair_count = int(report["q_pair_count"])
    replay_payload = slice_payload * q_pair_count
    return {
        "q_axis_count": int(report["q_axis_count"]),
        "q_pair_count": q_pair_count,
        "trace_count": int(report["trace_count"]),
        "family_basis_count": int(report["family_basis_count"]),
        "family_payload_bytes": family_payload,
        "slice_payload_bytes": slice_payload,
        "per_q_replay_payload_bytes": replay_payload,
        "family_to_replay_payload_ratio": float(family_payload) / float(replay_payload),
        "peak_slice_to_replay_payload_ratio": float(slice_payload) / float(replay_payload),
        "metal_forward_rows": len(rows),
        "metal_backward_rows": len(rows),
        "min_image_sum": min(float(row["image_sum"]) for row in rows),
        "max_image_sum": max(float(row["image_sum"]) for row in rows),
        "min_grad_coeff_abs_sum": min(float(row["grad_coeff_abs_sum"]) for row in rows),
        "min_grad_opacity_abs_sum": min(float(row["grad_opacity_abs_sum"]) for row in rows),
        "min_grad_color_abs_sum": min(float(row["grad_color_abs_sum"]) for row in rows),
    }


def run_report(
    *,
    q_axis_count: int = 5,
    frames: int = 4,
    image_size: int = 8,
    tile_size: int = 8,
    sigma_px: float = 1.7,
) -> dict[str, Any]:
    q_phase_values = torch.linspace(-0.30, 0.30, int(q_axis_count), dtype=torch.float32)
    q_height_values = torch.linspace(-0.24, 0.24, int(q_axis_count), dtype=torch.float32)
    config = _render_config(frames=int(frames), image_size=int(image_size), tile_size=int(tile_size))
    interval_metal_available = bool(torch.backends.mps.is_available() and has_projective_trace_cell_interval_metal())
    interval_backward_metal_available = bool(
        torch.backends.mps.is_available() and has_projective_trace_cell_interval_backward_metal()
    )
    if not interval_metal_available or not interval_backward_metal_available:
        return {
            "status": "failed",
            "benchmark": "star_uvt_projective_camera_family_2d_metal_lowering",
            "base_domain": "Q2 x Omega x T -> Omega x T Metal slice",
            "theory_contract": "A Q2 camera-family trace chart is lowered to an Omega x T slice of pi_* Gamma^* world primitives for the existing interval Metal forward/backward path.",
            "interval_metal_available": interval_metal_available,
            "interval_backward_metal_available": interval_backward_metal_available,
            "metal_ran": False,
            "errors": ["MPS interval Metal forward/backward is required for this saved evidence artifact."],
            "rows": [],
            "summary": {},
        }

    device = torch.device("mps")
    _apply_metal_tile_env(config)
    family_coeffs = _family_coeff_table(device=device)
    template_coeffs = lower_q2_family_coeffs(family_coeffs, q_phase=0.0, q_height=0.0)
    template_atlas = _atlas_from_coeffs(template_coeffs, frames=int(frames), device=device)
    times = torch.linspace(0.0, 1.0, int(frames), dtype=torch.float32, device=device).contiguous()
    rows = [
        _metal_row(
            q_phase=float(q_phase),
            q_height=float(q_height),
            family_coeffs=family_coeffs,
            times=times,
            config=config,
            sigma_px=float(sigma_px),
        )
        for q_phase in q_phase_values
        for q_height in q_height_values
    ]
    torch.mps.synchronize()
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_metal_lowering",
        "base_domain": "Q2 x Omega x T -> Omega x T Metal slice",
        "theory_contract": "A Q2 camera-family trace chart is lowered to an Omega x T slice of pi_* Gamma^* world primitives for the existing interval Metal forward/backward path. This is a slice-lowering smoke, not native Q2 Metal evaluation.",
        "interval_metal_available": interval_metal_available,
        "interval_backward_metal_available": interval_backward_metal_available,
        "metal_ran": True,
        "q_axis_count": int(q_axis_count),
        "q_pair_count": int(q_axis_count) * int(q_axis_count),
        "frames": int(frames),
        "image_size": int(image_size),
        "trace_count": TRACE_COUNT,
        "family_basis_count": Q2_BASIS_COUNT,
        "family_payload_bytes": _family_payload_bytes(family_coeffs),
        "slice_payload_bytes": _slice_payload_bytes(template_atlas),
        "rows": rows,
    }
    report["summary"] = summarize(report)
    errors = verify_camera_family_2d_metal_lowering_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _assert_summary_close(actual: Any, expected: Any, key: str, errors: list[str]) -> None:
    if isinstance(expected, float):
        if not _finite_float(actual) or abs(float(actual) - expected) > 1.0e-9:
            errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")


def verify_camera_family_2d_metal_lowering_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_metal_lowering":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T -> Omega x T Metal slice":
        errors.append(f"base_domain must name the Q2-to-interval-Metal lowering, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if not isinstance(theory_contract, str) or "pi_* Gamma^*" not in theory_contract or "slice-lowering" not in theory_contract:
        errors.append("theory_contract must preserve the pi_* Gamma^* slice-lowering contract")
    if report.get("metal_ran") is not True:
        errors.append("metal_ran must be true for this evidence artifact")
    if report.get("interval_metal_available") is not True or report.get("interval_backward_metal_available") is not True:
        errors.append("interval Metal forward/backward availability must both be true")

    q_axis_count = report.get("q_axis_count")
    q_pair_count = report.get("q_pair_count")
    if not isinstance(q_axis_count, int) or q_axis_count < 3:
        errors.append(f"q_axis_count must be an int >= 3, got {q_axis_count!r}")
        q_axis_count = 0
    if not isinstance(q_pair_count, int) or q_pair_count != int(q_axis_count) * int(q_axis_count):
        errors.append(f"q_pair_count must equal q_axis_count^2, got {q_pair_count!r}")
        q_pair_count = 0
    if report.get("trace_count") != TRACE_COUNT:
        errors.append(f"trace_count must be {TRACE_COUNT}, got {report.get('trace_count')!r}")
    if report.get("family_basis_count") != Q2_BASIS_COUNT:
        errors.append(f"family_basis_count must be {Q2_BASIS_COUNT}, got {report.get('family_basis_count')!r}")

    rows = report.get("rows")
    if not isinstance(rows, list):
        errors.append("rows must be a list")
        rows = []
    elif len(rows) != int(q_pair_count):
        errors.append(f"rows must contain one Metal row per q-pair, expected {q_pair_count}, got {len(rows)}")

    seen_pairs: set[tuple[float, float]] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {idx} must be an object")
            continue
        q_phase = row.get("q_phase")
        q_height = row.get("q_height")
        if not _finite_float(q_phase) or not _finite_float(q_height):
            errors.append(f"row {idx} q coordinates must be finite")
        else:
            seen_pairs.add((round(float(q_phase), 6), round(float(q_height), 6)))
        if row.get("ordered_primitive_ids") != [0, 1]:
            errors.append(f"row {idx} must preserve stable front-to-back order [0, 1]")
        for key in (
            "coeff_checksum",
            "image_sum",
            "image_max",
            "grad_coeff_abs_sum",
            "grad_opacity_abs_sum",
            "grad_color_abs_sum",
        ):
            if not _finite_float(row.get(key)):
                errors.append(f"row {idx} {key} must be finite, got {row.get(key)!r}")
        for key in ("image_sum", "image_max", "grad_coeff_abs_sum", "grad_opacity_abs_sum", "grad_color_abs_sum"):
            if _finite_float(row.get(key)) and float(row[key]) <= 1.0e-6:
                errors.append(f"row {idx} {key} must be nonzero, got {row[key]!r}")

    if rows and len(seen_pairs) != len(rows):
        errors.append("q-pair rows must be unique")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        try:
            expected = summarize(report)
            for key, value in expected.items():
                _assert_summary_close(summary.get(key), value, key, errors)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"summary could not be recomputed: {exc}")
        if (
            _finite_float(summary.get("family_to_replay_payload_ratio"))
            and float(summary["family_to_replay_payload_ratio"]) >= 0.35
        ):
            errors.append("family-to-replay payload ratio must stay below 0.35 for the Q2 lowering grid")
        if (
            _finite_float(summary.get("peak_slice_to_replay_payload_ratio"))
            and float(summary["peak_slice_to_replay_payload_ratio"]) >= 0.10
        ):
            errors.append("peak slice-to-replay payload ratio must stay below 0.10")
        for key in ("min_image_sum", "min_grad_coeff_abs_sum", "min_grad_opacity_abs_sum", "min_grad_color_abs_sum"):
            if _finite_float(summary.get(key)) and float(summary[key]) <= 1.0e-6:
                errors.append(f"summary {key} must be nonzero")
    return errors


def assert_camera_family_2d_metal_lowering_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_metal_lowering_report(report)
    if errors:
        raise AssertionError("camera-family 2D Metal lowering report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Camera-Family 2D Metal Lowering",
        "",
        "This is a slice-lowering smoke: it does not claim native Q2 Metal evaluation.",
        "",
        "## Contract",
        "",
        report["theory_contract"],
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
    parser.add_argument("--verify-report", type=Path, default=None)
    parser.add_argument("--q-axis-count", type=int, default=5)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_camera_family_2d_metal_lowering_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(q_axis_count=int(args.q_axis_count), frames=int(args.frames))
    assert_camera_family_2d_metal_lowering_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
