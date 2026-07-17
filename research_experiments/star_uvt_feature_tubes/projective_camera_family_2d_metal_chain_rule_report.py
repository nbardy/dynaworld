from __future__ import annotations

import argparse
import json
import math
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
    UVTRenderConfig,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
    render_projective_trace_cell_interval_atlas_metal,
)

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (  # noqa: E402
    Q2_BASIS_COUNT,
    TRACE_COUNT,
    _apply_metal_tile_env,
    _atlas_from_coeffs,
    _family_coeff_table,
    _family_payload_bytes,
    _q2_basis,
    _render_config,
    _slice_payload_bytes,
    _tensor_bytes,
    lower_q2_family_coeffs,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_camera_family_2d_metal_chain_rule"
)


def _q_grid(q_axis_count: int) -> list[tuple[float, float]]:
    q_phase_values = torch.linspace(-0.30, 0.30, int(q_axis_count), dtype=torch.float32)
    q_height_values = torch.linspace(-0.24, 0.24, int(q_axis_count), dtype=torch.float32)
    return [(float(q_phase), float(q_height)) for q_phase in q_phase_values for q_height in q_height_values]


def _grad_image(config: UVTRenderConfig, *, device: torch.device | str, row_index: int) -> torch.Tensor:
    sample_count = int(config.frames) * int(config.height) * int(config.width) * 3
    base = torch.linspace(-0.18, 0.31, steps=sample_count, dtype=torch.float32, device=device)
    # Make neighboring q-pairs produce different adjoints so the family reduction
    # is not only a repeated scalar multiple of one slice.
    scale = 1.0 + 0.017 * float(row_index)
    bias = 0.0025 * float((row_index % 5) - 2)
    return (scale * base + bias).reshape(int(config.frames), int(config.height), int(config.width), 3).contiguous()


def _row_objective(
    family_coeffs: torch.Tensor,
    *,
    q_phase: float,
    q_height: float,
    grad_image: torch.Tensor,
    times: torch.Tensor,
    config: UVTRenderConfig,
    sigma_px: float,
) -> float:
    coeffs = lower_q2_family_coeffs(family_coeffs, q_phase=q_phase, q_height=q_height)
    atlas = _atlas_from_coeffs(coeffs, frames=int(config.frames), device=times.device)
    image = render_projective_trace_cell_interval_atlas_metal(atlas, times, config, sigma_px=float(sigma_px))
    value = torch.sum(image * grad_image)
    torch.mps.synchronize()
    return float(value.detach().cpu().item())


def _chain_rule_row(
    family_coeffs: torch.Tensor,
    *,
    q_phase: float,
    q_height: float,
    grad_image: torch.Tensor,
    times: torch.Tensor,
    config: UVTRenderConfig,
    sigma_px: float,
) -> tuple[dict[str, Any], torch.Tensor, int]:
    basis = _q2_basis(q_phase, q_height, device=family_coeffs.device)
    coeffs = lower_q2_family_coeffs(family_coeffs, q_phase=q_phase, q_height=q_height)
    atlas = _atlas_from_coeffs(coeffs, frames=int(config.frames), device=times.device)
    image = render_projective_trace_cell_interval_atlas_metal(atlas, times, config, sigma_px=float(sigma_px))
    objective = torch.sum(image * grad_image)
    grads = direct_backward_projective_trace_cell_interval_atlas_metal(
        atlas,
        times,
        grad_image,
        config,
        sigma_px=float(sigma_px),
    )
    torch.mps.synchronize()
    family_grad = grads.grad_coeffs[:, :, None] * basis[None, None, :]
    row = {
        "q_phase": float(q_phase),
        "q_height": float(q_height),
        "basis_abs_sum": float(basis.abs().sum().detach().cpu().item()),
        "objective": float(objective.detach().cpu().item()),
        "slice_payload_bytes": _slice_payload_bytes(atlas),
        "slice_grad_payload_bytes": _tensor_bytes(grads.grad_coeffs),
        "grad_coeff_abs_sum": float(grads.grad_coeffs.abs().sum().detach().cpu().item()),
        "family_grad_abs_sum": float(family_grad.abs().sum().detach().cpu().item()),
        "image_sum": float(image.sum().detach().cpu().item()),
    }
    return row, family_grad, int(_tensor_bytes(grads.grad_coeffs))


def _finite_difference_row(
    *,
    family_coeffs: torch.Tensor,
    shared_grad: torch.Tensor,
    q_pairs: list[tuple[float, float]],
    grad_images: list[torch.Tensor],
    times: torch.Tensor,
    config: UVTRenderConfig,
    sigma_px: float,
    index: tuple[int, int, int],
    eps: float,
) -> dict[str, Any]:
    perturb = torch.zeros_like(family_coeffs)
    perturb[index] = float(eps)
    plus = family_coeffs + perturb
    minus = family_coeffs - perturb
    plus_value = 0.0
    minus_value = 0.0
    for row_index, (q_phase, q_height) in enumerate(q_pairs):
        plus_value += _row_objective(
            plus,
            q_phase=q_phase,
            q_height=q_height,
            grad_image=grad_images[row_index],
            times=times,
            config=config,
            sigma_px=sigma_px,
        )
        minus_value += _row_objective(
            minus,
            q_phase=q_phase,
            q_height=q_height,
            grad_image=grad_images[row_index],
            times=times,
            config=config,
            sigma_px=sigma_px,
        )
    fd = (plus_value - minus_value) / (2.0 * float(eps))
    analytic = float(shared_grad[index].detach().cpu().item())
    abs_error = abs(analytic - fd)
    rel_error = abs_error / max(abs(fd), abs(analytic), 1.0e-6)
    return {
        "trace": int(index[0]),
        "coeff": int(index[1]),
        "basis": int(index[2]),
        "analytic_grad": analytic,
        "finite_difference_grad": float(fd),
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
    }


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    finite_differences = report["finite_differences"]
    q_pair_count = int(report["q_pair_count"])
    shared_grad_payload = int(report["shared_family_gradient_payload_bytes"])
    per_q_replay_grad_payload = int(report["per_q_replay_gradient_payload_bytes"])
    return {
        "q_axis_count": int(report["q_axis_count"]),
        "q_pair_count": q_pair_count,
        "trace_count": int(report["trace_count"]),
        "family_basis_count": int(report["family_basis_count"]),
        "metal_forward_rows": len(rows),
        "metal_backward_rows": len(rows),
        "shared_family_gradient_payload_bytes": shared_grad_payload,
        "per_q_replay_gradient_payload_bytes": per_q_replay_grad_payload,
        "shared_to_replay_gradient_payload_ratio": float(shared_grad_payload) / float(per_q_replay_grad_payload),
        "family_payload_bytes": int(report["family_payload_bytes"]),
        "slice_payload_bytes": int(report["slice_payload_bytes"]),
        "peak_slice_to_replay_payload_ratio": float(report["slice_payload_bytes"])
        / float(int(report["slice_payload_bytes"]) * q_pair_count),
        "total_objective": float(sum(float(row["objective"]) for row in rows)),
        "min_slice_grad_coeff_abs_sum": min(float(row["grad_coeff_abs_sum"]) for row in rows),
        "shared_family_grad_abs_sum": float(report["shared_family_grad_abs_sum"]),
        "max_finite_difference_abs_error": max(float(row["abs_error"]) for row in finite_differences),
        "max_finite_difference_rel_error": max(float(row["rel_error"]) for row in finite_differences),
        "finite_difference_count": len(finite_differences),
    }


def run_report(
    *,
    q_axis_count: int = 5,
    frames: int = 4,
    image_size: int = 8,
    tile_size: int = 8,
    sigma_px: float = 1.7,
    finite_difference_eps: float = 1.0e-2,
) -> dict[str, Any]:
    config = _render_config(frames=int(frames), image_size=int(image_size), tile_size=int(tile_size))
    interval_metal_available = bool(torch.backends.mps.is_available() and has_projective_trace_cell_interval_metal())
    interval_backward_metal_available = bool(
        torch.backends.mps.is_available() and has_projective_trace_cell_interval_backward_metal()
    )
    if not interval_metal_available or not interval_backward_metal_available:
        return {
            "status": "failed",
            "benchmark": "star_uvt_projective_camera_family_2d_metal_chain_rule",
            "base_domain": "Q2 x Omega x T shared backward from Omega x T Metal slices",
            "theory_contract": "Per-slice interval Metal VJPs accumulate through d coeff_slice / d family_coeff into one shared Q2 x Omega x T adjoint for pi_* Gamma^* traces.",
            "interval_metal_available": interval_metal_available,
            "interval_backward_metal_available": interval_backward_metal_available,
            "metal_ran": False,
            "errors": ["MPS interval Metal forward/backward is required for this saved evidence artifact."],
            "rows": [],
            "finite_differences": [],
            "summary": {},
        }

    device = torch.device("mps")
    _apply_metal_tile_env(config)
    family_coeffs = _family_coeff_table(device=device)
    template_coeffs = lower_q2_family_coeffs(family_coeffs, q_phase=0.0, q_height=0.0)
    template_atlas = _atlas_from_coeffs(template_coeffs, frames=int(frames), device=device)
    times = torch.linspace(0.0, 1.0, int(frames), dtype=torch.float32, device=device).contiguous()
    q_pairs = _q_grid(int(q_axis_count))
    grad_images = [_grad_image(config, device=device, row_index=row_index) for row_index in range(len(q_pairs))]
    rows: list[dict[str, Any]] = []
    shared_grad = torch.zeros_like(family_coeffs)
    slice_grad_payload_bytes = 0
    for row_index, (q_phase, q_height) in enumerate(q_pairs):
        row, row_family_grad, row_grad_payload_bytes = _chain_rule_row(
            family_coeffs,
            q_phase=q_phase,
            q_height=q_height,
            grad_image=grad_images[row_index],
            times=times,
            config=config,
            sigma_px=float(sigma_px),
        )
        rows.append(row)
        shared_grad = shared_grad + row_family_grad
        slice_grad_payload_bytes = max(slice_grad_payload_bytes, int(row_grad_payload_bytes))
    finite_difference_indices = (
        (0, 3, 0),
        (1, 3, 0),
        (0, 4, 0),
        (0, 5, 0),
        (1, 4, 0),
        (1, 5, 0),
        (1, 1, 0),
        (0, 3, 4),
        (0, 3, 1),
    )
    finite_differences = [
        _finite_difference_row(
            family_coeffs=family_coeffs,
            shared_grad=shared_grad,
            q_pairs=q_pairs,
            grad_images=grad_images,
            times=times,
            config=config,
            sigma_px=float(sigma_px),
            index=index,
            eps=float(finite_difference_eps),
        )
        for index in finite_difference_indices
    ]
    torch.mps.synchronize()
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_metal_chain_rule",
        "base_domain": "Q2 x Omega x T shared backward from Omega x T Metal slices",
        "theory_contract": "Per-slice interval Metal VJPs accumulate through d coeff_slice / d family_coeff into one shared Q2 x Omega x T adjoint for pi_* Gamma^* traces. This is shared-family chain-rule accumulation over Metal slices, not native Q2 Metal evaluation.",
        "interval_metal_available": interval_metal_available,
        "interval_backward_metal_available": interval_backward_metal_available,
        "metal_ran": True,
        "q_axis_count": int(q_axis_count),
        "q_pair_count": len(q_pairs),
        "frames": int(frames),
        "image_size": int(image_size),
        "trace_count": TRACE_COUNT,
        "family_basis_count": Q2_BASIS_COUNT,
        "family_payload_bytes": _family_payload_bytes(family_coeffs),
        "slice_payload_bytes": _slice_payload_bytes(template_atlas),
        "shared_family_gradient_payload_bytes": _tensor_bytes(shared_grad),
        "per_q_replay_gradient_payload_bytes": int(slice_grad_payload_bytes) * len(q_pairs),
        "shared_family_grad_abs_sum": float(shared_grad.abs().sum().detach().cpu().item()),
        "finite_difference_eps": float(finite_difference_eps),
        "rows": rows,
        "finite_differences": finite_differences,
    }
    report["summary"] = summarize(report)
    errors = verify_camera_family_2d_metal_chain_rule_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _assert_summary_close(actual: Any, expected: Any, key: str, errors: list[str]) -> None:
    if isinstance(expected, float):
        if not _finite_float(actual) or abs(float(actual) - expected) > 1.0e-8:
            errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")


def verify_camera_family_2d_metal_chain_rule_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_metal_chain_rule":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T shared backward from Omega x T Metal slices":
        errors.append(f"base_domain must name the Q2 shared-backward lowering, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "pi_* Gamma^*" not in theory_contract
        or "chain-rule" not in theory_contract
    ):
        errors.append("theory_contract must preserve the pi_* Gamma^* chain-rule accumulation contract")
    if report.get("metal_ran") is not True:
        errors.append("metal_ran must be true for this evidence artifact")
    if report.get("interval_metal_available") is not True or report.get("interval_backward_metal_available") is not True:
        errors.append("interval Metal forward/backward availability must both be true")

    q_axis_count = report.get("q_axis_count")
    q_pair_count = report.get("q_pair_count")
    if not isinstance(q_axis_count, int) or q_axis_count < 5:
        errors.append(f"q_axis_count must be an int >= 5, got {q_axis_count!r}")
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
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {idx} must be an object")
            continue
        for key in (
            "q_phase",
            "q_height",
            "basis_abs_sum",
            "objective",
            "image_sum",
            "grad_coeff_abs_sum",
            "family_grad_abs_sum",
        ):
            if not _finite_float(row.get(key)):
                errors.append(f"row {idx} {key} must be finite, got {row.get(key)!r}")
        for key in ("image_sum", "grad_coeff_abs_sum", "family_grad_abs_sum"):
            if _finite_float(row.get(key)) and float(row[key]) <= 1.0e-6:
                errors.append(f"row {idx} {key} must be nonzero, got {row[key]!r}")

    finite_differences = report.get("finite_differences")
    if not isinstance(finite_differences, list):
        errors.append("finite_differences must be a list")
        finite_differences = []
    elif len(finite_differences) < 4:
        errors.append("finite_differences must include at least four shared-family coefficient checks")
    for idx, row in enumerate(finite_differences):
        if not isinstance(row, dict):
            errors.append(f"finite_difference row {idx} must be an object")
            continue
        for key in ("analytic_grad", "finite_difference_grad", "abs_error", "rel_error"):
            if not _finite_float(row.get(key)):
                errors.append(f"finite_difference row {idx} {key} must be finite, got {row.get(key)!r}")
        if _finite_float(row.get("rel_error")) and float(row["rel_error"]) > 1.0e-3:
            errors.append(f"finite_difference row {idx} rel_error too high: {row['rel_error']!r}")

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
            _finite_float(summary.get("shared_to_replay_gradient_payload_ratio"))
            and float(summary["shared_to_replay_gradient_payload_ratio"]) >= 0.30
        ):
            errors.append("shared/replay gradient payload ratio must stay below 0.30")
        if (
            _finite_float(summary.get("peak_slice_to_replay_payload_ratio"))
            and float(summary["peak_slice_to_replay_payload_ratio"]) >= 0.10
        ):
            errors.append("peak slice/replay payload ratio must stay below 0.10")
        if (
            _finite_float(summary.get("shared_family_grad_abs_sum"))
            and float(summary["shared_family_grad_abs_sum"]) <= 1.0e-6
        ):
            errors.append("shared family gradient must be nonzero")
        if (
            _finite_float(summary.get("max_finite_difference_rel_error"))
            and float(summary["max_finite_difference_rel_error"]) > 1.0e-3
        ):
            errors.append("max finite-difference relative error must stay below 1e-3")
    return errors


def assert_camera_family_2d_metal_chain_rule_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_metal_chain_rule_report(report)
    if errors:
        raise AssertionError("camera-family 2D Metal chain-rule report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Camera-Family 2D Metal Chain Rule",
        "",
        "This is a shared-family backward smoke over Metal slices. It does not claim native Q2 Metal evaluation.",
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
        assert_camera_family_2d_metal_chain_rule_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(q_axis_count=int(args.q_axis_count), frames=int(args.frames))
    assert_camera_family_2d_metal_chain_rule_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
