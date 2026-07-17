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
    ProjectiveTraceCellTraceAtlas,
    ProjectiveTraceTileTimeCell,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
    render_projective_trace_cell_interval_atlas_metal,
)

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_chain_rule_report import (  # noqa: E402
    _grad_image,
    _q_grid,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (  # noqa: E402
    Q2_BASIS_COUNT,
    TRACE_COUNT,
    _apply_metal_tile_env,
    _depth_interval,
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
    / "2026-05-25_star_uvt_projective_camera_family_2d_materialized_batch"
)


def _trace_color(*, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([[1.0, 0.1, 0.05], [0.05, 0.25, 1.0]], dtype=torch.float32, device=device)


def _trace_opacity(*, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([0.65, 0.45], dtype=torch.float32, device=device)


def _batched_atlas_from_family(
    family_coeffs: torch.Tensor,
    q_pairs: list[tuple[float, float]],
    *,
    frames_per_q: int,
    device: torch.device | str,
) -> ProjectiveTraceCellTraceAtlas:
    coeff_rows: list[torch.Tensor] = []
    cells: list[ProjectiveTraceTileTimeCell] = []
    source_window_indices: list[int] = []
    source_primitive_ids: list[int] = []
    active_start: list[int] = []
    active_stop: list[int] = []
    local_times_cpu = torch.linspace(0.0, 1.0, int(frames_per_q), dtype=torch.float32)

    for q_index, (q_phase, q_height) in enumerate(q_pairs):
        coeffs = lower_q2_family_coeffs(family_coeffs, q_phase=q_phase, q_height=q_height)
        coeff_rows.append(coeffs)
        start = q_index * int(frames_per_q)
        stop = start + int(frames_per_q)
        first_trace = q_index * TRACE_COUNT
        coeffs_cpu = coeffs.detach().cpu()
        mid_t = torch.tensor(0.5, dtype=torch.float32, device=coeffs.device)
        mid_depth = coeffs[:, 6] + coeffs[:, 7] * mid_t + coeffs[:, 8] * mid_t * mid_t
        local_order = [int(idx) for idx in torch.argsort(mid_depth).detach().cpu().tolist()]
        ordered_primitive_ids = tuple(first_trace + idx for idx in local_order)
        depth_intervals = tuple(
            _depth_interval(coeffs_cpu[idx], local_times_cpu) for idx in range(int(coeffs.shape[0]))
        )
        cells.append(
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=start,
                stop=stop,
                primitive_ids=tuple(first_trace + idx for idx in range(TRACE_COUNT)),
                ordered_primitive_ids=ordered_primitive_ids,
                depth_intervals=depth_intervals,
                fallback=False,
                fallback_reasons=(),
            )
        )
        source_window_indices.extend([q_index] * TRACE_COUNT)
        source_primitive_ids.extend(range(TRACE_COUNT))
        active_start.extend([start] * TRACE_COUNT)
        active_stop.extend([stop] * TRACE_COUNT)

    coeffs_batched = torch.cat(coeff_rows, dim=0).to(device=device, dtype=torch.float32).contiguous()
    opacity = _trace_opacity(device=device).repeat(len(q_pairs)).contiguous()
    color = _trace_color(device=device).repeat((len(q_pairs), 1)).contiguous()
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs_batched,
        opacity=opacity,
        color=color,
        cells=cells,
        source_window_indices=tuple(source_window_indices),
        source_primitive_ids=tuple(source_primitive_ids),
        active_start=tuple(active_start),
        active_stop=tuple(active_stop),
    )


def _materialized_payload_bytes(atlas: ProjectiveTraceCellTraceAtlas) -> int:
    return _slice_payload_bytes(atlas)


def _render_slice_reference(
    family_coeffs: torch.Tensor,
    q_pairs: list[tuple[float, float]],
    *,
    times: torch.Tensor,
    config,
    sigma_px: float,
) -> tuple[torch.Tensor, torch.Tensor, int, list[dict[str, float]]]:
    from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (
        _atlas_from_coeffs,
    )

    images: list[torch.Tensor] = []
    shared_grad = torch.zeros_like(family_coeffs)
    slice_payload_bytes = 0
    rows: list[dict[str, float]] = []
    for q_index, (q_phase, q_height) in enumerate(q_pairs):
        basis = _q2_basis(q_phase, q_height, device=family_coeffs.device)
        coeffs = lower_q2_family_coeffs(family_coeffs, q_phase=q_phase, q_height=q_height)
        atlas = _atlas_from_coeffs(coeffs, frames=int(config.frames), device=times.device)
        image = render_projective_trace_cell_interval_atlas_metal(atlas, times, config, sigma_px=float(sigma_px))
        grad_image = _grad_image(config, device=times.device, row_index=q_index)
        grads = direct_backward_projective_trace_cell_interval_atlas_metal(
            atlas,
            times,
            grad_image,
            config,
            sigma_px=float(sigma_px),
        )
        images.append(image)
        shared_grad = shared_grad + grads.grad_coeffs[:, :, None] * basis[None, None, :]
        slice_payload_bytes = max(slice_payload_bytes, _slice_payload_bytes(atlas))
        rows.append(
            {
                "q_phase": float(q_phase),
                "q_height": float(q_height),
                "image_sum": float(image.sum().detach().cpu().item()),
                "grad_coeff_abs_sum": float(grads.grad_coeffs.abs().sum().detach().cpu().item()),
            }
        )
    return torch.stack(images, dim=0).contiguous(), shared_grad, slice_payload_bytes, rows


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    q_pair_count = int(report["q_pair_count"])
    materialized_payload = int(report["materialized_trace_payload_bytes"])
    replay_payload = int(report["per_q_replay_trace_payload_bytes"])
    family_payload = int(report["family_payload_bytes"])
    materialized_grad_payload = int(report["materialized_gradient_payload_bytes"])
    shared_grad_payload = int(report["shared_family_gradient_payload_bytes"])
    replay_grad_payload = int(report["per_q_replay_gradient_payload_bytes"])
    return {
        "q_axis_count": int(report["q_axis_count"]),
        "q_pair_count": q_pair_count,
        "trace_count": int(report["trace_count"]),
        "family_basis_count": int(report["family_basis_count"]),
        "frames_per_q": int(report["frames_per_q"]),
        "batched_frames": int(report["batched_frames"]),
        "slice_forward_launches": q_pair_count,
        "batched_forward_launches": 1,
        "forward_launch_ratio": 1.0 / float(q_pair_count),
        "slice_backward_launches": q_pair_count,
        "batched_backward_launches": 1,
        "backward_launch_ratio": 1.0 / float(q_pair_count),
        "family_payload_bytes": family_payload,
        "materialized_trace_payload_bytes": materialized_payload,
        "per_q_replay_trace_payload_bytes": replay_payload,
        "materialized_to_replay_trace_payload_ratio": float(materialized_payload) / float(replay_payload),
        "family_to_materialized_trace_payload_ratio": float(family_payload) / float(materialized_payload),
        "shared_family_gradient_payload_bytes": shared_grad_payload,
        "materialized_gradient_payload_bytes": materialized_grad_payload,
        "per_q_replay_gradient_payload_bytes": replay_grad_payload,
        "materialized_to_replay_gradient_payload_ratio": float(materialized_grad_payload) / float(replay_grad_payload),
        "shared_to_materialized_gradient_payload_ratio": float(shared_grad_payload) / float(materialized_grad_payload),
        "max_batched_vs_slice_image_abs_error": float(report["max_batched_vs_slice_image_abs_error"]),
        "max_batched_vs_slice_image_rel_error": float(report["max_batched_vs_slice_image_rel_error"]),
        "max_batched_vs_slice_shared_grad_abs_error": float(report["max_batched_vs_slice_shared_grad_abs_error"]),
        "max_batched_vs_slice_shared_grad_rel_error": float(report["max_batched_vs_slice_shared_grad_rel_error"]),
        "batched_image_sum": float(report["batched_image_sum"]),
        "batched_grad_coeff_abs_sum": float(report["batched_grad_coeff_abs_sum"]),
        "batched_shared_family_grad_abs_sum": float(report["batched_shared_family_grad_abs_sum"]),
    }


def run_report(
    *,
    q_axis_count: int = 5,
    frames: int = 4,
    image_size: int = 8,
    tile_size: int = 8,
    sigma_px: float = 1.7,
) -> dict[str, Any]:
    slice_config = _render_config(frames=int(frames), image_size=int(image_size), tile_size=int(tile_size))
    q_pairs = _q_grid(int(q_axis_count))
    batched_config = _render_config(
        frames=int(frames) * len(q_pairs),
        image_size=int(image_size),
        tile_size=int(tile_size),
    )
    interval_metal_available = bool(torch.backends.mps.is_available() and has_projective_trace_cell_interval_metal())
    interval_backward_metal_available = bool(
        torch.backends.mps.is_available() and has_projective_trace_cell_interval_backward_metal()
    )
    if not interval_metal_available or not interval_backward_metal_available:
        return {
            "status": "failed",
            "benchmark": "star_uvt_projective_camera_family_2d_materialized_batch",
            "base_domain": "Q2 x Omega x T materialized single-launch Metal batch",
            "theory_contract": "A Q2 camera-family trace grid is materialized into one Omega x T interval Metal atlas to test launch reuse for pi_* Gamma^* traces while leaving native family-coefficient evaluation open.",
            "interval_metal_available": interval_metal_available,
            "interval_backward_metal_available": interval_backward_metal_available,
            "metal_ran": False,
            "errors": ["MPS interval Metal forward/backward is required for this saved evidence artifact."],
            "rows": [],
            "summary": {},
        }

    device = torch.device("mps")
    _apply_metal_tile_env(batched_config)
    family_coeffs = _family_coeff_table(device=device)
    slice_times = torch.linspace(0.0, 1.0, int(frames), dtype=torch.float32, device=device).contiguous()
    batched_times = slice_times.repeat(len(q_pairs)).contiguous()
    slice_images, slice_shared_grad, slice_payload_bytes, rows = _render_slice_reference(
        family_coeffs,
        q_pairs,
        times=slice_times,
        config=slice_config,
        sigma_px=float(sigma_px),
    )
    batched_atlas = _batched_atlas_from_family(
        family_coeffs,
        q_pairs,
        frames_per_q=int(frames),
        device=device,
    )
    batched_image = render_projective_trace_cell_interval_atlas_metal(
        batched_atlas,
        batched_times,
        batched_config,
        sigma_px=float(sigma_px),
    )
    grad_images = [_grad_image(slice_config, device=device, row_index=row_index) for row_index in range(len(q_pairs))]
    batched_grad_image = torch.cat(grad_images, dim=0).contiguous()
    batched_grads = direct_backward_projective_trace_cell_interval_atlas_metal(
        batched_atlas,
        batched_times,
        batched_grad_image,
        batched_config,
        sigma_px=float(sigma_px),
    )
    q_basis = torch.stack(
        [_q2_basis(q_phase, q_height, device=device) for q_phase, q_height in q_pairs],
        dim=0,
    )
    batched_grad_coeffs = batched_grads.grad_coeffs.reshape(len(q_pairs), TRACE_COUNT, 9)
    batched_shared_grad = torch.einsum("qnk,qb->nkb", batched_grad_coeffs, q_basis)
    torch.mps.synchronize()

    batched_image_q = batched_image.reshape(len(q_pairs), int(frames), int(image_size), int(image_size), 3)
    image_delta = (batched_image_q - slice_images).abs()
    image_ref = torch.maximum(batched_image_q.abs(), slice_images.abs()).amax().clamp_min(1.0e-6)
    grad_delta = (batched_shared_grad - slice_shared_grad).abs()
    grad_ref = torch.maximum(batched_shared_grad.abs(), slice_shared_grad.abs()).amax().clamp_min(1.0e-6)
    materialized_payload_bytes = _materialized_payload_bytes(batched_atlas)
    materialized_grad_payload_bytes = _tensor_bytes(batched_grads.grad_coeffs)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_materialized_batch",
        "base_domain": "Q2 x Omega x T materialized single-launch Metal batch",
        "theory_contract": "A Q2 camera-family trace grid is materialized into one Omega x T interval Metal atlas to test launch reuse for pi_* Gamma^* traces while leaving native family-coefficient evaluation open. This is a single-launch materialized baseline, not native Q2/Qn Metal evaluation.",
        "interval_metal_available": interval_metal_available,
        "interval_backward_metal_available": interval_backward_metal_available,
        "metal_ran": True,
        "q_axis_count": int(q_axis_count),
        "q_pair_count": len(q_pairs),
        "frames_per_q": int(frames),
        "batched_frames": int(frames) * len(q_pairs),
        "image_size": int(image_size),
        "trace_count": TRACE_COUNT,
        "family_basis_count": Q2_BASIS_COUNT,
        "family_payload_bytes": _family_payload_bytes(family_coeffs),
        "slice_trace_payload_bytes": int(slice_payload_bytes),
        "materialized_trace_payload_bytes": int(materialized_payload_bytes),
        "per_q_replay_trace_payload_bytes": int(slice_payload_bytes) * len(q_pairs),
        "shared_family_gradient_payload_bytes": _tensor_bytes(slice_shared_grad),
        "materialized_gradient_payload_bytes": int(materialized_grad_payload_bytes),
        "per_q_replay_gradient_payload_bytes": int(_tensor_bytes(batched_grads.grad_coeffs)),
        "max_batched_vs_slice_image_abs_error": float(image_delta.max().detach().cpu().item()),
        "max_batched_vs_slice_image_rel_error": float((image_delta.max() / image_ref).detach().cpu().item()),
        "max_batched_vs_slice_shared_grad_abs_error": float(grad_delta.max().detach().cpu().item()),
        "max_batched_vs_slice_shared_grad_rel_error": float((grad_delta.max() / grad_ref).detach().cpu().item()),
        "batched_image_sum": float(batched_image.sum().detach().cpu().item()),
        "batched_grad_coeff_abs_sum": float(batched_grads.grad_coeffs.abs().sum().detach().cpu().item()),
        "batched_shared_family_grad_abs_sum": float(batched_shared_grad.abs().sum().detach().cpu().item()),
        "rows": rows,
    }
    report["summary"] = summarize(report)
    errors = verify_camera_family_2d_materialized_batch_report(report)
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


def verify_camera_family_2d_materialized_batch_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_materialized_batch":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T materialized single-launch Metal batch":
        errors.append(f"base_domain must name the materialized Q2 single-launch batch, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "pi_* Gamma^*" not in theory_contract
        or "materialized" not in theory_contract
        or "not native Q2/Qn" not in theory_contract
    ):
        errors.append("theory_contract must preserve the pi_* Gamma^* materialized-baseline contract")
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
    if report.get("batched_frames") != int(report.get("frames_per_q", 0)) * int(q_pair_count):
        errors.append("batched_frames must equal frames_per_q * q_pair_count")

    rows = report.get("rows")
    if not isinstance(rows, list):
        errors.append("rows must be a list")
        rows = []
    elif len(rows) != int(q_pair_count):
        errors.append(f"rows must contain one slice-reference row per q-pair, expected {q_pair_count}, got {len(rows)}")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {idx} must be an object")
            continue
        for key in ("q_phase", "q_height", "image_sum", "grad_coeff_abs_sum"):
            if not _finite_float(row.get(key)):
                errors.append(f"row {idx} {key} must be finite, got {row.get(key)!r}")
        for key in ("image_sum", "grad_coeff_abs_sum"):
            if _finite_float(row.get(key)) and float(row[key]) <= 1.0e-6:
                errors.append(f"row {idx} {key} must be nonzero, got {row[key]!r}")

    for key in (
        "family_payload_bytes",
        "materialized_trace_payload_bytes",
        "per_q_replay_trace_payload_bytes",
        "shared_family_gradient_payload_bytes",
        "materialized_gradient_payload_bytes",
        "per_q_replay_gradient_payload_bytes",
    ):
        if not isinstance(report.get(key), int) or int(report[key]) <= 0:
            errors.append(f"{key} must be a positive integer")
    for key in (
        "max_batched_vs_slice_image_abs_error",
        "max_batched_vs_slice_image_rel_error",
        "max_batched_vs_slice_shared_grad_abs_error",
        "max_batched_vs_slice_shared_grad_rel_error",
        "batched_image_sum",
        "batched_grad_coeff_abs_sum",
        "batched_shared_family_grad_abs_sum",
    ):
        if not _finite_float(report.get(key)):
            errors.append(f"{key} must be finite, got {report.get(key)!r}")

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
        if _finite_float(summary.get("forward_launch_ratio")) and float(summary["forward_launch_ratio"]) >= 0.10:
            errors.append("forward launch ratio must stay below 0.10 for the Q2 grid")
        if _finite_float(summary.get("backward_launch_ratio")) and float(summary["backward_launch_ratio"]) >= 0.10:
            errors.append("backward launch ratio must stay below 0.10 for the Q2 grid")
        if (
            _finite_float(summary.get("materialized_to_replay_trace_payload_ratio"))
            and abs(float(summary["materialized_to_replay_trace_payload_ratio"]) - 1.0) > 1.0e-6
        ):
            errors.append("materialized/replay trace payload ratio must remain 1.0; this artifact must not imply native family compression")
        if (
            _finite_float(summary.get("family_to_materialized_trace_payload_ratio"))
            and float(summary["family_to_materialized_trace_payload_ratio"]) >= 0.35
        ):
            errors.append("family/materialized trace payload ratio must stay below 0.35 to expose the remaining native-family gap")
        if (
            _finite_float(summary.get("shared_to_materialized_gradient_payload_ratio"))
            and float(summary["shared_to_materialized_gradient_payload_ratio"]) >= 0.30
        ):
            errors.append("shared/materialized gradient payload ratio must stay below 0.30")
        for key in (
            "max_batched_vs_slice_image_abs_error",
            "max_batched_vs_slice_image_rel_error",
            "max_batched_vs_slice_shared_grad_abs_error",
            "max_batched_vs_slice_shared_grad_rel_error",
        ):
            threshold = 1.0e-5 if key.endswith("abs_error") else 1.0e-6
            if _finite_float(summary.get(key)) and float(summary[key]) > threshold:
                errors.append(f"{key} must stay below {threshold:g}")
        for key in ("batched_image_sum", "batched_grad_coeff_abs_sum", "batched_shared_family_grad_abs_sum"):
            if _finite_float(summary.get(key)) and float(summary[key]) <= 1.0e-6:
                errors.append(f"summary {key} must be nonzero")
    return errors


def assert_camera_family_2d_materialized_batch_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_materialized_batch_report(report)
    if errors:
        raise AssertionError("camera-family 2D materialized batch report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Camera-Family 2D Materialized Metal Batch",
        "",
        "This is a single-launch materialized baseline: it does not claim native Q2/Qn Metal evaluation.",
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
        assert_camera_family_2d_materialized_batch_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(q_axis_count=int(args.q_axis_count), frames=int(args.frames))
    assert_camera_family_2d_materialized_batch_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
