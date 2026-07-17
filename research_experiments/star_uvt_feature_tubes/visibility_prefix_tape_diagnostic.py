from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import write_report_json, write_report_text
    from .support_target_patch_diagnostic import (
        _colorize_sparse,
        _limit_points,
        _load_final_case,
        _recompute_support_birth_target_points,
        _selected_tube_ids,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import write_report_json, write_report_text
    from support_target_patch_diagnostic import (
        _colorize_sparse,
        _limit_points,
        _load_final_case,
        _recompute_support_birth_target_points,
        _selected_tube_ids,
    )


def _psnr(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(float(mse))


def _quadratic(q_uvt: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return (
        q_uvt[..., 0] * delta[..., 0] * delta[..., 0]
        + 2.0 * q_uvt[..., 1] * delta[..., 0] * delta[..., 1]
        + 2.0 * q_uvt[..., 2] * delta[..., 0] * delta[..., 2]
        + q_uvt[..., 3] * delta[..., 1] * delta[..., 1]
        + 2.0 * q_uvt[..., 4] * delta[..., 1] * delta[..., 2]
        + q_uvt[..., 5] * delta[..., 2] * delta[..., 2]
    )


def _depth_at(ma: torch.Tensor, depth0: torch.Tensor, depth_beta: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return depth0.unsqueeze(0) + ((points.unsqueeze(1) - ma.unsqueeze(0)) * depth_beta.unsqueeze(0)).sum(dim=-1)


def _pixel_ids_for_points(
    points: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    points_cpu = points.detach().to(device="cpu", dtype=torch.float32)
    frame = torch.round(points_cpu[:, 2] + 0.5 * float(int(frames) - 1)).to(torch.int64).clamp(0, int(frames) - 1)
    x = torch.floor(points_cpu[:, 0]).to(torch.int64).clamp(0, int(width) - 1)
    y = torch.floor(points_cpu[:, 1]).to(torch.int64).clamp(0, int(height) - 1)
    return (frame * int(height) * int(width) + y * int(width) + x).to(torch.int64)


def _compute_prefix_tape_tensors(
    *,
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    points: torch.Tensor,
    config: Any,
    selected_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if points.dim() != 2 or int(points.shape[1]) != 3:
        raise ValueError(f"points must have shape [P,3], got {tuple(points.shape)}")
    if int(points.shape[0]) <= 0:
        raise ValueError("visibility prefix diagnostic requires at least one point")
    device = ma.device
    points = points.to(device=device, dtype=torch.float32).contiguous()
    delta = points.unsqueeze(1) - ma.detach().unsqueeze(0)
    qv = _quadratic(q_uvt.detach().unsqueeze(0), delta)
    raw_alpha = opacity.detach().unsqueeze(0) * torch.exp(torch.clamp(-0.5 * qv, min=-80.0, max=0.0))
    alpha = torch.clamp(raw_alpha, min=0.0, max=float(config.max_alpha))
    alpha = torch.where(alpha >= float(config.alpha_threshold), alpha, torch.zeros_like(alpha))
    depth = _depth_at(ma.detach(), depth0.detach(), depth_beta.detach(), points)
    depth_for_sort = torch.where(alpha > 0.0, depth, torch.full_like(depth, float("inf")))
    order = torch.argsort(depth_for_sort, dim=1, stable=True)
    ordered_alpha = alpha.gather(1, order)
    ordered_depth = depth.gather(1, order)
    one_minus = (1.0 - ordered_alpha).clamp_min(0.0)
    trans_after = torch.cumprod(one_minus, dim=1)
    prefix = torch.cat((torch.ones_like(trans_after[:, :1]), trans_after[:, :-1]), dim=1)
    weight = prefix * ordered_alpha
    if float(config.transmittance_threshold) > 0.0:
        weight = torch.where(prefix > float(config.transmittance_threshold), weight, torch.zeros_like(weight))

    selected_mask_by_id = torch.zeros((int(ma.shape[0]),), dtype=torch.bool, device=device)
    if int(selected_ids.numel()) > 0:
        selected_mask_by_id.index_fill_(0, selected_ids.to(device=device, dtype=torch.int64), True)
    ordered_selected = selected_mask_by_id.index_select(0, order.reshape(-1)).reshape_as(order)
    selected_alpha = torch.where(ordered_selected, ordered_alpha, torch.zeros_like(ordered_alpha))
    selected_weight = torch.where(ordered_selected, weight, torch.zeros_like(weight))
    selected_alpha_max, selected_alpha_rank = selected_alpha.max(dim=1)
    selected_prefix_at_alpha_max = prefix.gather(1, selected_alpha_rank.unsqueeze(1)).squeeze(1)
    selected_weight_sum = selected_weight.sum(dim=1)
    selected_alpha_sum = selected_alpha.sum(dim=1)
    final_alpha = weight.sum(dim=1).clamp(0.0, 1.0)
    selected_weight_share = selected_weight_sum / final_alpha.clamp_min(1.0e-8)
    top_weight, top_rank = weight.max(dim=1)
    top_tube_id = order.gather(1, top_rank.unsqueeze(1)).squeeze(1)
    top_is_selected = selected_mask_by_id.index_select(0, top_tube_id)
    return {
        "order": order,
        "ordered_alpha": ordered_alpha,
        "ordered_depth": ordered_depth,
        "prefix": prefix,
        "weight": weight,
        "ordered_selected": ordered_selected,
        "selected_alpha_max": selected_alpha_max,
        "selected_alpha_sum": selected_alpha_sum,
        "selected_prefix_at_alpha_max": selected_prefix_at_alpha_max,
        "selected_weight_sum": selected_weight_sum,
        "selected_weight_share": selected_weight_share,
        "final_alpha": final_alpha,
        "top_weight": top_weight,
        "top_tube_id": top_tube_id,
        "top_is_selected": top_is_selected,
    }


def _feature_values_from_tape(feature: torch.Tensor, tape: dict[str, torch.Tensor]) -> torch.Tensor:
    weight_by_id = torch.zeros_like(tape["weight"])
    weight_by_id.scatter_add_(1, tape["order"], tape["weight"])
    return weight_by_id @ feature.detach()


def _front_mass_before_first_selected(tape: dict[str, torch.Tensor]) -> list[float]:
    weights = tape["weight"].detach().to(device="cpu", dtype=torch.float32)
    selected = tape["ordered_selected"].detach().to(device="cpu")
    values: list[float] = []
    for row in range(int(weights.shape[0])):
        selected_ranks = torch.nonzero(selected[row], as_tuple=False).flatten()
        if int(selected_ranks.numel()) == 0:
            values.append(0.0)
            continue
        first_rank = int(selected_ranks[0].item())
        values.append(float(weights[row, :first_rank].sum().item()))
    return values


def _top_contributors(tape: dict[str, torch.Tensor], *, point_index: int, top_k: int) -> list[dict[str, Any]]:
    weights = tape["weight"][point_index].detach().to(device="cpu", dtype=torch.float32)
    k = min(max(int(top_k), 1), int(weights.numel()))
    top_weight, top_rank = torch.topk(weights, k=k, largest=True, sorted=True)
    ordered_ids = tape["order"][point_index].detach().to(device="cpu", dtype=torch.int64)
    ordered_alpha = tape["ordered_alpha"][point_index].detach().to(device="cpu", dtype=torch.float32)
    ordered_depth = tape["ordered_depth"][point_index].detach().to(device="cpu", dtype=torch.float32)
    prefix = tape["prefix"][point_index].detach().to(device="cpu", dtype=torch.float32)
    selected = tape["ordered_selected"][point_index].detach().to(device="cpu")
    rows: list[dict[str, Any]] = []
    for weight_value, rank_value in zip(top_weight.tolist(), top_rank.tolist(), strict=False):
        rank = int(rank_value)
        rows.append(
            {
                "rank": rank,
                "tube_id": int(ordered_ids[rank].item()),
                "selected": bool(selected[rank].item()),
                "depth": float(ordered_depth[rank].item()),
                "alpha": float(ordered_alpha[rank].item()),
                "prefix": float(prefix[rank].item()),
                "weight": float(weight_value),
            }
        )
    return rows


def _sample_rows(
    *,
    points: torch.Tensor,
    target_values: torch.Tensor,
    pred_rgb: torch.Tensor,
    forced_rgb: torch.Tensor,
    tape: dict[str, torch.Tensor],
    frames: int,
    height: int,
    width: int,
    sample_count: int,
    top_k: int,
) -> list[dict[str, Any]]:
    count = min(int(sample_count), int(points.shape[0]))
    if count <= 0:
        return []
    selected = torch.linspace(0, int(points.shape[0]) - 1, count).round().to(torch.int64)
    points_cpu = points.detach().to(device="cpu", dtype=torch.float32)
    target_cpu = target_values.detach().to(device="cpu", dtype=torch.float32)
    pred_cpu = pred_rgb.detach().to(device="cpu", dtype=torch.float32)
    forced_cpu = forced_rgb.detach().to(device="cpu", dtype=torch.float32)
    final_alpha = tape["final_alpha"].detach().to(device="cpu", dtype=torch.float32)
    selected_alpha = tape["selected_alpha_max"].detach().to(device="cpu", dtype=torch.float32)
    selected_weight = tape["selected_weight_sum"].detach().to(device="cpu", dtype=torch.float32)
    selected_share = tape["selected_weight_share"].detach().to(device="cpu", dtype=torch.float32)
    selected_prefix = tape["selected_prefix_at_alpha_max"].detach().to(device="cpu", dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    for idx in selected.tolist():
        point = points_cpu[int(idx)]
        frame = int(round(float(point[2].item()) + 0.5 * float(int(frames) - 1)))
        x = int(math.floor(float(point[0].item())))
        y = int(math.floor(float(point[1].item())))
        frame = max(0, min(int(frames) - 1, frame))
        x = max(0, min(int(width) - 1, x))
        y = max(0, min(int(height) - 1, y))
        residual = (pred_cpu[int(idx)] - target_cpu[int(idx)]).abs().mean()
        rows.append(
            {
                "point_index": int(idx),
                "frame": frame,
                "x": x,
                "y": y,
                "target_rgb": [float(v) for v in target_cpu[int(idx)].tolist()],
                "pred_rgb": [float(v) for v in pred_cpu[int(idx)].tolist()],
                "forced_rgb": [float(v) for v in forced_cpu[int(idx)].tolist()],
                "l1_residual": float(residual.item()),
                "alpha": float(final_alpha[int(idx)].item()),
                "selected_alpha_max": float(selected_alpha[int(idx)].item()),
                "selected_prefix_at_alpha_max": float(selected_prefix[int(idx)].item()),
                "selected_weight_sum": float(selected_weight[int(idx)].item()),
                "selected_weight_share": float(selected_share[int(idx)].item()),
                "top_contributors": _top_contributors(tape, point_index=int(idx), top_k=top_k),
            }
        )
    return rows


def _summarize_case(
    *,
    label: str,
    case: dict[str, Any],
    points: torch.Tensor,
    tape: dict[str, torch.Tensor],
    target_values: torch.Tensor,
    forced_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    target_meta: dict[str, Any],
    sample_count: int,
    top_k: int,
) -> dict[str, Any]:
    alpha = tape["final_alpha"].to(dtype=forced_rgb.dtype).unsqueeze(1)
    target_background_rgb = target_values + alpha * (forced_rgb - target_values)
    black_mse = float((pred_rgb - target_values).square().mean().detach().cpu().item())
    forced_mse = float((forced_rgb - target_values).square().mean().detach().cpu().item())
    oracle_mse = float((target_background_rgb - target_values).square().mean().detach().cpu().item())
    selected_alpha_max = tape["selected_alpha_max"]
    selected_weight_sum = tape["selected_weight_sum"]
    selected_weight_share = tape["selected_weight_share"]
    selected_prefix = tape["selected_prefix_at_alpha_max"]
    alpha_threshold = float(case["uvt_config"].alpha_threshold)
    selected_present = selected_alpha_max >= alpha_threshold
    selected_hidden = (selected_alpha_max >= 0.01) & (selected_prefix < 0.5)
    selected_contributes = selected_weight_sum >= 0.05
    front_mass = torch.tensor(_front_mass_before_first_selected(tape), dtype=torch.float32)
    return {
        "label": label,
        "config_path": case["config_path"],
        "checkpoint": case["checkpoint"],
        "point_count": int(points.shape[0]),
        "selected_tube_ids": [int(item) for item in _selected_tube_ids(case).detach().cpu().tolist()],
        "target_point_meta": target_meta,
        "normal_black_psnr": _psnr(black_mse),
        "forced_alpha_1_psnr": _psnr(forced_mse),
        "target_background_oracle_psnr": _psnr(oracle_mse),
        "alpha_mean": float(tape["final_alpha"].mean().detach().cpu().item()),
        "alpha_gt_0_1": float((tape["final_alpha"] > 0.1).to(torch.float32).mean().detach().cpu().item()),
        "selected_alpha_max_mean": float(selected_alpha_max.mean().detach().cpu().item()),
        "selected_alpha_max_max": float(selected_alpha_max.max().detach().cpu().item()),
        "selected_weight_sum_mean": float(selected_weight_sum.mean().detach().cpu().item()),
        "selected_weight_share_mean": float(selected_weight_share.mean().detach().cpu().item()),
        "selected_prefix_at_alpha_max_mean": float(selected_prefix.mean().detach().cpu().item()),
        "front_mass_before_first_selected_mean": float(front_mass.mean().item()),
        "selected_absent_fraction": float((~selected_present).to(torch.float32).mean().detach().cpu().item()),
        "selected_hidden_fraction": float(selected_hidden.to(torch.float32).mean().detach().cpu().item()),
        "selected_contributes_fraction": float(selected_contributes.to(torch.float32).mean().detach().cpu().item()),
        "top_contributor_selected_fraction": float(
            tape["top_is_selected"].to(torch.float32).mean().detach().cpu().item()
        ),
        "samples": _sample_rows(
            points=points,
            target_values=target_values,
            pred_rgb=pred_rgb,
            forced_rgb=forced_rgb,
            tape=tape,
            frames=int(case["feature_config"].frames),
            height=int(case["feature_config"].height),
            width=int(case["feature_config"].width),
            sample_count=sample_count,
            top_k=top_k,
        ),
    }


def _analyze_case(
    label: str,
    config_path: Path,
    *,
    max_points: int,
    top_k: int,
    sample_count: int,
) -> dict[str, Any]:
    case = _load_final_case(config_path)
    target_points, target_meta = _recompute_support_birth_target_points(case)
    points = _limit_points(target_points, int(max_points))
    ma, q_uvt, depth0, depth_beta, opacity, feature = case["model"].tensors()
    selected_ids = _selected_tube_ids(case)
    with torch.no_grad():
        tape = _compute_prefix_tape_tensors(
            ma=ma,
            q_uvt=q_uvt,
            depth0=depth0,
            depth_beta=depth_beta,
            opacity=opacity,
            points=points,
            config=case["uvt_config"],
            selected_ids=selected_ids,
        )
        feature_values = _feature_values_from_tape(feature, tape)
        forced_rgb = _colorize_sparse(feature_values, case["colorizer"])
        alpha = tape["final_alpha"].to(dtype=forced_rgb.dtype).unsqueeze(1)
        pred_rgb = alpha * forced_rgb
        pixel_ids = _pixel_ids_for_points(
            points,
            frames=int(case["feature_config"].frames),
            height=int(case["feature_config"].height),
            width=int(case["feature_config"].width),
        ).to(device=case["target_rgb"].device)
        target_values = case["target_rgb"].permute(0, 2, 3, 1).reshape(-1, 3).index_select(0, pixel_ids)
        return _summarize_case(
            label=label,
            case=case,
            points=points,
            tape=tape,
            target_values=target_values,
            forced_rgb=forced_rgb,
            pred_rgb=pred_rgb,
            target_meta=target_meta,
            sample_count=sample_count,
            top_k=top_k,
        )


def _parse_case(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path)
    path = Path(raw)
    return path.stem, path


def _make_read(cases: list[dict[str, Any]]) -> str:
    reads: list[str] = []
    for case in cases:
        if case["selected_absent_fraction"] > 0.5:
            reads.append(
                f"`{case['label']}` selected support is mostly absent on sampled target rays: "
                f"{case['selected_absent_fraction']:.1%} have no selected tube above threshold."
            )
        elif case["selected_hidden_fraction"] > 0.5:
            reads.append(
                f"`{case['label']}` selected support is mostly prefix-hidden: "
                f"selected alpha max mean {case['selected_alpha_max_mean']:.4f}, prefix at selected max "
                f"{case['selected_prefix_at_alpha_max_mean']:.4f}, and hidden fraction "
                f"{case['selected_hidden_fraction']:.1%}."
            )
        else:
            reads.append(
                f"`{case['label']}` selected support is present but weakly decisive: selected weight share mean "
                f"{case['selected_weight_share_mean']:.4f}, top contributor selected on "
                f"{case['top_contributor_selected_fraction']:.1%} of sampled target rays."
            )
        reads.append(
            f"Sampled-ray normal/forced/oracle PSNR is "
            f"{case['normal_black_psnr']:.3f}/{case['forced_alpha_1_psnr']:.3f}/"
            f"{case['target_background_oracle_psnr']:.3f}."
        )
    return " ".join(reads)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = [
        [
            "label",
            "points",
            "normal",
            "forced",
            "target-bg",
            "alpha",
            "sel alpha",
            "sel weight",
            "sel share",
            "sel prefix",
            "absent",
            "hidden",
            "top selected",
        ]
    ]
    for case in result["cases"]:
        rows.append(
            [
                case["label"],
                str(case["point_count"]),
                f"{case['normal_black_psnr']:.3f}",
                f"{case['forced_alpha_1_psnr']:.3f}",
                f"{case['target_background_oracle_psnr']:.3f}",
                f"{case['alpha_mean']:.4f}",
                f"{case['selected_alpha_max_mean']:.4f}",
                f"{case['selected_weight_sum_mean']:.4f}",
                f"{case['selected_weight_share_mean']:.4f}",
                f"{case['selected_prefix_at_alpha_max_mean']:.4f}",
                f"{case['selected_absent_fraction']:.3f}",
                f"{case['selected_hidden_fraction']:.3f}",
                f"{case['top_contributor_selected_fraction']:.3f}",
            ]
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |"

    lines = [
        "# STAR UVT Visibility Prefix Tape Diagnostic",
        "",
        f"Date: {result['date']}",
        "",
        "## Purpose",
        "",
        "For selected support-target rays, reconstruct the alpha-over prefix tape:",
        "contributors sorted by depth, prefix transmittance before each contributor,",
        "per-contributor weight, and whether the selected born tubes are absent,",
        "prefix-hidden, or simply not dominant.",
        "",
        "## Results",
        "",
        fmt(rows[0]),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in rows[1:])
    lines.extend(["", "## Read", "", result["read"], "", "## Inputs", ""])
    for case in result["cases"]:
        lines.extend(
            [
                f"- `{case['label']}` config: `{case['config_path']}`",
                f"- `{case['label']}` checkpoint: `{case['checkpoint']}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="label=config.jsonc or config.jsonc")
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--date", default="2026-05-26")
    args = parser.parse_args()
    cases = [
        _analyze_case(
            label,
            path,
            max_points=int(args.max_points),
            top_k=int(args.top_k),
            sample_count=int(args.sample_count),
        )
        for label, path in (_parse_case(raw) for raw in args.case)
    ]
    result = {
        "date": args.date,
        "max_points": int(args.max_points),
        "top_k": int(args.top_k),
        "sample_count": int(args.sample_count),
        "cases": cases,
    }
    result["read"] = _make_read(cases)
    write_report_json(Path(args.out_json), result)
    _write_markdown(Path(args.out_md), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
