from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import write_report_json, write_report_text
from research_experiments.star_uvt_feature_tubes.dense_alpha_failure_diagnostic import _make_case
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    chunked_uvt_config,
    direct_atomic_feature_backward,
    direct_atomic_feature_backward_cached_bins,
    render_uvt_feature_alpha_all_pixels_with_bins,
    render_uvt_feature_tubes,
    shift_ma_for_frame_chunk,
)
from star_uvt_runtime import sync_device as _sync_device


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return max(values) if values else 0.0


def _time_call(device: torch.device, fn: Any) -> tuple[Any, float]:
    _sync_device(device)
    t0 = time.perf_counter()
    out = fn()
    _sync_device(device)
    return out, (time.perf_counter() - t0) * 1000.0


def _chunk_inputs(case: dict[str, Any], frame_start: int, chunk_frames: int) -> tuple[Any, ...]:
    model = case["model"]
    uvt_config = case["uvt_config"]
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
    if chunk_frames == int(uvt_config.frames):
        return ma, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature, uvt_config
    ma_chunk = shift_ma_for_frame_chunk(
        ma,
        global_frames=uvt_config.frames,
        frame_start=frame_start,
        chunk_frames=chunk_frames,
    )
    return (
        ma_chunk,
        q_uvt,
        depth0.detach(),
        depth_beta.detach(),
        opacity,
        feature,
        chunked_uvt_config(uvt_config, chunk_frames=chunk_frames),
    )


def _estimate_bytes(chunk_frames: int, height: int, width: int, feature_dim: int) -> dict[str, int]:
    pixels = int(chunk_frames * height * width)
    return {
        "dense_feature_image": pixels * feature_dim * 4,
        "dense_zero_grad_feature": pixels * feature_dim * 4,
        "alpha_image": pixels * 4,
        "alpha_sparse_pixel_ids": pixels * 4,
        "alpha_sparse_dummy_feature_values": pixels * 4,
        "alpha_sparse_zero_grad_feature": pixels * 4,
    }


def _profile_chunk(
    case: dict[str, Any],
    *,
    frame_start: int,
    chunk_frames: int,
    alpha_target: float,
    backward_mode: str,
) -> dict[str, Any]:
    device = case["device"]
    frames = int(case["feature_config"].frames)
    height = int(case["feature_config"].height)
    width = int(case["feature_config"].width)
    feature_dim = int(case["feature_config"].feature_dim)
    loss_denominator = float(frames * height * width)
    ma, q_uvt, depth0, depth_beta, opacity, feature, config = _chunk_inputs(case, frame_start, chunk_frames)

    dense_render, dense_render_ms = _time_call(
        device,
        lambda: render_uvt_feature_tubes(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            feature,
            config,
            return_bins=True,
        ),
    )
    alpha_render, alpha_render_ms = _time_call(
        device,
        lambda: render_uvt_feature_alpha_all_pixels_with_bins(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            config,
        ),
    )

    alpha_diff = dense_render.alpha - float(alpha_target)
    grad_alpha = ((2.0 / loss_denominator) * alpha_diff).contiguous()
    dense_grad_feature = torch.zeros_like(dense_render.feature_image)
    dummy_feature = torch.zeros((ma.shape[0], 1), dtype=torch.float32, device=device)
    dummy_grad_feature = torch.zeros(
        (int(config.frames), 1, int(config.height), int(config.width)),
        dtype=torch.float32,
        device=device,
    )

    dense_current_grads, dense_current_backward_ms = _time_call(
        device,
        lambda: direct_atomic_feature_backward(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            feature,
            dense_grad_feature,
            grad_alpha,
            config,
            backward_mode=backward_mode,
        ),
    )
    dense_cached_grads, dense_cached_backward_ms = _time_call(
        device,
        lambda: direct_atomic_feature_backward_cached_bins(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            feature,
            dense_grad_feature,
            grad_alpha,
            dense_render.tile_counts,
            dense_render.tile_tube_ids,
            dense_render.tile_depths,
            dense_render.tile_unstable,
            config,
            backward_mode=backward_mode,
        ),
    )
    alpha_grads, alpha_backward_ms = _time_call(
        device,
        lambda: direct_atomic_feature_backward_cached_bins(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            dummy_feature,
            dummy_grad_feature,
            grad_alpha,
            alpha_render.tile_counts,
            alpha_render.tile_tube_ids,
            alpha_render.tile_depths,
            alpha_render.tile_unstable,
            config,
            backward_mode=backward_mode,
        ),
    )

    alpha_max_abs_diff = float((dense_render.alpha - alpha_render.alpha).abs().max().detach().cpu().item())
    grad_names = ("ma", "q_uvt", "opacity")
    dense_current_vs_cached = {
        name: float((dense_current_grads[i] - dense_cached_grads[i]).abs().max().detach().cpu().item())
        for i, name in enumerate(grad_names)
    }
    dense_cached_vs_alpha = {
        name: float((dense_cached_grads[i] - alpha_grads[i]).abs().max().detach().cpu().item())
        for i, name in enumerate(grad_names)
    }
    tile_count = dense_render.tile_counts.detach()
    dense_overflow = int(dense_render.tile_overflow.sum().detach().cpu().item())
    alpha_overflow = int(alpha_render.tile_overflow.sum().detach().cpu().item())
    return {
        "frame_start": int(frame_start),
        "chunk_frames": int(chunk_frames),
        "alpha_loss": float(alpha_diff.square().sum().detach().cpu().item() / loss_denominator),
        "alpha_max_abs_diff": alpha_max_abs_diff,
        "dense_current_vs_cached_grad_max_abs": dense_current_vs_cached,
        "dense_cached_vs_alpha_grad_max_abs": dense_cached_vs_alpha,
        "dense_overflow_tiles": dense_overflow,
        "alpha_overflow_tiles": alpha_overflow,
        "tile_count_max": int(tile_count.max().detach().cpu().item()),
        "tile_count_p95": float(torch.quantile(tile_count.to(torch.float32), 0.95).detach().cpu().item()),
        "timing_ms": {
            "dense_render_with_bins": dense_render_ms,
            "dense_current_backward_rebin_f32": dense_current_backward_ms,
            "dense_cached_backward_f32": dense_cached_backward_ms,
            "alpha_sparse_f1_render_with_bins": alpha_render_ms,
            "alpha_cached_backward_f1": alpha_backward_ms,
            "dense_current_total": dense_render_ms + dense_current_backward_ms,
            "dense_cached_total": dense_render_ms + dense_cached_backward_ms,
            "alpha_sparse_total": alpha_render_ms + alpha_backward_ms,
        },
        "estimated_bytes": _estimate_bytes(chunk_frames, height, width, feature_dim),
    }


def _summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    timing_keys = tuple(chunks[0]["timing_ms"].keys()) if chunks else ()
    return {
        "chunk_count": len(chunks),
        "timing_ms_sum": {key: sum(float(row["timing_ms"][key]) for row in chunks) for key in timing_keys},
        "timing_ms_mean": {key: _mean([float(row["timing_ms"][key]) for row in chunks]) for key in timing_keys},
        "timing_ms_max": {key: _max([float(row["timing_ms"][key]) for row in chunks]) for key in timing_keys},
        "alpha_max_abs_diff_max": _max([float(row["alpha_max_abs_diff"]) for row in chunks]),
        "dense_cached_vs_alpha_grad_max_abs": {
            name: _max([float(row["dense_cached_vs_alpha_grad_max_abs"][name]) for row in chunks])
            for name in ("ma", "q_uvt", "opacity")
        },
        "dense_current_vs_cached_grad_max_abs": {
            name: _max([float(row["dense_current_vs_cached_grad_max_abs"][name]) for row in chunks])
            for name in ("ma", "q_uvt", "opacity")
        },
        "dense_overflow_tiles_sum": sum(int(row["dense_overflow_tiles"]) for row in chunks),
        "alpha_overflow_tiles_sum": sum(int(row["alpha_overflow_tiles"]) for row in chunks),
        "tile_count_max": max(int(row["tile_count_max"]) for row in chunks) if chunks else 0,
        "tile_count_p95_max": _max([float(row["tile_count_p95"]) for row in chunks]),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    timing_sum = summary["timing_ms_sum"]
    timing_mean = summary["timing_ms_mean"]
    dense_total = float(timing_sum["dense_current_total"])
    alpha_total = float(timing_sum["alpha_sparse_total"])
    ratio = dense_total / alpha_total if alpha_total > 0.0 else math.inf
    lines = [
        "# STAR UVT Alpha-Only Visibility Profile",
        "",
        "## Verdict",
        "",
        f"- Pass: `{report['pass']}`",
        f"- Chunks profiled: `{summary['chunk_count']}`",
        f"- Alpha parity max abs diff: `{summary['alpha_max_abs_diff_max']:.6g}`",
        "- Dense cached vs alpha F1 backward max abs diff: "
        f"ma `{summary['dense_cached_vs_alpha_grad_max_abs']['ma']:.6g}`, "
        f"q `{summary['dense_cached_vs_alpha_grad_max_abs']['q_uvt']:.6g}`, "
        f"opacity `{summary['dense_cached_vs_alpha_grad_max_abs']['opacity']:.6g}`",
        f"- Dense-current total: `{dense_total:.1f}ms`; alpha-sparse F1 total: `{alpha_total:.1f}ms`; ratio dense/alpha: `{ratio:.3f}x`",
        "",
        "## Timing",
        "",
        "| bucket | sum ms | mean chunk ms | max chunk ms |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "dense_render_with_bins",
        "dense_current_backward_rebin_f32",
        "dense_cached_backward_f32",
        "alpha_sparse_f1_render_with_bins",
        "alpha_cached_backward_f1",
        "dense_current_total",
        "dense_cached_total",
        "alpha_sparse_total",
    ):
        lines.append(
            f"| `{key}` | {float(timing_sum[key]):.1f} | {float(timing_mean[key]):.1f} | "
            f"{float(summary['timing_ms_max'][key]):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Tile And Memory",
            "",
            f"- Dense overflow tiles: `{summary['dense_overflow_tiles_sum']}`",
            f"- Alpha overflow tiles: `{summary['alpha_overflow_tiles_sum']}`",
            f"- Tile max / p95 max: `{summary['tile_count_max']}` / `{summary['tile_count_p95_max']:.1f}`",
            "- Per-chunk estimated dense feature image bytes: "
            f"`{report['per_chunk_estimated_bytes']['dense_feature_image']}`",
            "- Per-chunk estimated alpha-sparse feature-value plus pixel-id bytes: "
            f"`{report['per_chunk_estimated_bytes']['alpha_sparse_dummy_feature_values'] + report['per_chunk_estimated_bytes']['alpha_sparse_pixel_ids']}`",
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Inputs",
            "",
            f"- Config: `{report['config_path']}`",
            f"- Checkpoint: `{report['checkpoint']}`",
            f"- Backward mode: `{report['backward_mode']}`",
            f"- Alpha target: `{report['alpha_target']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alpha-target", type=float, default=0.75)
    parser.add_argument("--backward-mode", default="gradcache_skip_feature_grad")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means all chunks")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    case = _make_case(args.config)
    device = case["device"]
    if device.type != "mps":
        raise RuntimeError("alpha-only visibility profile requires MPS")
    frames = int(case["feature_config"].frames)
    chunk_size = case["cfg"]["train"]["frame_chunk_size"]
    chunk_size = frames if chunk_size is None else min(int(chunk_size), frames)
    starts = list(range(0, frames, chunk_size))
    if int(args.max_chunks) > 0:
        starts = starts[: int(args.max_chunks)]

    chunks = [
        _profile_chunk(
            case,
            frame_start=frame_start,
            chunk_frames=min(chunk_size, frames - frame_start),
            alpha_target=float(args.alpha_target),
            backward_mode=str(args.backward_mode),
        )
        for frame_start in starts
    ]
    summary = _summarize_chunks(chunks)
    alpha_ok = summary["alpha_max_abs_diff_max"] <= 1.0e-5
    grad_ok = max(float(v) for v in summary["dense_cached_vs_alpha_grad_max_abs"].values()) <= 5.0e-4
    overflow_ok = summary["dense_overflow_tiles_sum"] == 0 and summary["alpha_overflow_tiles_sum"] == 0
    dense_total = float(summary["timing_ms_sum"]["dense_current_total"])
    alpha_total = float(summary["timing_ms_sum"]["alpha_sparse_total"])
    speed_ok = alpha_total < dense_total
    if speed_ok:
        interpretation = (
            "The existing sparse-pixel F1 path is a viable alpha-only speed gate: it preserves alpha/backward "
            "parity and is faster than the dense F32 alpha baseline for the profiled chunks. It is still a "
            "visibility diagnostic, not a quality promotion."
        )
    else:
        interpretation = (
            "The existing sparse-pixel F1 path preserves the alpha-only math but does not beat the dense F32 "
            "alpha baseline for the profiled chunks. Avoid promoting it into the trainer; a real tile-level "
            "alpha kernel or support-model change is needed before spending more on dense alpha losses."
        )
    report = {
        "config_path": str(args.config),
        "checkpoint": case["checkpoint"],
        "alpha_target": float(args.alpha_target),
        "backward_mode": str(args.backward_mode),
        "chunk_size": int(chunk_size),
        "profiled_frame_starts": starts,
        "chunks": chunks,
        "summary": summary,
        "per_chunk_estimated_bytes": chunks[0]["estimated_bytes"] if chunks else {},
        "pass": bool(alpha_ok and grad_ok and overflow_ok and speed_ok),
        "checks": {
            "alpha_parity": bool(alpha_ok),
            "backward_parity": bool(grad_ok),
            "no_overflow": bool(overflow_ok),
            "speedup": bool(speed_ok),
        },
        "interpretation": interpretation,
    }
    write_report_json(args.out_json, report)
    _write_markdown(args.out_md, report)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "pass": report["pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
