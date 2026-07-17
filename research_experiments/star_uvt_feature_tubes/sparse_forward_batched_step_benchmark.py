from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


try:
    from .report_artifacts import ROOT, distribution_stats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, distribution_stats, write_report_json, write_report_text
from research_experiments.star_uvt_feature_tubes.sparse_forward_batched_target_vjp_profile import (
    DEFAULT_CONFIG,
    _load_sparse_forward_chunks,
    _profile_batched,
    _target_context,
)
from research_experiments.star_uvt_feature_tubes.star_uvt_feature1_wholegraph_profile import (
    _chunk_render_inputs,
    _load_case,
)
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    direct_atomic_feature_sparse_pixels_backward_cached_bins,
    render_uvt_feature_sparse_pixels_with_bins,
)
from star_uvt_checkpoints import set_optimizer_lr as _set_optimizer_lr
from star_uvt_runtime import sync_device as _sync_device
from star_uvt_schedules import (
    _feature_target_weights_for_step,
    _feature_target_weight_schedule,
    _optimizer_lr_for_step,
    _optimizer_lr_schedule,
)
from star_uvt_sparse_grid import _sparse_target_grid_pixel_ids


DEFAULT_OUT_BASE = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_step_benchmark"


def _mean_without_first(rows: list[dict[str, float]], key: str) -> float | None:
    if len(rows) <= 1:
        return None
    values = [float(row[key]) for row in rows[1:] if key in row]
    return None if not values else statistics.fmean(values)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _model_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    return {
        f"model.{name}": float(param.grad.detach().norm().cpu().item())
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _render_step_chunks(case: dict[str, Any], chunk_size: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    feature_config = case["feature_config"]
    uvt_config = case["uvt_config"]
    model = case["model"]
    device = case["device"]
    target_feature = case["target_feature"]
    chunks: list[dict[str, Any]] = []
    render_ms = 0.0
    for frame_start in range(0, int(feature_config.frames), chunk_size):
        chunk_frames = min(chunk_size, int(feature_config.frames) - frame_start)
        target_feature_chunk = target_feature.chunk(frame_start, chunk_frames)
        input_shape = (
            int(chunk_frames),
            int(feature_config.feature_dim),
            int(feature_config.height),
            int(feature_config.width),
        )
        target_shape = tuple(int(item) for item in target_feature_chunk.shape)
        pixel_ids = _sparse_target_grid_pixel_ids(
            input_shape=input_shape,
            target_shape=target_shape,
            mode=target_feature.grid_mode,
            device=device,
        )
        render_inputs, render_config = _chunk_render_inputs(model, uvt_config, frame_start, chunk_frames)
        _sync_device(device)
        t0 = time.perf_counter()
        render = render_uvt_feature_sparse_pixels_with_bins(*render_inputs, pixel_ids, render_config)
        _sync_device(device)
        render_ms += (time.perf_counter() - t0) * 1000.0
        chunks.append(
            {
                "frame_start": int(frame_start),
                "chunk_frames": int(chunk_frames),
                "input_shape": input_shape,
                "target_shape": target_shape,
                "pixel_ids": pixel_ids,
                "feature_values": render.feature_values,
                "render_inputs": render_inputs,
                "render_config": render_config,
                "tile_counts": render.tile_counts,
                "tile_tube_ids": render.tile_tube_ids,
                "tile_depths": render.tile_depths,
                "tile_unstable": render.tile_unstable,
            }
        )
    return chunks, {"render_forward_ms": render_ms}


def _run_step(
    case: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    chunk_size: int,
    global_step: int,
) -> dict[str, Any]:
    cfg = case["cfg"]
    model = case["model"]
    device = case["device"]
    weight_stage = _feature_target_weights_for_step(_feature_target_weight_schedule(cfg), global_step)
    lr_stage = _optimizer_lr_for_step(_optimizer_lr_schedule(cfg), global_step)
    _set_optimizer_lr(optimizer, lr_stage.lr)
    context = _target_context(case)
    context["feature_loss_weight"] = float(weight_stage.loss_weight)
    context["rgb_probe_loss_weight"] = float(weight_stage.rgb_probe_loss_weight)
    optimizer.zero_grad(set_to_none=True)
    _sync_device(device)
    step_t0 = time.perf_counter()
    chunks, render_timing = _render_step_chunks(case, chunk_size)
    _sync_device(device)
    loss_t0 = time.perf_counter()
    batched = _profile_batched(case, chunks, context)
    _sync_device(device)
    loss_t1 = time.perf_counter()
    renderer_backward_ms = 0.0
    param_backward_ms = 0.0
    for chunk, pack in zip(chunks, batched["packs"], strict=True):
        render_inputs = chunk["render_inputs"]
        _sync_device(device)
        renderer_t0 = time.perf_counter()
        grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = direct_atomic_feature_sparse_pixels_backward_cached_bins(
            *render_inputs,
            pack["pixel_ids"],
            pack["grad_feature_values"],
            pack["grad_alpha_values"],
            chunk["tile_counts"],
            chunk["tile_tube_ids"],
            chunk["tile_depths"],
            chunk["tile_unstable"],
            chunk["render_config"],
        )
        _sync_device(device)
        renderer_t1 = time.perf_counter()
        torch.autograd.backward(
            (render_inputs[0], render_inputs[1], render_inputs[4], render_inputs[5]),
            (grad_ma, grad_q, grad_opacity, grad_feature),
        )
        _sync_device(device)
        param_t1 = time.perf_counter()
        renderer_backward_ms += (renderer_t1 - renderer_t0) * 1000.0
        param_backward_ms += (param_t1 - renderer_t1) * 1000.0
    _sync_device(device)
    opt_t0 = time.perf_counter()
    optimizer.step()
    _sync_device(device)
    opt_t1 = time.perf_counter()
    return {
        "global_step": int(global_step),
        "loss": float(batched["loss"]),
        "feature_target_loss": float(batched["feature_target_loss"]),
        "rgb_probe_loss": float(batched["rgb_probe_loss"]),
        "timing_ms": {
            "render_forward_ms": float(render_timing["render_forward_ms"]),
            "batched_loss_vjp_wall_ms": (loss_t1 - loss_t0) * 1000.0,
            "feature_target_ms": float(batched["timing_ms"]["feature_target_ms"]),
            "rgb_probe_loss_ms": float(batched["timing_ms"]["rgb_probe_loss_ms"]),
            "image_vjp_ms": float(batched["timing_ms"]["image_vjp_ms"]),
            "renderer_backward_ms": renderer_backward_ms,
            "param_backward_ms": param_backward_ms,
            "backward_ms": renderer_backward_ms + param_backward_ms,
            "optimizer_ms": (opt_t1 - opt_t0) * 1000.0,
            "step_ms": (opt_t1 - step_t0) * 1000.0,
        },
        "grad_norms": _model_grad_norms(model),
    }


def benchmark(config_path: Path, *, steps: int) -> dict[str, Any]:
    case = _load_case(config_path)
    cfg = case["cfg"]
    feature_config = case["feature_config"]
    chunk_size_cfg = cfg["train"]["frame_chunk_size"]
    chunk_size = int(feature_config.frames) if chunk_size_cfg is None else min(int(chunk_size_cfg), int(feature_config.frames))
    if chunk_size <= 0:
        raise ValueError("train.frame_chunk_size must be null or positive")
    optimizer = torch.optim.Adam(case["model"].parameters(), lr=float(cfg["train"]["lr"]))
    # Match the loaded checkpoint optimizer-LR semantics used by the trainer: the
    # config schedule wins after resume.
    rows = []
    for step in range(steps):
        rows.append(_run_step(case, optimizer=optimizer, chunk_size=chunk_size, global_step=int(cfg["train"]["global_step_offset"]) + step))
    timing_keys = (
        "render_forward_ms",
        "batched_loss_vjp_wall_ms",
        "feature_target_ms",
        "rgb_probe_loss_ms",
        "image_vjp_ms",
        "renderer_backward_ms",
        "param_backward_ms",
        "backward_ms",
        "optimizer_ms",
        "step_ms",
    )
    timing = {key: distribution_stats([float(row["timing_ms"][key]) for row in rows]) for key in timing_keys}
    no_first = {key: _mean_without_first([row["timing_ms"] for row in rows], key) for key in timing_keys}
    preload = _load_sparse_forward_chunks(case, chunk_size)
    pass_flag = (
        len(rows) == steps
        and rows[-1]["loss"] < rows[0]["loss"]
        and int(preload["tile_overflow_sum"]) == 0
        and all(row["grad_norms"].get("model.raw_feature", 0.0) > 0.0 for row in rows)
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate": "star_uvt_sparse_forward_batched_step_benchmark",
        "config": str(config_path),
        "steps": int(steps),
        "frames": int(feature_config.frames),
        "size": int(feature_config.height),
        "feature_dim": int(feature_config.feature_dim),
        "tubes": int(cfg["feature_uvt"]["tube_count"]),
        "chunk_size": int(chunk_size),
        "chunk_count": int(len(rows) and int(feature_config.frames) // int(chunk_size)),
        "start_loss": rows[0]["loss"],
        "end_loss": rows[-1]["loss"],
        "start_feature_target_loss": rows[0]["feature_target_loss"],
        "end_feature_target_loss": rows[-1]["feature_target_loss"],
        "start_rgb_probe_loss": rows[0]["rgb_probe_loss"],
        "end_rgb_probe_loss": rows[-1]["rgb_probe_loss"],
        "timing_ms": timing,
        "no_first_timing_ms": no_first,
        "step_rows": rows,
        "sparse_pixel_count": int(preload["total_sparse_pixels"]),
        "sparse_pixel_fraction": float(preload["total_sparse_pixels"]) / float(preload["total_dense_pixels"]),
        "max_tile_count": int(preload["max_tile_count"]),
        "tile_overflow_sum": int(preload["tile_overflow_sum"]),
        "pass": pass_flag,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    timing = result["timing_ms"]
    no_first = result["no_first_timing_ms"]
    rows = []
    for key in (
        "step_ms",
        "render_forward_ms",
        "batched_loss_vjp_wall_ms",
        "feature_target_ms",
        "rgb_probe_loss_ms",
        "image_vjp_ms",
        "backward_ms",
        "renderer_backward_ms",
        "param_backward_ms",
        "optimizer_ms",
    ):
        rows.append(
            "| "
            + " | ".join(
                (
                    key,
                    _fmt(timing[key]["mean"]),
                    _fmt(no_first[key]),
                    _fmt(timing[key]["min"]),
                    _fmt(timing[key]["max"]),
                )
            )
            + " |"
        )
    lines = [
        "# STAR UVT Sparse-Forward Batched Step Benchmark",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Scope",
        "",
        "Runs a 5-step optimizer benchmark from the same 1300-step checkpoint using sparse feature forward, batched target-grid/frozen-probe loss+VJP across all chunks, and the existing sparse-pixel renderer backward.",
        "This is a harness gate before trainer integration.",
        "",
        "## Result",
        "",
        "| metric | mean | no-first mean | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Validation",
        "",
        f"- pass: `{result['pass']}`",
        f"- loss: `{result['start_loss']:.6f} -> {result['end_loss']:.6f}`",
        f"- feature loss: `{result['start_feature_target_loss']:.6f} -> {result['end_feature_target_loss']:.6f}`",
        f"- RGB-probe loss: `{result['start_rgb_probe_loss']:.6f} -> {result['end_rgb_probe_loss']:.6f}`",
        f"- sparse pixels: `{result['sparse_pixel_count']}` (`{result['sparse_pixel_fraction']:.6f}` of dense)",
        f"- tile overflow / max tile: `{result['tile_overflow_sum']}` / `{result['max_tile_count']}`",
        "",
        "## Decision",
        "",
    ]
    if result["pass"]:
        lines.extend(
            [
                "The batched loss+VJP ordering is trainable in the benchmark harness and should be compared against the repeat-3 trainer distribution.",
                "If it beats the repeat-3 no-first step distribution after a trainer-mode integration, it can become the next selected fast path; otherwise keep it as a native-kernel preflight.",
            ]
        )
    else:
        lines.append("Do not integrate this ordering until the validation failure above is fixed.")
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    args = parser.parse_args()
    result = benchmark(args.config, steps=args.steps)
    args.out_base.parent.mkdir(parents=True, exist_ok=True)
    write_report_json(args.out_base.with_suffix(".json"), result)
    write_markdown(args.out_base.with_suffix(".md"), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
